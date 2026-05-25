// lib/screens/study/study_screen.dart
// StudyScreen — subject cards grid with visual identity cards
// StudyChaptersPage — Chapters list - each card expands to reveal
// StudyChapterHub — Full hub for one chapter/sub-chapter
// ─────────────────────────────────────────────────────────────────

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:go_router/go_router.dart';

import '../../services/app_state.dart';
import '../../services/curriculum_catalog_service.dart';
import '../../services/daily_plan_service.dart';
import '../../services/firestore_service.dart';
import '../../models/daily_plan_task.dart';
import '../../theme/app_theme.dart';
import '../../theme/subject_themes.dart';
import '../../utils/layout_utils.dart';

import '../../widgets/common/rose_loader.dart';
import '../../widgets/common/daily_plan_panel.dart';
import 'glossary_screen.dart';
import '../../services/cambridge_notes_service.dart';

class StudyScreen extends ConsumerStatefulWidget {
  const StudyScreen({super.key});

  @override
  ConsumerState<StudyScreen> createState() => _StudyScreenState();
}

class _StudyScreenState extends ConsumerState<StudyScreen> {
  Map<String, List<String>> _subjects = {};
  List<String> _allBoardSubjects = const [];
  List<String> _selectedSubjects = const [];
  List<_SubjectPriority> _rankedSubjects = const [];
  bool _loading = true;
  String _lastAuthSignature = '';
  ProviderSubscription<AuthState>? _authSub;
  final _catalog = CurriculumCatalogService.instance;
  final _dailyPlanService = DailyPlanService();
  final Map<String, String> _subjectCodeMap = {};

  @override
  void initState() {
    super.initState();

    _authSub = ref.listenManual<AuthState>(authStateProvider, (_, next) {
      if (next.isLoading) return;
      final board = next.user?.board.trim() ?? '';
      final subjects = (next.user?.subjects ?? const <String>[])
          .map(normalizeSubject)
          .toList()
        ..sort();
      final signature = '$board|${subjects.join('||')}';
      if (signature == _lastAuthSignature && _rankedSubjects.isNotEmpty) return;
      _lastAuthSignature = signature;
      _load();
    });
    _load().timeout(const Duration(seconds: 10), onTimeout: () {
      if (mounted) {
        setState(() => _loading = false);
      }
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Force refresh when dependencies change
    _load();
  }

  @override
  void dispose() {
    _authSub?.close();
    super.dispose();
  }

  static String normalizeSubject(String name) {
    final cleaned = name.trim();
    if (cleaned.isEmpty) return '';
    final parts = cleaned.split(' ');
    if (parts.length == 1) return cleaned;
    final last = parts.last;
    if (last.length <= 3) {
      return parts.take(parts.length - 1).join(' ');
    }
    return cleaned;
  }

  Future<void> _load() async {
    try {
      await Future.wait([
        _catalog.initializeLocalData(),
        SharedPreferences.getInstance(),
        _catalog.loadBoards(),
      ], eagerError: false);

      final profile = ref.read(authStateProvider).user;
      final gateProfile = ref.read(authGateProvider).valueOrNull?.profile;
      final prefs = await SharedPreferences.getInstance();
      var board = [
        profile?.board ?? '',
        gateProfile?.board ?? '',
        prefs.getString('userBoard') ?? '',
      ].firstWhere((value) => value.trim().isNotEmpty, orElse: () => '');

      if (board.isEmpty) {
        board = 'IGCSE';
        await prefs.setString('userBoard', board);
      }

      final profileSubjects = profile?.subjects ?? [];
      final prefsSubjects = prefs.getStringList('userSubjects') ?? [];
      var selectedSubjects = <String>{
        ...profileSubjects.map(normalizeSubject),
        ...prefsSubjects.map(normalizeSubject),
      }.where((s) => s.trim().isNotEmpty).toList()
        ..sort();

      final uid = profile?.uid ?? gateProfile?.uid;

      if ((selectedSubjects.isEmpty || board.isEmpty) && uid != null) {
        try {
          final doc = await AxonPaths.privateUserDoc(uid).get();
          final remote = doc.data();
          if (remote != null) {
            final remoteBoard = (remote['board'] ?? '').toString().trim();
            final remoteSubjects = (remote['subjects'] as List? ?? const [])
                .map((e) => normalizeSubject(e.toString()))
                .where((item) => item.trim().isNotEmpty)
                .toSet()
                .toList()
              ..sort();

            if (board.trim().isEmpty && remoteBoard.isNotEmpty) {
              board = remoteBoard;
              await prefs.setString('userBoard', remoteBoard);
            }
            if (selectedSubjects.isEmpty && remoteSubjects.isNotEmpty) {
              selectedSubjects = remoteSubjects;
              await prefs.setStringList('userSubjects', remoteSubjects);
            }
          }
        } catch (_) {}
      }

      final canonicalBoard = await _catalog.canonicalBoardLabel(board);
      final allSubjectsForBoard =
          await _catalog.subjectsForBoard(canonicalBoard);
      _allBoardSubjects = allSubjectsForBoard;

      final allSubjectsData = await _catalog.getAllSubjects();
      for (final s in allSubjectsData) {
        if (_allBoardSubjects.contains(s.name)) {
          _subjectCodeMap[s.name] = s.code;
        }
      }

      if (selectedSubjects.isEmpty && _allBoardSubjects.isNotEmpty) {
        selectedSubjects = _allBoardSubjects.take(5).toList();
      }

      final visible = <String, List<String>>{};
      for (final subject in selectedSubjects) {
        final subjectCode = _subjectCodeMap[subject] ?? '';
        final chapters = await _catalog.getChapters(subjectCode);
        visible[subject] = chapters.map((c) => c.title).toList();
      }

      final ranked = <_SubjectPriority>[];
      for (var i = 0; i < selectedSubjects.length; i++) {
        final subject = selectedSubjects[i];
        final chapters = visible[subject] ?? [];
        final subjectCode = _subjectCodeMap[subject] ?? '';

        ranked.add(_SubjectPriority(
          subject: subject,
          subjectCode: subjectCode,
          chapters: chapters,
          lastStudiedAt: null,
          nextExamAt: null,
          totalStudySeconds: 0,
          progress: 0.0,
          urgencyScore: i.toDouble(),
          isPrimarySubject: i == 0,
        ));
      }

      ranked.sort((a, b) => a.urgencyScore.compareTo(b.urgencyScore));

      if (mounted) {
        setState(() {
          _subjects = visible;
          _selectedSubjects = selectedSubjects;
          _rankedSubjects = ranked;
          _loading = false;
        });
      }
    } catch (e) {
      debugPrint('StudyScreen._load error: $e');
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  void _showSubjectOptions(BuildContext context, String subject) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (sheetContext) => Container(
        margin: const EdgeInsets.symmetric(horizontal: 12),
        decoration: BoxDecoration(
          color: AxonColors.background,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          border: Border.all(color: AxonColors.divider, width: 0.5),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              margin: const EdgeInsets.only(top: 12),
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.white24,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const SizedBox(width: 20),
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: AxonColors.accent.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(Icons.auto_stories_rounded,
                      color: AxonColors.accent, size: 22),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(subject,
                          style: GoogleFonts.googleSans(
                              color: AxonColors.textPrimary,
                              fontSize: 16,
                              fontWeight: FontWeight.w700)),
                      Text('Subject Options',
                          style: GoogleFonts.googleSans(
                              color: AxonColors.textTertiary, fontSize: 11)),
                    ],
                  ),
                ),
                GestureDetector(
                  onTap: () => Navigator.pop(sheetContext),
                  child: Container(
                    width: 32,
                    height: 32,
                    margin: const EdgeInsets.only(right: 12),
                    decoration: BoxDecoration(
                        color: AxonColors.surfaceElevated,
                        borderRadius: BorderRadius.circular(8)),
                    child: Icon(Icons.close_rounded,
                        color: AxonColors.textTertiary, size: 18),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _optionTile(sheetContext, Icons.push_pin_rounded, AxonColors.accent,
                'Pin to Top', 'Keep this subject at the top of your list', () {
              _reorderSubject(subject, 0);
              Navigator.pop(sheetContext);
            }),
            _optionTile(
                sheetContext,
                Icons.star_rounded,
                Colors.amber,
                'Mark as Priority',
                'Prioritize this subject in your daily plan', () {
              Navigator.pop(sheetContext);
            }),
            _optionTile(
                sheetContext,
                Icons.refresh_rounded,
                AxonColors.textSecondary,
                'Reset Mastery',
                'Clear progress data for this subject', () {
              Navigator.pop(sheetContext);
            }),
            _optionTile(
                sheetContext,
                Icons.delete_outline_rounded,
                const Color(0xFFE11D48),
                'Remove Subject',
                'Remove this subject from your list', () {
              _removeSubject(subject);
              Navigator.pop(sheetContext);
            }),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  Widget _optionTile(BuildContext context, IconData icon, Color iconColor,
      String title, String subtitle, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        margin: const EdgeInsets.symmetric(horizontal: 8),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: iconColor.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: iconColor, size: 18),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style: GoogleFonts.googleSans(
                          color: AxonColors.textPrimary,
                          fontSize: 13,
                          fontWeight: FontWeight.w600)),
                  Text(subtitle,
                      style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary, fontSize: 10)),
                ],
              ),
            ),
            Icon(Icons.chevron_right_rounded,
                color: AxonColors.textTertiary, size: 18),
          ],
        ),
      ),
    );
  }

  void _reorderSubject(String subject, int newIndex) {
    setState(() {
      _selectedSubjects.remove(subject);
      _selectedSubjects.insert(
          newIndex.clamp(0, _selectedSubjects.length), subject);
      _reloadRanked();
    });
  }

  void _removeSubject(String subject) {
    setState(() {
      _selectedSubjects.remove(subject);
      _subjects.remove(subject);
      _reloadRanked();
    });
  }

  void _reloadRanked() {
    final ranked = <_SubjectPriority>[];
    for (var i = 0; i < _selectedSubjects.length; i++) {
      final subj = _selectedSubjects[i];
      ranked.add(_SubjectPriority(
        subject: subj,
        subjectCode: _subjectCodeMap[subj] ?? '',
        chapters: _subjects[subj] ?? [],
        lastStudiedAt: null,
        nextExamAt: null,
        totalStudySeconds: 0,
        progress: 0.0,
        urgencyScore: i.toDouble(),
        isPrimarySubject: i == 0,
      ));
    }
    _rankedSubjects = ranked;
  }

  void _handleTaskTap(BuildContext context, DailyPlanTask task) {
    context.push(task.taskType.buildRoute(task.subject));
  }

  Future<void> _addSubject(String name) async {
    if (name.isEmpty) return;
    final normalized = normalizeSubject(name);
    if (_selectedSubjects.contains(normalized)) return;
    final next = [..._selectedSubjects, normalized];
    await ref.read(authStateProvider.notifier).updateStudySubjects(next);
    await _load();
  }

  void _showAddSubject() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: AxonColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (_) => _AddSubjectBottomSheet(
        availableSubjects: _allBoardSubjects
            .where((s) => !_selectedSubjects.contains(s))
            .toList(),
        subjectCodeMap: _subjectCodeMap,
        onSubjectSelected: _addSubject,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final authGate = ref.watch(authGateProvider).valueOrNull;
    final uid =
        authGate?.profile?.uid ?? ref.watch(authStateProvider).user?.uid;
    final isDark = AxonThemeMode.isDark;
    final bg = isDark ? const Color(0xFF000000) : const Color(0xFFFFFFFF);
    final textPrimary = isDark ? Colors.white : const Color(0xFF212529);

    return Scaffold(
      backgroundColor: bg,
      body: Container(
        color: bg,
        child: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'STUDY HUB',
                            style: GoogleFonts.googleSans(
                              color: textPrimary,
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      constraints:
                          const BoxConstraints(minWidth: 40, minHeight: 40),
                      padding: EdgeInsets.zero,
                      icon: Icon(Icons.add_rounded,
                          color: AxonColors.electricCyan, size: 24),
                      onPressed: _showAddSubject,
                      tooltip: 'Add Subject',
                    ),
                  ],
                ),
              ),
              Expanded(
                child: SingleChildScrollView(
                  padding:
                      EdgeInsets.only(bottom: bottomDockClearance(context)),
                  child: Column(
                    children: [
                      const SizedBox(height: 16),
                      if (uid != null) ...[
                        const SizedBox(height: 16),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 20),
                          child: DailyPlanPanel(
                              uid: uid,
                              service: _dailyPlanService,
                              onTaskTap: _handleTaskTap),
                        ),
                      ],
                      if (_rankedSubjects.isNotEmpty) ...[
                        const SizedBox(height: 12),
                        _QuickJumpStrip(subjects: _rankedSubjects),
                      ],
                      const SizedBox(height: 16),
                      _buildSubjectSection(),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSubjectSection() {
    if (_loading) {
      return const Center(child: RoseLoader(size: 24));
    }
    if (_subjects.isEmpty) {
      return _EmptyStudy(onAdd: _showAddSubject);
    }
    return _SubjectPriorityView(
      subjects: _rankedSubjects,
      onTap: (ranked) => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => StudyChaptersPage(
            subject: ranked.subject,
            subjectCode: ranked.subjectCode,
            chapters: ranked.chapters,
          ),
        ),
      ),
      onLongPress: (ranked) => _showSubjectOptions(context, ranked.subject),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Data model
// ─────────────────────────────────────────────────────────────────

class _SubjectPriority {
  final String subject;
  final String subjectCode;
  final List<String> chapters;
  final DateTime? lastStudiedAt;
  final DateTime? nextExamAt;
  final int totalStudySeconds;
  final double progress;
  final double urgencyScore;
  final bool isPrimarySubject;

  const _SubjectPriority({
    required this.subject,
    required this.subjectCode,
    required this.chapters,
    this.lastStudiedAt,
    this.nextExamAt,
    this.totalStudySeconds = 0,
    this.progress = 0.0,
    this.urgencyScore = 0.0,
    this.isPrimarySubject = false,
  });
}

// ─────────────────────────────────────────────────────────────────
// Quick Jump Strip
// ─────────────────────────────────────────────────────────────────

class _QuickJumpStrip extends StatelessWidget {
  final List<_SubjectPriority> subjects;

  const _QuickJumpStrip({required this.subjects});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 36,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount: subjects.length.clamp(0, 6),
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (ctx, i) {
          final s = subjects[i];
          return GestureDetector(
            onTap: () => context
                .push('/study/chapters/${Uri.encodeComponent(s.subject)}'),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: AxonColors.surfaceElevated,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: AxonColors.divider),
              ),
              child: Text(
                s.subject,
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Subject Priority View (list)
// ─────────────────────────────────────────────────────────────────

class _SubjectPriorityView extends StatelessWidget {
  final List<_SubjectPriority> subjects;
  final void Function(_SubjectPriority) onTap;
  final void Function(_SubjectPriority)? onLongPress;

  const _SubjectPriorityView({
    required this.subjects,
    required this.onTap,
    this.onLongPress,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: subjects
          .map((s) => _SubjectPriorityCard(
                subject: s,
                onTap: () => onTap(s),
                onLongPress: onLongPress != null ? () => onLongPress!(s) : null,
              ))
          .toList(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Subject Priority Card
// ─────────────────────────────────────────────────────────────────

class _SubjectPriorityCard extends StatelessWidget {
  final _SubjectPriority subject;
  final VoidCallback onTap;
  final VoidCallback? onLongPress;

  const _SubjectPriorityCard({
    required this.subject,
    required this.onTap,
    this.onLongPress,
  });

  @override
  Widget build(BuildContext context) {
    final theme = themeFor(subject.subject);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 6),
      child: GestureDetector(
        onTap: onTap,
        onLongPress: onLongPress,
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.white12),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: theme.primary.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(theme.icon, color: theme.primary, size: 20),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      subject.subject,
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '${subject.chapters.length} chapters',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
              if (subject.isPrimarySubject)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: AxonColors.electricCyan.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    'PRIMARY',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.electricCyan,
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              const SizedBox(width: 8),
              Icon(Icons.chevron_right,
                  color: AxonColors.textTertiary, size: 20),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Study Chapters Page
// ─────────────────────────────────────────────────────────────────

class StudyChaptersPage extends StatefulWidget {
  final String subject;
  final String subjectCode;
  final List<String> chapters;

  const StudyChaptersPage({
    super.key,
    required this.subject,
    required this.subjectCode,
    required this.chapters,
  });

  @override
  State<StudyChaptersPage> createState() => _StudyChaptersPageState();
}

class _StudyChaptersPageState extends State<StudyChaptersPage> {
  final _scrollController = ScrollController();
  String _searchQuery = '';
  final _searchController = TextEditingController();
  double _subjectProgress = 0.0;

  @override
  void initState() {
    super.initState();
    _loadSubjectProgress();
  }

  Future<void> _loadSubjectProgress() async {
    final prefs = await SharedPreferences.getInstance();
    final key =
        'subject_progress_${widget.subject.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '_')}';
    final saved = prefs.getDouble(key) ?? 0.0;
    if (mounted) {
      setState(() => _subjectProgress = saved);
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _searchController.dispose();
    super.dispose();
  }

  List<String> get _filteredChapters {
    if (_searchQuery.isEmpty) return widget.chapters;
    return widget.chapters
        .where((c) => c.toLowerCase().contains(_searchQuery.toLowerCase()))
        .toList();
  }

  bool get _isComputerScience {
    final subj = widget.subject.toLowerCase();
    return subj.contains('computer science') ||
        subj == 'cs' ||
        subj == 'compsci' ||
        subj == '0478' ||
        subj == '9618';
  }

  @override
  Widget build(BuildContext context) {
    final theme = themeFor(widget.subject);

    return Scaffold(
      backgroundColor: AxonColors.background,
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              theme.primary.withValues(alpha: 0.15),
              AxonColors.background,
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        IconButton(
                          icon:
                              const Icon(Icons.arrow_back, color: Colors.white),
                          onPressed: () => Navigator.pop(context),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                widget.subject,
                                style: GoogleFonts.googleSans(
                                  color: Colors.white,
                                  fontSize: 20,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                              Text(
                                widget.subjectCode,
                                style: GoogleFonts.googleSans(
                                  color: Colors.white54,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 12),
                      decoration: BoxDecoration(
                        color: AxonColors.surfaceElevated,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.search,
                              color: Colors.white38, size: 20),
                          const SizedBox(width: 12),
                          Expanded(
                            child: TextField(
                              controller: _searchController,
                              onChanged: (v) =>
                                  setState(() => _searchQuery = v),
                              style:
                                  GoogleFonts.googleSans(color: Colors.white),
                              decoration: InputDecoration(
                                hintText: 'Search chapters...',
                                hintStyle: GoogleFonts.googleSans(
                                    color: Colors.white38),
                                border: InputBorder.none,
                                isDense: true,
                                contentPadding: EdgeInsets.zero,
                              ),
                            ),
                          ),
                          if (_searchQuery.isNotEmpty)
                            GestureDetector(
                              onTap: () {
                                _searchController.clear();
                                setState(() => _searchQuery = '');
                              },
                              child: const Icon(Icons.close,
                                  color: Colors.white38, size: 18),
                            ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                    _PyqSubjectCard(
                      subjectCode: widget.subjectCode,
                      subjectName: widget.subject,
                      chapters: widget.chapters,
                      theme: theme,
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: _SubjectActionButton(
                            icon: Icons.quiz_outlined,
                            label: 'Quiz',
                            theme: theme,
                            onTap: () {},
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _SubjectActionButton(
                            icon: Icons.library_books_outlined,
                            label: 'Resources',
                            theme: theme,
                            onTap: () {},
                          ),
                        ),
                        if (_isComputerScience) ...[
                          const SizedBox(width: 12),
                          Expanded(
                            child: _SubjectActionButton(
                              icon: Icons.book_rounded,
                              label: 'Glossary',
                              theme: theme,
                              onTap: () async {
                                final container =
                                    ProviderScope.containerOf(context);
                                container
                                    .read(navbarVisibleProvider.notifier)
                                    .state = false;
                                await Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                        builder: (_) =>
                                            const GlossaryScreen()));
                                container
                                    .read(navbarVisibleProvider.notifier)
                                    .state = true;
                              },
                            ),
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          '${_filteredChapters.length} chapters',
                          style: GoogleFonts.googleSans(
                              color: Colors.white54, fontSize: 12),
                        ),
                        const Spacer(),
                        Text(
                          '${(_subjectProgress * 100).toInt()}% covered',
                          style: GoogleFonts.googleSans(
                              color: theme.primary,
                              fontSize: 12,
                              fontWeight: FontWeight.w700),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0, end: _subjectProgress),
                      duration: const Duration(milliseconds: 600),
                      curve: Curves.easeOutCubic,
                      builder: (context, value, child) {
                        return LinearProgressIndicator(
                          value: value,
                          backgroundColor: Colors.white12,
                          valueColor: AlwaysStoppedAnimation(theme.primary),
                          minHeight: 3,
                          borderRadius: BorderRadius.circular(2),
                        );
                      },
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.builder(
                  controller: _scrollController,
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 100),
                  itemCount: _filteredChapters.length,
                  itemBuilder: (context, index) {
                    final chapter = _filteredChapters[index];
                    return _ChapterTile(
                      chapter: chapter,
                      index: index + 1,
                      subject: widget.subject,
                      theme: theme,
                      onTap: () => _openChapterHub(chapter),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _openChapterHub(String chapter) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => StudyChapterHub(
          subject: widget.subject,
          subjectCode: widget.subjectCode,
          chapter: chapter,
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Chapter Tile
// ─────────────────────────────────────────────────────────────────

class _ChapterTile extends StatelessWidget {
  final String chapter;
  final int index;
  final String subject;
  final dynamic theme;
  final VoidCallback onTap;

  const _ChapterTile({
    required this.chapter,
    required this.index,
    required this.subject,
    required this.theme,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AxonColors.surfaceElevated,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.white12),
        ),
        child: Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: theme.primary.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Center(
                child: Text(
                  '$index',
                  style: GoogleFonts.googleSans(
                    color: theme.primary,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    chapter,
                    style: GoogleFonts.googleSans(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subject,
                    style: GoogleFonts.googleSans(
                      color: Colors.white38,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildMiniAction(Icons.menu_book, 'Notes'),
                const SizedBox(width: 8),
                _buildMiniAction(Icons.quiz, 'Quiz'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMiniAction(IconData icon, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white10,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: Colors.white54, size: 14),
          const SizedBox(width: 4),
          Text(
            label,
            style: GoogleFonts.googleSans(color: Colors.white54, fontSize: 11),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Study Chapter Hub
// ─────────────────────────────────────────────────────────────────

class StudyChapterHub extends StatefulWidget {
  final String subject;
  final String subjectCode;
  final String chapter;

  const StudyChapterHub({
    super.key,
    required this.subject,
    required this.subjectCode,
    required this.chapter,
  });

  @override
  State<StudyChapterHub> createState() => _StudyChapterHubState();
}

class _StudyChapterHubState extends State<StudyChapterHub> {
  bool _isLoading = true;
  List<CurriculumNoteSubchapter> _subchapters = [];
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadNotes();
  }

  Future<void> _loadNotes() async {
    try {
      await CambridgeNotesService.instance.initialize();

      final aliases = {
        '0478': '9618',
        '9608': '9618',
        '9471': '9618',
        '9609': '9618',
        '0625': '9702',
        '0652': '9702',
        '0620': '9701',
        '0610': '9700',
        '0580': '9709',
        '0455': '9708',
        '0470': '9389',
        '0460': '9696',
        '0500': '9093',
      };

      CurriculumNotes? notes =
          CambridgeNotesService.instance.forSubject(widget.subjectCode);
      if (notes == null && aliases.containsKey(widget.subjectCode)) {
        notes = CambridgeNotesService.instance
            .forSubject(aliases[widget.subjectCode]!);
      }
      if (notes == null) {
        for (final code in CambridgeNotesService.instance.allSubjectCodes()) {
          final subjectNotes = CambridgeNotesService.instance.forSubject(code);
          if (subjectNotes != null &&
              (subjectNotes.subjectName
                      .toLowerCase()
                      .contains(widget.subject.toLowerCase()) ||
                  widget.subject
                      .toLowerCase()
                      .contains(subjectNotes.subjectName.toLowerCase()))) {
            notes = subjectNotes;
            break;
          }
        }
      }

      if (notes == null) {
        if (mounted) {
          setState(() {
            _isLoading = false;
            _error = 'No notes available for ${widget.subject}';
          });
        }
        return;
      }

      CurriculumNoteChapter? matchedChapter;
      double bestScore = 0;
      for (final ch in notes.chapters) {
        final score =
            _similarity(ch.title.toLowerCase(), widget.chapter.toLowerCase());
        if (score > bestScore) {
          bestScore = score;
          matchedChapter = ch;
        }
      }

      if (bestScore < 0.3) {
        for (final ch in notes.chapters) {
          final chWords = ch.title
              .toLowerCase()
              .split(RegExp(r'[^a-z0-9]+'))
              .where((s) => s.isNotEmpty)
              .toSet();
          final titleWords = widget.chapter
              .toLowerCase()
              .split(RegExp(r'[^a-z0-9]+'))
              .where((s) => s.isNotEmpty)
              .toSet();
          final overlap = chWords.intersection(titleWords).length;
          final union = chWords.union(titleWords).length;
          final jaccard =
              union > 0 ? overlap.toDouble() / union.toDouble() : 0.0;
          if (jaccard > bestScore) {
            bestScore = jaccard;
            matchedChapter = ch;
          }
        }
      }

      if (mounted) {
        setState(() {
          _subchapters = matchedChapter?.subchapters ?? [];
          _isLoading = false;
          if (_subchapters.isEmpty) {
            _error = 'No notes found for "$widget.chapter"';
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _error = 'Failed to load notes: $e';
        });
      }
    }
  }

  double _similarity(String a, String b) {
    if (a == b) return 1.0;
    if (a.contains(b) || b.contains(a)) return 0.8;
    final aWords =
        a.split(RegExp(r'[^a-z0-9]+')).where((s) => s.isNotEmpty).toSet();
    final bWords =
        b.split(RegExp(r'[^a-z0-9]+')).where((s) => s.isNotEmpty).toSet();
    if (aWords.isEmpty || bWords.isEmpty) return 0.0;
    int matches = 0;
    for (final w in aWords) {
      if (bWords.contains(w)) matches++;
    }
    return matches.toDouble() /
        (aWords.length + bWords.length - matches).toDouble();
  }

  @override
  Widget build(BuildContext context) {
    final theme = themeFor(widget.subject);

    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              widget.chapter,
              style: GoogleFonts.googleSans(color: Colors.white, fontSize: 18),
            ),
            Text(
              widget.subject,
              style:
                  GoogleFonts.googleSans(color: Colors.white54, fontSize: 12),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.timer_outlined, color: theme.primary),
            onPressed: () => context.push('/study/timer'),
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: RoseLoader(size: 24))
          : _error != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(40),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.description_outlined,
                            color: Colors.white38, size: 48),
                        const SizedBox(height: 16),
                        Text(
                          _error!,
                          textAlign: TextAlign.center,
                          style: GoogleFonts.googleSans(
                              color: Colors.white54, fontSize: 14),
                        ),
                      ],
                    ),
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 100),
                  itemCount: _subchapters.length,
                  itemBuilder: (context, index) {
                    final sub = _subchapters[index];
                    return _SubchapterNoteCard(
                      subchapter: sub,
                      index: index + 1,
                      theme: theme,
                    );
                  },
                ),
    );
  }
}

class _SubchapterNoteCard extends StatelessWidget {
  final CurriculumNoteSubchapter subchapter;
  final int index;
  final dynamic theme;

  const _SubchapterNoteCard({
    required this.subchapter,
    required this.index,
    required this.theme,
  });

  String _cleanNotes(String text) {
    if (text.isEmpty) return 'No notes available for this topic.';

    final noiseMarkers = [
      '\n\n---\n\n',
      'BBC Homepage',
      'Skip to content',
      'Accessibility Help',
      'Your account',
      'Home\nNews',
      'Search the BBC',
      'Department of Computer Science',
      'Study at Cambridge',
      'About the University',
      'Research at Cambridge',
      'Search site',
      'Undergraduate',
      'Graduate',
      'Continuing education',
      'Executive and professional',
      'How the University and Colleges work',
      'Visiting the University',
      'Term dates and',
      'XtremePapers',
      'PapaCambridge',
      'GCE Guide',
      'talks.cam',
      'Apollo',
      'local PDF',
      'We need your support',
      'For more than 16 years',
      'Click here to Donate',
      'Community platform by XenForo',
      'Design by:',
      'Pixel Exit',
      'XenPorta',
      'Jason Axelrod',
      '8WAYRUN',
      'Log in',
      'Register',
      'What\'s new',
      'Latest activity',
      'Search forums',
      'New posts',
      'Search profile posts',
      'Current visitors',
      'Contact us',
      'Terms and rules',
      'Privacy policy',
      'Copyright',
      'Read about our approach',
      'external linking',
      '© 20',
      '© 202',
      'Reactions:',
      'Thought blocker',
      'Dead!',
      'officially dead',
      'Loading…',
      'Go to page',
      'Next',
      'Last',
      'You must log in',
      'Home\nMembers',
      'Home\nForums',
      'Qualifications, Exams',
      'RSS',
      'MB, download:',
      'Duration:',
      'restricted to users',
      'Raven login',
      'Input your search term',
      'Search results for',
      'Terms of Use',
      'About the BBC',
      'Advertise with us',
      'Parental Guidance',
      'BBC emails for you',
      'Student\nTeacher',
      'Study tools',
      'Join now for free',
      'A Level',
      'OCR',
      'Exchange & Transport',
      'on books with Teachers',
      'Aug ',
      'Replies',
      'Views',
      'Mar ',
      'Feb ',
      'May ',
      'Jan ',
    ];

    String cleaned = text;
    int earliestIndex = cleaned.length;
    for (final marker in noiseMarkers) {
      final idx = cleaned.indexOf(marker);
      if (idx != -1 && idx < earliestIndex) {
        earliestIndex = idx;
      }
    }

    if (earliestIndex < cleaned.length) {
      cleaned = cleaned.substring(0, earliestIndex).trim();
    }

    cleaned = cleaned.replaceAll(RegExp(r'\n{3,}'), '\n\n');
    cleaned = cleaned.replaceAll(RegExp(r'[ \t]+\n'), '\n');
    cleaned = cleaned.trim();

    if (cleaned.isEmpty) {
      return 'No notes available for this topic.';
    }

    return cleaned;
  }

  @override
  Widget build(BuildContext context) {
    final cleanedText = _cleanNotes(subchapter.text);

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: AxonColors.surfaceElevated,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: theme.primary.withValues(alpha: 0.15)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Container(
                  width: 28,
                  height: 28,
                  decoration: BoxDecoration(
                    color: theme.primary.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Center(
                    child: Text(
                      '$index',
                      style: GoogleFonts.googleSans(
                        color: theme.primary,
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    subchapter.title,
                    style: GoogleFonts.googleSans(
                      color: Colors.white,
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
          ),
          Divider(height: 1, color: theme.primary.withValues(alpha: 0.1)),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              cleanedText,
              style: GoogleFonts.googleSans(
                color: Colors.white70,
                fontSize: 13,
                height: 1.6,
              ),
            ),
          ),
        ],
      ),
    );
  }
}


class _EmptyStudy extends StatelessWidget {
  final VoidCallback onAdd;

  const _EmptyStudy({required this.onAdd});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: AxonColors.electricCyan.withValues(alpha: 0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.menu_book_rounded,
                size: 64,
                color: AxonColors.electricCyan.withValues(alpha: 0.6),
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Start Your Study Journey',
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Text(
              'Add subjects to track your progress, get personalized study plans, and master any topic.',
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontSize: 14,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: onAdd,
              icon: const Icon(Icons.add),
              label: const Text('Add Your First Subject'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AxonColors.electricCyan,
                foregroundColor: Colors.black,
                padding:
                    const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(duration: 400.ms);
  }
}

// ─────────────────────────────────────────────────────────────────
// Add Subject Bottom Sheet
// ─────────────────────────────────────────────────────────────────

class _AddSubjectBottomSheet extends StatefulWidget {
  final List<String> availableSubjects;
  final Map<String, String> subjectCodeMap;
  final void Function(String) onSubjectSelected;

  const _AddSubjectBottomSheet({
    required this.availableSubjects,
    required this.subjectCodeMap,
    required this.onSubjectSelected,
  });

  @override
  State<_AddSubjectBottomSheet> createState() => _AddSubjectBottomSheetState();
}

class _AddSubjectBottomSheetState extends State<_AddSubjectBottomSheet> {
  final _searchController = TextEditingController();
  String _searchQuery = '';

  List<String> get _filteredSubjects {
    if (_searchQuery.isEmpty) return widget.availableSubjects;
    return widget.availableSubjects
        .where((s) => s.toLowerCase().contains(_searchQuery.toLowerCase()))
        .toList();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.height * 0.75,
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.white24,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Text(
                  'Add Subject',
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  decoration: BoxDecoration(
                    color: AxonColors.surfaceElevated,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: TextField(
                    controller: _searchController,
                    onChanged: (v) => setState(() => _searchQuery = v),
                    style: GoogleFonts.googleSans(color: Colors.white),
                    decoration: InputDecoration(
                      hintText: 'Search subjects...',
                      hintStyle: GoogleFonts.googleSans(color: Colors.white38),
                      border: InputBorder.none,
                      prefixIcon:
                          const Icon(Icons.search, color: Colors.white38),
                    ),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: _filteredSubjects.isEmpty
                ? Center(
                    child: Text(
                      'No subjects available',
                      style: GoogleFonts.googleSans(color: Colors.white54),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                    itemCount: _filteredSubjects.length,
                    itemBuilder: (ctx, i) {
                      final subject = _filteredSubjects[i];
                      final subjectCode = widget.subjectCodeMap[subject] ?? '';
                      final theme = themeFor(subject);
                      return ListTile(
                        leading: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: theme.primary.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child:
                              Icon(theme.icon, color: theme.primary, size: 20),
                        ),
                        title: Text(
                          subject,
                          style: GoogleFonts.googleSans(
                              color: Colors.white, fontWeight: FontWeight.w600),
                        ),
                        subtitle: Text(
                          subjectCode,
                          style: GoogleFonts.googleSans(
                              color: Colors.white38, fontSize: 12),
                        ),
                        trailing:
                            Icon(Icons.add, color: AxonColors.electricCyan),
                        onTap: () {
                          widget.onSubjectSelected(subject);
                          Navigator.pop(context);
                        },
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _SubjectActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final dynamic theme;
  final VoidCallback onTap;

  const _SubjectActionButton({
    required this.icon,
    required this.label,
    required this.theme,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: theme.primary.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: theme.primary.withValues(alpha: 0.3)),
        ),
        child: Column(
          children: [
            Icon(icon, color: theme.primary, size: 24),
            const SizedBox(height: 6),
            Text(
              label,
              style: GoogleFonts.googleSans(
                color: theme.primary,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PyqSubjectCard extends StatefulWidget {
  final String subjectCode;
  final String subjectName;
  final List<String> chapters;
  final dynamic theme;

  const _PyqSubjectCard({
    required this.subjectCode,
    required this.subjectName,
    required this.chapters,
    required this.theme,
  });

  @override
  State<_PyqSubjectCard> createState() => _PyqSubjectCardState();
}

class _PyqSubjectCardState extends State<_PyqSubjectCard> {
  double _subjectProgress = 0.0;

  @override
  void initState() {
    super.initState();
    _loadProgress();
  }

  Future<void> _loadProgress() async {
    final prefs = await SharedPreferences.getInstance();
    final key =
        'subject_progress_${widget.subjectName.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '_')}';
    final saved = prefs.getDouble(key) ?? 0.0;
    if (mounted) {
      setState(() {
        _subjectProgress = saved;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        context.push('/catalog').then((_) => _loadProgress());
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: widget.theme.primary.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(12),
          border:
              Border.all(color: widget.theme.primary.withValues(alpha: 0.3)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.description_outlined,
                    color: widget.theme.primary, size: 20),
                const SizedBox(width: 10),
                Text(
                  'Past Papers',
                  style: GoogleFonts.googleSans(
                    color: widget.theme.primary,
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                Icon(
                  Icons.arrow_forward_rounded,
                  color: widget.theme.primary.withValues(alpha: 0.6),
                  size: 18,
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '${widget.chapters.length} chapters',
                        style: GoogleFonts.googleSans(
                          color: widget.theme.primary.withValues(alpha: 0.7),
                          fontSize: 11,
                        ),
                      ),
                      const SizedBox(height: 6),
                      TweenAnimationBuilder<double>(
                        tween: Tween(begin: 0, end: _subjectProgress),
                        duration: const Duration(milliseconds: 600),
                        curve: Curves.easeOutCubic,
                        builder: (context, value, child) {
                          return LinearProgressIndicator(
                            value: value,
                            backgroundColor:
                                widget.theme.primary.withValues(alpha: 0.1),
                            valueColor:
                                AlwaysStoppedAnimation(widget.theme.primary),
                            minHeight: 4,
                            borderRadius: BorderRadius.circular(2),
                          );
                        },
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                Text(
                  '${(_subjectProgress * 100).toInt()}%',
                  style: GoogleFonts.googleSans(
                    color: widget.theme.primary,
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
