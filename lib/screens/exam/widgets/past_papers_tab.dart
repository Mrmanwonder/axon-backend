// lib/screens/exam/widgets/past_papers_tab.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../router/app_router.dart';
import '../../../services/app_state.dart';
import '../../../services/exam_planner_service.dart';
import '../../../services/exam_planner_repository.dart';
import '../../../services/mock_exam_service.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/layout_utils.dart';

class PastPapersTab extends ConsumerWidget {
  const PastPapersTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (error, _) => Center(child: Text('Error: $error')),
      data: (data) {
        final papers = data.pastPapers;
        final completed = papers.where((p) => p.isCompleted).toList();
        final suggested = papers.where((p) => !p.isCompleted).toList();
        final subjects = papers.map((p) => p.subject).toSet().toList();

        return PastPapersVault(
            papers: papers,
            completed: completed,
            suggested: suggested,
            subjects: subjects);
      },
    );
  }
}

class PastPapersVault extends ConsumerStatefulWidget {
  final List<PastPaperPack> papers;
  final List<PastPaperPack> completed;
  final List<PastPaperPack> suggested;
  final List<String> subjects;

  const PastPapersVault({
    required this.papers,
    required this.completed,
    required this.suggested,
    required this.subjects,
  });

  @override
  ConsumerState<PastPapersVault> createState() => PastPapersVaultState();
}

class PastPapersVaultState extends ConsumerState<PastPapersVault>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  String? _selectedSubject;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    if (widget.subjects.isNotEmpty) {
      _selectedSubject = widget.subjects.first;
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _buildVaultHeader(),
        _buildSubjectFilter(),
        Expanded(
          child: TabBarView(
            controller: _tabController,
            children: [
              _buildSuggestedTab(),
              _buildCompletedTab(),
              _buildAllPapersTab(),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildVaultHeader() {
    return Container(
      margin: const EdgeInsets.fromLTRB(20, 16, 20, 8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(16),
      ),
      child: TabBar(
        controller: _tabController,
        indicatorSize: TabBarIndicatorSize.tab,
        dividerColor: Colors.transparent,
        indicator: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          color: AxonColors.accent.withValues(alpha: 0.2),
        ),
        labelColor: AxonColors.accent,
        unselectedLabelColor: AxonColors.textSecondary,
        labelStyle:
            GoogleFonts.robotoMono(fontSize: 10, fontWeight: FontWeight.w700),
        tabs: [
          Tab(text: 'SUGGESTED (${widget.suggested.length})'),
          Tab(text: 'COMPLETED (${widget.completed.length})'),
          Tab(text: 'ALL PAPERS'),
        ],
      ),
    );
  }

  Widget _buildSubjectFilter() {
    if (widget.subjects.isEmpty) return const SizedBox.shrink();
    return Container(
      height: 40,
      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: widget.subjects.length + 1,
        itemBuilder: (context, i) {
          if (i == 0) {
            final isSelected = _selectedSubject == null;
            return GestureDetector(
              key: const ValueKey('subject-all'),
              onTap: () => setState(() => _selectedSubject = null),
              child: Container(
                margin: const EdgeInsets.only(right: 8),
                padding: const EdgeInsets.symmetric(horizontal: 14),
                decoration: BoxDecoration(
                  color: isSelected
                      ? AxonColors.accent.withValues(alpha: 0.2)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(10),
                ),
                alignment: Alignment.center,
                child: Text(
                  'All',
                  style: GoogleFonts.googleSans(
                    color: isSelected
                        ? AxonColors.accent
                        : AxonColors.textSecondary,
                    fontSize: 12,
                  ),
                ),
              ),
            );
          }
          final subject = widget.subjects[i - 1];
          final isSelected = subject == _selectedSubject;
          return GestureDetector(
            key: ValueKey('subject-$subject'),
            onTap: () => setState(() => _selectedSubject = subject),
            child: Container(
              margin: const EdgeInsets.only(right: 8),
              padding: const EdgeInsets.symmetric(horizontal: 14),
              decoration: BoxDecoration(
                color: isSelected
                    ? AxonColors.accent.withValues(alpha: 0.2)
                    : Colors.transparent,
                borderRadius: BorderRadius.circular(10),
              ),
              alignment: Alignment.center,
              child: Text(
                subject,
                style: GoogleFonts.googleSans(
                  color:
                      isSelected ? AxonColors.accent : AxonColors.textSecondary,
                  fontSize: 12,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  List<PastPaperPack> _filterBySubject(List<PastPaperPack> papers) {
    if (_selectedSubject == null) return papers;
    return papers.where((p) => p.subject == _selectedSubject).toList();
  }

  Widget _buildSuggestedTab() {
    final filtered = _filterBySubject(widget.suggested);
    if (filtered.isEmpty) {
      return _buildEmptyState('No suggested papers',
          'Complete more papers to get personalized suggestions');
    }

    return ListView.builder(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      itemCount: filtered.length,
      itemBuilder: (context, i) => SuggestedPaperCard(
        key: ValueKey('suggested-${filtered[i].id}'),
        paper: filtered[i],
      ),
    );
  }

  Widget _buildCompletedTab() {
    final filtered = _filterBySubject(widget.completed);
    if (filtered.isEmpty) {
      return _buildEmptyState(
          'No completed papers', 'Start practicing to track your progress');
    }

    return ListView.builder(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      itemCount: filtered.length,
      itemBuilder: (context, i) => CompletedPaperCard(
        key: ValueKey('completed-${filtered[i].id}'),
        paper: filtered[i],
        onTap: () => _showAnalysis(context, filtered[i]),
      ),
    );
  }

  Widget _buildAllPapersTab() {
    final filtered = _filterBySubject(widget.papers);
    if (filtered.isEmpty) {
      return _buildEmptyState(
          'No papers available', 'Set your board in settings');
    }

    return ListView.builder(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      itemCount: filtered.length,
      itemBuilder: (context, i) => PastPaperCard(
        key: ValueKey('paper-${filtered[i].id}'),
        paper: filtered[i],
        onResultSubmit: (result) async {
          await ref
              .read(examPlannerProvider.notifier)
              .savePastPaperResult(filtered[i].id, result.score, null);
          await ref.read(metricsProvider.notifier).updateMockScoreWithChapter(
                score: result.score.toDouble(),
                subject: filtered[i].subject,
                chapter: result.reflection,
              );
        },
      ),
    );
  }

  Widget _buildEmptyState(String title, String subtitle) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.description_outlined,
              size: 48, color: AxonColors.textTertiary),
          const SizedBox(height: 16),
          Text(title,
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              )),
          const SizedBox(height: 8),
          Text(subtitle,
              style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary,
                fontSize: 13,
              )),
        ],
      ),
    );
  }

  void _showAnalysis(BuildContext context, PastPaperPack paper) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) =>
          PaperAnalysisSheet(paper: paper, allPapers: widget.papers),
    );
  }
}

class PastPaperCard extends StatelessWidget {
  final PastPaperPack paper;
  final Future<void> Function(MockExamResult) onResultSubmit;

  const PastPaperCard({required this.paper, required this.onResultSubmit, super.key});

  @override
  Widget build(BuildContext context) {
    final scorePercent = paper.score != null
        ? (paper.score! / paper.maxMarks * 100).round()
        : null;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: paper.isCompleted
                  ? (scorePercent != null && scorePercent >= 60
                      ? Color(0xFF4CAF50).withValues(alpha: 0.2)
                      : Color(0xFFFF9800).withValues(alpha: 0.2))
                  : AxonColors.accent.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Center(
              child: paper.isCompleted && scorePercent != null
                  ? Text(
                      '$scorePercent%',
                      style: GoogleFonts.googleSans(
                        color: scorePercent >= 60
                            ? Color(0xFF4CAF50)
                            : Color(0xFFFF9800),
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                      ),
                    )
                  : Icon(Icons.description_outlined,
                      color: AxonColors.accent, size: 24),
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${paper.subject} ${paper.year}',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${paper.variant} • ${paper.maxMarks} marks • ${paper.duration} min',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          if (!paper.isCompleted)
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                GestureDetector(
                  onTap: () => _startSimulation(context),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: AxonColors.accent.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      'START',
                      style: GoogleFonts.robotoMono(
                        color: AxonColors.accent,
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                GestureDetector(
                  onTap: () => _showScoreDialog(context),
                  child: Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.06),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Icon(Icons.edit_outlined,
                        color: AxonColors.textSecondary, size: 18),
                  ),
                ),
              ],
            )
          else
            Icon(Icons.check_circle, color: Color(0xFF4CAF50), size: 24),
        ],
      ),
    );
  }

  Future<void> _startSimulation(BuildContext context) async {
    final result = await context.push<MockExamResult>(
      AppRoutes.mockSimulation,
      extra: {'paper': paper},
    );
    if (result == null) return;
    await onResultSubmit(result);
  }

  void _showScoreDialog(BuildContext context) {
    final controller = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AxonColors.surface,
        title: Text('Enter Score',
            style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary, fontWeight: FontWeight.w700)),
        content: TextField(
          controller: controller,
          keyboardType: TextInputType.number,
          decoration: InputDecoration(
            hintText: '/${paper.maxMarks}',
            suffixText: 'marks',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              final score = int.tryParse(controller.text);
              if (score != null && score >= 0 && score <= paper.maxMarks) {
                await onResultSubmit(
                  MockExamResult(
                    score: score,
                    reflection: 'Manual score entry for ${paper.subject}',
                    violationCount: 0,
                  ),
                );
                if (!context.mounted) return;
                Navigator.pop(context);
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }
}

class SuggestedPaperCard extends ConsumerWidget {
  final PastPaperPack paper;
  const SuggestedPaperCard({required this.paper, super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return GestureDetector(
      onTap: () => _startSimulation(context, ref),
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.03),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AxonColors.accent.withValues(alpha: 0.2)),
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: AxonColors.accent.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(Icons.play_arrow_rounded,
                  color: AxonColors.accent, size: 24),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${paper.subject} ${paper.year} ${paper.variant}',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      StatChip(
                          icon: Icons.timer_outlined,
                          label: '${paper.duration} min'),
                      const SizedBox(width: 8),
                      StatChip(
                          icon: Icons.grade_outlined,
                          label: '${paper.maxMarks} marks'),
                    ],
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: AxonColors.accent,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                'START',
                style: GoogleFonts.robotoMono(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _startSimulation(BuildContext context, WidgetRef ref) async {
    final result = await context.push<MockExamResult>(
      AppRoutes.mockSimulation,
      extra: {'paper': paper},
    );
    if (result == null) return;
    await ref
        .read(examPlannerProvider.notifier)
        .savePastPaperResult(paper.id, result.score, null);
    await ref.read(metricsProvider.notifier).updateMockScoreWithChapter(
          score: result.score.toDouble(),
          subject: paper.subject,
          chapter: result.reflection,
        );
  }
}

class CompletedPaperCard extends StatelessWidget {
  final PastPaperPack paper;
  final VoidCallback onTap;
  const CompletedPaperCard({required this.paper, required this.onTap, super.key});

  @override
  Widget build(BuildContext context) {
    final scorePercent =
        paper.score != null ? (paper.score! / paper.maxMarks * 100).round() : 0;
    final isPassing = scorePercent >= 60;
    final gradeColor =
        isPassing ? const Color(0xFF4CAF50) : const Color(0xFFFF9800);

    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.03),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          children: [
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: gradeColor.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    '$scorePercent%',
                    style: GoogleFonts.googleSans(
                      color: gradeColor,
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  Text(
                    '${paper.score}/${paper.maxMarks}',
                    style: GoogleFonts.googleSans(
                      color: gradeColor.withValues(alpha: 0.7),
                      fontSize: 9,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${paper.subject} ${paper.year}',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      StatChip(
                          icon: Icons.calendar_today,
                          label: _formatDate(paper.attemptedAt)),
                      const SizedBox(width: 8),
                      StatChip(
                          icon: Icons.schedule, label: '${paper.duration} min'),
                    ],
                  ),
                  if (paper.attemptedAt != null) ...[
                    const SizedBox(height: 4),
                    Text(
                      _getTimeAgo(paper.attemptedAt!),
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary,
                        fontSize: 10,
                      ),
                    ),
                  ],
                ],
              ),
            ),
            Icon(Icons.chevron_right, color: AxonColors.textTertiary, size: 24),
          ],
        ),
      ),
    );
  }

  String _formatDate(DateTime? dt) {
    if (dt == null) return '';
    return '${dt.day}/${dt.month}/${dt.year}';
  }

  String _getTimeAgo(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inDays > 30) return '${(diff.inDays / 30).floor()} months ago';
    if (diff.inDays > 0) return '${diff.inDays} days ago';
    if (diff.inHours > 0) return '${diff.inHours} hours ago';
    return 'Just now';
  }
}

class PaperAnalysisSheet extends StatelessWidget {
  final PastPaperPack paper;
  final List<PastPaperPack> allPapers;
  const PaperAnalysisSheet({required this.paper, required this.allPapers, super.key});

  @override
  Widget build(BuildContext context) {
    final scorePercent =
        paper.score != null ? (paper.score! / paper.maxMarks * 100).round() : 0;
    final isPassing = scorePercent >= 60;
    final gradeColor =
        isPassing ? const Color(0xFF4CAF50) : const Color(0xFFFF9800);
    final grade = _getGrade(scorePercent);

    return Container(
      height: MediaQuery.of(context).size.height * 0.75,
      decoration: const BoxDecoration(
        color: Color(0xFF0E0E0E),
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 100),
            decoration: BoxDecoration(
              border: Border(
                  bottom:
                      BorderSide(color: Colors.white.withValues(alpha: 0.05))),
            ),
            child: Row(
              children: [
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.close, color: Colors.white54),
                ),
                const SizedBox(width: 8),
                Text(
                  'Analysis',
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 100),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildScoreCard(scorePercent, grade, gradeColor),
                  const SizedBox(height: 20),
                  _buildStatsGrid(),
                  const SizedBox(height: 20),
                  _buildGuidanceSection(isPassing),
                  const SizedBox(height: 20),
                  _buildSuggestedNextSection(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScoreCard(int percent, String grade, Color color) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [color.withValues(alpha: 0.2), Colors.transparent],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${paper.subject} ${paper.year}',
                    style: GoogleFonts.googleSans(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  Text(
                    '${paper.variant} Paper ${paper.paperNumber}',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textSecondary,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  grade,
                  style: GoogleFonts.googleSans(
                    color: color,
                    fontSize: 24,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                '$percent%',
                style: GoogleFonts.googleSans(
                  color: color,
                  fontSize: 48,
                  fontWeight: FontWeight.w900,
                ),
              ),
              const SizedBox(width: 12),
              Text(
                '/${paper.maxMarks}',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary,
                  fontSize: 24,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
          if (paper.attemptedAt != null) ...[
            const SizedBox(height: 8),
            Text(
              'Attempted ${_formatDateTime(paper.attemptedAt!)}',
              style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary,
                fontSize: 12,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStatsGrid() {
    return Row(
      children: [
        Expanded(
          child: StatCard(
            icon: Icons.timer_outlined,
            label: 'Duration',
            value: '${paper.duration} min',
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: StatCard(
            icon: Icons.grade_outlined,
            label: 'Max Marks',
            value: '${paper.maxMarks}',
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: StatCard(
            icon: Icons.note_outlined,
            label: 'Score',
            value: '${paper.score ?? 0}',
          ),
        ),
      ],
    );
  }

  Widget _buildGuidanceSection(bool isPassing) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.lightbulb_outline, color: AxonColors.accent, size: 20),
              const SizedBox(width: 8),
              Text(
                'Guidance',
                style: GoogleFonts.googleSans(
                  color: AxonColors.accent,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            isPassing
                ? 'Great work! You passed this paper. Focus on timing and accuracy to improve further.'
                : 'Focus on understanding the topics you struggled with. Review the mark scheme and practice similar questions.',
            style: GoogleFonts.googleSans(
              color: AxonColors.textSecondary,
              fontSize: 13,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSuggestedNextSection() {
    final sameSubject = allPapers
        .where((p) => p.subject == paper.subject && !p.isCompleted)
        .take(3)
        .toList();

    if (sameSubject.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.03),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          children: [
            const Icon(Icons.check_circle, color: Color(0xFF4CAF50), size: 20),
            const SizedBox(width: 8),
            Text(
              'All ${paper.subject} papers completed!',
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontSize: 13,
              ),
            ),
          ],
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Suggested Next',
          style: GoogleFonts.googleSans(
            color: AxonColors.textPrimary,
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 12),
        ...sameSubject.map((p) => SuggestedPaperCard(paper: p)),
      ],
    );
  }

  String _getGrade(int percent) {
    if (percent >= 90) return 'A*';
    if (percent >= 80) return 'A';
    if (percent >= 70) return 'B';
    if (percent >= 60) return 'C';
    if (percent >= 50) return 'D';
    return 'U';
  }

  String _formatDateTime(DateTime dt) {
    return '${dt.day}/${dt.month}/${dt.year} at ${dt.hour}:${dt.minute.toString().padLeft(2, '0')}';
  }
}

class StatChip extends StatelessWidget {
  final IconData icon;
  final String label;
  const StatChip({required this.icon, required this.label, super.key});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: AxonColors.textTertiary, size: 12),
        const SizedBox(width: 4),
        Text(label,
            style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary,
              fontSize: 11,
            )),
      ],
    );
  }
}

class StatCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  const StatCard(
      {required this.icon, required this.label, required this.value, super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Icon(icon, color: AxonColors.textTertiary, size: 18),
          const SizedBox(height: 8),
          Text(
            value,
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          Text(
            label,
            style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary,
              fontSize: 10,
            ),
          ),
        ],
      ),
    );
  }
}
