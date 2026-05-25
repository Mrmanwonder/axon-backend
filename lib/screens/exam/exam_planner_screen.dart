// lib/screens/exam/exam_planner_screen.dart
// ─────────────────────────────────────────────────────────────────
// Full Exam Planner Screen
// Countdown, milestones, past papers, checklists, formulas
// ─────────────────────────────────────────────────────────────────

import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../router/app_router.dart';
import '../../services/app_state.dart';
import '../../services/formula_drill_service.dart';
import '../../services/exam_planner_service.dart';
import '../../services/exam_planner_repository.dart';
import '../../services/mock_exam_service.dart';
import '../../theme/app_theme.dart';
import '../study/formula_drill_screen.dart';
import '../../widgets/common/rose_loader.dart';
import '../../widgets/common/blur_reveal_widget.dart';
import '../../widgets/math/math_expression.dart';
import '../../utils/layout_utils.dart';

TextDecoration? _lineThrough(bool condition) =>
    condition ? TextDecoration.lineThrough : null;
Color _phaseColor(bool isPast, Color phaseColor) =>
    isPast ? AxonColors.textTertiary : phaseColor;
Color _phaseColorWithAlpha(bool isPast, Color phaseColor, double alpha) =>
    isPast
        ? AxonColors.textTertiary.withValues(alpha: alpha)
        : phaseColor.withValues(alpha: alpha);

class ExamPlannerScreen extends ConsumerStatefulWidget {
  const ExamPlannerScreen({super.key});

  @override
  ConsumerState<ExamPlannerScreen> createState() => _ExamPlannerScreenState();
}

class _ExamPlannerScreenState extends ConsumerState<ExamPlannerScreen>
    with TickerProviderStateMixin {
  late TabController _tabController;
  late AnimationController _transitionController;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 5, vsync: this);

    _transitionController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(
          parent: _transitionController, curve: Curves.easeOutCubic),
    );

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _transitionController.forward();
      }
    });
  }

  @override
  void dispose() {
    _tabController.dispose();
    _transitionController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.oxfordBlueDark,
      resizeToAvoidBottomInset: false,
      body: Stack(
        children: [
          Positioned(
            top: -100,
            right: -50,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: AxonColors.accent.withValues(alpha: 0.15),
              ),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 80, sigmaY: 80),
                child: Container(),
              ),
            ),
          ),
          SafeArea(
            child: FadeTransition(
              opacity: _scaleAnimation,
              child: ScaleTransition(
                scale: _scaleAnimation,
                child: Column(
                  children: [
                    _buildExamPlannerHeader(),
                    Expanded(
                      child: TabBarView(
                        controller: _tabController,
                        children: const [
                          _CountdownTab(),
                          _MilestonesTab(),
                          _PastPapersTab(),
                          _ChecklistsTab(),
                          _FormulasTab(),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildExamPlannerHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new_rounded,
                color: Colors.white, size: 20),
            onPressed: () {
              if (Navigator.of(context).canPop()) {
                Navigator.of(context).pop();
              } else {
                // Navigate to home if can't pop
                context.go('/home');
              }
            },
          ),
          const SizedBox(width: 4),
          Expanded(
            child: Text(
              'EXAM PLANNER',
              style: GoogleFonts.googleSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AnimatedDrillButton extends StatefulWidget {
  final VoidCallback? onPressed;
  const _AnimatedDrillButton({this.onPressed});

  @override
  State<_AnimatedDrillButton> createState() => _AnimatedDrillButtonState();
}

class _AnimatedDrillButtonState extends State<_AnimatedDrillButton> {
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: widget.onPressed,
      child: Container(
        width: 48,
        height: 48,
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.3),
            width: 1,
          ),
        ),
        child: Icon(
          Icons.bolt_rounded,
          color: Colors.white,
          size: 24,
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// COUNTDOWN TAB
// ─────────────────────────────────────────────────────────────────

class _CountdownTab extends ConsumerWidget {
  const _CountdownTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: RoseLoader(size: 24)),
      error: (error, _) => _buildErrorState(error.toString(), ref),
      data: (data) {
        if (data.countdown == null || data.countdown!.daysRemaining == 0) {
          return _buildNoExamSet();
        }

        final countdown = data.countdown!;
        return SingleChildScrollView(
          padding:
              EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              BlurFadeSlideWidget(
                sessionKey: 'exam_planner_countdown',
                child: _buildMainCountdown(countdown),
              ),
              const SizedBox(height: 24),
              BlurFadeSlideWidget(
                sessionKey: 'exam_planner_phases',
                delay: const Duration(milliseconds: 100),
                child: _buildPhasesTimeline(countdown),
              ),
              const SizedBox(height: 24),
              BlurFadeSlideWidget(
                sessionKey: 'exam_planner_focus',
                delay: const Duration(milliseconds: 200),
                child: _buildCurrentPhase(context, countdown),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildErrorState(String error, WidgetRef ref) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.error_outline, size: 48, color: AxonColors.textTertiary),
          const SizedBox(height: 16),
          Text('Failed to load: $error',
              style: TextStyle(color: AxonColors.textSecondary)),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: () => ref.read(examPlannerProvider.notifier).retry(),
            child: const Text('RETRY'),
          ),
        ],
      ),
    );
  }

  Widget _buildNoExamSet() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: AxonColors.accent.withValues(alpha: 0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.event_note, size: 48, color: AxonColors.accent),
            ),
            const SizedBox(height: 24),
            Text(
              'No Exam Date Set',
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Set your exam date in settings to see countdown',
              textAlign: TextAlign.center,
              style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary,
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMainCountdown(ExamCountdown countdown) {
    final days = countdown.daysRemaining;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(32),
        gradient: LinearGradient(
          colors: [AxonColors.accent, AxonColors.accent.withValues(alpha: 0.7)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: AxonColors.accent.withValues(alpha: 0.4),
            blurRadius: 30,
            offset: const Offset(0, 15),
          ),
        ],
      ),
      child: Column(
        children: [
          Text(
            '$days',
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 84,
              fontWeight: FontWeight.w900,
              height: 1,
            ),
          ).animate().scale(duration: 400.ms, curve: Curves.easeOutBack),
          Text(
            'DAYS TO GO',
            style: GoogleFonts.googleSans(
              color: Colors.white.withValues(alpha: 0.8),
              fontSize: 12,
              fontWeight: FontWeight.w800,
              letterSpacing: 4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPhasesTimeline(ExamCountdown countdown) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Revision Phases',
          style: GoogleFonts.googleSans(
            color: AxonColors.textPrimary,
            fontSize: 16,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 12),
        ...countdown.phases.map((phase) {
          final isActive = countdown.currentPhase?.name == phase.name;
          final isPast = countdown.daysRemaining < phase.startDay;

          return Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: isActive
                  ? Color(phase.color).withValues(alpha: 0.15)
                  : isPast
                      ? AxonColors.surface.withValues(alpha: 0.05)
                      : AxonColors.surface.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: isActive
                    ? Color(phase.color).withValues(alpha: 0.4)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color:
                        _phaseColorWithAlpha(isPast, Color(phase.color), 0.2),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    isPast ? Icons.check_rounded : Icons.schedule,
                    color: _phaseColor(isPast, Color(phase.color)),
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        phase.name,
                        style: GoogleFonts.googleSans(
                          color: isPast
                              ? AxonColors.textTertiary
                              : AxonColors.textPrimary,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          decoration: _lineThrough(isPast),
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'Day ${phase.startDay} - ${phase.endDay} • ${phase.dailyHours}h/day',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
                if (isActive)
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Color(phase.color).withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      'NOW',
                      style: GoogleFonts.googleSans(
                        color: Color(phase.color),
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
              ],
            ),
          );
        }),
      ],
    );
  }

  Widget _buildCurrentPhase(BuildContext context, ExamCountdown countdown) {
    final phase = countdown.currentPhase;
    if (phase == null) return const SizedBox.shrink();

    return Container(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Color(phase.color).withValues(alpha: 0.1),
            AxonColors.surface.withValues(alpha: 0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.lightbulb_outline,
                  color: Color(phase.color), size: 20),
              const SizedBox(width: 8),
              Text(
                'Current Focus',
                style: GoogleFonts.googleSans(
                  color: Color(phase.color),
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            phase.focus,
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            phase.description,
            style: GoogleFonts.googleSans(
              color: AxonColors.textSecondary,
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// MILESTONES TAB
// ─────────────────────────────────────────────────────────────────

class _MilestonesTab extends ConsumerWidget {
  const _MilestonesTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: RoseLoader(size: 24)),
      error: (error, _) => Center(child: Text('Error: $error')),
      data: (data) {
        final milestones = data.milestones;
        final completed = milestones.where((m) => m.isCompleted).length;

        return SingleChildScrollView(
          padding:
              EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildProgressHeader(context, completed, milestones.length),
              const SizedBox(height: 20),
              ...milestones.map((m) => _MilestoneCard(
                    milestone: m,
                    onComplete: () async {
                      await ref
                          .read(examPlannerProvider.notifier)
                          .completeMilestone(m.id);
                    },
                  )),
            ],
          ),
        );
      },
    );
  }

  Widget _buildProgressHeader(BuildContext context, int completed, int total) {
    final progress = total > 0 ? completed / total : 0.0;

    return Container(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      decoration: BoxDecoration(
        color: AxonColors.surface.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Progress',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Text(
                '$completed / $total',
                style: GoogleFonts.googleSans(
                  color: AxonColors.accent,
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: progress,
              minHeight: 8,
              backgroundColor: AxonColors.surface.withValues(alpha: 0.2),
              color: AxonColors.accent,
            ),
          ),
        ],
      ),
    );
  }
}

class _MilestoneCard extends StatelessWidget {
  final Milestone milestone;
  final VoidCallback onComplete;

  const _MilestoneCard({required this.milestone, required this.onComplete});

  static const Map<MilestoneType, IconData> iconMap = {
    MilestoneType.syllabus: Icons.book_rounded,
    MilestoneType.mockTest: Icons.assignment_rounded,
    MilestoneType.analysis: Icons.analytics_rounded,
    MilestoneType.practice: Icons.edit_note_rounded,
    MilestoneType.formula: Icons.functions_rounded,
    MilestoneType.timing: Icons.timer_rounded,
    MilestoneType.consolidation: Icons.layers_rounded,
    MilestoneType.general: Icons.flag_rounded,
  };

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: null,
          borderRadius: BorderRadius.circular(16),
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: milestone.isCompleted
                  ? AxonColors.accent.withValues(alpha: 0.1)
                  : AxonColors.surface.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: milestone.isCompleted
                    ? AxonColors.accent.withValues(alpha: 0.3)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: milestone.isCompleted
                        ? AxonColors.accent.withValues(alpha: 0.2)
                        : AxonColors.surface.withValues(alpha: 0.15),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    milestone.isCompleted
                        ? Icons.check_rounded
                        : iconMap[milestone.type] ?? Icons.flag_rounded,
                    color: milestone.isCompleted
                        ? AxonColors.accent
                        : AxonColors.textSecondary,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        milestone.title,
                        style: GoogleFonts.googleSans(
                          color: milestone.isCompleted
                              ? AxonColors.accent
                              : AxonColors.textPrimary,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          decoration: _lineThrough(milestone.isCompleted),
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        milestone.description,
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 12,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Target: Day ${milestone.targetDay}',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 11,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
                if (!milestone.isCompleted)
                  Icon(Icons.radio_button_unchecked,
                      color: AxonColors.textTertiary, size: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// PAST PAPERS TAB
// ─────────────────────────────────────────────────────────────────

class _PastPapersTab extends ConsumerWidget {
  const _PastPapersTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: RoseLoader(size: 24)),
      error: (error, _) => Center(child: Text('Error: $error')),
      data: (data) {
        final papers = data.pastPapers;
        final completed = papers.where((p) => p.isCompleted).toList();
        final suggested = papers.where((p) => !p.isCompleted).toList();
        final subjects = papers.map((p) => p.subject).toSet().toList();

        return _PastPapersVault(
            papers: papers,
            completed: completed,
            suggested: suggested,
            subjects: subjects);
      },
    );
  }
}

class _PastPapersVault extends ConsumerStatefulWidget {
  final List<PastPaperPack> papers;
  final List<PastPaperPack> completed;
  final List<PastPaperPack> suggested;
  final List<String> subjects;

  const _PastPapersVault({
    required this.papers,
    required this.completed,
    required this.suggested,
    required this.subjects,
  });

  @override
  ConsumerState<_PastPapersVault> createState() => _PastPapersVaultState();
}

class _PastPapersVaultState extends ConsumerState<_PastPapersVault>
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
      itemBuilder: (context, i) => _SuggestedPaperCard(paper: filtered[i]),
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
      itemBuilder: (context, i) => _CompletedPaperCard(
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
      itemBuilder: (context, i) => _PastPaperCard(
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
          _PaperAnalysisSheet(paper: paper, allPapers: widget.papers),
    );
  }
}

class _PastPaperCard extends StatelessWidget {
  final PastPaperPack paper;
  final Future<void> Function(MockExamResult) onResultSubmit;

  const _PastPaperCard({required this.paper, required this.onResultSubmit});

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

// ─────────────────────────────────────────────────────────────────
// CHECKLISTS TAB
// ─────────────────────────────────────────────────────────────────

class _ChecklistsTab extends ConsumerWidget {
  const _ChecklistsTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: RoseLoader(size: 24)),
      error: (error, _) => Center(child: Text('Error: $error')),
      data: (data) {
        final checklists = data.checklists;
        final subjects = checklists.keys.toList();

        return _ChecklistsContent(checklists: checklists, subjects: subjects);
      },
    );
  }
}

class _ChecklistsContent extends ConsumerStatefulWidget {
  final Map<String, StrategyChecklist> checklists;
  final List<String> subjects;

  const _ChecklistsContent({required this.checklists, required this.subjects});

  @override
  ConsumerState<_ChecklistsContent> createState() => _ChecklistsContentState();
}

class _ChecklistsContentState extends ConsumerState<_ChecklistsContent> {
  String _selectedSubject = 'general';

  @override
  void initState() {
    super.initState();
    if (widget.subjects.isNotEmpty) {
      _selectedSubject = widget.subjects.first;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          height: 44,
          margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: widget.subjects.length,
            itemBuilder: (context, i) {
              final subject = widget.subjects[i];
              final isSelected = subject == _selectedSubject;
              return GestureDetector(
                onTap: () => setState(() => _selectedSubject = subject),
                child: Container(
                  margin: const EdgeInsets.only(right: 8),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? AxonColors.accent.withValues(alpha: 0.2)
                        : AxonColors.surface.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: isSelected
                          ? AxonColors.accent.withValues(alpha: 0.4)
                          : Colors.transparent,
                    ),
                  ),
                  alignment: Alignment.center,
                  child: Text(
                    _capitalize(subject),
                    style: GoogleFonts.googleSans(
                      color: isSelected
                          ? AxonColors.accent
                          : AxonColors.textSecondary,
                      fontSize: 13,
                      fontWeight:
                          isSelected ? FontWeight.w600 : FontWeight.w500,
                    ),
                  ),
                ),
              );
            },
          ),
        ),
        Expanded(
          child: _buildChecklist(),
        ),
      ],
    );
  }

  String _capitalize(String s) => s[0].toUpperCase() + s.substring(1);

  Widget _buildChecklist() {
    final checklist = widget.checklists[_selectedSubject];
    if (checklist == null) {
      return const Center(child: RoseLoader(size: 24));
    }

    final completed = checklist.items.where((i) => i.isChecked).length;
    final progress =
        checklist.items.isNotEmpty ? completed / checklist.items.length : 0.0;

    return SingleChildScrollView(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AxonColors.surface.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Completion',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Text(
                      '$completed/${checklist.items.length}',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.accent,
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.0, end: progress),
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.easeOutCubic,
                  builder: (context, value, child) {
                    return ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: value,
                        minHeight: 6,
                        backgroundColor:
                            AxonColors.surface.withValues(alpha: 0.2),
                        color: AxonColors.accent,
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          ...checklist.items.map((item) => _ChecklistItemCard(
                item: item,
                onToggle: (checked) async {
                  final subjectKey = _selectedSubject.toLowerCase();
                  await ref
                      .read(examPlannerProvider.notifier)
                      .updateChecklistItem(subjectKey, item.id, checked);
                },
              )),
        ],
      ),
    );
  }
}

class _ChecklistItemCard extends StatelessWidget {
  final ChecklistItem item;
  final Function(bool) onToggle;

  const _ChecklistItemCard({required this.item, required this.onToggle});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        GestureDetector(
          onTap: () => onToggle(!item.isChecked),
          behavior: HitTestBehavior.opaque,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 8),
            child: Row(
              children: [
                _AnimatedCheckbox(isCompleted: item.isChecked),
                const SizedBox(width: 16),
                Expanded(
                  child: Stack(
                    children: [
                      Text(
                        item.text,
                        style: GoogleFonts.googleSans(
                          color: item.isChecked ? Colors.white24 : Colors.white,
                          fontSize: 15,
                        ),
                      ),
                      Positioned.fill(
                        child: _AnimatedStrike(isCompleted: item.isChecked),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        Divider(
            height: 1, color: Colors.white.withValues(alpha: 0.03), indent: 48),
      ],
    );
  }
}

class _AnimatedCheckbox extends StatelessWidget {
  final bool isCompleted;
  const _AnimatedCheckbox({required this.isCompleted});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {},
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 22,
        height: 22,
        decoration: BoxDecoration(
          color: isCompleted ? const Color(0xFF3A86FF) : Colors.transparent,
          shape: BoxShape.circle,
          border: Border.all(
            color: isCompleted ? const Color(0xFF3A86FF) : Colors.white38,
            width: 2,
          ),
        ),
        child: isCompleted
            ? const Icon(Icons.check, color: Colors.white, size: 14)
            : null,
      ),
    );
  }
}

class _AnimatedStrike extends StatelessWidget {
  final bool isCompleted;
  const _AnimatedStrike({required this.isCompleted});

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: isCompleted ? 1.0 : 0.0),
      duration: const Duration(milliseconds: 350),
      curve: Curves.easeOutCubic,
      builder: (context, value, child) {
        return CustomPaint(
          painter: _StrikePainter(progress: value),
        );
      },
    );
  }
}

class _StrikePainter extends CustomPainter {
  final double progress;
  _StrikePainter({required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    if (progress == 0) return;
    final paint = Paint()
      ..color = Colors.white38
      ..strokeWidth = 1.2
      ..strokeCap = StrokeCap.round;

    final lineCount = (size.height / 20).floor() + 2;
    final lineHeight = size.height / lineCount;

    for (var i = 0; i < lineCount; i++) {
      final y = (i * lineHeight) + (lineHeight / 2);
      final lineProgress = lineCount == 1
          ? progress
          : (i < lineCount - 1)
              ? 1.0
              : progress;
      canvas.drawLine(
        Offset(0, y + 2),
        Offset(size.width * lineProgress, y + 2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(_StrikePainter oldDelegate) =>
      oldDelegate.progress != progress;
}

// ─────────────────────────────────────────────────────────────────
// FORMULAS TAB
// ─────────────────────────────────────────────────────────────────

class _FormulasTab extends ConsumerWidget {
  const _FormulasTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: RoseLoader(size: 24)),
      error: (error, _) => Center(child: Text('Error: $error')),
      data: (data) {
        final formulas = data.formulas;
        final topics = formulas.keys.toList();

        if (topics.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.functions_rounded, color: Colors.white24, size: 48),
                const SizedBox(height: 16),
                Text('No formulas available',
                    style: TextStyle(color: Colors.white54)),
                const SizedBox(height: 8),
                MathExpression(
                  formulaTex: r'\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1',
                  fontSize: 20,
                ),
              ],
            ),
          );
        }

        return _FormulasContent(formulas: formulas, topics: topics);
      },
    );
  }
}

class _FormulasContent extends ConsumerStatefulWidget {
  final Map<String, FormulaSheet> formulas;
  final List<String> topics;

  const _FormulasContent({required this.formulas, required this.topics});

  @override
  ConsumerState<_FormulasContent> createState() => _FormulasContentState();
}

class _FormulasContentState extends ConsumerState<_FormulasContent> {
  String _selectedTopic = 'algebra';
  final _searchController = TextEditingController();
  String _searchQuery = '';

  @override
  void initState() {
    super.initState();
    if (widget.topics.isNotEmpty) {
      _selectedTopic = widget.topics.first;
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  String _formatTopic(String topic) {
    return topic
        .split('_')
        .map((w) => w[0].toUpperCase() + w.substring(1))
        .join(' ');
  }

  Future<void> _openDrill(List<FormulaItem> filtered) async {
    final topics = filtered.map((item) => item.topicName).toSet().toList();

    List<FormulaDrillCard> allDueCards = [];
    for (final topic in topics) {
      final cards = await FormulaDrillService.instance.getDueCards(
        widget.formulas,
        topicKey: topic,
      );
      allDueCards.addAll(cards);
    }

    final allowedIds =
        filtered.map((item) => '${item.topicName}::${item.name}').toSet();
    final deck =
        allDueCards.where((card) => allowedIds.contains(card.id)).toList();
    if (!mounted || deck.isEmpty) return;

    ref.read(navbarVisibleProvider.notifier).state = false;
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (context) => FormulaDrillScreen(
          title: filtered.length == 1 || topics.length == 1
              ? '${_formatTopic(topics.isNotEmpty ? (topics.first ?? '') : '')} Drill'
              : 'Formula Drill',
          cards: deck,
          onReview: FormulaDrillService.instance.reviewCard,
        ),
      ),
    );
    if (mounted) {
      ref.read(navbarVisibleProvider.notifier).state = true;
      setState(() {});
    }
  }

  Future<void> _toggleFavorite(FormulaItem formula) async {
    await ref
        .read(examPlannerProvider.notifier)
        .toggleFormulaFavorite(_selectedTopic, formula.name);
  }

  @override
  Widget build(BuildContext context) {
    List<FormulaItem> filtered = [];

    if (_searchQuery.isNotEmpty) {
      for (final entry in widget.formulas.entries) {
        final topicFormulas = entry.value.formulas
            .where((f) =>
                f.name.toLowerCase().contains(_searchQuery.toLowerCase()) ||
                f.expression.toLowerCase().contains(_searchQuery.toLowerCase()))
            .toList();
        for (final f in topicFormulas) {
          filtered.add(FormulaItem(
            name: f.name,
            expression: f.expression,
            explanation: f.explanation,
            isFavorite: f.isFavorite,
            topicName: entry.key,
          ));
        }
      }
    } else {
      final sheet = widget.formulas[_selectedTopic];
      filtered = sheet?.formulas.toList() ?? [];
    }

    return Column(
      children: [
        Container(
          color: const Color(0xFF0A0A0A),
          padding: const EdgeInsets.only(top: 8),
          child: Column(
            children: [
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _searchController,
                        onChanged: (v) => setState(() => _searchQuery = v),
                        style: const TextStyle(color: Colors.white),
                        decoration: InputDecoration(
                          hintText: 'Search formulas...',
                          hintStyle: TextStyle(
                              color: Colors.white.withValues(alpha: 0.3)),
                          prefixIcon: Icon(Icons.search,
                              color: Colors.white.withValues(alpha: 0.5),
                              size: 20),
                          suffixIcon: _searchQuery.isNotEmpty
                              ? IconButton(
                                  icon: const Icon(Icons.clear,
                                      size: 18, color: Colors.white54),
                                  onPressed: () {
                                    _searchController.clear();
                                    setState(() => _searchQuery = '');
                                  },
                                )
                              : null,
                          filled: true,
                          fillColor: Colors.white.withValues(alpha: 0.05),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: const EdgeInsets.symmetric(
                              horizontal: 20, vertical: 14),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    _AnimatedDrillButton(
                        onPressed: filtered.isEmpty
                            ? null
                            : () => _openDrill(filtered)),
                  ],
                ),
              ),
              SizedBox(
                height: 40,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  itemCount: widget.topics.length,
                  itemBuilder: (context, i) {
                    final topic = widget.topics[i];
                    final isSelected = topic == _selectedTopic;
                    return GestureDetector(
                      onTap: () => setState(() => _selectedTopic = topic),
                      child: Container(
                        margin: const EdgeInsets.only(right: 8),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          color: isSelected
                              ? const Color(0xFF3A86FF).withValues(alpha: 0.2)
                              : Colors.white.withValues(alpha: 0.03),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: isSelected
                                ? const Color(0xFF3A86FF).withValues(alpha: 0.4)
                                : Colors.transparent,
                          ),
                        ),
                        alignment: Alignment.center,
                        child: Text(
                          _formatTopic(topic),
                          style: TextStyle(
                            color: isSelected
                                ? const Color(0xFF3A86FF)
                                : Colors.white54,
                            fontSize: 12,
                            fontWeight:
                                isSelected ? FontWeight.w600 : FontWeight.w500,
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: filtered.isEmpty
              ? const Center(
                  child: Text('No formulas found',
                      style: TextStyle(color: Colors.white54)))
              : ListView.builder(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 100),
                  itemCount: filtered.length,
                  itemBuilder: (context, i) => _FormulaCard(
                    formula: filtered[i],
                    onFavorite: () => _toggleFavorite(filtered[i]),
                  ),
                ),
        ),
      ],
    );
  }
}

class _FormulaCard extends StatelessWidget {
  final FormulaItem formula;
  final VoidCallback onFavorite;

  const _FormulaCard({required this.formula, required this.onFavorite});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.15)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  formula.name.toUpperCase(),
                  style: GoogleFonts.robotoMono(
                      color: Colors.white38, fontSize: 8, letterSpacing: 1.5),
                ),
                const SizedBox(height: 8),
                MathExpression(
                  formulaTex: formula.latexExpression,
                  tintColor: const Color(0xFF3A86FF),
                  fontSize: 18,
                ),
              ],
            ),
          ),
          IconButton(
            onPressed: onFavorite,
            icon: Icon(
              formula.isFavorite
                  ? Icons.star_rounded
                  : Icons.star_outline_rounded,
              color: formula.isFavorite ? Colors.amber : Colors.white10,
              size: 20,
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// VAULT CARD WIDGETS
// ─────────────────────────────────────────────────────────────────

class _SuggestedPaperCard extends ConsumerWidget {
  final PastPaperPack paper;
  const _SuggestedPaperCard({required this.paper});

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
                      _StatChip(
                          icon: Icons.timer_outlined,
                          label: '${paper.duration} min'),
                      const SizedBox(width: 8),
                      _StatChip(
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

class _StatChip extends StatelessWidget {
  final IconData icon;
  final String label;
  const _StatChip({required this.icon, required this.label});

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

class _CompletedPaperCard extends StatelessWidget {
  final PastPaperPack paper;
  final VoidCallback onTap;
  const _CompletedPaperCard({required this.paper, required this.onTap});

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
                      _StatChip(
                          icon: Icons.calendar_today,
                          label: _formatDate(paper.attemptedAt)),
                      const SizedBox(width: 8),
                      _StatChip(
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

class _PaperAnalysisSheet extends StatelessWidget {
  final PastPaperPack paper;
  final List<PastPaperPack> allPapers;
  const _PaperAnalysisSheet({required this.paper, required this.allPapers});

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
          child: _StatCard(
            icon: Icons.timer_outlined,
            label: 'Duration',
            value: '${paper.duration} min',
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _StatCard(
            icon: Icons.grade_outlined,
            label: 'Max Marks',
            value: '${paper.maxMarks}',
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _StatCard(
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
        ...sameSubject.map((p) => _SuggestedPaperCard(paper: p)),
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

class _StatCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  const _StatCard(
      {required this.icon, required this.label, required this.value});

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
