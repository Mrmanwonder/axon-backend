import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../models/daily_plan_task.dart';
import '../../models/models.dart';
import '../../services/app_state.dart';
import '../../services/exam_service.dart';
import '../../theme/app_theme.dart';
import '../../theme/task_type_theme.dart';
import '../../utils/layout_utils.dart';

class DashboardScreen extends ConsumerStatefulWidget {
  const DashboardScreen({super.key});

  @override
  ConsumerState<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends ConsumerState<DashboardScreen> {
  Timer? _clockTimer;
  DateTime _now = DateTime.now();
  String? _examFutureKey;
  Future<BoardFetchResult>? _examFuture;

  @override
  void initState() {
    super.initState();
    _clockTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      if (mounted) setState(() => _now = DateTime.now());
    });
  }

  @override
  void dispose() {
    _clockTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final user = ref.watch(authStateProvider).user;
    final metrics = ref.watch(metricsProvider);
    final tasks = ref.watch(todayPlanProvider);
    final isDark = AxonThemeMode.isDark;
    return Scaffold(
      backgroundColor: AxonColors.background,
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: () async {
            setState(() {
              _examFutureKey = null;
              _examFuture = null;
            });
            ref.invalidate(todayPlanProvider);
          },
          child: SingleChildScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: EdgeInsets.only(
              left: 20,
              right: 20,
              top: 20,
              bottom: bottomDockClearance(context) + 20,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeader(user, metrics),
                const SizedBox(height: 20),
                _buildCommandPanel(user, metrics, tasks),
                const SizedBox(height: 16),
                _buildExamPanel(user),
                const SizedBox(height: 16),
                _buildDailyPlanPanel(tasks),
                const SizedBox(height: 16),
                _buildPortfolioPanel(user),
                const SizedBox(height: 16),
                _buildToolkitPanel(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(UserProfile? user, MetricsState metrics) {
    final greeting = _greetingFor(_now);
    final name = (user?.displayName.trim().isNotEmpty ?? false)
        ? user!.displayName.trim()
        : 'Student';
    final board =
        (user?.board.trim().isNotEmpty ?? false) ? user!.board.trim() : 'CAIE';
    final readiness =
        (metrics.predictedPerformance * 100).clamp(0, 100).round();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '$greeting, $name',
          style: GoogleFonts.googleSans(
            color: AxonColors.textPrimary,
            fontSize: 24,
            fontWeight: FontWeight.w800,
            height: 1.05,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          '$board · ${DateFormat('EEE, d MMM').format(_now)}',
          style: GoogleFonts.googleSans(
            color: AxonColors.textTertiary,
            fontSize: 13,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            _StatusChip(
              icon: Icons.auto_graph_rounded,
              label: 'Readiness',
              value: readiness > 0 ? '$readiness%' : 'Learning',
              color: AxonColors.electricCyan,
            ),
            const SizedBox(width: 10),
            _StatusChip(
              icon: Icons.local_fire_department_rounded,
              label: 'Streak',
              value: '${metrics.consistencyStreak}d',
              color: const Color(0xFFF59E0B),
            ),
            const SizedBox(width: 10),
            _StatusChip(
              icon: Icons.book_rounded,
              label: 'Subjects',
              value: '${user?.subjects.length ?? 0}',
              color: const Color(0xFF10B981),
            ),
          ],
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            _ActionChip(
              icon: Icons.timer_rounded,
              label: 'Start focus',
              onTap: () => context.push('/timer'),
            ),
            const SizedBox(width: 10),
            _ActionChip(
              icon: Icons.auto_awesome_rounded,
              label: 'Ask Axon',
              onTap: () => context.push('/ai'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildCommandPanel(
    UserProfile? user,
    MetricsState metrics,
    AsyncValue<List<DailyPlanTask>> tasks,
  ) {
    final targetHours = user?.targetStudyHours ?? metrics.targetStudyHours;
    final activeHours = metrics.activeStudyHours;
    final focusProgress =
        targetHours > 0 ? (activeHours / targetHours).clamp(0.0, 1.0) : 0.0;
    final openTasks =
        tasks.valueOrNull?.where((task) => !task.isCompleted).length;
    final readiness =
        metrics.predictedPerformance <= 0 ? 0.0 : metrics.predictedPerformance;

    return _HubCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _IconBox(icon: Icons.bolt_rounded, color: AxonColors.electricCyan),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Today command center',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _primarySignal(metrics, openTasks),
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          _buildReadinessBar(readiness),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: _MetricTile(
                  icon: Icons.track_changes_rounded,
                  label: 'Focus today',
                  value: '${activeHours.toStringAsFixed(1)}h',
                  detail: 'Target ${targetHours.toStringAsFixed(1)}h',
                  color: const Color(0xFF10B981),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _MetricTile(
                  icon: Icons.auto_graph_rounded,
                  label: 'Focus ratio',
                  value:
                      '${(metrics.sevenDayAvgFocus * 100).clamp(0, 100).round()}%',
                  detail: '7 day trend',
                  color: const Color(0xFF14B8A6),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _MetricTile(
                  icon: Icons.pending_actions_rounded,
                  label: 'Open tasks',
                  value: openTasks == null ? '-' : '$openTasks',
                  detail: 'Daily plan',
                  color: const Color(0xFFF59E0B),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: focusProgress,
              minHeight: 6,
              backgroundColor: AxonColors.divider,
              valueColor:
                  const AlwaysStoppedAnimation<Color>(Color(0xFF10B981)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildReadinessBar(double readiness) {
    final percent = (readiness * 100).clamp(0, 100).round();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              readiness == 0 ? 'Building baseline' : '$percent% exam readiness',
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
            const Spacer(),
            Text(
              AxonColors.performanceLabel(readiness),
              style: GoogleFonts.googleSans(
                color: readiness == 0
                    ? AxonColors.textTertiary
                    : AxonColors.performanceColor(readiness),
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: readiness <= 0 ? 0.08 : readiness.clamp(0.0, 1.0),
            minHeight: 8,
            backgroundColor: AxonColors.divider,
            valueColor: AlwaysStoppedAnimation<Color>(
              readiness <= 0
                  ? AxonColors.textTertiary
                  : AxonColors.performanceColor(readiness),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildExamPanel(UserProfile? user) {
    final future = _examFutureFor(user);
    if (future == null) {
      return _HubCard(
        child: _buildEmptyExam(),
      );
    }

    return FutureBuilder<BoardFetchResult>(
      future: future,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return _HubCard(
            child: _buildLoadingExam(),
          );
        }

        final result = snapshot.data;
        final event = _nextExam(result?.events ?? const [], user);
        if (event == null) {
          return _HubCard(
            child: _buildEmptyExam(
              message: result?.error ?? 'No upcoming exams found.',
            ),
          );
        }

        final days = _daysUntil(event.startDate);
        return _HubCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _sectionHeader('NEXT PAPER'),
              const SizedBox(height: 16),
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    days <= 0 ? 'Today' : '$days',
                    style: GoogleFonts.googleSans(
                      color: _urgencyColor(days),
                      fontSize: days <= 0 ? 32 : 44,
                      fontWeight: FontWeight.w900,
                      height: 0.95,
                    ),
                  ),
                  if (days > 0) ...[
                    const SizedBox(width: 8),
                    Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Text(
                        'days',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 12),
              Text(
                event.subject.isNotEmpty ? event.subject : 'Cambridge paper',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                [
                  if (event.label.isNotEmpty) event.label,
                  DateFormat('EEE, d MMM yyyy').format(event.startDate),
                ].join(' - '),
                style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildEmptyExam({String? message}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionHeader('EXAM SCHEDULE'),
        const SizedBox(height: 12),
        Icon(Icons.event_note_rounded,
            color: AxonColors.textTertiary, size: 32),
        const SizedBox(height: 8),
        Text(
          message ??
              'Add your board and subjects to unlock date-aware planning.',
          style: GoogleFonts.googleSans(
            color: AxonColors.textTertiary,
            fontSize: 13,
          ),
        ),
        const SizedBox(height: 12),
        TextButton(
          onPressed: () => context.push('/exam/planner'),
          style: TextButton.styleFrom(
            foregroundColor: AxonColors.electricCyan,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(10),
              side: BorderSide(color: AxonColors.electricCyan.withValues(alpha: 0.3)),
            ),
          ),
          child: const Text('Open Planner',
              style: TextStyle(fontWeight: FontWeight.w700)),
        ),
      ],
    );
  }

  Widget _buildLoadingExam() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionHeader('EXAM SCHEDULE'),
        const SizedBox(height: 16),
        _SkeletonLine(),
        const SizedBox(height: 10),
        _SkeletonLine(widthFactor: 0.6),
      ],
    );
  }

  Widget _buildDailyPlanPanel(AsyncValue<List<DailyPlanTask>> tasks) {
    return _HubCard(
      child: tasks.when(
        data: (taskList) {
          final done = taskList.where((task) => task.isCompleted).length;
          final total = taskList.length;
          if (taskList.isEmpty) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _sectionHeader('DAILY PLAN'),
                const SizedBox(height: 12),
                Icon(Icons.checklist_rounded,
                    color: AxonColors.textTertiary, size: 32),
                const SizedBox(height: 8),
                Text(
                  'No tasks scheduled for today.',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 12),
                TextButton(
                  onPressed: _generatePlan,
                  style: TextButton.styleFrom(
                    foregroundColor: AxonColors.electricCyan,
                    padding:
                        const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                      side: BorderSide(
                          color: AxonColors.electricCyan.withValues(alpha: 0.3)),
                    ),
                  ),
                  child: const Text('Generate Plan',
                      style: TextStyle(fontWeight: FontWeight.w700)),
                ),
              ],
            );
          }

          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    'DAILY PLAN',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    '$done/$total',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              _buildPhaseBanner(taskList),
              const SizedBox(height: 14),
              ...taskList.take(6).map((task) => _TaskRow(
                    task: task,
                    onToggle: () => _toggleTask(task),
                    onOpen: () => _navigateToTask(task),
                  )),
              if (taskList.length > 6) ...[
                const SizedBox(height: 8),
                Text(
                  '+${taskList.length - 6} more in Study',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ],
          );
        },
        loading: () => Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _sectionHeader('DAILY PLAN'),
            const SizedBox(height: 16),
            _SkeletonLine(),
            const SizedBox(height: 10),
            _SkeletonLine(widthFactor: 0.7),
            const SizedBox(height: 10),
            _SkeletonLine(widthFactor: 0.5),
          ],
        ),
        error: (error, _) => Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _sectionHeader('DAILY PLAN'),
            const SizedBox(height: 12),
            Icon(Icons.error_outline_rounded,
                color: const Color(0xFFE11D48), size: 32),
            const SizedBox(height: 8),
            Text(
              error.toString(),
              style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary,
                fontSize: 13,
              ),
            ),
            const SizedBox(height: 12),
            TextButton(
              onPressed: () => ref.invalidate(todayPlanProvider),
              style: TextButton.styleFrom(
                foregroundColor: AxonColors.electricCyan,
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                  side: BorderSide(
                      color: AxonColors.electricCyan.withValues(alpha: 0.3)),
                ),
              ),
              child: const Text('Retry',
                  style: TextStyle(fontWeight: FontWeight.w700)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPortfolioPanel(UserProfile? user) {
    final badges = user?.badges.length ?? 0;
    final xp = user?.xp ?? 0;
    final level = user?.level ?? 1;

    return _HubCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'PORTFOLIO',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.5,
                ),
              ),
              const Spacer(),
              GestureDetector(
                onTap: () => context.push('/analysis/admissions'),
                child: Text(
                  'OPEN',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.electricCyan,
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _PortfolioMetric(
                  label: 'Level',
                  value: '$level',
                  color: const Color(0xFF8B5CF6),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _PortfolioMetric(
                  label: 'XP',
                  value: '$xp',
                  color: AxonColors.electricCyan,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _PortfolioMetric(
                  label: 'Badges',
                  value: '$badges',
                  color: const Color(0xFFF59E0B),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _InlineNavTile(
            icon: Icons.add_photo_alternate_rounded,
            title: 'Add achievement evidence',
            onTap: () => context.push('/achievements'),
          ),
        ],
      ),
    );
  }

  Widget _buildToolkitPanel() {
    return _HubCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionHeader('STUDY TOOLS'),
          const SizedBox(height: 8),
          _InlineNavTile(
            icon: Icons.fact_check_rounded,
            title: 'Past paper vault',
            onTap: () => context.push('/study/papers'),
          ),
          _InlineNavTile(
            icon: Icons.style_rounded,
            title: 'Flashcards',
            onTap: () => context.push('/study/flashcards'),
          ),
          _InlineNavTile(
            icon: Icons.psychology_alt_rounded,
            title: 'Active recall',
            onTap: () => context.push('/study/recall'),
          ),
        ],
      ),
    );
  }

  // ───────────────────────── helpers ─────────────────────────

  Widget _sectionHeader(String label) {
    return Text(
      label,
      style: GoogleFonts.googleSans(
        color: AxonColors.textTertiary,
        fontSize: 11,
        fontWeight: FontWeight.w700,
        letterSpacing: 0.5,
      ),
    );
  }

  Future<BoardFetchResult>? _examFutureFor(UserProfile? user) {
    final board = user?.board.trim() ?? '';
    if (user == null || board.isEmpty) return null;

    final key = '${user.uid}|$board|${user.subjects.join(',')}';
    if (_examFutureKey == key && _examFuture != null) return _examFuture;

    _examFutureKey = key;
    _examFuture = ExamService()
        .fetchBoardDates(board, subjects: user.subjects)
        .timeout(
          const Duration(seconds: 8),
          onTimeout: () => BoardFetchResult(
            board: board,
            sourceUrl: '',
            sources: const [],
            events: const [],
            error: 'Date sync timed out.',
          ),
        );
    return _examFuture;
  }

  ExamEvent? _nextExam(List<ExamEvent> events, UserProfile? user) {
    final today = DateTime(_now.year, _now.month, _now.day);
    final subjects = (user?.subjects ?? const <String>[])
        .map((subject) => subject.toLowerCase().trim())
        .where((subject) => subject.isNotEmpty)
        .toList();

    final upcoming = events.where((event) {
      final eventDay = DateTime(
        event.startDate.year,
        event.startDate.month,
        event.startDate.day,
      );
      if (eventDay.isBefore(today)) return false;
      if (subjects.isEmpty) return true;
      final subject = event.subject.toLowerCase();
      return subjects.any((candidate) =>
          subject.contains(candidate) || candidate.contains(subject));
    }).toList()
      ..sort((a, b) => a.startDate.compareTo(b.startDate));

    return upcoming.isEmpty ? null : upcoming.first;
  }

  String _primarySignal(MetricsState metrics, int? openTasks) {
    if (openTasks != null && openTasks > 0) {
      return '$openTasks priority ${openTasks == 1 ? 'task' : 'tasks'} waiting.';
    }
    if (metrics.activeStudyHours > 0) {
      return '${metrics.activeStudyHours.toStringAsFixed(1)} focus hours logged today.';
    }
    return 'Ready for the first focused session.';
  }

  String _greetingFor(DateTime now) {
    if (now.hour < 12) return 'Good morning';
    if (now.hour < 17) return 'Good afternoon';
    return 'Good evening';
  }

  int _daysUntil(DateTime date) {
    final today = DateTime(_now.year, _now.month, _now.day);
    final examDay = DateTime(date.year, date.month, date.day);
    return examDay.difference(today).inDays;
  }

  Color _urgencyColor(int days) {
    if (days <= 7) return const Color(0xFFE11D48);
    if (days <= 21) return const Color(0xFFF59E0B);
    return const Color(0xFF10B981);
  }

  void _navigateToTask(DailyPlanTask task) {
    context.push(task.taskType.buildRoute(task.subject));
  }

  Future<void> _toggleTask(DailyPlanTask task) async {
    try {
      await ref.read(dailyPlanServiceProvider).updateTask(
        task.copyWith(isCompleted: !task.isCompleted),
      );
      ref.invalidate(todayPlanProvider);
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('last_active_task_id', task.id);
    } catch (error) {
      debugPrint('Toggle task failed: $error');
    }
  }

  Future<void> _generatePlan() async {
    try {
      await ref.read(ensureTodayPlanProvider)(null);
      ref.invalidate(todayPlanProvider);
    } catch (error) {
      debugPrint('Generate plan failed: $error');
    }
  }

  Widget _buildPhaseBanner(List<DailyPlanTask> tasks) {
    if (tasks.isEmpty) return const SizedBox.shrink();
    final phase = tasks.first.phase;
    final int days;
    switch (phase) {
      case StudyPhase.t7MockSprint: days = 3;
      case StudyPhase.t14DeepDive: days = 10;
      default: days = 60;
    }
    final color = _urgencyColor(days);
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.flag_rounded, size: 12, color: color),
            const SizedBox(width: 6),
            Text(
              phase.label,
              style: GoogleFonts.googleSans(
                color: color,
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ────────────────────────── Shared Widgets ──────────────────────────

class _HubCard extends StatelessWidget {
  final Widget child;
  const _HubCard({required this.child, super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AxonColors.surfaceElevated,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AxonColors.divider),
      ),
      child: child,
    );
  }
}

class _IconBox extends StatelessWidget {
  final IconData icon;
  final Color color;
  const _IconBox({required this.icon, required this.color, super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Icon(icon, color: color, size: 20),
    );
  }
}

class _StatusChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;

  const _StatusChip({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AxonColors.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AxonColors.divider),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 16),
          const SizedBox(width: 6),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                label,
                style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary,
                  fontSize: 9,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                value,
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ActionChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ActionChip({
    required this.icon,
    required this.label,
    required this.onTap,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: AxonColors.electricCyan.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
              color: AxonColors.electricCyan.withValues(alpha: 0.3)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: AxonColors.electricCyan, size: 18),
            const SizedBox(width: 8),
            Text(
              label,
              style: GoogleFonts.googleSans(
                color: AxonColors.electricCyan,
                fontSize: 13,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final String detail;
  final Color color;

  const _MetricTile({
    required this.icon,
    required this.label,
    required this.value,
    required this.detail,
    required this.color,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(height: 8),
          Text(
            label,
            style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary,
              fontSize: 10,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 18,
              fontWeight: FontWeight.w900,
            ),
          ),
          Text(
            detail,
            style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary,
              fontSize: 10,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

class _PortfolioMetric extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _PortfolioMetric({
    required this.label,
    required this.value,
    required this.color,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            value,
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 18,
              fontWeight: FontWeight.w900,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary,
              fontSize: 10,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _InlineNavTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final VoidCallback onTap;

  const _InlineNavTile({
    required this.icon,
    required this.title,
    required this.onTap,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(10),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 10),
        child: Row(
          children: [
            Icon(icon, color: AxonColors.electricCyan, size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                title,
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            Icon(Icons.chevron_right_rounded,
                color: AxonColors.textTertiary, size: 18),
          ],
        ),
      ),
    );
  }
}

class _TaskRow extends StatelessWidget {
  final DailyPlanTask task;
  final VoidCallback onToggle;
  final VoidCallback onOpen;

  const _TaskRow({
    required this.task,
    required this.onToggle,
    required this.onOpen,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    final timeLabel = '${task.startTime.hour.toString().padLeft(2, '0')}:${task.startTime.minute.toString().padLeft(2, '0')} - ${task.endTime.hour.toString().padLeft(2, '0')}:${task.endTime.minute.toString().padLeft(2, '0')}';
    final typeColor = taskTypeColor(task.taskType);

    return InkWell(
      onTap: onOpen,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          children: [
            GestureDetector(
              onTap: onToggle,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                width: 26,
                height: 26,
                decoration: BoxDecoration(
                  color: task.isCompleted
                      ? const Color(0xFF10B981)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: task.isCompleted
                        ? const Color(0xFF10B981)
                        : Colors.white24,
                    width: 1.4,
                  ),
                ),
                child: task.isCompleted
                    ? Icon(Icons.check_rounded,
                        color: AxonColors.textPrimary, size: 16)
                    : null,
              ),
            ),
            const SizedBox(width: 10),
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: typeColor.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(taskTypeIcon(task.taskType), color: typeColor, size: 16),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    task.title,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.googleSans(
                      color: task.isCompleted
                          ? AxonColors.textTertiary
                          : AxonColors.textPrimary,
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      decoration: task.isCompleted
                          ? TextDecoration.lineThrough
                          : null,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    [
                      task.taskType.label,
                      timeLabel,
                    ].join(' · '),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary,
                      fontSize: 10,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Container(
              width: 6,
              height: 6,
              decoration: BoxDecoration(
                color: intensityColor(task.intensityLabel),
                shape: BoxShape.circle,
              ),
            ),
            if (task.priority == Priority.high) ...[
              const SizedBox(width: 6),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                decoration: BoxDecoration(
                  color: const Color(0xFFE11D48).withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  'High',
                  style: GoogleFonts.googleSans(
                    color: const Color(0xFFE11D48),
                    fontSize: 9,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _SkeletonLine extends StatelessWidget {
  final double widthFactor;
  const _SkeletonLine({this.widthFactor = 0.72, super.key});

  @override
  Widget build(BuildContext context) {
    return FractionallySizedBox(
      widthFactor: widthFactor,
      child: Container(
        height: 12,
        decoration: BoxDecoration(
          color: AxonColors.surfaceHighlight,
          borderRadius: BorderRadius.circular(6),
        ),
      ),
    );
  }
}
