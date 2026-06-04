// lib/screens/analysis/analysis_screen.dart

import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';

import '../../models/admissions_models.dart';
import '../../models/models.dart';
import '../../router/app_router.dart';
import '../../services/admissions_service.dart';
import '../../services/alert_service.dart';
import '../../services/app_state.dart';
import '../../services/app_usage_service.dart';
import '../../services/ask_axon_context_service.dart';
import '../../services/report_card_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/layout_utils.dart';
import '../../widgets/ask_axon_logo.dart';

class AnalysisScreen extends ConsumerWidget {
  const AnalysisScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final metrics = ref.watch(metricsProvider);
    final auth = ref.watch(authStateProvider);
    final user = auth.user;
    final style =
        user?.motivationStyle ?? MotivationStyle.positiveReinforcement;
    final score = metrics.predictedPerformance.clamp(0.0, 1.0);
    final hasData = metrics.hasData;

    return Scaffold(
      backgroundColor: Colors.transparent,
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
      floatingActionButton: Padding(
        padding: EdgeInsets.only(
          bottom: bottomDockClearance(context, extra: 0) -
              MediaQuery.of(context).padding.bottom -
              16,
        ),
        child: AskAxonOrbButton(
          isWorking: false,
          enableLongPressBuildUp: true,
          onTap: () => context.push(
            AppRoutes.ai,
            extra: {
              'title': 'Ask Axon',
              'contextFuture': AskAxonContextService.instance.buildContext(
                metrics: metrics,
                currentSubject: metrics.primarySubject,
              ),
              'motivationStyle': style,
            },
          ),
        ),
      ),
      body: Container(
        decoration: BoxDecoration(gradient: AxonGradients.backgroundGradient),
        child: SafeArea(
          child: RefreshIndicator(
            color: AxonColors.accent,
            onRefresh: () => ref.read(metricsProvider.notifier).refreshMetrics(),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final isWide = constraints.maxWidth >= 900;
                final horizontalPadding = isWide ? 32.0 : 20.0;

                return ListView(
                  physics: const AlwaysScrollableScrollPhysics(),
                  padding: EdgeInsets.fromLTRB(
                    horizontalPadding,
                    20,
                    horizontalPadding,
                    bottomDockClearance(context),
                  ),
                  children: [
                    _AnalyticsHeader(
                      metrics: metrics,
                      score: score,
                      style: style,
                      hasData: hasData,
                    ),
                    const SizedBox(height: 20),
                    if (!hasData) ...[
                      _EmptyAnalytics(
                        onStartTimer: () => context.push('/timer'),
                        onImportReport: () =>
                            ReportCardService().importReportCardAndUpdateScore(
                          context,
                          ref,
                        ),
                      ),
                      const SizedBox(height: 20),
                    ],
                    if (isWide)
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            flex: 6,
                            child: Column(
                              children: [
                                _ReadinessCard(metrics: metrics, score: score),
                                const SizedBox(height: 16),
                                _TrendCard(history: metrics.weekHistory),
                              ],
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            flex: 4,
                            child: Column(
                              children: [
                                _VitalsPanel(metrics: metrics, hasData: hasData),
                                const SizedBox(height: 16),
                                _InsightCard(
                                  score: score,
                                  style: style,
                                  metrics: metrics,
                                ),
                                const SizedBox(height: 16),
                                _UniversityTracker(uid: user?.uid),
                              ],
                            ),
                          ),
                        ],
                      )
                    else ...[
                      _ReadinessCard(metrics: metrics, score: score),
                      const SizedBox(height: 16),
                      _VitalsPanel(metrics: metrics, hasData: hasData),
                      const SizedBox(height: 16),
                      _TrendCard(history: metrics.weekHistory),
                      const SizedBox(height: 16),
                      _InsightCard(score: score, style: style, metrics: metrics),
                      const SizedBox(height: 16),
                      _UniversityTracker(uid: user?.uid),
                    ],
                    const SizedBox(height: 16),
                    _ActionStrip(
                      hasData: hasData,
                      onTimeline: () => context.push('/calendar'),
                      onTimer: () => context.push('/timer'),
                      onReportCard: () =>
                          ReportCardService().importReportCardAndUpdateScore(
                        context,
                        ref,
                      ),
                      onUsageAccess: () => AppUsageService().requestUsageAccess(),
                    ),
                  ],
                );
              },
            ),
          ),
        ),
      ),
    );
  }
}

class _AnalyticsHeader extends StatelessWidget {
  final MetricsState metrics;
  final double score;
  final MotivationStyle style;
  final bool hasData;

  const _AnalyticsHeader({
    required this.metrics,
    required this.score,
    required this.style,
    required this.hasData,
  });

  @override
  Widget build(BuildContext context) {
    final color =
        hasData ? AxonColors.performanceColor(score) : AxonColors.textTertiary;
    final subject = metrics.primarySubject.trim().isEmpty
        ? 'General'
        : metrics.primarySubject;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Analytics',
                    style: GoogleFonts.inter(
                      color: AxonColors.textPrimary,
                      fontSize: 30,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 0,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    hasData
                        ? '$subject readiness, focus, sleep, and screen-time signals'
                        : 'Connect your study signals to unlock live performance analytics',
                    style: GoogleFonts.inter(
                      color: AxonColors.textSecondary,
                      fontSize: 13,
                      height: 1.4,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 16),
            _ScorePill(score: score, color: color, hasData: hasData),
          ],
        ),
        const SizedBox(height: 16),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _StatusChip(
              icon: Icons.speed_rounded,
              label: hasData ? AxonColors.performanceLabel(score) : 'No data',
              color: color,
            ),
            _StatusChip(
              icon: Icons.local_fire_department_rounded,
              label: '${metrics.consistencyStreak} day streak',
              color: AxonColors.warning,
            ),
            _StatusChip(
              icon: Icons.psychology_rounded,
              label: style.displayName,
              color: AxonColors.accent,
            ),
          ],
        ),
      ],
    ).animateIf(PageIntroService.shouldAnimate('analysis_header'), [
      (a) => a.fadeIn(duration: 260.ms).slideY(begin: 0.04, end: 0),
    ]);
  }
}

class _ScorePill extends StatelessWidget {
  final double score;
  final Color color;
  final bool hasData;

  const _ScorePill({
    required this.score,
    required this.color,
    required this.hasData,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 84,
      height: 84,
      decoration: BoxDecoration(
        color: AxonColors.surfaceElevated.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: color.withValues(alpha: 0.28)),
        boxShadow: [
          BoxShadow(
            color: color.withValues(alpha: 0.12),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            blurRadius: 30,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Stack(
        alignment: Alignment.center,
        children: [
          SizedBox(
            width: 58,
            height: 58,
            child: CircularProgressIndicator(
              value: hasData ? score : 0,
              strokeWidth: 5,
              color: color,
              backgroundColor: AxonColors.divider.withValues(alpha: 0.5),
              strokeCap: StrokeCap.round,
            ),
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                hasData ? '${(score * 100).round()}' : '--',
                style: GoogleFonts.inter(
                  color: AxonColors.textPrimary,
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  height: 1,
                ),
              ),
              Text(
                'score',
                style: GoogleFonts.inter(
                  color: AxonColors.textTertiary,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;

  const _StatusChip({
    required this.icon,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.18)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 15),
          const SizedBox(width: 6),
          Text(
            label,
            style: GoogleFonts.inter(
              color: AxonColors.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class _EmptyAnalytics extends StatelessWidget {
  final VoidCallback onStartTimer;
  final VoidCallback onImportReport;

  const _EmptyAnalytics({
    required this.onStartTimer,
    required this.onImportReport,
  });

  @override
  Widget build(BuildContext context) {
    return _Panel(
      borderColor: AxonColors.accent.withValues(alpha: 0.25),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _IconTile(icon: Icons.insights_rounded, color: AxonColors.accent),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'No analytics signal yet',
                      style: GoogleFonts.inter(
                        color: AxonColors.textPrimary,
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Run a study session or import a report card to build your performance model.',
                      style: GoogleFonts.inter(
                        color: AxonColors.textSecondary,
                        fontSize: 13,
                        height: 1.45,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _CompactButton(
                icon: Icons.timer_rounded,
                label: 'Start Timer',
                onTap: onStartTimer,
                color: AxonColors.accent,
              ),
              _CompactButton(
                icon: Icons.upload_file_rounded,
                label: 'Import Report',
                onTap: onImportReport,
                color: AxonColors.warning,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ReadinessCard extends StatelessWidget {
  final MetricsState metrics;
  final double score;

  const _ReadinessCard({
    required this.metrics,
    required this.score,
  });

  @override
  Widget build(BuildContext context) {
    final color = AxonColors.performanceColor(score);
    final activeProgress = metrics.targetStudyHours <= 0
        ? 0.0
        : (metrics.activeStudyHours / metrics.targetStudyHours)
            .clamp(0.0, 1.0)
            .toDouble();

    return _Panel(
      padding: const EdgeInsets.all(20),
      borderColor: color.withValues(alpha: 0.25),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _IconTile(icon: Icons.monitor_heart_rounded, color: color),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Readiness Index',
                      style: GoogleFonts.inter(
                        color: AxonColors.textPrimary,
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      _readinessSummary(metrics, score),
                      style: GoogleFonts.inter(
                        color: AxonColors.textSecondary,
                        fontSize: 12,
                        height: 1.35,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Flexible(
                child: Text(
                  '${(score * 100).round()}',
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.inter(
                    color: color,
                    fontSize: 52,
                    fontWeight: FontWeight.w900,
                    height: 0.95,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Padding(
                padding: const EdgeInsets.only(bottom: 7),
                child: Text(
                  '/100',
                  style: GoogleFonts.inter(
                    color: AxonColors.textTertiary,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              const Spacer(),
              _DeltaText(delta: metrics.delta),
            ],
          ),
          const SizedBox(height: 20),
          _SignalProgress(
            label: 'Study load',
            value: activeProgress,
            display:
                '${metrics.activeStudyHours.toStringAsFixed(1)}h / ${metrics.targetStudyHours.toStringAsFixed(1)}h',
            color: AxonColors.accent,
          ),
          const SizedBox(height: 12),
          _SignalProgress(
            label: 'Syllabus coverage',
            value: (metrics.syllabusCoverage / 100).clamp(0.0, 1.0).toDouble(),
            display: '${metrics.syllabusCoverage.round()}%',
            color: AxonColors.success,
          ),
          const SizedBox(height: 12),
          _SignalProgress(
            label: 'Stress control',
            value: (1 - (metrics.stressLevel / 10).clamp(0.0, 1.0)).toDouble(),
            display: '${metrics.stressLevel.toStringAsFixed(1)}/10',
            color: metrics.stressLevel >= 7
                ? AxonColors.error
                : AxonColors.warning,
          ),
        ],
      ),
    ).animateIf(PageIntroService.shouldAnimate('analysis_readiness'), [
      (a) => a.fadeIn(delay: 80.ms).slideY(begin: 0.04, end: 0),
    ]);
  }

  String _readinessSummary(MetricsState metrics, double score) {
    if (metrics.sleepHours > 0 && metrics.sleepHours < 6) {
      return 'Sleep is the main limiter. Keep heavy work short today.';
    }
    if (metrics.screenTimeHours > 6) {
      return 'Screen time is high. Protect focus blocks from context switching.';
    }
    if (score >= 0.75) {
      return 'Good operating state. Use the window for high-yield work.';
    }
    return 'Model confidence is building. Add one clean study signal today.';
  }
}

class _DeltaText extends StatelessWidget {
  final String delta;

  const _DeltaText({required this.delta});

  @override
  Widget build(BuildContext context) {
    final text = delta.trim().isEmpty ? 'No recent delta' : delta.trim();
    return ConstrainedBox(
      constraints: const BoxConstraints(maxWidth: 180),
      child: Text(
        text,
        textAlign: TextAlign.right,
        maxLines: 3,
        overflow: TextOverflow.ellipsis,
        style: GoogleFonts.inter(
          color: AxonColors.textTertiary,
          fontSize: 11,
          height: 1.35,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }
}

class _SignalProgress extends StatelessWidget {
  final String label;
  final double value;
  final String display;
  final Color color;

  const _SignalProgress({
    required this.label,
    required this.value,
    required this.display,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              label,
              style: GoogleFonts.inter(
                color: AxonColors.textSecondary,
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
            const Spacer(),
            Text(
              display,
              style: GoogleFonts.inter(
                color: color,
                fontSize: 12,
                fontWeight: FontWeight.w800,
              ),
            ),
          ],
        ),
        const SizedBox(height: 7),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: value.clamp(0.0, 1.0),
            minHeight: 6,
            color: color,
            backgroundColor: AxonColors.divider.withValues(alpha: 0.4),
          ),
        ),
      ],
    );
  }
}

class _VitalsPanel extends ConsumerWidget {
  final MetricsState metrics;
  final bool hasData;

  const _VitalsPanel({
    required this.metrics,
    required this.hasData,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return _Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _SectionTitle(
            title: "Today's Vitals",
            trailing: hasData ? 'Live model inputs' : 'Waiting for input',
          ),
          const SizedBox(height: 14),
          GridView(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
              maxCrossAxisExtent: 190,
              mainAxisExtent: 118,
              crossAxisSpacing: 10,
              mainAxisSpacing: 10,
            ),
            children: [
              _VitalTile(
                icon: Icons.bedtime_rounded,
                label: 'Sleep',
                value: hasData ? metrics.sleepHours.toStringAsFixed(1) : '--',
                unit: 'hrs',
                sublabel: hasData
                    ? 'Avg ${metrics.sevenDayAvgSleep.toStringAsFixed(1)}h'
                    : 'Tap to set',
                color: AxonColors.accent,
                onTap: () => _showSleepSheet(context, ref, metrics.sleepHours),
              ),
              _VitalTile(
                icon: Icons.phone_android_rounded,
                label: 'Screen',
                value:
                    hasData ? metrics.screenTimeHours.toStringAsFixed(1) : '--',
                unit: 'hrs',
                sublabel: hasData ? 'Device signal' : 'Grant access',
                color: metrics.screenTimeHours > 6
                    ? AxonColors.error
                    : AxonColors.textSecondary,
                onTap: () => AppUsageService().requestUsageAccess(),
              ),
              _VitalTile(
                icon: Icons.assignment_turned_in_rounded,
                label: 'Mock',
                value: hasData ? metrics.mockScore.toStringAsFixed(0) : '--',
                unit: '%',
                sublabel:
                    metrics.mockScore >= 70 ? 'Above target' : 'Needs work',
                color: metrics.mockScore >= 70
                    ? AxonColors.success
                    : AxonColors.warning,
                onTap: () => ReportCardService().importReportCardAndUpdateScore(
                  context,
                  ref,
                ),
              ),
              _VitalTile(
                icon: Icons.center_focus_strong_rounded,
                label: 'Focus',
                value: hasData
                    ? '${(metrics.sevenDayAvgFocus * 100).round()}'
                    : '--',
                unit: '%',
                sublabel: '7-day average',
                color: AxonColors.accent,
              ),
            ],
          ),
        ],
      ),
    ).animateIf(PageIntroService.shouldAnimate('analysis_vitals'), [
      (a) => a.fadeIn(delay: 140.ms),
    ]);
  }

  void _showSleepSheet(BuildContext context, WidgetRef ref, double current) {
    var value = current <= 0 ? 7.0 : current;
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      barrierColor: Colors.black54,
      builder: (sheetContext) {
        return _GlassSheet(
          child: StatefulBuilder(
            builder: (context, setState) => Padding(
              padding: EdgeInsets.fromLTRB(
                20,
                14,
                20,
                24 + MediaQuery.of(sheetContext).viewInsets.bottom,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 36,
                      height: 4,
                      decoration: BoxDecoration(
                        color: AxonColors.divider,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  Text(
                    'Sleep Hours',
                    style: GoogleFonts.inter(
                      color: AxonColors.textPrimary,
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '${value.toStringAsFixed(1)} hrs',
                    style: GoogleFonts.inter(
                      color: AxonColors.accent,
                      fontSize: 34,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                  Slider(
                    min: 0,
                    max: 12,
                    divisions: 48,
                    value: value.clamp(0, 12),
                    activeColor: AxonColors.accent,
                    inactiveColor: AxonColors.divider,
                    onChanged: (next) {
                      setState(() => value = next);
                      ref.read(metricsProvider.notifier).updateSleep(next);
                    },
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class _VitalTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final String unit;
  final String sublabel;
  final Color color;
  final VoidCallback? onTap;

  const _VitalTile({
    required this.icon,
    required this.label,
    required this.value,
    required this.unit,
    required this.sublabel,
    required this.color,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Ink(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated.withValues(alpha: 0.6),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: AxonColors.divider.withValues(alpha: 0.4),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(icon, color: color, size: 18),
                  const Spacer(),
                  if (onTap != null)
                    Icon(
                      Icons.chevron_right_rounded,
                      color: AxonColors.textTertiary,
                      size: 18,
                    ),
                ],
              ),
              const Spacer(),
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Flexible(
                    child: Text(
                      value,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.inter(
                        color: AxonColors.textPrimary,
                        fontSize: 24,
                        fontWeight: FontWeight.w900,
                        height: 1,
                      ),
                    ),
                  ),
                  const SizedBox(width: 3),
                  Padding(
                    padding: const EdgeInsets.only(bottom: 2),
                    child: Text(
                      unit,
                      style: GoogleFonts.inter(
                        color: AxonColors.textTertiary,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              Text(
                label,
                style: GoogleFonts.inter(
                  color: AxonColors.textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                sublabel,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.inter(
                  color: AxonColors.textTertiary,
                  fontSize: 10,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _TrendCard extends StatelessWidget {
  final List<DailyMetrics> history;

  const _TrendCard({required this.history});

  @override
  Widget build(BuildContext context) {
    return _Panel(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _SectionTitle(
              title: '7-Day Trajectory', trailing: 'Focus score'),
          const SizedBox(height: 16),
          SizedBox(
            height: 210,
            child: history.isEmpty ? const _EmptyTrend() : _TrendChart(history),
          ),
        ],
      ),
    ).animateIf(PageIntroService.shouldAnimate('analysis_trend'), [
      (a) => a.fadeIn(delay: 180.ms),
    ]);
  }
}

class _EmptyTrend extends StatelessWidget {
  const _EmptyTrend();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.show_chart_rounded,
              color: AxonColors.textTertiary, size: 28),
          const SizedBox(height: 8),
          Text(
            'Start a study session to build the trend line.',
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              color: AxonColors.textTertiary,
              fontSize: 12,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}

class _TrendChart extends StatelessWidget {
  final List<DailyMetrics> history;

  const _TrendChart(this.history);

  @override
  Widget build(BuildContext context) {
    final visible =
        history.length > 7 ? history.sublist(history.length - 7) : history;
    final spots = <FlSpot>[];
    final labels = <int, String>{};
    final formatter = DateFormat('EEE');

    for (var i = 0; i < visible.length; i++) {
      final entry = visible[i];
      spots.add(FlSpot(
        i.toDouble(),
        entry.predictedPerformance.clamp(0.0, 1.0),
      ));
      labels[i] = formatter.format(entry.date);
    }

    return LineChart(
      LineChartData(
        minX: 0,
        maxX: math.max(1, visible.length - 1).toDouble(),
        minY: 0,
        maxY: 1,
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: 0.25,
          getDrawingHorizontalLine: (_) => FlLine(
            color: AxonColors.divider.withValues(alpha: 0.3),
            strokeWidth: 0.8,
          ),
        ),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          topTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 34,
              interval: 0.5,
              getTitlesWidget: (value, _) => Text(
                '${(value * 100).round()}',
                style: GoogleFonts.inter(
                  color: AxonColors.textTertiary,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 28,
              interval: 1,
              getTitlesWidget: (value, _) {
                final label = labels[value.round()];
                if (label == null) return const SizedBox.shrink();
                return Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    label,
                    style: GoogleFonts.inter(
                      color: AxonColors.textTertiary,
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                );
              },
            ),
          ),
        ),
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipItems: (items) => items
                .map(
                  (item) => LineTooltipItem(
                    '${(item.y * 100).round()}%',
                    GoogleFonts.inter(
                      color: AxonColors.accent,
                      fontSize: 12,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                )
                .toList(),
          ),
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            curveSmoothness: 0.28,
            color: AxonColors.accent,
            barWidth: 3,
            dotData: FlDotData(
              show: true,
              getDotPainter: (spot, _, __, ___) => FlDotCirclePainter(
                radius: 4,
                color: AxonColors.performanceColor(spot.y),
                strokeWidth: 1.5,
                strokeColor: AxonColors.surfaceElevated,
              ),
            ),
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  AxonColors.accent.withValues(alpha: 0.18),
                  AxonColors.accent.withValues(alpha: 0.02),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _InsightCard extends StatelessWidget {
  final double score;
  final MotivationStyle style;
  final MetricsState metrics;

  const _InsightCard({
    required this.score,
    required this.style,
    required this.metrics,
  });

  @override
  Widget build(BuildContext context) {
    final color = AxonColors.performanceColor(score);
    final title = PerformanceState.getTitle(score: score, style: style);
    final message = PerformanceState.getMessage(score: score, style: style);

    return _Panel(
      borderColor: color.withValues(alpha: 0.25),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _IconTile(icon: Icons.auto_awesome_rounded, color: color),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  title,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.inter(
                    color: color,
                    fontSize: 15,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            message,
            style: GoogleFonts.inter(
              color: AxonColors.textSecondary,
              fontSize: 13,
              height: 1.55,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 14),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _StatusChip(
                icon: Icons.tune_rounded,
                label: style.displayName,
                color: color,
              ),
              _StatusChip(
                icon: Icons.subject_rounded,
                label: metrics.primarySubject.isEmpty
                    ? 'General'
                    : metrics.primarySubject,
                color: AxonColors.accent,
              ),
            ],
          ),
        ],
      ),
    ).animateIf(PageIntroService.shouldAnimate('analysis_insight'), [
      (a) => a.fadeIn(delay: 220.ms),
    ]);
  }
}

class _ActionStrip extends StatelessWidget {
  final bool hasData;
  final VoidCallback onTimeline;
  final VoidCallback onTimer;
  final VoidCallback onReportCard;
  final VoidCallback onUsageAccess;

  const _ActionStrip({
    required this.hasData,
    required this.onTimeline,
    required this.onTimer,
    required this.onReportCard,
    required this.onUsageAccess,
  });

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 10,
      runSpacing: 10,
      children: [
        _CompactButton(
          icon: Icons.timeline_rounded,
          label: 'Exam Timeline',
          onTap: onTimeline,
          color: AxonColors.warning,
        ),
        _CompactButton(
          icon: Icons.timer_rounded,
          label: 'Focus Timer',
          onTap: onTimer,
          color: AxonColors.accent,
        ),
        _CompactButton(
          icon: Icons.upload_file_rounded,
          label: hasData ? 'Update Report' : 'Import Report',
          onTap: onReportCard,
          color: AxonColors.success,
        ),
        _CompactButton(
          icon: Icons.phone_android_rounded,
          label: 'Screen Sync',
          onTap: onUsageAccess,
          color: AxonColors.textSecondary,
        ),
        _CompactButton(
          icon: Icons.emoji_events_rounded,
          label: 'Achievements',
          onTap: () => context.push('/achievements'),
          color: AxonColors.warning,
        ),
      ],
    );
  }
}

class _CompactButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final Color color;

  const _CompactButton({
    required this.icon,
    required this.label,
    required this.onTap,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: onTap,
        child: Ink(
          height: 42,
          padding: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.09),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: color.withValues(alpha: 0.2)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: color, size: 17),
              const SizedBox(width: 8),
              Text(
                label,
                style: GoogleFonts.inter(
                  color: AxonColors.textPrimary,
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _UniversityTracker extends ConsumerWidget {
  final String? uid;

  const _UniversityTracker({this.uid});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(authStateProvider).user;
    final effectiveUid = uid ?? user?.uid ?? '';
    final metrics = ref.watch(metricsProvider);

    if (effectiveUid.isEmpty) {
      return _Panel(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            _IconTile(
                icon: Icons.school_rounded, color: AxonColors.textTertiary),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                'Sign in to save university targets.',
                style: GoogleFonts.inter(
                  color: AxonColors.textSecondary,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      );
    }

    return StreamBuilder<List<AdmissionsTarget>>(
      stream: AdmissionsService().watchTargets(effectiveUid),
      builder: (context, snapshot) {
        final service = AdmissionsService();
        final targets = snapshot.data ?? [];
        if (targets.isEmpty) {
          return _Panel(
            padding: const EdgeInsets.all(14),
            borderColor: AxonColors.accent.withValues(alpha: 0.22),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    _IconTile(
                      icon: Icons.school_rounded,
                      color: AxonColors.accent,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'University Target',
                            style: GoogleFonts.inter(
                              color: AxonColors.textPrimary,
                              fontSize: 14,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                          const SizedBox(height: 3),
                          Text(
                            'Add a target to generate admissions milestones.',
                            style: GoogleFonts.inter(
                              color: AxonColors.textSecondary,
                              fontSize: 12,
                              height: 1.35,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    _CompactButton(
                      icon: Icons.add_rounded,
                      label: 'Add Target',
                      onTap: () => _showTargetSheet(
                        context,
                        service: service,
                        uid: effectiveUid,
                        metrics: metrics,
                      ),
                      color: AxonColors.accent,
                    ),
                    _CompactButton(
                      icon: Icons.open_in_new_rounded,
                      label: 'Open Tracker',
                      onTap: () => context.push(
                        AppRoutes.admissions,
                        extra: {'uid': effectiveUid},
                      ),
                      color: AxonColors.warning,
                    ),
                  ],
                ),
              ],
            ),
          );
        }

        final primary = targets.first;
        final progress = primary.readinessScore.clamp(0.0, 1.0);
        return Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(14),
            onTap: () => context.push(
              AppRoutes.admissions,
              extra: {'uid': effectiveUid},
            ),
            child: Ink(
              decoration: BoxDecoration(
                color: AxonColors.surfaceElevated.withValues(alpha: 0.75),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                  color: AxonColors.divider.withValues(alpha: 0.5),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.25),
                    blurRadius: 24,
                    offset: const Offset(0, 8),
                  ),
                  BoxShadow(
                    color: AxonColors.accent.withValues(alpha: 0.06),
                    blurRadius: 40,
                    offset: const Offset(0, 16),
                  ),
                ],
              ),
              child: Padding(
                padding: const EdgeInsets.all(14),
                child: Row(
                  children: [
                    _IconTile(
                      icon: Icons.school_rounded,
                      color: AxonColors.accent,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Expanded(
                                child: Text(
                                  'Target University',
                                  style: GoogleFonts.inter(
                                    color: AxonColors.textTertiary,
                                    fontSize: 11,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ),
                              if (primary.classification.isNotEmpty)
                                _MiniBadge(primary.classification),
                            ],
                          ),
                          const SizedBox(height: 3),
                          Text(
                            primary.universityName,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.inter(
                              color: AxonColors.textPrimary,
                              fontSize: 14,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                          if (primary.courseName.isNotEmpty) ...[
                            const SizedBox(height: 2),
                            Text(
                              primary.courseName,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.inter(
                                color: AxonColors.textSecondary,
                                fontSize: 12,
                              ),
                            ),
                          ],
                          const SizedBox(height: 8),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: progress,
                              minHeight: 5,
                              color: AxonColors.accent,
                              backgroundColor: AxonColors.divider.withValues(alpha: 0.4),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 10),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        if (primary.deadlineAt != null)
                          Text(
                            _formatDeadline(primary.deadlineAt!),
                            style: GoogleFonts.inter(
                              color: AxonColors.warning,
                              fontSize: 12,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        const SizedBox(height: 8),
                        Icon(
                          Icons.chevron_right_rounded,
                          color: AxonColors.textTertiary,
                          size: 20,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ).animateIf(PageIntroService.shouldAnimate('analysis_university'), [
          (a) => a.fadeIn(delay: 260.ms),
        ]);
      },
    );
  }

  Future<void> _showTargetSheet(
    BuildContext context, {
    required AdmissionsService service,
    required String uid,
    required MetricsState metrics,
  }) async {
    final universityController = TextEditingController();
    final courseController = TextEditingController();
    final countryController = TextEditingController(text: 'UK');
    final deadlineController = TextEditingController();
    var classification = 'match';

    final shouldSave = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      barrierColor: Colors.black54,
      builder: (sheetContext) {
        return _GlassSheet(
          child: StatefulBuilder(
            builder: (context, setState) => Padding(
              padding: EdgeInsets.fromLTRB(
                20,
                14,
                20,
                24 + MediaQuery.of(sheetContext).viewInsets.bottom,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 36,
                      height: 4,
                      decoration: BoxDecoration(
                        color: AxonColors.divider,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  Text(
                    'Add University Target',
                    style: GoogleFonts.inter(
                      color: AxonColors.textPrimary,
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 16),
                  _TargetField(
                    controller: universityController,
                    label: 'University',
                    hint: 'Imperial College London',
                  ),
                  const SizedBox(height: 10),
                  _TargetField(
                    controller: courseController,
                    label: 'Course',
                    hint: 'Computer Science',
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Expanded(
                        child: _TargetField(
                          controller: countryController,
                          label: 'Country',
                          hint: 'UK',
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: _TargetField(
                          controller: deadlineController,
                          label: 'Deadline',
                          hint: '2027-01-15',
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    children: ['safety', 'match', 'reach'].map((value) {
                      final selected = classification == value;
                      return ChoiceChip(
                        selected: selected,
                        label: Text(value.toUpperCase()),
                        onSelected: (_) => setState(() => classification = value),
                        selectedColor: AxonColors.accent.withValues(alpha: 0.18),
                        backgroundColor: AxonColors.surfaceHighlight.withValues(alpha: 0.5),
                        labelStyle: GoogleFonts.inter(
                          color: selected
                              ? AxonColors.accent
                              : AxonColors.textSecondary,
                          fontSize: 11,
                          fontWeight: FontWeight.w800,
                        ),
                        side: BorderSide(
                          color:
                              selected ? AxonColors.accent : AxonColors.divider.withValues(alpha: 0.4),
                        ),
                      );
                    }).toList(),
                  ),
                  const SizedBox(height: 18),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed: () {
                        if (universityController.text.trim().isEmpty ||
                            courseController.text.trim().isEmpty) {
                          return;
                        }
                        Navigator.of(sheetContext).pop(true);
                      },
                      icon: const Icon(Icons.check_rounded),
                      label: const Text('SAVE TARGET'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );

    if (shouldSave != true) return;
    final deadline = DateTime.tryParse(deadlineController.text.trim());
    try {
      await service.createManualTarget(
        uid: uid,
        universityName: universityController.text,
        courseName: courseController.text,
        country: countryController.text,
        classification: classification,
        deadlineAt: deadline,
        readinessScore: metrics.predictedPerformance.clamp(0.0, 1.0),
      );
      if (!context.mounted) return;
      AlertService.showSuccess(
        context,
        'Target saved',
        'AXON created milestones for ${universityController.text.trim()}.',
      );
    } catch (e) {
      if (!context.mounted) return;
      AlertService.showError(
        context,
        'Could not save target',
        e.toString(),
      );
    }
  }

  String _formatDeadline(DateTime deadline) {
    final diff = deadline.difference(DateTime.now());
    if (diff.isNegative) return 'Passed';
    if (diff.inDays == 0) return 'Today';
    if (diff.inDays == 1) return 'Tomorrow';
    if (diff.inDays < 30) return '${diff.inDays}d left';
    if (diff.inDays < 365) return '${(diff.inDays / 30).round()}mo left';
    return '${(diff.inDays / 365).round()}y left';
  }
}

class _MiniBadge extends StatelessWidget {
  final String label;

  const _MiniBadge(this.label);

  @override
  Widget build(BuildContext context) {
    final normalized = label.trim().isEmpty ? 'target' : label.trim();
    final color = switch (normalized.toLowerCase()) {
      'safety' => AxonColors.success,
      'reach' => AxonColors.warning,
      _ => AxonColors.accent,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withValues(alpha: 0.18)),
      ),
      child: Text(
        normalized.toUpperCase(),
        style: GoogleFonts.inter(
          color: color,
          fontSize: 9,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }
}

class _TargetField extends StatelessWidget {
  final TextEditingController controller;
  final String label;
  final String hint;

  const _TargetField({
    required this.controller,
    required this.label,
    required this.hint,
  });

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      style: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 13,
        fontWeight: FontWeight.w600,
      ),
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        labelStyle: GoogleFonts.inter(color: AxonColors.textTertiary),
        hintStyle: GoogleFonts.inter(color: AxonColors.textTertiary),
        filled: true,
        fillColor: AxonColors.surfaceHighlight.withValues(alpha: 0.6),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(color: AxonColors.divider.withValues(alpha: 0.5)),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(color: AxonColors.divider.withValues(alpha: 0.5)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(color: AxonColors.accent),
        ),
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String title;
  final String? trailing;

  const _SectionTitle({
    required this.title,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            title,
            style: GoogleFonts.inter(
              color: AxonColors.textPrimary,
              fontSize: 15,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
        if (trailing != null)
          Text(
            trailing!,
            style: GoogleFonts.inter(
              color: AxonColors.textTertiary,
              fontSize: 11,
              fontWeight: FontWeight.w700,
            ),
          ),
      ],
    );
  }
}

class _IconTile extends StatelessWidget {
  final IconData icon;
  final Color color;

  const _IconTile({
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 38,
      height: 38,
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.18)),
      ),
      child: Icon(icon, color: color, size: 20),
    );
  }
}

class _Panel extends StatelessWidget {
  final Widget child;
  final EdgeInsets padding;
  final Color? borderColor;

  const _Panel({
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.borderColor,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(14),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 16, sigmaY: 16),
        child: Container(
          width: double.infinity,
          padding: padding,
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated.withValues(alpha: 0.72),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: borderColor ?? AxonColors.divider.withValues(alpha: 0.45),
              width: 0.5,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.28),
                blurRadius: 24,
                offset: const Offset(0, 8),
              ),
              BoxShadow(
                color: (borderColor ?? AxonColors.accent).withValues(alpha: 0.05),
                blurRadius: 40,
                offset: const Offset(0, 16),
              ),
            ],
          ),
          child: child,
        ),
      ),
    );
  }
}

class _GlassSheet extends StatelessWidget {
  final Widget child;

  const _GlassSheet({required this.child});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(top: Radius.circular(18)),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated.withValues(alpha: 0.88),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(18)),
            border: Border(
              top: BorderSide(
                color: AxonColors.divider.withValues(alpha: 0.4),
                width: 0.5,
              ),
            ),
          ),
          child: child,
        ),
      ),
    );
  }
}
