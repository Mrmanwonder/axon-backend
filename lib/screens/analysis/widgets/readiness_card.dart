import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../services/app_state.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/layout_utils.dart';
import 'common_components.dart';

class ReadinessCard extends StatelessWidget {
  final MetricsState metrics;
  final double score;

  const ReadinessCard({
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

    return Panel(
      padding: const EdgeInsets.all(20),
      borderColor: color.withValues(alpha: 0.25),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              IconTile(icon: Icons.monitor_heart_rounded, color: color),
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
              Text(
                '${(score * 100).round()}',
                style: GoogleFonts.inter(
                  color: color,
                  fontSize: 52,
                  fontWeight: FontWeight.w900,
                  height: 0.95,
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
              DeltaText(delta: metrics.delta),
            ],
          ),
          const SizedBox(height: 20),
          SignalProgress(
            label: 'Study load',
            value: activeProgress,
            display:
                '${metrics.activeStudyHours.toStringAsFixed(1)}h / ${metrics.targetStudyHours.toStringAsFixed(1)}h',
            color: AxonColors.accent,
          ),
          const SizedBox(height: 12),
          SignalProgress(
            label: 'Syllabus coverage',
            value: (metrics.syllabusCoverage / 100).clamp(0.0, 1.0).toDouble(),
            display: '${metrics.syllabusCoverage.round()}%',
            color: AxonColors.success,
          ),
          const SizedBox(height: 12),
          SignalProgress(
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

class DeltaText extends StatelessWidget {
  final String delta;

  const DeltaText({required this.delta});

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

class SignalProgress extends StatelessWidget {
  final String label;
  final double value;
  final String display;
  final Color color;

  const SignalProgress({
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
