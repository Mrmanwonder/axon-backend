import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../models/models.dart';
import '../../../theme/app_theme.dart';
import '../../../services/app_state.dart';
import '../../../utils/layout_utils.dart';
import 'common_components.dart';

class AnalyticsHeader extends StatelessWidget {
  final MetricsState metrics;
  final double score;
  final MotivationStyle style;
  final bool hasData;

  const AnalyticsHeader({
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
            ScorePill(score: score, color: color, hasData: hasData),
          ],
        ),
        const SizedBox(height: 16),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            StatusChip(
              icon: Icons.speed_rounded,
              label: hasData ? AxonColors.performanceLabel(score) : 'No data',
              color: color,
            ),
            StatusChip(
              icon: Icons.local_fire_department_rounded,
              label: '${metrics.consistencyStreak} day streak',
              color: AxonColors.warning,
            ),
            StatusChip(
              icon: Icons.psychology_rounded,
              label: style.displayName,
              color: AxonColors.accent,
            ),
          ],
        ),
      ],
    );
  }
}

class ScorePill extends StatelessWidget {
  final double score;
  final Color color;
  final bool hasData;

  const ScorePill({
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
            color: const Color(0x4D000000),
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
    ).animateIf(PageIntroService.shouldAnimate('analysis_header'), [
      (a) => a.fadeIn(duration: 260.ms).slideY(begin: 0.04, end: 0),
    ]);
  }
}
