import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../models/models.dart';
import '../../../theme/app_theme.dart';
import '../../../services/app_state.dart';
import '../../../utils/layout_utils.dart';
import 'common_components.dart';

class InsightCard extends StatelessWidget {
  final double score;
  final MotivationStyle style;
  final MetricsState metrics;

  const InsightCard({
    required this.score,
    required this.style,
    required this.metrics,
  });

  @override
  Widget build(BuildContext context) {
    final color = AxonColors.performanceColor(score);
    final title = PerformanceState.getTitle(score: score, style: style);
    final message = PerformanceState.getMessage(score: score, style: style);

    return Panel(
      borderColor: color.withValues(alpha: 0.25),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              IconTile(icon: Icons.auto_awesome_rounded, color: color),
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
              StatusChip(
                icon: Icons.tune_rounded,
                label: style.displayName,
                color: color,
              ),
              StatusChip(
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
