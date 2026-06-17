import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme/app_theme.dart';

class ActionStrip extends StatelessWidget {
  final bool hasData;
  final VoidCallback onTimeline;
  final VoidCallback onTimer;
  final VoidCallback onReportCard;
  final VoidCallback onUsageAccess;

  const ActionStrip({
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
        CompactButton(
          icon: Icons.timeline_rounded,
          label: 'Exam Timeline',
          onTap: onTimeline,
          color: AxonColors.warning,
        ),
        CompactButton(
          icon: Icons.timer_rounded,
          label: 'Focus Timer',
          onTap: onTimer,
          color: AxonColors.accent,
        ),
        CompactButton(
          icon: Icons.upload_file_rounded,
          label: hasData ? 'Update Report' : 'Import Report',
          onTap: onReportCard,
          color: AxonColors.success,
        ),
        CompactButton(
          icon: Icons.phone_android_rounded,
          label: 'Screen Sync',
          onTap: onUsageAccess,
          color: AxonColors.textSecondary,
        ),
        CompactButton(
          icon: Icons.emoji_events_rounded,
          label: 'Achievements',
          onTap: () => context.push('/achievements'),
          color: AxonColors.warning,
        ),
      ],
    );
  }
}

class CompactButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final Color color;

  const CompactButton({
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
