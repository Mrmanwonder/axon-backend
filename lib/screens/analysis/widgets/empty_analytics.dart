import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme/app_theme.dart';
import 'action_strip.dart';
import 'common_components.dart';

class EmptyAnalytics extends StatelessWidget {
  final VoidCallback onStartTimer;
  final VoidCallback onImportReport;

  const EmptyAnalytics({
    required this.onStartTimer,
    required this.onImportReport,
  });

  @override
  Widget build(BuildContext context) {
    return Panel(
      borderColor: AxonColors.accent.withValues(alpha: 0.25),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              IconTile(icon: Icons.insights_rounded, color: AxonColors.accent),
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
              CompactButton(
                icon: Icons.timer_rounded,
                label: 'Start Timer',
                onTap: onStartTimer,
                color: AxonColors.accent,
              ),
              CompactButton(
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
