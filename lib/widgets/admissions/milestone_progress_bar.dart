import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../models/admissions_models.dart';
import '../../theme/app_theme.dart';

class MilestoneProgressBar extends StatelessWidget {
  const MilestoneProgressBar({
    super.key,
    required this.progress,
  });

  final AdmissionsMilestoneProgress progress;

  @override
  Widget build(BuildContext context) {
    final ratio = progress.progressRatio.clamp(0.0, 1.0);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                'Admissions Milestones',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            Text(
              '${progress.progressLabel} • ${progress.blockedCount} blocked • ${progress.evidenceCount} evidence',
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontSize: 12,
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        ClipRRect(
          borderRadius: BorderRadius.circular(999),
          child: LinearProgressIndicator(
            value: ratio,
            minHeight: 10,
            backgroundColor: AxonColors.divider,
            valueColor: AlwaysStoppedAnimation<Color>(
              ratio >= 0.75
                  ? const Color(0xFF27AE60)
                  : ratio >= 0.4
                      ? const Color(0xFFF0AD4E)
                      : const Color(0xFFD9534F),
            ),
          ),
        ),
      ],
    );
  }
}
