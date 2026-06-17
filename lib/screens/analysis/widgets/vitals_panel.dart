import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../services/app_state.dart';
import '../../../services/app_usage_service.dart';
import '../../../services/report_card_service.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/layout_utils.dart';
import 'common_components.dart';

class VitalsPanel extends ConsumerWidget {
  final MetricsState metrics;
  final bool hasData;

  const VitalsPanel({
    required this.metrics,
    required this.hasData,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SectionTitle(
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
              VitalTile(
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
              VitalTile(
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
              VitalTile(
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
              VitalTile(
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
      barrierColor: const Color(0x8A000000),
      builder: (sheetContext) {
        return GlassSheet(
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

class VitalTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final String unit;
  final String sublabel;
  final Color color;
  final VoidCallback? onTap;

  const VitalTile({
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
