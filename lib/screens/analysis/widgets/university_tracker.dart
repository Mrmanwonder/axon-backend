import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../models/models.dart';
import '../../../theme/app_theme.dart';
import '../../../router/app_router.dart';
import '../../../services/app_state.dart';
import '../../../services/admissions_service.dart';
import '../../../services/alert_service.dart';
import '../../../utils/layout_utils.dart';
import 'action_strip.dart';
import 'common_components.dart';

class UniversityTracker extends ConsumerWidget {
  final String? uid;

  const UniversityTracker({this.uid});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(authStateProvider).user;
    final effectiveUid = uid ?? user?.uid ?? '';
    final metrics = ref.watch(metricsProvider);

    if (effectiveUid.isEmpty) {
      return Panel(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            IconTile(
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
          return Panel(
            padding: const EdgeInsets.all(14),
            borderColor: AxonColors.accent.withValues(alpha: 0.22),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    IconTile(
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
                    CompactButton(
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
                    CompactButton(
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
                boxShadow: SpatialGlow.medium(AxonColors.accent),
              ),
              child: Padding(
                padding: const EdgeInsets.all(14),
                child: Row(
                  children: [
                    IconTile(
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
                                MiniBadge(primary.classification),
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
                    'Add University Target',
                    style: GoogleFonts.inter(
                      color: AxonColors.textPrimary,
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 16),
                  TargetField(
                    controller: universityController,
                    label: 'University',
                    hint: 'Imperial College London',
                  ),
                  const SizedBox(height: 10),
                  TargetField(
                    controller: courseController,
                    label: 'Course',
                    hint: 'Computer Science',
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Expanded(
                        child: TargetField(
                          controller: countryController,
                          label: 'Country',
                          hint: 'UK',
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: TargetField(
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

class MiniBadge extends StatelessWidget {
  final String label;

  const MiniBadge(this.label);

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

class TargetField extends StatelessWidget {
  final TextEditingController controller;
  final String label;
  final String hint;

  const TargetField({
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
