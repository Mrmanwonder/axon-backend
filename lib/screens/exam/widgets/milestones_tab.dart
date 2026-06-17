// lib/screens/exam/widgets/milestones_tab.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../services/exam_planner_service.dart';
import '../../../services/exam_planner_repository.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/layout_utils.dart';
import '../../../widgets/common/rose_loader.dart';

TextDecoration? _lineThrough(bool condition) =>
    condition ? TextDecoration.lineThrough : null;

class MilestonesTab extends ConsumerWidget {
  const MilestonesTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: RoseLoader(size: 24)),
      error: (error, _) => Center(child: Text('Error: $error')),
      data: (data) {
        final milestones = data.milestones;
        final completed = milestones.where((m) => m.isCompleted).length;

        return SingleChildScrollView(
          padding:
              EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildProgressHeader(context, completed, milestones.length),
              const SizedBox(height: 20),
              ...milestones.map((m) => MilestoneCard(
                    milestone: m,
                    onComplete: () async {
                      await ref
                          .read(examPlannerProvider.notifier)
                          .completeMilestone(m.id);
                    },
                  )),
            ],
          ),
        );
      },
    );
  }

  Widget _buildProgressHeader(BuildContext context, int completed, int total) {
    final progress = total > 0 ? completed / total : 0.0;

    return Container(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      decoration: BoxDecoration(
        color: AxonColors.surface.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Progress',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Text(
                '$completed / $total',
                style: GoogleFonts.googleSans(
                  color: AxonColors.accent,
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: progress,
              minHeight: 8,
              backgroundColor: AxonColors.surface.withValues(alpha: 0.2),
              color: AxonColors.accent,
            ),
          ),
        ],
      ),
    );
  }
}

class MilestoneCard extends StatelessWidget {
  final Milestone milestone;
  final VoidCallback onComplete;

  const MilestoneCard({required this.milestone, required this.onComplete, super.key});

  static const Map<MilestoneType, IconData> iconMap = {
    MilestoneType.syllabus: Icons.book_rounded,
    MilestoneType.mockTest: Icons.assignment_rounded,
    MilestoneType.analysis: Icons.analytics_rounded,
    MilestoneType.practice: Icons.edit_note_rounded,
    MilestoneType.formula: Icons.functions_rounded,
    MilestoneType.timing: Icons.timer_rounded,
    MilestoneType.consolidation: Icons.layers_rounded,
    MilestoneType.general: Icons.flag_rounded,
  };

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: null,
          borderRadius: BorderRadius.circular(16),
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: milestone.isCompleted
                  ? AxonColors.accent.withValues(alpha: 0.1)
                  : AxonColors.surface.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: milestone.isCompleted
                    ? AxonColors.accent.withValues(alpha: 0.3)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: milestone.isCompleted
                        ? AxonColors.accent.withValues(alpha: 0.2)
                        : AxonColors.surface.withValues(alpha: 0.15),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    milestone.isCompleted
                        ? Icons.check_rounded
                        : iconMap[milestone.type] ?? Icons.flag_rounded,
                    color: milestone.isCompleted
                        ? AxonColors.accent
                        : AxonColors.textSecondary,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        milestone.title,
                        style: GoogleFonts.googleSans(
                          color: milestone.isCompleted
                              ? AxonColors.accent
                              : AxonColors.textPrimary,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          decoration: _lineThrough(milestone.isCompleted),
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        milestone.description,
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 12,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Target: Day ${milestone.targetDay}',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 11,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
                if (!milestone.isCompleted)
                  Icon(Icons.radio_button_unchecked,
                      color: AxonColors.textTertiary, size: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
