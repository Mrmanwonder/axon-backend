// lib/screens/exam/widgets/countdown_tab.dart

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../services/exam_planner_service.dart';
import '../../../services/exam_planner_repository.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/layout_utils.dart';
import '../../../widgets/common/blur_reveal_widget.dart';
import '../../../widgets/common/rose_loader.dart';

TextDecoration? _lineThrough(bool condition) =>
    condition ? TextDecoration.lineThrough : null;
Color _phaseColor(bool isPast, Color phaseColor) =>
    isPast ? AxonColors.textTertiary : phaseColor;
Color _phaseColorWithAlpha(bool isPast, Color phaseColor, double alpha) =>
    isPast
        ? AxonColors.textTertiary.withValues(alpha: alpha)
        : phaseColor.withValues(alpha: alpha);

class CountdownTab extends ConsumerWidget {
  const CountdownTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: RoseLoader(size: 24)),
      error: (error, _) => _buildErrorState(error.toString(), ref),
      data: (data) {
        if (data.countdown == null || data.countdown!.daysRemaining == 0) {
          return _buildNoExamSet();
        }

        final countdown = data.countdown!;
        return SingleChildScrollView(
          padding:
              EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              BlurFadeSlideWidget(
                sessionKey: 'exam_planner_countdown',
                child: _buildMainCountdown(countdown),
              ),
              const SizedBox(height: 24),
              BlurFadeSlideWidget(
                sessionKey: 'exam_planner_phases',
                delay: const Duration(milliseconds: 100),
                child: _buildPhasesTimeline(countdown),
              ),
              const SizedBox(height: 24),
              BlurFadeSlideWidget(
                sessionKey: 'exam_planner_focus',
                delay: const Duration(milliseconds: 200),
                child: _buildCurrentPhase(context, countdown),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildErrorState(String error, WidgetRef ref) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.error_outline, size: 48, color: AxonColors.textTertiary),
          const SizedBox(height: 16),
          Text('Failed to load: $error',
              style: TextStyle(color: AxonColors.textSecondary)),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: () => ref.read(examPlannerProvider.notifier).retry(),
            child: const Text('RETRY'),
          ),
        ],
      ),
    );
  }

  Widget _buildNoExamSet() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: AxonColors.accent.withValues(alpha: 0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.event_note, size: 48, color: AxonColors.accent),
            ),
            const SizedBox(height: 24),
            Text(
              'No Exam Date Set',
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Set your exam date in settings to see countdown',
              textAlign: TextAlign.center,
              style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary,
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMainCountdown(ExamCountdown countdown) {
    final days = countdown.daysRemaining;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(32),
        gradient: LinearGradient(
          colors: [AxonColors.accent, AxonColors.accent.withValues(alpha: 0.7)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: AxonColors.accent.withValues(alpha: 0.4),
            blurRadius: 30,
            offset: const Offset(0, 15),
          ),
        ],
      ),
      child: Column(
        children: [
          Text(
            '$days',
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 84,
              fontWeight: FontWeight.w900,
              height: 1,
            ),
          ).animate().scale(duration: 400.ms, curve: Curves.easeOutBack),
          Text(
            'DAYS TO GO',
            style: GoogleFonts.googleSans(
              color: Colors.white.withValues(alpha: 0.8),
              fontSize: 12,
              fontWeight: FontWeight.w800,
              letterSpacing: 4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPhasesTimeline(ExamCountdown countdown) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Revision Phases',
          style: GoogleFonts.googleSans(
            color: AxonColors.textPrimary,
            fontSize: 16,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 12),
        ...countdown.phases.map((phase) {
          final isActive = countdown.currentPhase?.name == phase.name;
          final isPast = countdown.daysRemaining < phase.startDay;

          return Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: isActive
                  ? Color(phase.color).withValues(alpha: 0.15)
                  : isPast
                      ? AxonColors.surface.withValues(alpha: 0.05)
                      : AxonColors.surface.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: isActive
                    ? Color(phase.color).withValues(alpha: 0.4)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color:
                        _phaseColorWithAlpha(isPast, Color(phase.color), 0.2),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    isPast ? Icons.check_rounded : Icons.schedule,
                    color: _phaseColor(isPast, Color(phase.color)),
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        phase.name,
                        style: GoogleFonts.googleSans(
                          color: isPast
                              ? AxonColors.textTertiary
                              : AxonColors.textPrimary,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          decoration: _lineThrough(isPast),
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'Day ${phase.startDay} - ${phase.endDay} • ${phase.dailyHours}h/day',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
                if (isActive)
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Color(phase.color).withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      'NOW',
                      style: GoogleFonts.googleSans(
                        color: Color(phase.color),
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
              ],
            ),
          );
        }),
      ],
    );
  }

  Widget _buildCurrentPhase(BuildContext context, ExamCountdown countdown) {
    final phase = countdown.currentPhase;
    if (phase == null) return const SizedBox.shrink();

    return Container(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Color(phase.color).withValues(alpha: 0.1),
            AxonColors.surface.withValues(alpha: 0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.lightbulb_outline,
                  color: Color(phase.color), size: 20),
              const SizedBox(width: 8),
              Text(
                'Current Focus',
                style: GoogleFonts.googleSans(
                  color: Color(phase.color),
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            phase.focus,
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            phase.description,
            style: GoogleFonts.googleSans(
              color: AxonColors.textSecondary,
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }
}
