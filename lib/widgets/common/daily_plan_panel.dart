import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../services/daily_plan_service.dart';
import '../../models/daily_plan_task.dart';
import '../../theme/app_theme.dart';
import '../../theme/task_type_theme.dart';
import 'rose_loader.dart';

class DailyPlanPanel extends ConsumerStatefulWidget {
  final String uid;
  final DailyPlanService service;
  final void Function(BuildContext context, DailyPlanTask task)? onTaskTap;

  const DailyPlanPanel({
    super.key,
    required this.uid,
    required this.service,
    this.onTaskTap,
  });

  @override
  ConsumerState<DailyPlanPanel> createState() => _DailyPlanPanelState();
}

class _DailyPlanPanelState extends ConsumerState<DailyPlanPanel> {
  bool _isGenerating = false;

  Future<void> _generatePlan() async {
    setState(() => _isGenerating = true);
    try {
      final tasks = await widget.service.ensureTodayPlan(widget.uid);
      if (!mounted) return;
      if (tasks.isEmpty) {
        _showAlert(
          'Plan not created',
          'Add subjects or exam dates, then try again.',
          isError: true,
        );
      } else {
        _showAlert(
          'Daily plan ready',
          '${tasks.length} tasks scheduled for today.',
        );
      }
    } catch (e) {
      if (mounted) {
        _showAlert('Daily plan failed', e.toString(), isError: true);
      }
    } finally {
      if (mounted) setState(() => _isGenerating = false);
    }
  }

  void _showAlert(String title, String message, {bool isError = false}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? AxonColors.error : AxonColors.success,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<List<DailyPlanTask>>(
      stream: widget.service.watchTodayPlan(widget.uid),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: RoseLoader(size: 24));
        }

        final tasks = snapshot.data ?? [];
        if (tasks.isEmpty) {
          return _buildEmptyState();
        }

        final nextTask = tasks.cast<DailyPlanTask?>().firstWhere(
              (t) => !(t!.isCompleted),
              orElse: () => tasks.last,
            )!;

        final phase = nextTask.phase;
        return Column(
          children: [
            if (phase.label.isNotEmpty) ...[
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: AxonColors.electricCyan.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.flag_rounded, size: 12, color: AxonColors.electricCyan),
                    const SizedBox(width: 6),
                    Text(
                      phase.label,
                      style: GoogleFonts.googleSans(
                        color: AxonColors.electricCyan,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
            ],
            DailyPlanCard(
              task: nextTask,
              totalTasks: tasks.length,
              completedTasks: tasks.where((t) => t.isCompleted).length,
              onTap: widget.onTaskTap,
            ),
          ],
        );
      },
    );
  }

  Widget _buildEmptyState() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AxonColors.surfaceElevated,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Column(
        children: [
          Icon(Icons.calendar_today_outlined,
              color: AxonColors.textTertiary, size: 32),
          const SizedBox(height: 12),
          Text(
            'No plan for today',
            style: GoogleFonts.googleSans(
                color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 4),
          Text(
            'Generate a personalized plan based on your subjects.',
            textAlign: TextAlign.center,
            style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary, fontSize: 13),
          ),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _isGenerating ? null : _generatePlan,
              style: ElevatedButton.styleFrom(
                backgroundColor: AxonColors.electricCyan,
                foregroundColor: Colors.black,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
              child: _isGenerating
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.black))
                  : const Text('Generate Daily Plan',
                      style: TextStyle(fontWeight: FontWeight.bold)),
            ),
          ),
        ],
      ),
    );
  }
}

class DailyPlanCard extends StatelessWidget {
  final DailyPlanTask task;
  final int totalTasks;
  final int completedTasks;
  final void Function(BuildContext context, DailyPlanTask task)? onTap;

  const DailyPlanCard({
    super.key,
    required this.task,
    required this.totalTasks,
    required this.completedTasks,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final now = DateTime.now();
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    final dateStr =
        '${days[now.weekday - 1]}, ${months[now.month - 1]} ${now.day}';

    return GestureDetector(
      onTap: () => onTap?.call(context, task),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              AxonColors.electricCyan.withValues(alpha: 0.15),
              AxonColors.surfaceElevated,
            ],
          ),
          borderRadius: BorderRadius.circular(24),
          border:
              Border.all(color: AxonColors.electricCyan.withValues(alpha: 0.3)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: AxonColors.electricCyan.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    dateStr,
                    style: GoogleFonts.googleSans(
                      color: AxonColors.electricCyan,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                Text(
                  '$completedTasks/$totalTasks Completed',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              task.subject,
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontSize: 14,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: taskTypeColor(task.taskType).withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(taskTypeIcon(task.taskType),
                      color: taskTypeColor(task.taskType), size: 18),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        task.title,
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textPrimary,
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        task.taskType.label,
                        style: GoogleFonts.googleSans(
                          color: taskTypeColor(task.taskType),
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.white12,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(Icons.play_arrow_rounded,
                      color: Colors.white, size: 20),
                ),
              ],
            ),
            if (task.description.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                task.description,
                style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary,
                  fontSize: 13,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
            const SizedBox(height: 14),
            Row(
              children: [
                Icon(Icons.access_time_rounded,
                    color: AxonColors.textTertiary, size: 14),
                const SizedBox(width: 4),
                Text(
                  '${task.startTime.hour.toString().padLeft(2, '0')}:${task.startTime.minute.toString().padLeft(2, '0')} - ${task.endTime.hour.toString().padLeft(2, '0')}:${task.endTime.minute.toString().padLeft(2, '0')}',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 12,
                  ),
                ),
                const Spacer(),
                Text(
                  'CONTINUE',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.electricCyan,
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
