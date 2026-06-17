// lib/screens/exam/widgets/checklists_tab.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../services/exam_planner_service.dart';
import '../../../services/exam_planner_repository.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/layout_utils.dart';
import '../../../widgets/common/rose_loader.dart';

class ChecklistsTab extends ConsumerWidget {
  const ChecklistsTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: RoseLoader(size: 24)),
      error: (error, _) => Center(child: Text('Error: $error')),
      data: (data) {
        final checklists = data.checklists;
        final subjects = checklists.keys.toList();

        return ChecklistsContent(checklists: checklists, subjects: subjects);
      },
    );
  }
}

class ChecklistsContent extends ConsumerStatefulWidget {
  final Map<String, StrategyChecklist> checklists;
  final List<String> subjects;

  const ChecklistsContent({required this.checklists, required this.subjects});

  @override
  ConsumerState<ChecklistsContent> createState() => ChecklistsContentState();
}

class ChecklistsContentState extends ConsumerState<ChecklistsContent> {
  String _selectedSubject = 'general';

  @override
  void initState() {
    super.initState();
    if (widget.subjects.isNotEmpty) {
      _selectedSubject = widget.subjects.first;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          height: 44,
          margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: widget.subjects.length,
            itemBuilder: (context, i) {
              final subject = widget.subjects[i];
              final isSelected = subject == _selectedSubject;
              return GestureDetector(
                key: ValueKey('subject-tab-$subject'),
                onTap: () => setState(() => _selectedSubject = subject),
                child: Container(
                  margin: const EdgeInsets.only(right: 8),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? AxonColors.accent.withValues(alpha: 0.2)
                        : AxonColors.surface.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: isSelected
                          ? AxonColors.accent.withValues(alpha: 0.4)
                          : Colors.transparent,
                    ),
                  ),
                  alignment: Alignment.center,
                  child: Text(
                    _capitalize(subject),
                    style: GoogleFonts.googleSans(
                      color: isSelected
                          ? AxonColors.accent
                          : AxonColors.textSecondary,
                      fontSize: 13,
                      fontWeight:
                          isSelected ? FontWeight.w600 : FontWeight.w500,
                    ),
                  ),
                ),
              );
            },
          ),
        ),
        Expanded(
          child: _buildChecklist(),
        ),
      ],
    );
  }

  String _capitalize(String s) => s[0].toUpperCase() + s.substring(1);

  Widget _buildChecklist() {
    final checklist = widget.checklists[_selectedSubject];
    if (checklist == null) {
      return const Center(child: RoseLoader(size: 24));
    }

    final completed = checklist.items.where((i) => i.isChecked).length;
    final progress =
        checklist.items.isNotEmpty ? completed / checklist.items.length : 0.0;

    return SingleChildScrollView(
      padding: EdgeInsets.fromLTRB(20, 20, 20, bottomDockClearance(context)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AxonColors.surface.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Completion',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Text(
                      '$completed/${checklist.items.length}',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.accent,
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.0, end: progress),
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.easeOutCubic,
                  builder: (context, value, child) {
                    return ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: value,
                        minHeight: 6,
                        backgroundColor:
                            AxonColors.surface.withValues(alpha: 0.2),
                        color: AxonColors.accent,
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          ...checklist.items.map((item) => ChecklistItemCard(
                item: item,
                onToggle: (checked) async {
                  final subjectKey = _selectedSubject.toLowerCase();
                  await ref
                      .read(examPlannerProvider.notifier)
                      .updateChecklistItem(subjectKey, item.id, checked);
                },
              )),
        ],
      ),
    );
  }
}

class ChecklistItemCard extends StatelessWidget {
  final ChecklistItem item;
  final Function(bool) onToggle;

  const ChecklistItemCard({required this.item, required this.onToggle, super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        GestureDetector(
          onTap: () => onToggle(!item.isChecked),
          behavior: HitTestBehavior.opaque,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 8),
            child: Row(
              children: [
                AnimatedCheckbox(isCompleted: item.isChecked),
                const SizedBox(width: 16),
                Expanded(
                  child: Stack(
                    children: [
                      Text(
                        item.text,
                        style: GoogleFonts.googleSans(
                          color: item.isChecked ? Colors.white24 : Colors.white,
                          fontSize: 15,
                        ),
                      ),
                      Positioned.fill(
                        child: AnimatedStrike(isCompleted: item.isChecked),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        Divider(
            height: 1, color: Colors.white.withValues(alpha: 0.03), indent: 48),
      ],
    );
  }
}

class AnimatedCheckbox extends StatelessWidget {
  final bool isCompleted;
  const AnimatedCheckbox({required this.isCompleted, super.key});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {},
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 22,
        height: 22,
        decoration: BoxDecoration(
          color: isCompleted ? const Color(0xFF3A86FF) : Colors.transparent,
          shape: BoxShape.circle,
          border: Border.all(
            color: isCompleted ? const Color(0xFF3A86FF) : Colors.white38,
            width: 2,
          ),
        ),
        child: isCompleted
            ? const Icon(Icons.check, color: Colors.white, size: 14)
            : null,
      ),
    );
  }
}

class AnimatedStrike extends StatelessWidget {
  final bool isCompleted;
  const AnimatedStrike({required this.isCompleted, super.key});

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: isCompleted ? 1.0 : 0.0),
      duration: const Duration(milliseconds: 350),
      curve: Curves.easeOutCubic,
      builder: (context, value, child) {
        return CustomPaint(
          painter: StrikePainter(progress: value),
        );
      },
    );
  }
}

class StrikePainter extends CustomPainter {
  final double progress;
  StrikePainter({required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    if (progress == 0) return;
    final paint = Paint()
      ..color = Colors.white38
      ..strokeWidth = 1.2
      ..strokeCap = StrokeCap.round;

    final lineCount = (size.height / 20).floor() + 2;
    final lineHeight = size.height / lineCount;

    for (var i = 0; i < lineCount; i++) {
      final y = (i * lineHeight) + (lineHeight / 2);
      final lineProgress = lineCount == 1
          ? progress
          : (i < lineCount - 1)
              ? 1.0
              : progress;
      canvas.drawLine(
        Offset(0, y + 2),
        Offset(size.width * lineProgress, y + 2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(StrikePainter oldDelegate) =>
      oldDelegate.progress != progress;
}
