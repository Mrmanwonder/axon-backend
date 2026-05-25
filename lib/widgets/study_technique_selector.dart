import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../services/study_techniques_service.dart';
import '../../theme/app_theme.dart';

class StudyTechniqueSelector extends ConsumerStatefulWidget {
  final Function(String)? onTechniqueSelected;

  const StudyTechniqueSelector({super.key, this.onTechniqueSelected});

  @override
  ConsumerState<StudyTechniqueSelector> createState() =>
      _StudyTechniqueSelectorState();
}

class _StudyTechniqueSelectorState
    extends ConsumerState<StudyTechniqueSelector> {
  String? _selectedTechniqueId;
  Map<String, TechniqueProgress> _progressMap = {};

  @override
  void initState() {
    super.initState();
    _loadProgress();
  }

  Future<void> _loadProgress() async {
    final progress = await StudyTechniquesService.instance.getAllProgress();
    if (mounted) {
      setState(() => _progressMap = progress);
    }
  }

  IconData _getIcon(String iconName) {
    switch (iconName) {
      case 'timer':
        return Icons.timer;
      case 'repeat':
        return Icons.repeat;
      case 'lightbulb':
        return Icons.lightbulb;
      case 'inbox':
        return Icons.inbox;
      case 'shuffle':
        return Icons.shuffle;
      case 'brain':
        return Icons.psychology;
      case 'layers':
        return Icons.layers;
      case 'calendar_today':
        return Icons.calendar_today;
      default:
        return Icons.school;
    }
  }

  Color _getColor(String colorHex) {
    final hex = colorHex.replaceAll('#', '');
    return Color(int.parse('FF$hex', radix: 16));
  }

  @override
  Widget build(BuildContext context) {
    final techniques = StudyTechniquesService.instance.getAllTechniques();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Text(
            'Choose Your Technique',
            style: TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        Expanded(
          child: GridView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
              childAspectRatio: 1.1,
            ),
            itemCount: techniques.length,
            itemBuilder: (context, index) {
              final technique = techniques[index];
              final isSelected = _selectedTechniqueId == technique.id;
              final progress = _progressMap[technique.id];
              final color = _getColor(technique.color);

              return GestureDetector(
                onTap: () {
                  HapticFeedback.mediumImpact();
                  setState(() => _selectedTechniqueId = technique.id);
                  widget.onTechniqueSelected?.call(technique.id);
                },
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? color.withValues(alpha: 0.3)
                        : AxonColors.oxfordBlue.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: isSelected ? color : Colors.transparent,
                      width: 2,
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: color.withValues(alpha: 0.2),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Icon(
                                _getIcon(technique.icon),
                                color: color,
                                size: 20,
                              ),
                            ),
                            const Spacer(),
                            if (progress != null &&
                                progress.sessionsCompleted > 0)
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 6,
                                  vertical: 2,
                                ),
                                decoration: BoxDecoration(
                                  color: color.withValues(alpha: 0.2),
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: Text(
                                  '${progress.sessionsCompleted}x',
                                  style: TextStyle(
                                    color: color,
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                          ],
                        ),
                        const Spacer(),
                        Text(
                          technique.name,
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          technique.description,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: Colors.white54,
                            fontSize: 10,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

class TechniqueCard extends StatelessWidget {
  final StudyTechnique technique;
  final bool isSelected;
  final TechniqueProgress? progress;
  final VoidCallback onTap;

  const TechniqueCard({
    super.key,
    required this.technique,
    required this.isSelected,
    this.progress,
    required this.onTap,
  });

  IconData _getIcon(String iconName) {
    switch (iconName) {
      case 'timer':
        return Icons.timer;
      case 'repeat':
        return Icons.repeat;
      case 'lightbulb':
        return Icons.lightbulb;
      case 'inbox':
        return Icons.inbox;
      case 'shuffle':
        return Icons.shuffle;
      case 'brain':
        return Icons.psychology;
      case 'layers':
        return Icons.layers;
      case 'calendar_today':
        return Icons.calendar_today;
      default:
        return Icons.school;
    }
  }

  Color _getColor(String colorHex) {
    final hex = colorHex.replaceAll('#', '');
    return Color(int.parse('FF$hex', radix: 16));
  }

  @override
  Widget build(BuildContext context) {
    final color = _getColor(technique.color);

    return GestureDetector(
      onTap: () {
        HapticFeedback.mediumImpact();
        onTap();
      },
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isSelected
              ? color.withValues(alpha: 0.2)
              : Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected ? color : Colors.transparent,
            width: 2,
          ),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                _getIcon(technique.icon),
                color: color,
                size: 24,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    technique.name,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    technique.description,
                    style: const TextStyle(
                      color: Colors.white54,
                      fontSize: 12,
                    ),
                  ),
                  if (progress != null && progress!.sessionsCompleted > 0) ...[
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(Icons.check_circle, color: color, size: 14),
                        const SizedBox(width: 4),
                        Text(
                          '${progress!.sessionsCompleted} sessions',
                          style: TextStyle(color: color, fontSize: 11),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _formatDuration(progress!.totalTime),
                          style: const TextStyle(
                              color: Colors.white38, fontSize: 11),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
            Icon(
              isSelected ? Icons.check_circle : Icons.circle_outlined,
              color: isSelected ? color : Colors.white24,
            ),
          ],
        ),
      ),
    );
  }

  String _formatDuration(Duration duration) {
    final hours = duration.inHours;
    final minutes = duration.inMinutes.remainder(60);
    if (hours > 0) {
      return '${hours}h ${minutes}m';
    }
    return '${minutes}m';
  }
}
