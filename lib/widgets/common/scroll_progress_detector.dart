import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import '../../services/user_progress_service.dart';
import '../../services/app_state.dart';

class ScrollProgress {
  final double percentage;
  final int timeSpentSeconds;
  final DateTime lastUpdated;

  ScrollProgress({
    required this.percentage,
    required this.timeSpentSeconds,
    required this.lastUpdated,
  });
}

class ScrollProgressNotifier extends StateNotifier<ScrollProgress?> {
  ScrollProgressNotifier() : super(null);

  DateTime? _startTime;
  bool _isTracking = false;

  void startTracking() {
    _startTime = DateTime.now();
    _isTracking = true;
  }

  void stopTracking() {
    _isTracking = false;
    _startTime = null;
  }

  void updateScrollPosition(double position, double maxScroll) {
    if (!_isTracking || maxScroll <= 0) return;

    final percentage = (position / maxScroll * 100).clamp(0.0, 100.0);
    final timeSpent = _startTime != null
        ? DateTime.now().difference(_startTime!).inSeconds
        : 0;

    state = ScrollProgress(
      percentage: percentage,
      timeSpentSeconds: timeSpent,
      lastUpdated: DateTime.now(),
    );
  }
}

final scrollProgressProvider =
    StateNotifierProvider<ScrollProgressNotifier, ScrollProgress?>((ref) {
  return ScrollProgressNotifier();
});

class ScrollDetector extends ConsumerWidget {
  final String subjectCode;
  final String chapterNumber;
  final String subchapterNumber;
  final Widget child;

  const ScrollDetector({
    super.key,
    required this.subjectCode,
    required this.chapterNumber,
    required this.subchapterNumber,
    required this.child,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return NotificationListener<ScrollNotification>(
      onNotification: (notification) {
        if (notification is ScrollUpdateNotification) {
          final metrics = notification.metrics;
          final notifier = ref.read(scrollProgressProvider.notifier);

          notifier.updateScrollPosition(
            metrics.pixels,
            metrics.maxScrollExtent,
          );
        } else if (notification is ScrollStartNotification) {
          ref.read(scrollProgressProvider.notifier).startTracking();
        } else if (notification is ScrollEndNotification) {
          _saveProgress(ref);
          ref.read(scrollProgressProvider.notifier).stopTracking();
        }
        return false;
      },
      child: child,
    );
  }

  Future<void> _saveProgress(WidgetRef ref) async {
    final progress = ref.read(scrollProgressProvider);
    if (progress == null) return;

    // Save to local first
    final prefs = await SharedPreferences.getInstance();
    final key = 'scroll_${subjectCode}_${chapterNumber}_$subchapterNumber';
    await prefs.setString(
        key,
        jsonEncode({
          'percentage': progress.percentage,
          'timeSpentSeconds': progress.timeSpentSeconds,
          'lastUpdated': progress.lastUpdated.toIso8601String(),
        }));

    final uid = ref.read(authStateProvider).user?.uid;
    final chapter = int.tryParse(chapterNumber);
    final subchapter = int.tryParse(subchapterNumber);
    if (uid != null && chapter != null && subchapter != null) {
      await UserProgressService.instance.updateSubchapterProgress(
        uid: uid,
        subjectCode: subjectCode,
        chapterNumber: chapter,
        subchapterNumber: subchapter,
        scrollPercentage: progress.percentage,
        timeSpentSeconds: progress.timeSpentSeconds,
      );
    }
  }
}

// Progress indicator widget
class ProgressIndicatorBar extends StatelessWidget {
  final double percentage;
  final String label;
  final Color color;

  const ProgressIndicatorBar({
    super.key,
    required this.percentage,
    this.label = 'Progress',
    this.color = const Color(0xFF3A86FF),
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 12,
              ),
            ),
            Text(
              '${percentage.toStringAsFixed(1)}%',
              style: TextStyle(
                color: color,
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: percentage / 100,
            backgroundColor: Colors.white12,
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
      ],
    );
  }
}

// Hook for using scroll progress in any widget
final progressProvider =
    StateProvider.family<ScrollProgress?, String>((ref, key) => null);
