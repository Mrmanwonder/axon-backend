import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Snapshot of the user's current wellbeing & focus state.
class WellbeingSnapshot {
  final double screenTimeHours;
  final double studyHours;
  final double focusRatio;
  final int streak;
  final double sleepHours;
  final double stressLevel;
  final int distractionCount;
  final bool isOnTask;
  final int activeMinutes;
  final String primarySubject;

  const WellbeingSnapshot({
    this.screenTimeHours = 0,
    this.studyHours = 0,
    this.focusRatio = 0,
    this.streak = 0,
    this.sleepHours = 0,
    this.stressLevel = 0,
    this.distractionCount = 0,
    this.isOnTask = false,
    this.activeMinutes = 0,
    this.primarySubject = '',
  });

  double get wellbeingScore {
    final study = (studyHours / 6).clamp(0.0, 1.0);
    final screen = (1 - (screenTimeHours / 12)).clamp(0.0, 1.0);
    final focus = focusRatio;
    final sleep = (sleepHours / 8).clamp(0.0, 1.0);
    return (study * 0.35 + screen * 0.25 + focus * 0.25 + sleep * 0.15);
  }
}

/// Monitors device usage, study activity, and focus metrics
/// to emit [WellbeingSnapshot]s at a configurable interval.
class AppUsageService {
  AppUsageService();

  static const MethodChannel _channel = MethodChannel('com.axon.app/utils');
  Timer? _monitorTimer;
  StreamController<WellbeingSnapshot>? _controller;

  Stream<WellbeingSnapshot> watchWellbeing({int intervalSeconds = 30}) {
    _controller ??= StreamController<WellbeingSnapshot>.broadcast(
      onCancel: () => stopWellbeingMonitor(),
    );

    _emitSnapshot();
    _monitorTimer = Timer.periodic(
      Duration(seconds: intervalSeconds),
      (_) => _emitSnapshot(),
    );

    return _controller!.stream;
  }

  void stopWellbeingMonitor() {
    _monitorTimer?.cancel();
    _monitorTimer = null;
    _controller?.close();
    _controller = null;
  }

  Future<void> _emitSnapshot() async {
    try {
      final snapshot = await _buildSnapshot();
      _controller?.add(snapshot);
    } catch (e) {
      debugPrint('AppUsageService: snapshot build failed — $e');
    }
  }

  Future<WellbeingSnapshot> _buildSnapshot() async {
    final prefs = await SharedPreferences.getInstance();

    double screenTimeHours = 0;
    double studyHours = 0;
    double focusRatio = 0;
    int streak = 0;
    double sleepHours = 0;
    double stressLevel = 0;
    int distractionCount = 0;
    int activeMinutes = 0;
    String primarySubject = '';

    // Read metrics_state (written by MetricsNotifier)
    final metricsRaw = prefs.getString('metrics_state');
    if (metricsRaw != null) {
      try {
        final map = jsonDecode(metricsRaw) as Map<String, dynamic>;
        screenTimeHours = (map['screenTimeHours'] as num?)?.toDouble() ?? 0;
        studyHours = (map['activeStudyHours'] as num?)?.toDouble() ?? 0;
        focusRatio = (map['focusRatio'] as num?)?.toDouble() ?? 0;
        streak = (map['consistencyStreak'] as num?)?.toInt() ?? 0;
        sleepHours = (map['sleepHours'] as num?)?.toDouble() ?? 0;
        stressLevel = (map['stressLevel'] as num?)?.toDouble() ?? 0;
        primarySubject = (map['primarySubject'] as String?) ?? '';
      } catch (_) {}
    }

    // Read timer history for today's active minutes & distractions
    final todayStr = DateTime.now().toIso8601String().split('T').first;
    final historyRaw = prefs.getString('timer_history');
    if (historyRaw != null) {
      try {
        final list = jsonDecode(historyRaw) as List;
        int totalSeconds = 0;
        int distractions = 0;
        for (final entry in list) {
          final e = entry as Map<String, dynamic>;
          final date = e['date'] as String? ?? '';
          if (!date.startsWith(todayStr)) continue;
          totalSeconds += (e['duration'] as num?)?.toInt() ?? 0;
          distractions += (e['interruptionCount'] as num?)?.toInt() ?? 0;
        }
        activeMinutes = totalSeconds ~/ 60;
        distractionCount = distractions;
      } catch (_) {}
    }

    // Try platform channel for live screen time
    if (Platform.isAndroid) {
      try {
        final result =
            await _channel.invokeMethod<double>('getScreenTimeHours');
        if (result != null) screenTimeHours = result;
      } catch (_) {}
    }

    return WellbeingSnapshot(
      screenTimeHours: screenTimeHours,
      studyHours: studyHours,
      focusRatio: focusRatio,
      streak: streak,
      sleepHours: sleepHours,
      stressLevel: stressLevel,
      distractionCount: distractionCount,
      isOnTask: studyHours > 0,
      activeMinutes: activeMinutes,
      primarySubject: primarySubject,
    );
  }

  Future<bool> hasUsageAccess() async {
    if (!Platform.isAndroid) {
      return false;
    }
    try {
      final result = await _channel.invokeMethod<bool>('hasUsageAccess');
      return result ?? false;
    } catch (_) {
      return false;
    }
  }

  Future<void> requestUsageAccess() async {
    if (!Platform.isAndroid) return;
    try {
      await _channel.invokeMethod('requestUsageAccess');
    } catch (e) {
      debugPrint('AppUsageService: requestUsageAccess failed — $e');
    }
  }

  Future<double> getDailyScreenTimeHours() async {
    final snapshot = await _buildSnapshot();
    return snapshot.screenTimeHours;
  }

  Future<Map<String, dynamic>> getDetailedUsage(
      DateTime start, DateTime end) async {
    final prefs = await SharedPreferences.getInstance();
    final historyRaw = prefs.getString('timer_history');
    if (historyRaw == null) {
      return const {
        'total_usage_minutes': 0,
        'app_breakdown': <String, dynamic>{},
      };
    }

    try {
      final list = jsonDecode(historyRaw) as List;
      int totalSeconds = 0;
      final subjects = <String, int>{};

      for (final entry in list) {
        final e = entry as Map<String, dynamic>;
        final dateStr = e['date'] as String? ?? '';
        final entryDate = DateTime.tryParse(dateStr);
        if (entryDate == null ||
            entryDate.isBefore(start) ||
            entryDate.isAfter(end)) {
          continue;
        }

        final dur = (e['duration'] as num?)?.toInt() ?? 0;
        totalSeconds += dur;

        final subject = (e['subject'] as String?) ?? 'General';
        subjects[subject] = (subjects[subject] ?? 0) + dur;
      }

      return {
        'total_usage_minutes': totalSeconds ~/ 60,
        'app_breakdown': subjects.map((k, v) => MapEntry(k, v ~/ 60)),
      };
    } catch (_) {
      return const {
        'total_usage_minutes': 0,
        'app_breakdown': <String, dynamic>{},
      };
    }
  }
}
