import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:home_widget/home_widget.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../../models/study_activity.dart';
import 'firestore_service.dart';

class StudyActivityService {
  static final StudyActivityService _instance =
      StudyActivityService._internal();
  factory StudyActivityService() => _instance;
  StudyActivityService._internal();

  static const String _activityKey = 'study_activity_levels';
  static const String _activitySessionsKey = 'activity_sessions';
  static const String _lastSyncKey = 'study_activity_last_sync';
  static const int _defaultDays = 35;
  static const int _heatmapDays = 28;

  List<int> _cachedLevels = [];
  List<StudyHeatmapSession> _cachedSessions = [];

  Future<void> initialize() async {
    await _loadFromStorage();
  }

  Future<void> _loadFromStorage() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final data = prefs.getString(_activityKey);
      if (data != null && data.isNotEmpty) {
        _cachedLevels = (jsonDecode(data) as List).cast<int>();
      }
      final sessionData = prefs.getString(_activitySessionsKey);
      if (sessionData != null && sessionData.isNotEmpty) {
        _cachedSessions = (jsonDecode(sessionData) as List)
            .map(
              (item) => StudyHeatmapSession.fromJson(
                Map<String, dynamic>.from(item as Map),
              ),
            )
            .toList();
      }
    } catch (e) {
      _cachedLevels = [];
      _cachedSessions = [];
    }
  }

  Future<void> _saveToStorage() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_activityKey, jsonEncode(_cachedLevels));
    await prefs.setString(
      _activitySessionsKey,
      jsonEncode(_cachedSessions.map((session) => session.toJson()).toList()),
    );
    await prefs.setString(_lastSyncKey, DateTime.now().toIso8601String());
  }

  List<int> getRecentActivityLevels({int days = _defaultDays}) {
    if (_cachedLevels.isEmpty) {
      return List.generate(days, (_) => 0);
    }
    if (_cachedLevels.length >= days) {
      return _cachedLevels.sublist(_cachedLevels.length - days);
    }
    return [
      ...List.generate(days - _cachedLevels.length, (_) => 0),
      ..._cachedLevels,
    ];
  }

  double getMasteryPercentage() {
    final sessions = getRecentSessions();
    if (sessions.isEmpty) {
      final levels = getRecentActivityLevels();
      if (levels.isEmpty) return 0.0;
      final total = levels.reduce((a, b) => a + b);
      final maxPossible = levels.length * 3;
      return total / maxPossible;
    }
    final weightedTotal = sessions.fold<double>(
      0,
      (total, session) =>
          total + (session.minutes.clamp(0, 300) / 300) * session.quality,
    );
    return (weightedTotal / sessions.length).clamp(0.0, 1.0);
  }

  List<StudyHeatmapSession> getRecentSessions({int days = _heatmapDays}) {
    if (_cachedSessions.isEmpty) {
      return StudyHeatmapSession.generateLastNDaysFromLevels(
        getRecentActivityLevels(days: days),
        days: days,
      );
    }
    final sorted = List<StudyHeatmapSession>.from(_cachedSessions)
      ..sort((a, b) => a.date.compareTo(b.date));
    if (sorted.length >= days) {
      return sorted.sublist(sorted.length - days);
    }
    final now = DateTime.now();
    final padded = <StudyHeatmapSession>[];
    for (int i = days - sorted.length; i > 0; i--) {
      final date = DateTime(now.year, now.month, now.day)
          .subtract(Duration(days: sorted.length + i - 1));
      padded.add(StudyHeatmapSession(date: date, minutes: 0, quality: 0));
    }
    return [...padded, ...sorted];
  }

  Future<void> recordStudySession({
    required int chaptersCompleted,
    int? minutes,
    double? quality,
  }) async {
    int level = 0;
    if (chaptersCompleted > 0) level = 1;
    if (chaptersCompleted > 2) level = 2;
    if (chaptersCompleted > 4) level = 3;

    final newLevels = List<int>.from(_cachedLevels);
    if (newLevels.isEmpty) {
      newLevels.add(level);
    } else {
      newLevels[newLevels.length - 1] = level;
    }

    _cachedLevels = newLevels;
    _upsertSession(
      date: DateTime.now(),
      minutes: minutes ??
          switch (level) {
            0 => 0,
            1 => 60,
            2 => 180,
            _ => 300,
          },
      quality: quality ??
          switch (level) {
            0 => 0.0,
            1 => 0.35,
            2 => 0.65,
            _ => 1.0,
          },
    );
    await _saveToStorage();
    await syncToHomeWidget();
  }

  Future<void> addDailyActivity({
    required DateTime date,
    required int chapters,
    int? minutes,
    double? quality,
  }) async {
    final level = _calculateLevel(chapters);

    final newLevels = List<int>.from(_cachedLevels);
    newLevels.add(level);

    if (newLevels.length > _defaultDays) {
      _cachedLevels = newLevels.sublist(newLevels.length - _defaultDays);
    } else {
      _cachedLevels = newLevels;
    }

    _upsertSession(
      date: date,
      minutes: minutes ??
          switch (level) {
            0 => 0,
            1 => 60,
            2 => 180,
            _ => 300,
          },
      quality: quality ??
          switch (level) {
            0 => 0.0,
            1 => 0.35,
            2 => 0.65,
            _ => 1.0,
          },
    );
    await _saveToStorage();
    await syncToHomeWidget();
  }

  int _calculateLevel(int chapters) {
    if (chapters == 0) return 0;
    if (chapters <= 2) return 1;
    if (chapters <= 4) return 2;
    return 3;
  }

  Future<void> syncToHomeWidget() async {
    try {
      final levels = getRecentActivityLevels();
      final sessions = getRecentSessions();

      await HomeWidget.saveWidgetData(_activityKey, jsonEncode(levels));
      await HomeWidget.saveWidgetData(
        _activitySessionsKey,
        jsonEncode(sessions.map((session) => session.toJson()).toList()),
      );
      await HomeWidget.saveWidgetData(
        _lastSyncKey,
        DateTime.now().toIso8601String(),
      );
      await HomeWidget.saveWidgetData(
        'mastery_percentage',
        getMasteryPercentage(),
      );

      await HomeWidget.updateWidget(
        name: 'ContributionGraphWidgetProvider',
        iOSName: 'ContributionGraphWidget',
      );
    } catch (e) {
      print('Study activity widget sync error: $e');
    }
  }

  Future<void> refreshFromBackend() async {
    final uid = FirebaseAuth.instance.currentUser?.uid;

    if (uid == null) {
      _cachedLevels = List.filled(_heatmapDays, 0);
      _cachedSessions = [];
      await _saveToStorage();
      await syncToHomeWidget();
      return;
    }

    try {
      final docRef = AxonPaths.privateUserDoc(uid);
      final today = DateTime.now();
      final startDate = today.subtract(Duration(days: _heatmapDays));

      final snapshot = await docRef
          .collection('sessions')
          .where('timestamp', isGreaterThan: Timestamp.fromDate(startDate))
          .orderBy('timestamp')
          .get();

      final levelsMap = <String, int>{};

      for (final doc in snapshot.docs) {
        final timestamp = doc.data()['timestamp'] as Timestamp?;
        final minutes = (doc.data()['minutes'] as num?)?.toInt() ?? 0;

        if (timestamp != null && minutes > 0) {
          final date = timestamp.toDate();
          final key =
              '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
          levelsMap[key] = (levelsMap[key] ?? 0) + minutes;
        }
      }

      final levels = <int>[];
      for (int i = _heatmapDays - 1; i >= 0; i--) {
        final date = today.subtract(Duration(days: i));
        final key =
            '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
        final minutes = levelsMap[key] ?? 0;

        if (minutes == 0) {
          levels.add(0);
        } else if (minutes < 30) {
          levels.add(1);
        } else if (minutes < 60) {
          levels.add(2);
        } else if (minutes < 120) {
          levels.add(3);
        } else {
          levels.add(4);
        }
      }

      _cachedLevels = levels;
      _cachedSessions = StudyHeatmapSession.generateLastNDaysFromLevels(
        levels,
        days: _heatmapDays,
      );
      await _saveToStorage();
      await syncToHomeWidget();
    } catch (e) {
      debugPrint('Error loading study activity from Firestore: $e');
      _cachedLevels = List.filled(_heatmapDays, 0);
      _cachedSessions = [];
      await _saveToStorage();
    }
  }

  Map<String, dynamic> getWeeklyStats() {
    final levels = getRecentActivityLevels(days: 7);
    final total = levels.reduce((a, b) => a + b);
    final activeDays = levels.where((l) => l > 0).length;

    return {
      'totalScore': total,
      'activeDays': activeDays,
      'averageScore': activeDays > 0 ? total / activeDays : 0,
      'levels': levels,
    };
  }

  void _upsertSession({
    required DateTime date,
    required int minutes,
    required double quality,
  }) {
    final day = DateTime(date.year, date.month, date.day);
    final next = List<StudyHeatmapSession>.from(_cachedSessions);
    final index = next.indexWhere((session) =>
        session.date.year == day.year &&
        session.date.month == day.month &&
        session.date.day == day.day);

    final normalized = StudyHeatmapSession(
      date: day,
      minutes: minutes.clamp(0, 300).toInt(),
      quality: quality.clamp(0.0, 1.0).toDouble(),
    );

    if (index >= 0) {
      next[index] = normalized;
    } else {
      next.add(normalized);
    }

    next.sort((a, b) => a.date.compareTo(b.date));
    _cachedSessions = next.length > _heatmapDays
        ? next.sublist(next.length - _heatmapDays)
        : next;
  }
}
