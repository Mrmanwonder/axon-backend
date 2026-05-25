import 'dart:async';
import 'dart:convert';
import 'package:home_widget/home_widget.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/study_activity.dart';
import 'exam_planner_service.dart';
import 'exam_repository.dart';

class WidgetService {
  static final WidgetService _instance = WidgetService._internal();
  factory WidgetService() => _instance;
  WidgetService._internal();

  static const String _flipClockWidgetProvider =
      'FlipClockEvolutionWidgetProvider';
  static const String _industrialTimerWidgetProvider =
      'IndustrialTimerWidgetProvider';
  static const String _contributionGraphWidgetProvider =
      'ContributionGraphWidgetProvider';
  static const String _askAxonWidgetProvider = 'AskAxonWidgetProvider';
  static const String _examCalendarWidgetProvider =
      'ExamCalendarWidgetProvider';

  Future<void> initialize() async {
    await HomeWidget.setAppGroupId('com.example.axon');
  }

  Future<void> syncExamCountdown({
    required String examName,
    required String examCode,
    required DateTime examDate,
    String? manualSubject,
    String colorMode = 'focusBlue',
  }) async {
    // Run computation directly to avoid UI blocking
    await _syncExamCountdownCompute({
      'examName': examName,
      'examCode': examCode,
      'examDate': examDate.toIso8601String(),
      'manualSubject': manualSubject,
      'colorMode': colorMode,
    });
  }

  static Future<void> _syncExamCountdownCompute(
      Map<String, dynamic> params) async {
    final now = DateTime.now();
    final examDate = DateTime.parse(params['examDate'] as String);
    final difference = examDate.difference(now);

    final daysRemaining = difference.inDays.clamp(0, 999);
    final hoursRemaining = (difference.inHours % 24).clamp(0, 23);
    final minutesRemaining = (difference.inMinutes % 60).clamp(0, 59);
    final secondsRemaining = (difference.inSeconds % 60).clamp(0, 59);

    final prefs = await SharedPreferences.getInstance();
    unawaited(prefs.setString('exam_name', params['examName'] as String));
    unawaited(prefs.setString('exam_code', params['examCode'] as String));
    unawaited(prefs.setString('exam_date', examDate.toIso8601String()));
    unawaited(prefs.setInt('days_remaining', daysRemaining));
    unawaited(prefs.setInt('hours_remaining', hoursRemaining));
    unawaited(prefs.setInt('minutes_remaining', minutesRemaining));
    unawaited(prefs.setInt('seconds_remaining', secondsRemaining));
    if (params['manualSubject'] != null) {
      unawaited(
          prefs.setString('manual_subject', params['manualSubject'] as String));
    }
    unawaited(prefs.setString('color_mode', params['colorMode'] as String));
    unawaited(prefs.setString('last_sync', now.toIso8601String()));

    // Fire and forget widget syncs (non-blocking)
    unawaited(_syncHomeWidgetsNonBlocking(
        daysRemaining,
        hoursRemaining,
        minutesRemaining,
        secondsRemaining,
        params['manualSubject'],
        params['colorMode'] as String));
    unawaited(
        _syncCalendarWidgetNonBlocking(examDate, params['examCode'] as String));
  }

  Future<void> updateStudyTimer({
    required double progress,
    required String subject,
    required int elapsedMinutes,
  }) async {
    await HomeWidget.saveWidgetData<double>('study_progress', progress);
    await HomeWidget.saveWidgetData<String>('current_subject', subject);
    await HomeWidget.saveWidgetData<int>('elapsed_minutes', elapsedMinutes);
    await HomeWidget.updateWidget(
      name: 'StudyTimerProvider',
      androidName: 'StudyTimerProvider',
      qualifiedAndroidName: 'com.example.axon.StudyTimerProvider',
    );
  }

  static Future<void> _syncHomeWidgetsNonBlocking(int days, int hours,
      int minutes, int seconds, String? manualSubject, String colorMode) async {
    await HomeWidget.saveWidgetData('days_remaining', days);
    await HomeWidget.saveWidgetData('hours_remaining', hours);
    await HomeWidget.saveWidgetData('minutes_remaining', minutes);
    await HomeWidget.saveWidgetData('seconds_remaining', seconds);
    await HomeWidget.saveWidgetData('manual_subject', manualSubject);
    await HomeWidget.saveWidgetData('color_mode', colorMode);
    await HomeWidget.updateWidget(
      name: _flipClockWidgetProvider,
      androidName: _flipClockWidgetProvider,
      qualifiedAndroidName: 'com.example.axon.FlipClockEvolutionWidgetProvider',
    );
  }

  static Future<void> _syncCalendarWidgetNonBlocking(
      DateTime examDate, String examCode) async {
    final examRepo = ExamRepository.instance;
    await examRepo.initialize();
    final allExams = examRepo.allExams;
    final now = DateTime.now();
    final upcomingExams = allExams.where((e) => e.date.isAfter(now)).toList()
      ..sort((a, b) => a.date.compareTo(b.date));
    await HomeWidget.saveWidgetData('exam_count', upcomingExams.length);
    if (upcomingExams.isNotEmpty) {
      final next = upcomingExams.first;
      await HomeWidget.saveWidgetData(
          'next_exam_date', next.date.toIso8601String().split('T').first);
      await HomeWidget.saveWidgetData('next_exam_code', next.code);
    }
    await HomeWidget.updateWidget(
      name: _examCalendarWidgetProvider,
      androidName: _examCalendarWidgetProvider,
      qualifiedAndroidName: 'com.example.axon.ExamCalendarWidgetProvider',
    );
  }

  Future<void> syncTimerState({
    required String timerDisplay,
    required bool isRunning,
    int? totalSeconds,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('timer_display', timerDisplay);
    await prefs.setBool('is_running', isRunning);
    if (totalSeconds != null) {
      await prefs.setInt('total_seconds', totalSeconds);
    }

    await HomeWidget.saveWidgetData('timer_display', timerDisplay);
    await HomeWidget.saveWidgetData('is_running', isRunning);
    if (totalSeconds != null) {
      await HomeWidget.saveWidgetData('total_seconds', totalSeconds);
    }

    await HomeWidget.updateWidget(
      name: _industrialTimerWidgetProvider,
      androidName: _industrialTimerWidgetProvider,
      qualifiedAndroidName: 'com.example.axon.IndustrialTimerWidgetProvider',
    );
  }

  Future<void> syncContributionGraph({
    required List<int> activityLevels,
    required double masteryPercentage,
    List<StudyHeatmapSession>? sessions,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('activity_matrix', jsonEncode(activityLevels));
    await prefs.setDouble('mastery_percentage', masteryPercentage);
    if (sessions != null) {
      await prefs.setString(
        'activity_sessions',
        jsonEncode(sessions.map((session) => session.toJson()).toList()),
      );
    }

    await HomeWidget.saveWidgetData(
        'activity_matrix', jsonEncode(activityLevels));
    await HomeWidget.saveWidgetData('mastery_percentage', masteryPercentage);
    if (sessions != null) {
      await HomeWidget.saveWidgetData(
        'activity_sessions',
        jsonEncode(sessions.map((session) => session.toJson()).toList()),
      );
    }

    await HomeWidget.updateWidget(
      name: _contributionGraphWidgetProvider,
      androidName: _contributionGraphWidgetProvider,
      qualifiedAndroidName: 'com.example.axon.ContributionGraphWidgetProvider',
    );
  }

  Future<void> syncAskAxonWidget({
    String widgetType = 'compact',
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('ask_axon_widget_type', widgetType);

    await HomeWidget.saveWidgetData('ask_axon_widget_type', widgetType);

    await HomeWidget.updateWidget(
      name: _askAxonWidgetProvider,
      androidName: _askAxonWidgetProvider,
      qualifiedAndroidName: 'com.example.axon.AskAxonWidgetProvider',
    );
  }

  Future<void> syncFromExamConfig(ExamConfig config) async {
    if (config.examStartDate == null) return;

    final examName =
        config.subjects.isNotEmpty ? config.subjects.first : 'Exam';
    final examCode = _getSubjectCode(
        config.subjects.isNotEmpty ? config.subjects.first : '');

    await syncExamCountdown(
      examName: examName,
      examCode: examCode,
      examDate: config.examStartDate!,
      colorMode: 'focusBlue',
    );
  }

  String _getSubjectCode(String subject) {
    final codes = {
      'physics': '9702',
      'chemistry': '9701',
      'mathematics': '9709',
      'biology': '9700',
      'economics': '9708',
      'accounting': '9706',
      'business': '9609',
      'computer science': '9618',
      'english': '9093',
      'history': '9389',
    };

    final lowerSubject = subject.toLowerCase();
    for (final entry in codes.entries) {
      if (lowerSubject.contains(entry.key)) {
        return entry.value;
      }
    }
    return '---';
  }

  Future<void> updateAllWidgets() async {
    final examPlanner = ExamPlannerService();
    final config = await examPlanner.getExamConfig();

    if (config.examStartDate != null) {
      await syncFromExamConfig(config);
    }

    await _updateContributionGraphFromActivity();
  }

  Future<void> _updateContributionGraphFromActivity() async {
    final prefs = await SharedPreferences.getInstance();

    List<int> activityLevels = List.filled(35, 0);

    final activityRaw = prefs.getString('study_activity_levels');
    if (activityRaw != null && activityRaw.isNotEmpty) {
      try {
        final decoded = jsonDecode(activityRaw) as List;
        for (int i = 0; i < decoded.length && i < 35; i++) {
          activityLevels[i] = decoded[i] as int;
        }
      } catch (_) {}
    }

    final totalSessions = prefs.getInt('total_study_sessions') ?? 0;
    final masteryPercentage = (totalSessions / 100).clamp(0.0, 1.0);
    final heatmapSessions = StudyHeatmapSession.generateLastNDaysFromLevels(
      activityLevels,
      days: 28,
    );

    await syncContributionGraph(
      activityLevels: activityLevels,
      masteryPercentage: masteryPercentage,
      sessions: heatmapSessions,
    );
  }

  Future<void> recordStudySession({required int durationMinutes}) async {
    final prefs = await SharedPreferences.getInstance();

    List<int> activityLevels = List.filled(35, 0);
    final activityRaw = prefs.getString('study_activity_levels');
    if (activityRaw != null && activityRaw.isNotEmpty) {
      try {
        final decoded = jsonDecode(activityRaw) as List;
        for (int i = 0; i < decoded.length && i < 35; i++) {
          activityLevels[i] = decoded[i] as int;
        }
      } catch (_) {}
    }

    int todayIndex = 0;
    activityLevels[todayIndex] = _calculateActivityLevel(durationMinutes);

    await prefs.setString('study_activity_levels', jsonEncode(activityLevels));

    final totalSessions = prefs.getInt('total_study_sessions') ?? 0;
    await prefs.setInt('total_study_sessions', totalSessions + 1);

    await _updateContributionGraphFromActivity();
  }

  Future<void> syncContributionGraphFromActivity() async {
    final prefs = await SharedPreferences.getInstance();

    final now = DateTime.now();
    final activityLevels = <int>[];

    for (int i = 27; i >= 0; i--) {
      final dateKey =
          now.subtract(Duration(days: i)).toIso8601String().substring(0, 10);
      final level = prefs.getInt('activity_$dateKey') ?? 0;
      activityLevels.add(level.clamp(0, 3));
    }

    final totalMastery = activityLevels.isEmpty
        ? 0.0
        : activityLevels.reduce((a, b) => a + b) / (activityLevels.length * 3);

    await syncContributionGraph(
      activityLevels: activityLevels,
      masteryPercentage: totalMastery,
    );
  }

  int _calculateActivityLevel(int minutes) {
    if (minutes >= 120) return 3;
    if (minutes >= 60) return 2;
    if (minutes >= 30) return 1;
    return 0;
  }

  Future<Map<String, dynamic>> getWidgetData() async {
    final prefs = await SharedPreferences.getInstance();
    return {
      'exam_name': prefs.getString('exam_name') ?? 'NO EXAMS',
      'exam_code': prefs.getString('exam_code') ?? '---',
      'days_remaining': prefs.getInt('days_remaining') ?? 0,
      'hours_remaining': prefs.getInt('hours_remaining') ?? 0,
      'minutes_remaining': prefs.getInt('minutes_remaining') ?? 0,
      'seconds_remaining': prefs.getInt('seconds_remaining') ?? 0,
      'timer_display': prefs.getString('timer_display') ?? '00:00',
      'is_running': prefs.getBool('is_running') ?? false,
    };
  }
}

final widgetServiceProvider = WidgetService();
