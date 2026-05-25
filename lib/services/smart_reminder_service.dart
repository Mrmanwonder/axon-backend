// lib/services/smart_reminder_service.dart
// ─────────────────────────────────────────────────────────────────
// Smart Reminder Service
// Behavior-driven notifications — not time-driven.
//  • Neglected topic nudges  ("You haven't touched X in N days")
//  • Daily startup briefing  (morning intelligence digest)
//  • Exam urgency alerts     (chapter targets based on days remaining)
//  • Win notifications       (weak topic improved)
//  • Forgetfulness alerts    (spaced repetition decay predictions)
// ─────────────────────────────────────────────────────────────────

import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'notification_service.dart';
import 'analytics_service.dart';
import 'exam_planner_service.dart';
import 'coaching_persona_service.dart';

class _TopicMasteryData {
  final String subject;
  final String chapter;
  final DateTime lastStudied;
  final int studyMinutes;

  _TopicMasteryData({
    required this.subject,
    required this.chapter,
    required this.lastStudied,
    required this.studyMinutes,
  });
}

class SmartReminderService {
  static final SmartReminderService _instance =
      SmartReminderService._internal();
  factory SmartReminderService() => _instance;
  SmartReminderService._internal();

  static const String _notifiedNeglectKey = 'smart_notified_neglect';
  static const String _notifiedBriefingKey = 'smart_notified_briefing';
  static const String _notifiedExamUrgencyKey = 'smart_notified_exam_urgency';
  static const String _notifiedWinKey = 'smart_notified_wins';
  static const String _chapterScoresKey = 'smart_chapter_scores';
  static const String _studyStreakKey = 'smart_study_streak';
  static const String _lastCheckDateKey = 'smart_last_check';

  // ─────────────────────────────────────────────────────────────────
  // PUBLIC ENTRY POINT — call on app startup
  // ─────────────────────────────────────────────────────────────────

  Future<void> onAppStartup() async {
    try {
      await _checkDailyBriefing();
    } catch (e) {
      debugPrint('[SmartReminderService] Daily briefing check failed: $e');
    }
    try {
      await _checkNeglectedTopics();
    } catch (e) {
      debugPrint('[SmartReminderService] Neglected topics check failed: $e');
    }
    try {
      await _checkExamUrgency();
    } catch (e) {
      debugPrint('[SmartReminderService] Exam urgency check failed: $e');
    }
    try {
      await _checkWinNotifications();
    } catch (e) {
      debugPrint('[SmartReminderService] Win notifications check failed: $e');
    }
    try {
      await _checkForgetfulnessAlerts();
    } catch (e) {
      debugPrint(
          '[SmartReminderService] Forgetfulness alerts check failed: $e');
    }
    try {
      await _updateLastCheckDate();
    } catch (e) {
      debugPrint('[SmartReminderService] Last check date update failed: $e');
    }
  }

  Future<void> onSessionComplete({
    required String subject,
    required String chapter,
    required int durationMinutes,
    required double focusQuality,
  }) async {
    await _trackStudySession(subject, chapter);
    await _checkWinNotifications();
  }

  Future<void> onMockScoreRecorded({
    required String subject,
    required String chapter,
    required int score,
  }) async {
    final scores = await _getStoredChapterScores();
    final key = '${subject}_$chapter';
    scores[key] = {
      'subject': subject,
      'chapter': chapter,
      'score': score,
      'date': DateTime.now().toIso8601String(),
    };
    await _saveChapterScores(scores);
    await _checkWinNotifications();
  }

  // ─────────────────────────────────────────────────────────────────
  // NEGLECTED TOPIC NUDGES
  // "You haven't touched X in 5 days"
  // ─────────────────────────────────────────────────────────────────

  Future<void> _checkNeglectedTopics() async {
    final sessions = await _getSessionHistory();
    final now = DateTime.now();
    final catalog = await _loadStudyCatalog();

    final lastStudiedPerChapter = <String, DateTime>{};
    for (final session in sessions) {
      final date = _parseDate(session);
      final subject = session['subject']?.toString() ?? '';
      final chapter = session['chapter']?.toString() ?? '';
      if (subject.isEmpty || chapter.isEmpty) continue;
      final key = '${subject}_$chapter';
      if (!lastStudiedPerChapter.containsKey(key) ||
          date.isAfter(lastStudiedPerChapter[key]!)) {
        lastStudiedPerChapter[key] = date;
      }
    }

    final notified = await _getNotifiedNeglect();
    final neglectDaysThreshold = 5;
    final urgentNeglectDaysThreshold = 10;

    for (final subject in catalog.keys) {
      final chapters = _readCatalogChapters(catalog[subject]);
      for (final chapter in chapters) {
        final key = '${subject}_$chapter';
        final lastStudied = lastStudiedPerChapter[key];
        if (lastStudied == null) continue;

        final daysSince = now.difference(lastStudied).inDays;
        final alreadyNotified = notified.contains(key);

        if (daysSince >= urgentNeglectDaysThreshold && !alreadyNotified) {
          await _sendNeglectNotification(
            subject: subject,
            chapter: chapter,
            daysSince: daysSince,
            urgency: _NeglectUrgency.urgent,
          );
          notified.add(key);
        } else if (daysSince >= neglectDaysThreshold &&
            daysSince < urgentNeglectDaysThreshold &&
            !alreadyNotified) {
          await _sendNeglectNotification(
            subject: subject,
            chapter: chapter,
            daysSince: daysSince,
            urgency: _NeglectUrgency.normal,
          );
          notified.add(key);
        }
      }
    }

    await _saveNotifiedNeglect(notified);
  }

  // ─────────────────────────────────────────────────────────────────
  // FORGETFULNESS ALERTS
  // "Your mastery of 'Vectors' is dropping. Do a 5-minute drill now."
  // Based on spaced repetition decay predictions
  // ─────────────────────────────────────────────────────────────────

  Future<void> _checkForgetfulnessAlerts() async {
    final sessions = await _getSessionHistory();
    final now = DateTime.now();
    final notified = await _getNotifiedNeglect();

    final topicMasteryMap = <String, _TopicMasteryData>{};

    for (final session in sessions) {
      final date = _parseDate(session);
      final subject = session['subject']?.toString() ?? '';
      final chapter = session['chapter']?.toString() ?? '';
      final duration = (session['duration'] as num?)?.toInt() ?? 0;

      if (subject.isEmpty || chapter.isEmpty) continue;

      final key = '${subject}_$chapter';
      final existing = topicMasteryMap[key];
      if (existing == null || date.isAfter(existing.lastStudied)) {
        topicMasteryMap[key] = _TopicMasteryData(
          subject: subject,
          chapter: chapter,
          lastStudied: date,
          studyMinutes: duration,
        );
      } else {
        topicMasteryMap[key] = _TopicMasteryData(
          subject: existing.subject,
          chapter: existing.chapter,
          lastStudied: existing.lastStudied,
          studyMinutes: existing.studyMinutes + duration,
        );
      }
    }

    for (final entry in topicMasteryMap.entries) {
      final key = entry.key;
      if (notified.contains('forget_$key')) continue;

      final data = entry.value;
      final daysSince = now.difference(data.lastStudied).inDays;

      if (daysSince >= 10 && daysSince <= 14) {
        final studyRate = data.studyMinutes / 60 / (daysSince + 1);
        final hoursRemaining =
            ((0.85 - (0.3 * math.exp(-0.012 * daysSince))) / studyRate)
                .clamp(0.5, 20.0);

        final personaSvc = CoachingPersonaService();
        final title = personaSvc.getNotificationTitle('forgetfulness');
        final hoursStr = hoursRemaining.toStringAsFixed(1);

        await NotificationService.show(
          id: 4101 + key.hashCode,
          title: title,
          body:
              "Mastery of '${data.chapter}' is dropping after $daysSince days. About $hoursStr focused hours will restore your target readiness.",
          channel: AxonChannel.studyReminder,
          payload: '/study',
        );

        notified.add('forget_$key');
      }
    }

    await _saveNotifiedNeglect(notified);
  }

  Future<void> _sendNeglectNotification({
    required String subject,
    required String chapter,
    required int daysSince,
    required _NeglectUrgency urgency,
  }) async {
    final personaSvc = CoachingPersonaService();
    final personaMsg =
        personaSvc.getNeglectReminder(subject, chapter, daysSince);
    final title = personaSvc.getNotificationTitle('neglect');

    await NotificationService.show(
      id: 4001 + subject.hashCode + chapter.hashCode,
      title: title,
      body: personaMsg,
      channel: AxonChannel.studyReminder,
      payload: '/study',
    );
  }

  // ─────────────────────────────────────────────────────────────────
  // DAILY STARTUP BRIEFING
  // Fires once per day on first app open
  // ─────────────────────────────────────────────────────────────────

  Future<void> _checkDailyBriefing() async {
    final today = _todayKey();
    final notified = await _getNotifiedBriefings();
    if (notified.contains(today)) return;

    final sessions = await _getSessionHistory();
    final yesterday = DateTime.now().subtract(const Duration(days: 1));
    final yesterdaySessions = sessions.where((s) {
      final date = _parseDate(s);
      return _sameDay(date, yesterday);
    }).toList();

    final todaySessions = sessions.where((s) {
      final date = _parseDate(s);
      return _sameDay(date, DateTime.now());
    }).toList();

    final yesterdayHours = yesterdaySessions.fold(
        0.0, (sum, s) => sum + ((s['duration'] ?? 0) / 60));
    final todayHours =
        todaySessions.fold(0.0, (sum, s) => sum + ((s['duration'] ?? 0) / 60));
    final yesterdayAvgFocus = yesterdaySessions.isEmpty
        ? 0.0
        : yesterdaySessions
                .map((s) => (s['focusQuality'] ?? 0.0) as double)
                .reduce((a, b) => a + b) /
            yesterdaySessions.length;
    final targetHours = await _getTargetHours();
    final streak = await _getStudyStreak();

    final analytics = AnalyticsService();
    final delayedReport = await analytics.getDelayedChapters();
    final weakSubject = delayedReport.ignored.isNotEmpty
        ? delayedReport.ignored.first.subject
        : null;

    final body = _buildBriefingBody(
      yesterdayHours: yesterdayHours,
      targetHours: targetHours,
      yesterdayAvgFocus: yesterdayAvgFocus,
      streak: streak,
      weakSubject: weakSubject,
      todayHours: todayHours,
    );

    final personaSvc = CoachingPersonaService();
    final title = personaSvc.getNotificationTitle('briefing');

    await NotificationService.show(
      id: 4100,
      title: title,
      body: body,
      channel: AxonChannel.performanceDigest,
      payload: '/home',
    );

    notified.add(today);
    await _saveNotifiedBriefings(notified);
  }

  String _buildBriefingBody({
    required double yesterdayHours,
    required double targetHours,
    required double yesterdayAvgFocus,
    required int streak,
    required String? weakSubject,
    required double todayHours,
  }) {
    final buffer = StringBuffer();

    if (yesterdayHours > 0) {
      final diff = yesterdayHours - targetHours;
      if (diff >= 1) {
        buffer.write('You studied ${yesterdayHours.toStringAsFixed(1)}h '
            'yesterday — ${diff.toStringAsFixed(1)}h over target! ');
      } else if (diff >= 0) {
        buffer.write(
            '${yesterdayHours.toStringAsFixed(1)}h yesterday — on target. ');
      } else {
        buffer.write('${yesterdayHours.toStringAsFixed(1)}h yesterday — '
            '${(-diff).toStringAsFixed(1)}h short. ');
      }
    } else {
      buffer.write('No study sessions yesterday. ');
    }

    if (todayHours > 0) {
      buffer.write('Already at ${todayHours.toStringAsFixed(1)}h today. ');
    }

    if (yesterdayAvgFocus > 0) {
      final focusPct = (yesterdayAvgFocus * 100).round();
      if (focusPct >= 80) {
        buffer.write('Focus was $focusPct% — excellent. ');
      } else if (focusPct >= 60) {
        buffer.write('Focus was $focusPct%. ');
      }
    }

    if (weakSubject != null) {
      buffer.write('$weakSubject needs attention today. ');
    }

    if (streak > 0 && streak % 5 == 0) {
      buffer.write('$streak-day consistency — that\'s the way.');
    }

    if (buffer.isEmpty) return 'Open Axon to start your day right.';
    return buffer.toString().trim();
  }

  // ─────────────────────────────────────────────────────────────────
  // EXAM URGENCY ALERTS
  // Chapter-level targets based on days until exam
  // ─────────────────────────────────────────────────────────────────

  Future<void> _checkExamUrgency() async {
    final config = await ExamPlannerService().getExamConfig();
    if (config.examStartDate == null) return;

    final daysRemaining =
        config.examStartDate!.difference(DateTime.now()).inDays;
    if (daysRemaining < 0 || daysRemaining > 60) return;

    final notified = await _getNotifiedExamUrgency();

    if (daysRemaining <= 7 && !notified.contains('7_day')) {
      await _sendExamUrgentNotification(daysRemaining, config);
      notified.add('7_day');
    } else if (daysRemaining <= 14 && !notified.contains('14_day')) {
      await _sendExamWarningNotification(daysRemaining, config);
      notified.add('14_day');
    } else if (daysRemaining <= 30 && !notified.contains('30_day')) {
      await _sendExamPrepNotification(daysRemaining, config);
      notified.add('30_day');
    }

    if (daysRemaining <= 14) {
      final chapters = await _getChapterTargets(daysRemaining);
      for (final chapter in chapters.take(3)) {
        final chapterKey = '${chapter['subject']}_${chapter['chapter']}';
        if (!notified.contains(chapterKey)) {
          await _sendChapterTargetNotification(
            chapter['subject'] as String,
            chapter['chapter'] as String,
            daysRemaining,
          );
          notified.add(chapterKey);
        }
      }
    }

    await _saveNotifiedExamUrgency(notified);
  }

  Future<List<Map<String, dynamic>>> _getChapterTargets(
      int daysRemaining) async {
    final sessions = await _getSessionHistory();
    final catalog = await _loadStudyCatalog();

    final studyCounts = <String, int>{};
    for (final session in sessions) {
      final subject = session['subject']?.toString() ?? '';
      final chapter = session['chapter']?.toString() ?? '';
      if (subject.isEmpty || chapter.isEmpty) continue;
      final key = '${subject}_$chapter';
      studyCounts[key] = (studyCounts[key] ?? 0) + 1;
    }

    final targets = <Map<String, dynamic>>[];
    for (final entry in catalog.entries) {
      final subject = entry.key;
      final chapters = _readCatalogChapters(entry.value);
      for (final chapter in chapters) {
        final key = '${subject}_$chapter';
        final count = studyCounts[key] ?? 0;
        if (count < 2) {
          targets.add({'subject': subject, 'chapter': chapter, 'count': count});
        }
      }
    }
    targets.sort((a, b) => (a['count'] as int).compareTo(b['count'] as int));
    return targets;
  }

  Future<void> _sendExamUrgentNotification(int days, ExamConfig config) async {
    final personaSvc = CoachingPersonaService();
    final msg = personaSvc.getExamWarningMessage(days, config.subjects);
    final title = personaSvc.getNotificationTitle('exam');
    await NotificationService.show(
      id: 4200,
      title: title,
      body: msg,
      channel: AxonChannel.studyReminder,
      payload: '/exam-planner',
    );
  }

  Future<void> _sendExamWarningNotification(int days, ExamConfig config) async {
    final personaSvc = CoachingPersonaService();
    final msg = personaSvc.getExamWarningMessage(days, config.subjects);
    final title = personaSvc.getNotificationTitle('exam');
    await NotificationService.show(
      id: 4201,
      title: title,
      body: msg,
      channel: AxonChannel.studyReminder,
      payload: '/exam-planner',
    );
  }

  Future<void> _sendExamPrepNotification(int days, ExamConfig config) async {
    final personaSvc = CoachingPersonaService();
    final msg = personaSvc.getExamWarningMessage(days, config.subjects);
    final title = personaSvc.getNotificationTitle('exam');
    await NotificationService.show(
      id: 4202,
      title: title,
      body: msg,
      channel: AxonChannel.studyReminder,
      payload: '/exam-planner',
    );
  }

  Future<void> _sendChapterTargetNotification(
    String subject,
    String chapter,
    int daysRemaining,
  ) async {
    await NotificationService.show(
      id: 4300 + chapter.hashCode,
      title: 'Chapter target: $chapter',
      body: '$subject — $daysRemaining days left. '
          '"$chapter" hasn\'t been covered enough. '
          'Add it to today\'s plan.',
      channel: AxonChannel.studyReminder,
      payload: '/exam-planner',
    );
  }

  // ─────────────────────────────────────────────────────────────────
  // WIN NOTIFICATIONS
  // "Your weak topic just improved!"
  // ─────────────────────────────────────────────────────────────────

  Future<void> _checkWinNotifications() async {
    final previousScores = await _getStoredChapterScores();
    final currentScores = await _getCurrentChapterScores();
    final notified = await _getNotifiedWins();

    for (final entry in currentScores.entries) {
      final key = entry.key;
      final current = entry.value;
      final previous = previousScores[key];

      if (previous != null &&
          !notified.contains(key) &&
          current['score'] > previous['score']) {
        final improvement =
            (current['score'] as int) - (previous['score'] as int);

        if (improvement >= 5 ||
            (previous['score'] < 40 && current['score'] >= 40)) {
          final subject = current['subject'] as String;
          final chapter = current['chapter'] as String;

          await _sendWinNotification(
            subject: subject,
            chapter: chapter,
            previousScore: previous['score'] as int,
            currentScore: current['score'] as int,
            improvement: improvement,
            wasWeak: previous['score'] < 50,
          );
          notified.add(key);
        }
      }
    }

    await _saveNotifiedWins(notified);
    await _saveChapterScores(currentScores);
  }

  Future<void> _sendWinNotification({
    required String subject,
    required String chapter,
    required int previousScore,
    required int currentScore,
    required int improvement,
    required bool wasWeak,
  }) async {
    final personaSvc = CoachingPersonaService();
    final body =
        personaSvc.getWinMessage(subject, chapter, improvement.toDouble());
    final title = personaSvc.getNotificationTitle('win');

    await NotificationService.show(
      id: 4400 + chapter.hashCode,
      title: title,
      body: body,
      channel: AxonChannel.achievement,
      payload: '/analytics',
    );
  }

  // ─────────────────────────────────────────────────────────────────
  // STUDY STREAK TRACKING
  // ─────────────────────────────────────────────────────────────────

  Future<void> _trackStudySession(String subject, String chapter) async {
    final sessions = await _getSessionHistory();
    final now = DateTime.now();

    final hasTodaySession = sessions.any((s) {
      final date = _parseDate(s);
      return _sameDay(date, now);
    });

    if (!hasTodaySession) {
      final prefs = await SharedPreferences.getInstance();
      final lastActiveDay = prefs.getString(_studyStreakKey);

      if (lastActiveDay != null) {
        final lastDate = DateTime.tryParse(lastActiveDay);
        if (lastDate != null) {
          final yesterday = now.subtract(const Duration(days: 1));
          if (_sameDay(lastDate, yesterday)) {
            // streak continues — just update the key
          }
        }
      }
      await prefs.setString(_studyStreakKey, now.toIso8601String());
    }
  }

  Future<int> _getStudyStreak() async {
    final sessions = await _getSessionHistory();
    if (sessions.isEmpty) return 0;
    final now = DateTime.now();

    final daysWithStudy = <String>{};
    for (final s in sessions) {
      final date = _parseDate(s);
      daysWithStudy.add(_dateKey(date));
    }

    final sortedDays = daysWithStudy.toList()..sort((a, b) => b.compareTo(a));
    final today = _todayKey();
    final yesterday =
        _dateKey(DateTime.now().subtract(const Duration(days: 1)));

    if (sortedDays.isEmpty) return 0;
    if (sortedDays.first != today && sortedDays.first != yesterday) return 0;

    int streak = 0;
    var checkDate =
        sortedDays.first == today ? now : now.subtract(const Duration(days: 1));

    for (int i = 0; i < 365; i++) {
      final key = _dateKey(checkDate);
      if (daysWithStudy.contains(key)) {
        streak++;
        checkDate = checkDate.subtract(const Duration(days: 1));
      } else {
        break;
      }
    }

    return streak;
  }

  DateTime get now => DateTime.now();

  // ─────────────────────────────────────────────────────────────────
  // CHAPTER SCORE TRACKING
  // ─────────────────────────────────────────────────────────────────

  Future<Map<String, Map<String, dynamic>>> _getStoredChapterScores() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_chapterScoresKey);
    if (raw == null || raw.isEmpty) return {};

    try {
      final data = jsonDecode(raw) as Map<String, dynamic>;
      return data.map((k, v) =>
          MapEntry(k, Map<String, dynamic>.from(v as Map<String, dynamic>)));
    } catch (e) {
      debugPrint(
          '[SmartReminderService] Failed to load stored chapter scores: $e');
      return {};
    }
  }

  Future<void> _saveChapterScores(
      Map<String, Map<String, dynamic>> scores) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_chapterScoresKey, jsonEncode(scores));
  }

  Future<Map<String, Map<String, dynamic>>> _getCurrentChapterScores() async {
    final scores = await _getMockScores();
    final result = <String, Map<String, dynamic>>{};

    for (final score in scores) {
      final subject = score['subject']?.toString() ?? '';
      final chapter = score['chapter']?.toString() ?? '';
      if (chapter.isEmpty) continue;
      final key = '${subject}_$chapter';
      result[key] = {
        'subject': subject,
        'chapter': chapter,
        'score': score['score'] as int,
        'date': score['date']?.toString() ?? '',
      };
    }

    return result;
  }

  // ─────────────────────────────────────────────────────────────────
  // PERSISTENCE HELPERS
  // ─────────────────────────────────────────────────────────────────

  Future<Set<String>> _getNotifiedNeglect() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_notifiedNeglectKey);
    if (raw == null) return {};
    try {
      return Set<String>.from(jsonDecode(raw) as List);
    } catch (e) {
      debugPrint(
          '[SmartReminderService] Failed to load notified neglect set: $e');
      return {};
    }
  }

  Future<void> _saveNotifiedNeglect(Set<String> data) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_notifiedNeglectKey, jsonEncode(data.toList()));
  }

  Future<Set<String>> _getNotifiedBriefings() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_notifiedBriefingKey);
    if (raw == null) return {};
    try {
      return Set<String>.from(jsonDecode(raw) as List);
    } catch (e) {
      debugPrint(
          '[SmartReminderService] Failed to load notified briefings: $e');
      return {};
    }
  }

  Future<void> _saveNotifiedBriefings(Set<String> data) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_notifiedBriefingKey, jsonEncode(data.toList()));
  }

  Future<Set<String>> _getNotifiedExamUrgency() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_notifiedExamUrgencyKey);
    if (raw == null) return {};
    try {
      return Set<String>.from(jsonDecode(raw) as List);
    } catch (e) {
      debugPrint(
          '[SmartReminderService] Failed to load notified exam urgency: $e');
      return {};
    }
  }

  Future<void> _saveNotifiedExamUrgency(Set<String> data) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_notifiedExamUrgencyKey, jsonEncode(data.toList()));
  }

  Future<Set<String>> _getNotifiedWins() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_notifiedWinKey);
    if (raw == null) return {};
    try {
      return Set<String>.from(jsonDecode(raw) as List);
    } catch (e) {
      debugPrint('[SmartReminderService] Failed to load notified wins: $e');
      return {};
    }
  }

  Future<void> _saveNotifiedWins(Set<String> data) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_notifiedWinKey, jsonEncode(data.toList()));
  }

  Future<void> _updateLastCheckDate() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_lastCheckDateKey, _todayKey());
  }

  Future<double> _getTargetHours() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getDouble('userTargetHours') ??
        (prefs.getInt('userTargetHours') ?? 4).toDouble();
  }

  Future<List<Map<String, dynamic>>> _getSessionHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('timer_history');
    if (raw == null || raw.isEmpty) return [];

    try {
      final list = jsonDecode(raw) as List;
      return list.map((e) => Map<String, dynamic>.from(e)).toList();
    } catch (e) {
      debugPrint('[SmartReminderService] Failed to load session history: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> _getMockScores() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('mock_scores_history');
    if (raw == null || raw.isEmpty) return [];

    try {
      final list = jsonDecode(raw) as List;
      return list.map((e) => Map<String, dynamic>.from(e)).toList();
    } catch (e) {
      debugPrint('[SmartReminderService] Failed to load mock scores: $e');
      return [];
    }
  }

  Future<Map<String, dynamic>> _loadStudyCatalog() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('study_catalog');
    if (raw == null || raw.isEmpty) return {};

    try {
      return jsonDecode(raw) as Map<String, dynamic>;
    } catch (e) {
      debugPrint('[SmartReminderService] Failed to load study catalog: $e');
      return {};
    }
  }

  List<String> _readCatalogChapters(dynamic raw) {
    if (raw is List) {
      return raw
          .map((item) => item.toString().trim())
          .where((item) => item.isNotEmpty)
          .toList();
    }

    if (raw is Map) {
      final chapters = raw['chapters'];
      if (chapters is List) {
        return chapters
            .map((item) => item.toString().trim())
            .where((item) => item.isNotEmpty)
            .toList();
      }

      return raw.keys
          .map((item) => item.toString().trim())
          .where((item) => item.isNotEmpty && item != 'chapters')
          .toList();
    }

    return const <String>[];
  }

  // ─────────────────────────────────────────────────────────────────
  // UTILITIES
  // ─────────────────────────────────────────────────────────────────

  DateTime _parseDate(Map<String, dynamic> session) {
    try {
      return DateTime.parse(session['date']?.toString() ?? '');
    } catch (e) {
      debugPrint('[SmartReminderService] Failed to parse session date: $e');
      return DateTime.now();
    }
  }

  String _todayKey() => _dateKey(DateTime.now());

  String _dateKey(DateTime date) =>
      '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';

  bool _sameDay(DateTime a, DateTime b) =>
      a.year == b.year && a.month == b.month && a.day == b.day;
}

enum _NeglectUrgency { normal, urgent }

// Singleton provider
final smartReminderServiceProvider = SmartReminderService();
