import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

enum AxonChannel {
  studyReminder,
  crawlerAlert,
  performanceDigest,
  achievement,
  streakAlert,
  motivation,
  weeklyRecap
}

class NotificationService {
  NotificationService._();

  static final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();
  static bool _initialized = false;

  static Future<void> initialize() async {
    if (_initialized) return;

    const androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );
    const initSettings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _plugin.initialize(
      settings: initSettings,
      onDidReceiveNotificationResponse: _onNotificationTap,
    );
    await _createChannels();
    _initialized = true;
  }

  static Future<void> _createChannels() async {
    final android = _plugin.resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin>();
    if (android == null) return;

    for (final channel in AxonChannel.values) {
      await android.createNotificationChannel(
        AndroidNotificationChannel(
          channel.name,
          _channelName(channel),
          description: _channelDesc(channel),
          importance: Importance.high,
        ),
      );
    }
  }

  static String _channelName(AxonChannel channel) {
    switch (channel) {
      case AxonChannel.studyReminder:
        return 'Study Reminders';
      case AxonChannel.crawlerAlert:
        return 'Resource Alerts';
      case AxonChannel.performanceDigest:
        return 'Performance & Insights';
      case AxonChannel.achievement:
        return 'Achievements';
      case AxonChannel.streakAlert:
        return 'Streak Alerts';
      case AxonChannel.motivation:
        return 'Motivation';
      case AxonChannel.weeklyRecap:
        return 'Weekly Recaps';
    }
  }

  static String _channelDesc(AxonChannel channel) {
    switch (channel) {
      case AxonChannel.studyReminder:
        return 'Study reminders and nudges';
      case AxonChannel.crawlerAlert:
        return 'New study resources available';
      case AxonChannel.performanceDigest:
        return 'Performance analysis and daily briefings';
      case AxonChannel.achievement:
        return 'Level-ups, wins, and milestones';
      case AxonChannel.streakAlert:
        return 'Streak protection and milestone alerts';
      case AxonChannel.motivation:
        return 'Encouraging messages';
      case AxonChannel.weeklyRecap:
        return 'Weekly study summaries';
    }
  }

  static void _onNotificationTap(NotificationResponse response) {
    debugPrint('Notification tapped: payload=${response.payload}');
  }

  static Future<void> show({
    required int id,
    required String title,
    required String body,
    AxonChannel channel = AxonChannel.performanceDigest,
    String? payload,
  }) async {
    if (!_initialized) await initialize();

    final androidDetails = AndroidNotificationDetails(
      channel.name,
      _channelName(channel),
      channelDescription: _channelDesc(channel),
      importance: Importance.high,
      priority: Priority.high,
    );
    const iosDetails = DarwinNotificationDetails();
    final details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    try {
      await _plugin.show(
        id: id,
        title: title,
        body: body,
        notificationDetails: details,
      );
    } catch (e) {
      debugPrint('NotificationService: failed to show notification: $e');
    }
  }

  // ── Convenience methods ──────────────────────────────────────────

  static Future<void> showStudyReminder({
    required String subject,
    required String chapter,
    required double targetHours,
  }) =>
      show(
        id: 1001,
        title: 'Time to study - $subject',
        body:
            'Chapter: $chapter - Target: ${targetHours.toStringAsFixed(0)}h today.',
        channel: AxonChannel.studyReminder,
        payload: '/study',
      );

  static Future<void> showCrawlerAlert({
    required String subject,
    required String chapter,
    required int count,
  }) =>
      show(
        id: 2001,
        title: 'New resources - $subject',
        body: '$count new resources found for "$chapter".',
        channel: AxonChannel.crawlerAlert,
        payload: '/study',
      );

  static Future<void> showPerformanceDigest({
    required String label,
    required String xaiInsight,
  }) =>
      show(
        id: 3001,
        title: 'Axon Intelligence - $label',
        body: xaiInsight,
        channel: AxonChannel.performanceDigest,
        payload: '/home',
      );

  static Future<void> showMotivation({required String message}) => show(
      id: 4001,
      title: 'Keep going!',
      body: message,
      channel: AxonChannel.motivation,
      payload: '/home');

  static Future<void> showWeeklyRecap({required String summary}) => show(
      id: 5001,
      title: 'Weekly Recap',
      body: summary,
      channel: AxonChannel.weeklyRecap,
      payload: '/analysis');

  static Future<void> showWeeklySummaryNotification({
    required List<int> dailyMinutes,
    required int totalHours,
  }) =>
      show(
        id: 5002,
        title: 'Weekly Summary',
        body: 'Total: $totalHours this week',
        channel: AxonChannel.weeklyRecap,
        payload: '/analysis',
      );

  static Future<void> showStudyCheck({
    required bool studiedToday,
    String? nextSubject,
  }) =>
      show(
        id: 6001,
        title: studiedToday ? 'Great job today!' : 'Start studying!',
        body: studiedToday
            ? 'You have studied today.'
            : (nextSubject != null
                ? 'Ready to start $nextSubject?'
                : 'Tap to start your study session.'),
        channel: AxonChannel.studyReminder,
        payload: studiedToday ? '/home' : '/study',
      );

  static Future<void> showStreakReminder({
    required int currentStreak,
    required bool studiedToday,
  }) =>
      show(
        id: 7001,
        title: studiedToday ? 'Streak protected!' : 'Do not lose your streak!',
        body: studiedToday
            ? '$currentStreak day streak intact.'
            : 'You have a $currentStreak day streak.',
        channel: AxonChannel.streakAlert,
        payload: studiedToday ? '/home' : '/study',
      );

  static Future<void> showStreakBroken({required int lostStreak}) => show(
      id: 7002,
      title: 'Streak broken',
      body: 'Your $lostStreak day streak is over.',
      channel: AxonChannel.streakAlert,
      payload: '/study');

  static Future<void> showStreakMilestone({required int streakDays}) => show(
      id: 7003,
      title: 'Streak Milestone',
      body: '$streakDays day streak reached.',
      channel: AxonChannel.streakAlert,
      payload: '/home');

  static Future<void> showMilestoneLevelUp({
    required int level,
    required String title,
  }) =>
      show(
        id: 8001,
        title: 'Level Up',
        body: 'You are now $title (Level $level).',
        channel: AxonChannel.achievement,
        payload: '/home',
      );

  static Future<void> cancelStudyReminders() async => _plugin.cancelAll();
  static Future<void> cancelCrawlerAlerts() async => _plugin.cancelAll();
  static Future<void> cancelPerformanceDigest() async => _plugin.cancelAll();
  static Future<void> cancelAll() async => _plugin.cancelAll();
}
