import 'dart:convert';
import 'package:home_widget/home_widget.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/study_activity.dart';

class HomeWidgetService {
  static final HomeWidgetService instance = HomeWidgetService._();
  HomeWidgetService._();

  static const List<String> _androidWidgetNames = <String>[
    'IndustrialTimerWidgetProvider',
    'FlipClockEvolutionWidgetProvider',
    'ContributionGraphWidgetProvider',
    'AskAxonWidgetProvider',
    'ExamCalendarWidgetProvider',
  ];
  static const String iOSWidgetName = 'AxonWidget';

  static const String _activityMatrixKey = 'activity_matrix';
  static const String _lastSyncKey = 'last_widget_sync';

  Future<void> initialize() async {
    // HomeWidget.setAppGroupId is no longer needed when using direct prefs
  }

  SharedPreferences? _androidPrefs;

  Future<SharedPreferences> get _prefs async {
    _androidPrefs ??= await SharedPreferences.getInstance();
    return _androidPrefs!;
  }

  Future<void> syncContributionGraph(
    List<int> levels, {
    List<StudyHeatmapSession>? sessions,
  }) async {
    try {
      final prefs = await _prefs;
      await prefs.setString(_activityMatrixKey, jsonEncode(levels));
      if (sessions != null) {
        await prefs.setString(
          'activity_sessions',
          jsonEncode(sessions.map((session) => session.toJson()).toList()),
        );
      }
      await prefs.setString(_lastSyncKey, DateTime.now().toIso8601String());

      // Trigger widget update via HomeWidget (still needed for update signal)
      await HomeWidget.updateWidget(
        name: 'ContributionGraphWidgetProvider',
        androidName: 'ContributionGraphWidgetProvider',
        qualifiedAndroidName: 'com.example.axon.ContributionGraphWidgetProvider',
        iOSName: iOSWidgetName,
      );
    } catch (e) {
      print('HomeWidget contribution sync error: $e');
    }
  }

  Future<void> updateStudyActivity(List<StudyActivity> activities) async {
    final levels = activities.map((a) => a.level).toList();
    await syncContributionGraph(
      levels,
      sessions: StudyHeatmapSession.generateLastNDaysFromLevels(
        levels,
        days: 28,
      ),
    );
  }

  Future<List<int>> getRecentActivityLevels(int days) async {
    try {
      final data = await HomeWidget.getWidgetData<String>(
        _activityMatrixKey,
        defaultValue: '[]',
      );
      if (data != null && data.isNotEmpty) {
        final decoded = jsonDecode(data) as List;
        return decoded.cast<int>();
      }
    } catch (e) {
      print('Error reading activity levels: $e');
    }
    return List.generate(days, (_) => 0);
  }

  Future<void> handleWidgetTap() async {
    try {
      final uri = await HomeWidget.initiallyLaunchedFromHomeWidget();
      if (uri != null) {
        print('Launched from widget: $uri');
      }
    } catch (e) {
      print('Widget tap error: $e');
    }
  }

  Future<void> saveStudyData({
    String? currentSubject,
    String? currentChapter,
    int? remainingSeconds,
    bool? isRunning,
    double? progress,
    String? nextSessionTitle,
    String? nextSessionChapter,
    DateTime? nextSessionTime,
    DateTime? anchorExamDate,
    String? anchorExamName,
    String? examCode,
    double? readinessScore,
    Map<String, int>? weeklyProgress,
    List<String>? quickAccessResources,
    String? lastAxonPrompt,
    String? lastAxonResponse,
    bool? isListening,
    int? examCount,
    String? nextExamDate,
  }) async {
    final prefs = await _prefs;

    if (currentSubject != null) {
      await prefs.setString('current_subject', currentSubject);
    }
    if (currentChapter != null) {
      await prefs.setString('current_chapter', currentChapter);
    }
    if (remainingSeconds != null) {
      await prefs.setInt('remaining_seconds', remainingSeconds);
    }
    if (isRunning != null) {
      await prefs.setBool('is_running', isRunning);
    }
    if (progress != null) {
      await prefs.setDouble('current_progress', progress);
    }
    if (nextSessionTitle != null) {
      await prefs.setString('next_session_title', nextSessionTitle);
    }
    if (nextSessionChapter != null) {
      await prefs.setString('next_session_chapter', nextSessionChapter);
    }
    if (nextSessionTime != null) {
      await prefs.setString('next_session_time', nextSessionTime.toIso8601String());
    }
    if (anchorExamDate != null) {
      await prefs.setString('anchor_exam_date', anchorExamDate.toIso8601String());
    }
    if (anchorExamName != null) {
      await prefs.setString('anchor_exam_name', anchorExamName);
    }
    if (examCode != null) {
      await prefs.setString('exam_code', examCode);
    }
    if (readinessScore != null) {
      await prefs.setDouble('readiness_score', readinessScore);
    }
    if (weeklyProgress != null) {
      await prefs.setString('weekly_progress', jsonEncode(weeklyProgress));
    }
    if (quickAccessResources != null) {
      await prefs.setString('quick_access_resources', jsonEncode(quickAccessResources));
    }
    if (lastAxonPrompt != null) {
      await prefs.setString('last_axon_prompt', lastAxonPrompt);
    }
    if (lastAxonResponse != null) {
      await prefs.setString('last_axon_response', lastAxonResponse);
    }
    if (isListening != null) {
      await prefs.setBool('is_listening', isListening);
    }
    if (examCount != null) {
      await prefs.setInt('exam_count', examCount);
    }
    if (nextExamDate != null) {
      await prefs.setString('next_exam_date', nextExamDate);
    }
  }

  Future<void> updateWidgets() async {
    for (final widgetName in _androidWidgetNames) {
      await HomeWidget.updateWidget(
        name: widgetName,
        androidName: widgetName,
        qualifiedAndroidName: 'com.example.axon.$widgetName',
        iOSName: iOSWidgetName,
      );
    }
  }

  Future<void> registerBackgroundCallback(Function(Uri?) callback) async {
    await HomeWidget.registerInteractivityCallback(callback);
  }

  Future<Uri?> getWidgetLaunchUri() async {
    return await HomeWidget.initiallyLaunchedFromHomeWidget();
  }

  Future<void> clearWidgetData() async {
    final prefs = await _prefs;
    final keys = [
      'current_subject', 'current_chapter', 'remaining_seconds', 'is_running',
      'current_progress', 'next_session_title', 'next_session_chapter',
      'next_session_time', 'anchor_exam_date', 'anchor_exam_name', 'exam_code',
      'readiness_score', 'weekly_progress', 'quick_access_resources',
      'last_axon_prompt', 'last_axon_response', 'is_listening',
      'exam_count', 'next_exam_date',
    ];
    for (final key in keys) {
      await prefs.remove(key);
    }
  }

  Future<void> updateTimerWidget({
    required String subject,
    required String chapter,
    required int remainingSeconds,
    required double progress,
  }) async {
    await saveStudyData(
      currentSubject: subject,
      currentChapter: chapter,
      remainingSeconds: remainingSeconds,
      progress: progress,
    );
    await updateWidgets();
  }

  Future<void> updateNextSessionWidget({
    required String title,
    required String chapter,
    required DateTime time,
  }) async {
    await saveStudyData(
      nextSessionTitle: title,
      nextSessionChapter: chapter,
      nextSessionTime: time,
    );
    await updateWidgets();
  }

  Future<void> updateAnchorExamWidget({
    required DateTime examDate,
    required String examName,
    required double readinessScore,
  }) async {
    await saveStudyData(
      anchorExamDate: examDate,
      anchorExamName: examName,
      readinessScore: readinessScore,
    );
    await updateWidgets();
  }

  Future<void> updateWeeklyProgress(Map<String, int> progress) async {
    await saveStudyData(weeklyProgress: progress);
    await updateWidgets();
  }

  Future<void> updateQuickAccessResources(List<String> resources) async {
    await saveStudyData(quickAccessResources: resources);
    await updateWidgets();
  }

  Future<void> startDeepFocusSession({
    required String subject,
    required int durationMinutes,
  }) async {
    await saveStudyData(
      currentSubject: subject,
      remainingSeconds: durationMinutes * 60,
      progress: 0.0,
    );
    await updateWidgets();
  }

  Future<void> endDeepFocusSession() async {
    await saveStudyData(
      currentSubject: null,
      currentChapter: null,
      remainingSeconds: null,
      progress: null,
    );
    await updateWidgets();
  }
}

class DeepFocusWidgetData {
  final String subject;
  final int remainingSeconds;
  final double progress;

  DeepFocusWidgetData({
    required this.subject,
    required this.remainingSeconds,
    required this.progress,
  });

  String get formattedTime {
    final minutes = remainingSeconds ~/ 60;
    final seconds = remainingSeconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
  }

  String get formattedProgress => '${(progress * 100).toInt()}%';
}

class NextSessionWidgetData {
  final String title;
  final String chapter;
  final DateTime time;

  NextSessionWidgetData({
    required this.title,
    required this.chapter,
    required this.time,
  });

  String get formattedTime {
    final now = DateTime.now();
    final diff = time.difference(now);
    if (diff.isNegative) {
      return 'Now';
    }
    if (diff.inHours > 0) {
      return 'In ${diff.inHours}h';
    }
    return 'In ${diff.inMinutes}m';
  }
}

class AnchorExamWidgetData {
  final String examName;
  final DateTime examDate;
  final double readinessScore;

  AnchorExamWidgetData({
    required this.examName,
    required this.examDate,
    required this.readinessScore,
  });

  String get formattedCountdown {
    final now = DateTime.now();
    final diff = examDate.difference(now);
    if (diff.isNegative) {
      return 'Completed';
    }
    final days = diff.inDays;
    final hours = diff.inHours.remainder(24);
    final minutes = diff.inMinutes.remainder(60);
    return '${days}d ${hours}h ${minutes}m';
  }

  String get formattedReadiness => '${(readinessScore * 100).toInt()}%';

  bool get isUrgent => examDate.difference(DateTime.now()).inHours < 48;
}

class WeeklyProgressData {
  final Map<String, int> subjectProgress;

  WeeklyProgressData({required this.subjectProgress});

  static Future<WeeklyProgressData> load() async {
    final data = await HomeWidget.getWidgetData<String>('weekly_progress',
        defaultValue: '');
    if (data != null && data.isNotEmpty) {
      final decoded = jsonDecode(data) as Map<String, dynamic>;
      return WeeklyProgressData(
        subjectProgress: decoded.map((k, v) => MapEntry(k, v as int)),
      );
    }
    return WeeklyProgressData(subjectProgress: {});
  }

  int getTotalProgress() {
    if (subjectProgress.isEmpty) return 0;
    return subjectProgress.values.reduce((a, b) => a + b);
  }

  int getProgressForSubject(String subject) {
    return subjectProgress[subject] ?? 0;
  }
}
