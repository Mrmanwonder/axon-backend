// lib/providers/metrics_provider.dart
import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/models.dart';
import '../services/app_usage_service.dart';
import '../services/backend_health_service.dart';
import '../services/coach_report_service.dart';
import '../services/gamification_service.dart';
import '../services/prediction_service.dart';
import '../services/sync_service.dart';
import 'auth_provider.dart';
import 'timer_provider.dart';

final metricsProvider = StateNotifierProvider<MetricsNotifier, MetricsState>(
  (ref) => MetricsNotifier(ref),
);

class MetricsState {
  final bool hasData;
  final double sleepHours;
  final double screenTimeHours;
  final double subjectDifficulty;
  final double mockScore;
  final double predictedPerformance;
  final double sevenDayAvgSleep;
  final double sevenDayAvgFocus;
  final List<DailyMetrics> weekHistory;
  final String delta;
  final String primarySubject;
  final double syllabusCoverage;
  final double activeStudyHours;
  final double targetStudyHours;
  final double focusRatio;
  final int consistencyStreak;
  final double stressLevel;
  final bool showStreakHighlight;

  int get streak => consistencyStreak;

  const MetricsState({
    this.hasData = true,
    this.sleepHours = 0.0,
    this.screenTimeHours = 0.0,
    this.subjectDifficulty = 0.0,
    this.mockScore = 0.0,
    this.predictedPerformance = 0.0,
    this.sevenDayAvgSleep = 0.0,
    this.sevenDayAvgFocus = 0.0,
    this.weekHistory = const [],
    this.delta = '',
    this.primarySubject = '',
    this.syllabusCoverage = 0.0,
    this.activeStudyHours = 0.0,
    this.targetStudyHours = 6.0,
    this.focusRatio = 0.0,
    this.consistencyStreak = 0,
    this.stressLevel = 0.0,
    this.showStreakHighlight = false,
  });

  MetricsState copyWith({
    bool? hasData,
    double? sleepHours,
    double? screenTimeHours,
    double? subjectDifficulty,
    double? mockScore,
    double? predictedPerformance,
    double? sevenDayAvgSleep,
    double? sevenDayAvgFocus,
    List<DailyMetrics>? weekHistory,
    String? delta,
    String? primarySubject,
    double? syllabusCoverage,
    double? activeStudyHours,
    double? targetStudyHours,
    double? focusRatio,
    int? consistencyStreak,
    double? stressLevel,
    bool? showStreakHighlight,
  }) {
    return MetricsState(
      hasData: hasData ?? this.hasData,
      sleepHours: sleepHours ?? this.sleepHours,
      screenTimeHours: screenTimeHours ?? this.screenTimeHours,
      subjectDifficulty: subjectDifficulty ?? this.subjectDifficulty,
      mockScore: mockScore ?? this.mockScore,
      predictedPerformance: predictedPerformance ?? this.predictedPerformance,
      sevenDayAvgSleep: sevenDayAvgSleep ?? this.sevenDayAvgSleep,
      sevenDayAvgFocus: sevenDayAvgFocus ?? this.sevenDayAvgFocus,
      weekHistory: weekHistory ?? this.weekHistory,
      delta: delta ?? this.delta,
      primarySubject: primarySubject ?? this.primarySubject,
      syllabusCoverage: syllabusCoverage ?? this.syllabusCoverage,
      activeStudyHours: activeStudyHours ?? this.activeStudyHours,
      targetStudyHours: targetStudyHours ?? this.targetStudyHours,
      focusRatio: focusRatio ?? this.focusRatio,
      consistencyStreak: consistencyStreak ?? this.consistencyStreak,
      stressLevel: stressLevel ?? this.stressLevel,
      showStreakHighlight: showStreakHighlight ?? this.showStreakHighlight,
    );
  }
}

class MetricsNotifier extends StateNotifier<MetricsState> {
  final Ref _ref;
  final AppUsageService _appUsageService = AppUsageService();
  final PredictionService _predictionService = PredictionService();
  Timer? _screenTimeTimer;

  MetricsNotifier(this._ref) : super(const MetricsState()) {
    _loadFromPrefs();
    _startScreenTimePolling();
  }

  void syncWithProfile(UserProfile profile) {
    if (profile.targetStudyHours > 0 &&
        state.targetStudyHours != profile.targetStudyHours) {
      updateTargetStudyHours(profile.targetStudyHours.toDouble());
    }
    if (profile.subjects.isNotEmpty &&
        (state.primarySubject.isEmpty ||
            !profile.subjects.contains(state.primarySubject))) {
      updateSubject(profile.subjects.first);
    }
    if (state.consistencyStreak != profile.currentStreak) {
      updateConsistencyStreak(profile.currentStreak);
    }
  }

  Future<void> refreshMetrics() async {
    await syncScreenTimeFromDevice();
    await _computeSevenDayAvgFocus();
  }

  Future<void> _computeSevenDayAvgFocus() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString('timer_history');
      if (raw == null || raw.isEmpty) return;

      final history = jsonDecode(raw) as List;
      final sevenDaysAgo = DateTime.now().subtract(const Duration(days: 7));
      double totalFocus = 0;
      int count = 0;

      for (final entry in history) {
        if (entry is! Map) continue;
        final dateStr = entry['date']?.toString();
        if (dateStr == null) continue;
        final date = DateTime.tryParse(dateStr);
        if (date == null || date.isBefore(sevenDaysAgo)) continue;

        final quality = (entry['focusQuality'] ?? entry['intensityIndex']);
        if (quality is num) {
          totalFocus += quality.toDouble();
          count++;
        }
      }

      if (count > 0) {
        final avg = totalFocus / count;
        state = state.copyWith(sevenDayAvgFocus: avg.clamp(0.0, 1.0));
        await _saveToPrefs(state);
      }
    } catch (e) {
      debugPrint('Failed to compute 7-day avg focus: $e');
    }
  }

  Future<void> syncScreenTimeFromDevice() async {
    final hasAccess = await _appUsageService.hasUsageAccess();
    if (!hasAccess) {
      debugPrint('Screen time: Usage access denied');
      return;
    }
    final hours = await _appUsageService.getDailyScreenTimeHours();
    if (hours < 0) {
      debugPrint('Screen time fetch failed: $hours');
      return;
    }
    debugPrint('Screen time fetched: ${hours.toStringAsFixed(1)}h');
    await updateScreenTime(hours);
  }

  void _startScreenTimePolling() {
    _screenTimeTimer?.cancel();
    _screenTimeTimer = Timer.periodic(
      const Duration(minutes: 15),
      (_) => syncScreenTimeFromDevice(),
    );
  }

  Future<void> _loadFromPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('metrics_state');
    if (raw == null) return;
    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      final history = (map['weekHistory'] as List?)
              ?.map((e) => DailyMetrics.fromJson(Map<String, dynamic>.from(e)))
              .toList() ??
          const [];
      state = MetricsState(
        hasData: map['hasData'] ?? true,
        sleepHours: (map['sleepHours'] ?? 0).toDouble(),
        screenTimeHours: (map['screenTimeHours'] ?? 0).toDouble(),
        subjectDifficulty: (map['subjectDifficulty'] ?? 0).toDouble(),
        mockScore: (map['mockScore'] ?? 0).toDouble(),
        predictedPerformance: (map['predictedPerformance'] ?? 0).toDouble(),
        sevenDayAvgSleep: (map['sevenDayAvgSleep'] ?? 0).toDouble(),
        sevenDayAvgFocus: (map['sevenDayAvgFocus'] ?? 0).toDouble(),
        weekHistory: history,
        delta: (map['delta'] ?? '').toString(),
        primarySubject: (map['primarySubject'] ?? '').toString(),
        syllabusCoverage: (map['syllabusCoverage'] ?? 0).toDouble(),
        activeStudyHours: (map['activeStudyHours'] ?? 0).toDouble(),
        targetStudyHours: (map['targetStudyHours'] ?? 6).toDouble(),
        focusRatio: (map['focusRatio'] ?? 0).toDouble(),
        consistencyStreak: (map['consistencyStreak'] ?? 0).toInt(),
        stressLevel: (map['stressLevel'] ?? 0).toDouble(),
      );
    } catch (_) {
      // Ignore corrupt state
    }
  }

  Future<void> _saveToPrefs(MetricsState s) async {
    final prefs = await SharedPreferences.getInstance();
    final map = {
      'hasData': s.hasData,
      'sleepHours': s.sleepHours,
      'screenTimeHours': s.screenTimeHours,
      'subjectDifficulty': s.subjectDifficulty,
      'mockScore': s.mockScore,
      'predictedPerformance': s.predictedPerformance,
      'sevenDayAvgSleep': s.sevenDayAvgSleep,
      'sevenDayAvgFocus': s.sevenDayAvgFocus,
      'weekHistory': s.weekHistory.map((e) => e.toJson()).toList(),
      'delta': s.delta,
      'primarySubject': s.primarySubject,
      'syllabusCoverage': s.syllabusCoverage,
      'activeStudyHours': s.activeStudyHours,
      'targetStudyHours': s.targetStudyHours,
      'focusRatio': s.focusRatio,
      'consistencyStreak': s.consistencyStreak,
      'stressLevel': s.stressLevel,
    };
    await prefs.setString('metrics_state', jsonEncode(map));
  }

  double _computeScore({
    required double mockScore,
    required double studyHours,
    required double targetHours,
    required double focusRatio,
    required double sleepHours,
    required double screenTimeHours,
    double? syllabusCoverage,
    int? consistencyStreak,
    double? stressLevel,
  }) {
    return _predictionService.predict(
      mockScore: mockScore,
      studyHours: studyHours,
      targetHours: targetHours,
      focusRatio: focusRatio,
      sleepHours: sleepHours,
      screenTimeHours: screenTimeHours,
      syllabusCoverage: syllabusCoverage ?? state.syllabusCoverage,
      consistencyStreak: consistencyStreak ?? state.consistencyStreak,
      stressLevel: stressLevel ?? state.stressLevel,
    );
  }

  String _buildDelta({
    required double nextScore,
    required double sleepHours,
    required double screenTimeHours,
  }) {
    return _predictionService.explain(
      currentScore: nextScore,
      previousScore: state.predictedPerformance,
      sleepHours: sleepHours,
      sevenDayAvgSleep: state.sevenDayAvgSleep,
      screenTimeHours: screenTimeHours,
    );
  }

  void updateSleep(double hours) {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: hours,
      screenTimeHours: state.screenTimeHours,
    );
    final next = state.copyWith(
        hasData: true,
        sleepHours: hours,
        sevenDayAvgSleep:
            state.sevenDayAvgSleep == 0.0 ? hours : state.sevenDayAvgSleep,
        predictedPerformance: score,
        delta: _buildDelta(
          nextScore: score,
          sleepHours: hours,
          screenTimeHours: state.screenTimeHours,
        ));
    state = next;
    _saveToPrefs(next);
  }

  Future<void> updateScreenTime(double hours) async {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: hours,
    );
    final next = state.copyWith(
        hasData: true,
        screenTimeHours: hours,
        predictedPerformance: score,
        delta: _buildDelta(
          nextScore: score,
          sleepHours: state.sleepHours,
          screenTimeHours: hours,
        ));
    state = next;
    await _saveToPrefs(next);

    // Cloud sync if backend healthy
    final healthy = await BackendHealthService.instance.isFirestoreAvailable();
    if (healthy && _ref.read(authStateProvider).isAuthenticated) {
      final todayMetrics = DailyMetrics(
        date: DateTime.now(),
        sleepHours: next.sleepHours,
        screenTimeHours: next.screenTimeHours,
        primarySubject: next.primarySubject,
        subjectDifficulty: next.subjectDifficulty,
        studyIntensity: next.focusRatio, // approximate
        mockScore: next.mockScore,
        predictedPerformance: next.predictedPerformance,
        syllabusCoverage: next.syllabusCoverage,
        activeStudyHours: next.activeStudyHours,
        targetStudyHours: next.targetStudyHours,
        focusRatio: next.focusRatio,
        consistencyStreak: next.consistencyStreak,
        stressLevel: next.stressLevel,
      );
      unawaited(SyncService().syncDailyMetrics(todayMetrics));
      debugPrint('Screen time synced to cloud: ${hours.toStringAsFixed(1)}h');
    }
  }

  void updateMockScore(double score) {
    updateMockScoreWithChapter(
        score: score, subject: state.primarySubject, chapter: '');
  }

  Future<void> updateMockScoreWithChapter({
    required double score,
    required String subject,
    String? chapter,
  }) async {
    final chapterStr = chapter ?? '';
    if (chapterStr.isNotEmpty) {
      await _appendMockScore(subject, chapterStr, score.round());
    }
    final predictedScore = _computeScore(
      mockScore: score,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
    );
    final next = state.copyWith(
      hasData: true,
      mockScore: score,
      predictedPerformance: predictedScore,
      delta: _buildDelta(
        nextScore: predictedScore,
        sleepHours: state.sleepHours,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    await _saveToPrefs(next);
  }

  Future<void> _appendMockScore(
      String subject, String chapter, int score) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('mock_scores_history');
    List<dynamic> scores = [];
    if (raw != null && raw.isNotEmpty) {
      try {
        scores = jsonDecode(raw) as List;
      } catch (_) {}
    }
    scores.insert(0, {
      'subject': subject,
      'chapter': chapter,
      'score': score,
      'date': DateTime.now().toIso8601String(),
    });
    if (scores.length > 100) scores.removeLast();
    await prefs.setString('mock_scores_history', jsonEncode(scores));
  }

  void updateSyllabusCoverage(double value) {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
      syllabusCoverage: value,
    );
    final next = state.copyWith(
      hasData: true,
      syllabusCoverage: value,
      predictedPerformance: score,
      delta: _buildDelta(
        nextScore: score,
        sleepHours: state.sleepHours,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    _saveToPrefs(next);
  }

  void updateSubjectDifficulty(double value) {
    final next = state.copyWith(hasData: true, subjectDifficulty: value);
    state = next;
    _saveToPrefs(next);
  }

  void updateStressLevel(double value) {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
      stressLevel: value,
    );
    final next = state.copyWith(
      hasData: true,
      stressLevel: value,
      predictedPerformance: score,
      delta: _buildDelta(
        nextScore: score,
        sleepHours: state.sleepHours,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    _saveToPrefs(next);
  }

  void updateTargetStudyHours(double value) {
    final next = state.copyWith(hasData: true, targetStudyHours: value);
    state = next;
    _saveToPrefs(next);
  }

  void updateConsistencyStreak(int value) {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
      consistencyStreak: value,
    );
    final next = state.copyWith(
      hasData: true,
      consistencyStreak: value,
      predictedPerformance: score,
      showStreakHighlight: value > state.consistencyStreak,
      delta: _buildDelta(
        nextScore: score,
        sleepHours: state.sleepHours,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    _saveToPrefs(next);
  }

  void setShowStreakHighlight(bool value) {
    state = state.copyWith(showStreakHighlight: value);
    _saveToPrefs(state);
  }

  void updateSubject(String subject) {
    final next = state.copyWith(hasData: true, primarySubject: subject);
    state = next;
    _saveToPrefs(next);
  }

  Future<void> addStudySession(
      double intensityIndex, double studyHours, double focusRatio) async {
    final now = DateTime.now();
    final streak = await GamificationService.instance
        .logStudySession((studyHours * 60).round());
    final existingToday = state.weekHistory.cast<DailyMetrics?>().firstWhere(
          (h) => h != null && _sameDay(h.date, now),
          orElse: () => null,
        );
    final previousDayHours = existingToday?.activeStudyHours ?? 0.0;
    final cumulativeStudyHours = previousDayHours + studyHours;
    final previousActiveHours = state.activeStudyHours;
    final previousFocusRatio = existingToday?.focusRatio ?? 0.0;
    final previousIntensity = existingToday?.studyIntensity ?? 0.0;
    final weightedFocusRatio = cumulativeStudyHours <= 0
        ? focusRatio
        : ((previousFocusRatio * previousDayHours) +
                (focusRatio * studyHours)) /
            cumulativeStudyHours;
    final weightedIntensity = cumulativeStudyHours <= 0
        ? intensityIndex
        : ((previousIntensity * previousDayHours) +
                (intensityIndex * studyHours)) /
            cumulativeStudyHours;
    final newScore = _computeScore(
      mockScore: state.mockScore,
      studyHours: cumulativeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: weightedFocusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
      consistencyStreak: streak.currentStreak,
    );
    final updated = _upsertHistory(
        state.weekHistory,
        DailyMetrics(
          date: now,
          sleepHours: state.sleepHours,
          screenTimeHours: state.screenTimeHours,
          primarySubject:
              state.primarySubject.isEmpty ? 'General' : state.primarySubject,
          subjectDifficulty: state.subjectDifficulty,
          studyIntensity: weightedIntensity,
          mockScore: state.mockScore,
          predictedPerformance: newScore,
          syllabusCoverage: state.syllabusCoverage,
          activeStudyHours: cumulativeStudyHours,
          targetStudyHours: state.targetStudyHours,
          focusRatio: weightedFocusRatio,
          consistencyStreak: streak.currentStreak,
          stressLevel: state.stressLevel,
        ));

    final avgSleep = updated.isEmpty
        ? 0.0
        : updated.map((h) => h.sleepHours).reduce((a, b) => a + b) /
            updated.length;
    final avgFocus = updated.isEmpty
        ? 0.0
        : updated.map((h) => h.studyIntensity).reduce((a, b) => a + b) /
            updated.length;

    final next = state.copyWith(
      hasData: true,
      weekHistory: updated,
      sevenDayAvgSleep: avgSleep,
      sevenDayAvgFocus: avgFocus,
      predictedPerformance: newScore,
      activeStudyHours: cumulativeStudyHours,
      focusRatio: weightedFocusRatio,
      consistencyStreak: streak.currentStreak,
      showStreakHighlight: streak.currentStreak > state.consistencyStreak,
      delta: _predictionService.explain(
        currentScore: newScore,
        previousScore: state.predictedPerformance,
        sleepHours: state.sleepHours,
        sevenDayAvgSleep: avgSleep,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    await _saveToPrefs(next);
    await _maybeCelebrateDailyGoal(
      previousHours: previousActiveHours,
      currentHours: cumulativeStudyHours,
      targetHours: next.targetStudyHours,
      date: now,
    );
    await _maybeRewardMilestones(
      streak: streak,
      sessionMinutes: (studyHours * 60).round(),
      intensityIndex: intensityIndex,
      targetHours: next.targetStudyHours,
      currentHours: cumulativeStudyHours,
      focusRatio: weightedFocusRatio,
      screenTimeHours: next.screenTimeHours,
      date: now,
    );
    await CoachReportService.instance.maybeSendLossAversionReminder(
      metrics: next,
      style: _ref.read(authStateProvider).user?.motivationStyle ??
          MotivationStyle.positiveReinforcement,
      currentStreak: streak.currentStreak,
      board: _ref.read(authStateProvider).user?.board ?? '',
      subject: next.primarySubject,
    );
  }

  Future<void> _maybeCelebrateDailyGoal({
    required double previousHours,
    required double currentHours,
    required double targetHours,
    required DateTime date,
  }) async {
    if (targetHours <= 0) return;
    if (previousHours >= targetHours || currentHours < targetHours) return;

    final prefs = await SharedPreferences.getInstance();
    final todayKey = '${date.year}-${date.month}-${date.day}';
    if (prefs.getString('goal_unlock_chime_day') == todayKey) return;

    await prefs.setString('goal_unlock_chime_day', todayKey);
  }

  Future<void> _maybeRewardMilestones({
    required StudyStreak streak,
    required int sessionMinutes,
    required double intensityIndex,
    required double targetHours,
    required double currentHours,
    required double focusRatio,
    required double screenTimeHours,
    required DateTime date,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final xp = GamificationService.instance.calculateSessionXp(
      sessionMinutes,
      intensityIndex,
    );
    await GamificationService.instance.addXp(xp, reason: 'study_session');
    final unlocked = await GamificationService.instance.checkAndUnlockBadges();
    var milestoneTriggered = unlocked.isNotEmpty;

    final dayKey = '${date.year}-${date.month}-${date.day}';
    final perfectDay = targetHours > 0 &&
        currentHours >= targetHours &&
        focusRatio >= 0.78 &&
        screenTimeHours <= 5.5;
    if (perfectDay && prefs.getString('perfect_day_badge_day') != dayKey) {
      await prefs.setString('perfect_day_badge_day', dayKey);
      await GamificationService.instance.unlockBadge('perfect_day');
      milestoneTriggered = true;
    }

    if ((streak.currentStreak == 7 ||
            streak.currentStreak == 30 ||
            streak.currentStreak == 90) &&
        prefs.getString('streak_milestone_day') != dayKey) {
      await prefs.setString('streak_milestone_day', dayKey);
      milestoneTriggered = true;
    }
  }

  List<DailyMetrics> _upsertHistory(
      List<DailyMetrics> history, DailyMetrics entry) {
    final next = List<DailyMetrics>.from(history);
    final idx = next.indexWhere((h) => _sameDay(h.date, entry.date));
    if (idx >= 0) {
      next[idx] = entry;
    } else {
      next.add(entry);
    }
    next.sort((a, b) => a.date.compareTo(b.date));
    if (next.length > 7) {
      return next.sublist(next.length - 7);
    }
    return next;
  }

  bool _sameDay(DateTime a, DateTime b) =>
      a.year == b.year && a.month == b.month && a.day == b.day;

  @override
  void dispose() {
    _screenTimeTimer?.cancel();
    super.dispose();
  }
}
