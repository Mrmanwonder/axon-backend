// lib/providers/timer_provider.dart
import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../models/models.dart';
import '../services/axon_feedback_service.dart';
import '../services/firestore_service.dart';
import '../services/notification_service.dart';
import '../services/smart_reminder_service.dart';
import '../services/sync_service.dart';
import 'auth_provider.dart';
import 'metrics_provider.dart';

// Study Session Mode and Extensions
enum StudySessionMode {
  pomodoro,
  examSimulation,
  recallSprint,
  timedPaper,
}

extension StudySessionModeX on StudySessionMode {
  String get id => name;

  String get label {
    switch (this) {
      case StudySessionMode.pomodoro:
        return 'Pomodoro';
      case StudySessionMode.examSimulation:
        return 'Exam Simulation';
      case StudySessionMode.recallSprint:
        return 'Recall Sprint';
      case StudySessionMode.timedPaper:
        return 'Timed Paper';
    }
  }

  String get description {
    switch (this) {
      case StudySessionMode.pomodoro:
        return 'Structured work with frequent resets.';
      case StudySessionMode.examSimulation:
        return 'Long unbroken focus with low interruption tolerance.';
      case StudySessionMode.recallSprint:
        return 'Short, intense memory retrieval bursts.';
      case StudySessionMode.timedPaper:
        return 'Exam-paper pacing with stricter timing pressure.';
    }
  }

  int get fatigueThresholdMinutes {
    switch (this) {
      case StudySessionMode.pomodoro:
        return 25;
      case StudySessionMode.examSimulation:
        return 55;
      case StudySessionMode.recallSprint:
        return 18;
      case StudySessionMode.timedPaper:
        return 45;
    }
  }

  static StudySessionMode fromId(String raw) {
    return StudySessionMode.values.firstWhere(
      (mode) => mode.id == raw,
      orElse: () => StudySessionMode.pomodoro,
    );
  }
}

// Session Reflection
enum SessionReflection {
  easy,
  draining,
  confusing,
  productive,
}

extension SessionReflectionX on SessionReflection {
  String get label {
    switch (this) {
      case SessionReflection.easy:
        return 'Easy';
      case SessionReflection.draining:
        return 'Draining';
      case SessionReflection.confusing:
        return 'Confusing';
      case SessionReflection.productive:
        return 'Productive';
    }
  }
}

// Session Summary
class SessionSummary {
  final Map<String, dynamic> record;
  final double focusQuality;
  final double fatigueIndex;
  final int recommendedBreakMinutes;

  const SessionSummary({
    required this.record,
    required this.focusQuality,
    required this.fatigueIndex,
    required this.recommendedBreakMinutes,
  });
}

// Timer State
class TimerState {
  final bool isRunning;
  final Duration elapsed;
  final int breakCount;
  final List<Duration> focusSegments;
  final String subject;
  final String chapter;
  final double intensityIndex;
  final int pings;
  final DateTime? startedAt;
  final bool isRestoring;
  final String templateId;
  final String templateName;
  final Duration? targetDuration;
  final StudySessionMode mode;
  final bool antiDistractionLock;
  final int interruptionCount;
  final int distractionNudges;
  final double fatigueIndex;
  final double focusQuality;
  final int recommendedBreakMinutes;
  final bool pomodoroEnabled;
  final String? lastOpenedResource;
  final String? lastOpenedResourceType;
  final String activityType;

  const TimerState({
    this.isRunning = false,
    this.elapsed = Duration.zero,
    this.breakCount = 0,
    this.focusSegments = const [],
    this.subject = '',
    this.chapter = '',
    this.intensityIndex = 0.0,
    this.pings = 0,
    this.startedAt,
    this.isRestoring = false,
    this.templateId = '',
    this.templateName = '',
    this.targetDuration,
    this.mode = StudySessionMode.pomodoro,
    this.antiDistractionLock = false,
    this.interruptionCount = 0,
    this.distractionNudges = 0,
    this.fatigueIndex = 0.0,
    this.focusQuality = 1.0,
    this.recommendedBreakMinutes = 0,
    this.pomodoroEnabled = true,
    this.lastOpenedResource,
    this.lastOpenedResourceType,
    this.activityType = 'timer',
  });

  Duration get remaining {
    final target = targetDuration;
    if (target == null) return Duration.zero;
    final delta = target - elapsed;
    return delta.isNegative ? Duration.zero : delta;
  }

  bool get hasTargetDuration =>
      targetDuration != null && targetDuration! > Duration.zero;

  TimerState copyWith({
    bool? isRunning,
    Duration? elapsed,
    int? breakCount,
    List<Duration>? focusSegments,
    String? subject,
    String? chapter,
    double? intensityIndex,
    int? pings,
    DateTime? startedAt,
    bool clearStartedAt = false,
    bool? isRestoring,
    String? templateId,
    String? templateName,
    Duration? targetDuration,
    bool clearTargetDuration = false,
    StudySessionMode? mode,
    bool? antiDistractionLock,
    int? interruptionCount,
    int? distractionNudges,
    double? fatigueIndex,
    double? focusQuality,
    int? recommendedBreakMinutes,
    bool? pomodoroEnabled,
    String? lastOpenedResource,
    String? lastOpenedResourceType,
    String? activityType,
  }) {
    return TimerState(
      isRunning: isRunning ?? this.isRunning,
      elapsed: elapsed ?? this.elapsed,
      breakCount: breakCount ?? this.breakCount,
      focusSegments: focusSegments ?? this.focusSegments,
      subject: subject ?? this.subject,
      chapter: chapter ?? this.chapter,
      intensityIndex: intensityIndex ?? this.intensityIndex,
      pings: pings ?? this.pings,
      startedAt: clearStartedAt ? null : (startedAt ?? this.startedAt),
      isRestoring: isRestoring ?? this.isRestoring,
      templateId: templateId ?? this.templateId,
      templateName: templateName ?? this.templateName,
      targetDuration:
          clearTargetDuration ? null : (targetDuration ?? this.targetDuration),
      mode: mode ?? this.mode,
      antiDistractionLock: antiDistractionLock ?? this.antiDistractionLock,
      interruptionCount: interruptionCount ?? this.interruptionCount,
      distractionNudges: distractionNudges ?? this.distractionNudges,
      fatigueIndex: fatigueIndex ?? this.fatigueIndex,
      focusQuality: focusQuality ?? this.focusQuality,
      recommendedBreakMinutes:
          recommendedBreakMinutes ?? this.recommendedBreakMinutes,
      pomodoroEnabled: pomodoroEnabled ?? this.pomodoroEnabled,
      lastOpenedResource: lastOpenedResource ?? this.lastOpenedResource,
      lastOpenedResourceType:
          lastOpenedResourceType ?? this.lastOpenedResourceType,
      activityType: activityType ?? this.activityType,
    );
  }
}

// Session Signals (internal helper)
class _SessionSignals {
  final double intensity;
  final double focusQuality;
  final double fatigueIndex;
  final int recommendedBreakMinutes;

  const _SessionSignals({
    required this.intensity,
    required this.focusQuality,
    required this.fatigueIndex,
    required this.recommendedBreakMinutes,
  });
}

// Timer Notifier Provider
final timerProvider = StateNotifierProvider<TimerNotifier, TimerState>(
  (ref) => TimerNotifier(ref),
);

class TimerNotifier extends StateNotifier<TimerState> {
  final Ref _ref;
  Timer? _ticker;
  Timer? _cloudSyncTimer;
  static const _prefsKey = 'active_timer_state';
  final SyncService _syncService = SyncService();

  TimerNotifier(this._ref) : super(const TimerState(isRestoring: true)) {
    _restore();
  }

  Future<void> _restore() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_prefsKey);

    final recoverableSession = await _syncService.getRecoverableSession();

    if (recoverableSession != null && raw != null && raw.isNotEmpty) {
      try {
        final map = Map<String, dynamic>.from(jsonDecode(raw));
        final startedAtRaw = map['startedAt']?.toString();
        final startedAt = startedAtRaw == null || startedAtRaw.isEmpty
            ? null
            : DateTime.tryParse(startedAtRaw);

        state = TimerState(
          isRunning: map['isRunning'] == true,
          elapsed:
              Duration(seconds: (map['elapsedSeconds'] as num?)?.toInt() ?? 0),
          breakCount: (map['breakCount'] as num?)?.toInt() ?? 0,
          subject: (map['subject'] ?? recoverableSession.subject).toString(),
          chapter: (map['chapter'] ?? recoverableSession.chapter).toString(),
          intensityIndex: (map['intensityIndex'] as num?)?.toDouble() ?? 0.0,
          pings: (map['pings'] as num?)?.toInt() ?? 0,
          startedAt: startedAt,
          isRestoring: false,
          templateId: (map['templateId'] ?? '').toString(),
          templateName: (map['templateName'] ?? '').toString(),
          targetDuration: (map['targetDurationSeconds'] as num?) == null
              ? null
              : Duration(
                  seconds: (map['targetDurationSeconds'] as num).toInt()),
          mode: StudySessionModeX.fromId((map['modeId'] ?? '').toString()),
          antiDistractionLock: map['antiDistractionLock'] == true,
          interruptionCount: (map['interruptionCount'] as num?)?.toInt() ?? 0,
          distractionNudges: (map['distractionNudges'] as num?)?.toInt() ?? 0,
          fatigueIndex: (map['fatigueIndex'] as num?)?.toDouble() ?? 0.0,
          focusQuality: (map['focusQuality'] as num?)?.toDouble() ?? 1.0,
          recommendedBreakMinutes:
              (map['recommendedBreakMinutes'] as num?)?.toInt() ?? 0,
          pomodoroEnabled: map['pomodoroEnabled'] != false,
        );
        if (state.isRunning && state.startedAt != null) {
          state = state.copyWith(isRunning: false);
          await prefs.remove(_prefsKey);
        }
      } catch (_) {
        state = const TimerState();
        await prefs.remove(_prefsKey);
      }
    } else {
      state = const TimerState();
    }
  }

  void start(
    String subject,
    String chapter, {
    String templateId = '',
    String templateName = '',
    Duration? targetDuration,
    StudySessionMode mode = StudySessionMode.pomodoro,
    bool antiDistractionLock = false,
    bool pomodoroEnabled = true,
  }) {
    final startedAt = DateTime.now().subtract(state.elapsed);
    state = state.copyWith(
      isRunning: true,
      subject: subject,
      chapter: chapter,
      startedAt: startedAt,
      isRestoring: false,
      templateId: templateId,
      templateName: templateName,
      targetDuration: targetDuration,
      mode: mode,
      antiDistractionLock: antiDistractionLock,
      pomodoroEnabled: pomodoroEnabled,
      lastOpenedResource: null,
      lastOpenedResourceType: null,
    );
    _saveState();
    _syncActiveSession();
    AxonFeedbackService.startActiveLoop();
    _ticker?.cancel();
    _ticker = Timer.periodic(const Duration(seconds: 1), (t) => tick());
  }

  void trackOpenedResource(String resourceUrl, String resourceType) {
    state = state.copyWith(
      lastOpenedResource: resourceUrl,
      lastOpenedResourceType: resourceType,
    );
    _saveState();
  }

  void pause({bool countAsBreak = true}) {
    _syncElapsedFromClock();
    final signals = _computeSessionSignals(
      elapsed: state.elapsed,
      breakCount: state.breakCount + (countAsBreak ? 1 : 0),
      interruptionCount: state.interruptionCount,
      pings: state.pings,
      mode: state.mode,
    );
    state = state.copyWith(
      isRunning: false,
      breakCount: state.breakCount + (countAsBreak ? 1 : 0),
      clearStartedAt: true,
      intensityIndex: signals.intensity,
      focusQuality: signals.focusQuality,
      fatigueIndex: signals.fatigueIndex,
      recommendedBreakMinutes: signals.recommendedBreakMinutes,
    );
    _ticker?.cancel();
    AxonFeedbackService.stopActiveLoop();
    _saveState();
  }

  void tick() {
    if (state.isRunning) {
      final nextElapsed = _elapsedFromClock();
      var nextPings = state.pings;
      if (nextElapsed.inSeconds > 0 &&
          nextElapsed.inSeconds % 300 == 0 &&
          nextElapsed.inSeconds != state.elapsed.inSeconds) {
        nextPings++;
      }
      final signals = _computeSessionSignals(
        elapsed: nextElapsed,
        breakCount: state.breakCount,
        interruptionCount: state.interruptionCount,
        pings: nextPings,
        mode: state.mode,
      );
      state = state.copyWith(
        elapsed: nextElapsed,
        pings: nextPings,
        intensityIndex: signals.intensity,
        focusQuality: signals.focusQuality,
        fatigueIndex: signals.fatigueIndex,
        recommendedBreakMinutes: signals.recommendedBreakMinutes,
      );
      _saveState();
      if (state.hasTargetDuration && nextElapsed >= state.targetDuration!) {
        unawaited(stop());
      }
    }
  }

  void markInterruption() {
    if (!state.isRunning) return;
    final nextInterruptions = state.interruptionCount + 1;
    final nextNudges = state.antiDistractionLock
        ? state.distractionNudges + 1
        : state.distractionNudges;
    final signals = _computeSessionSignals(
      elapsed: _elapsedFromClock(),
      breakCount: state.breakCount,
      interruptionCount: nextInterruptions,
      pings: state.pings,
      mode: state.mode,
    );
    state = state.copyWith(
      elapsed: _elapsedFromClock(),
      interruptionCount: nextInterruptions,
      distractionNudges: nextNudges,
      intensityIndex: signals.intensity,
      focusQuality: signals.focusQuality,
      fatigueIndex: signals.fatigueIndex,
      recommendedBreakMinutes: signals.recommendedBreakMinutes,
    );
    _saveState();
  }

  Future<SessionSummary> stop() async {
    _syncElapsedFromClock();
    _ticker?.cancel();
    _cloudSyncTimer?.cancel();
    await AxonFeedbackService.stopActiveLoop();
    final signals = _computeSessionSignals(
      elapsed: state.elapsed,
      breakCount: state.breakCount,
      interruptionCount: state.interruptionCount,
      pings: state.pings,
      mode: state.mode,
    );
    final intensity = signals.intensity;
    final studyHours = state.elapsed.inSeconds / 3600.0;

    await _ref.read(metricsProvider.notifier).addStudySession(
          intensity,
          studyHours,
          signals.focusQuality,
        );

    final record = {
      'subject': state.subject,
      'chapter': state.chapter,
      'duration': state.elapsed.inSeconds,
      'date': DateTime.now().toIso8601String(),
      'intensity': intensity,
      'focusQuality': signals.focusQuality,
      'fatigueIndex': signals.fatigueIndex,
      'breaks': state.breakCount,
      'interruptions': state.interruptionCount,
      'distractionNudges': state.distractionNudges,
      'recommendedBreakMinutes': signals.recommendedBreakMinutes,
      'templateId': state.templateId,
      'templateName': state.templateName,
      'targetDurationSeconds': state.targetDuration?.inSeconds,
      'modeId': state.mode.id,
      'modeName': state.mode.label,
      'antiDistractionLock': state.antiDistractionLock,
      'reflection': '',
      'activityType': state.activityType,
    };
    await _syncSessionToCloud(record);

    // Write-through cache: persist cloud-facing session first, then update
    // local history for offline reads and fast UI restores.
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString('timer_history');
    List<dynamic> history = [];
    if (data != null) {
      try {
        history = jsonDecode(data);
      } catch (_) {}
    }
    history.insert(0, record);
    if (history.length > 50) history.removeLast();
    await prefs.setString('timer_history', jsonEncode(history));
    await prefs.remove(_prefsKey);

    await _syncService.clearActiveSession();

    await SmartReminderService().onSessionComplete(
      subject: state.subject,
      chapter: state.chapter,
      durationMinutes: state.elapsed.inMinutes,
      focusQuality: signals.focusQuality,
    );

    await _maybeNotifyIntensityDrop();

    state = const TimerState();
    return SessionSummary(
      record: record,
      focusQuality: signals.focusQuality,
      fatigueIndex: signals.fatigueIndex,
      recommendedBreakMinutes: signals.recommendedBreakMinutes,
    );
  }

  void reset() {
    _ticker?.cancel();
    _cloudSyncTimer?.cancel();
    AxonFeedbackService.stopActiveLoop();
    _clearSavedState();
    _syncService.clearActiveSession();
    state = const TimerState();
  }

  void updateSubject(String subject) {
    state = state.copyWith(subject: subject);
  }

  Future<void> _syncActiveSession() async {
    if (state.subject.isEmpty || state.startedAt == null) return;

    final sessionPings = <SessionPing>[];
    for (int i = 0; i < state.pings; i++) {
      sessionPings.add(SessionPing(
        timestamp: state.startedAt!.add(Duration(minutes: 5 * (i + 1))),
        isActive: true,
      ));
    }

    final session = ActiveSession(
      sessionId: 'session_${DateTime.now().millisecondsSinceEpoch}',
      subject: state.subject,
      chapter: state.chapter,
      startedAt: state.startedAt!,
      elapsed: state.elapsed,
      breakCount: state.breakCount,
      pings: sessionPings,
      deviceId: '',
      isSyncing: false,
      metadata: {
        'modeId': state.mode.id,
        'templateId': state.templateId,
        'antiDistractionLock': state.antiDistractionLock,
      },
    );

    await _syncService.saveActiveSession(session);

    _cloudSyncTimer?.cancel();
    _cloudSyncTimer = Timer.periodic(const Duration(seconds: 30), (_) async {
      if (state.isRunning) {
        final updatedSession = session.copyWith(
          elapsed: state.elapsed,
          breakCount: state.breakCount,
        );
        await _syncService.saveActiveSession(updatedSession);
      }
    });
  }

  Future<void> _syncSessionToCloud(Map<String, dynamic> record) async {
    final sessionId =
        record['id'] ?? 'session_${DateTime.now().millisecondsSinceEpoch}';
    final session = StudySession(
      id: sessionId,
      date: DateTime.now(),
      durationMinutes: ((record['duration'] ?? 0) / 60).round(),
      subject: record['subject'] ?? '',
      breakCount: record['breaks'] ?? 0,
      intensityIndex: (record['intensity'] ?? 0.0).toDouble(),
      pings: const [],
    );
    await _syncService.syncSession(session);

    final user = _firebaseAuthOrNull?.currentUser;
    if (user == null) return;

    final objectiveId = _objectiveIdFromRecord(record);
    await AxonPaths.privateUserCollection(user.uid, 'study_events')
        .doc(sessionId)
        .set({
      'id': sessionId,
      'subject': (record['subject'] ?? '').toString(),
      'chapter': (record['chapter'] ?? '').toString(),
      'objective_id': objectiveId,
      'topic_id': objectiveId,
      'type': 'study_session',
      'duration_minutes': ((record['duration'] ?? 0) / 60).round(),
      'occurred_at': DateTime.now().toIso8601String(),
      'accuracy_score': (record['focusQuality'] ?? 0.0).toDouble(),
      'intensity': (record['intensity'] ?? 0.0).toDouble(),
    }, SetOptions(merge: true));

    try {
      final token = await user.getIdToken();
      if (token != null && token.isNotEmpty) {
        await http
            .post(
              Uri.parse('https://axon-ml.onrender.com/analyze-study-pulse'),
              headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer $token',
              },
              body: jsonEncode({
                'user_id': user.uid,
                'session_id': sessionId,
              }),
            )
            .timeout(const Duration(seconds: 8));
      }
    } catch (_) {}
  }

  String _objectiveIdFromRecord(Map<String, dynamic> record) {
    final existing = (record['objective_id'] ?? '').toString().trim();
    if (existing.isNotEmpty) {
      return existing;
    }
    final chapter = (record['chapter'] ?? '').toString().trim();
    final subject = (record['subject'] ?? '').toString().trim();
    final board = (record['board'] ?? '').toString().trim().toUpperCase();
    final raw = chapter.isNotEmpty ? chapter : subject;
    final slug = raw
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^_|_$'), '');
    if (board.isNotEmpty && subject.isNotEmpty) {
      return '${board}_${subject.toUpperCase().replaceAll(' ', '_')}_$slug';
    }
    return slug;
  }

  Duration _elapsedFromClock() {
    final startedAt = state.startedAt;
    if (startedAt == null) {
      return state.elapsed;
    }
    return DateTime.now().difference(startedAt);
  }

  void _syncElapsedFromClock() {
    if (!state.isRunning || state.startedAt == null) return;
    state = state.copyWith(elapsed: _elapsedFromClock());
  }

  Future<void> _saveState() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _prefsKey,
      jsonEncode({
        'isRunning': state.isRunning,
        'elapsedSeconds': state.elapsed.inSeconds,
        'breakCount': state.breakCount,
        'subject': state.subject,
        'chapter': state.chapter,
        'intensityIndex': state.intensityIndex,
        'pings': state.pings,
        'startedAt': state.startedAt?.toIso8601String(),
        'templateId': state.templateId,
        'templateName': state.templateName,
        'targetDurationSeconds': state.targetDuration?.inSeconds,
        'modeId': state.mode.id,
        'antiDistractionLock': state.antiDistractionLock,
        'interruptionCount': state.interruptionCount,
        'distractionNudges': state.distractionNudges,
        'fatigueIndex': state.fatigueIndex,
        'focusQuality': state.focusQuality,
        'recommendedBreakMinutes': state.recommendedBreakMinutes,
        'pomodoroEnabled': state.pomodoroEnabled,
      }),
    );
  }

  _SessionSignals _computeSessionSignals({
    required Duration elapsed,
    required int breakCount,
    required int interruptionCount,
    required int pings,
    required StudySessionMode mode,
  }) {
    final totalMinutes = elapsed.inSeconds / 60.0;
    if (totalMinutes <= 0) {
      return const _SessionSignals(
        intensity: 0.0,
        focusQuality: 1.0,
        fatigueIndex: 0.0,
        recommendedBreakMinutes: 0,
      );
    }

    final breakPenalty = breakCount * 0.1;
    final interruptionPenalty = interruptionCount * 0.13;
    final threshold = mode.fatigueThresholdMinutes.toDouble();
    final overtimeRatio = totalMinutes <= threshold
        ? 0.0
        : ((totalMinutes - threshold) / threshold);
    final fatigueIndex = (overtimeRatio * 0.75 +
            (breakCount * 0.05) +
            (interruptionCount * 0.08))
        .clamp(0.0, 1.0);
    final pingBonus = pings == 0
        ? 0.0
        : (pings / ((totalMinutes / 5.0).ceil().clamp(1, 9999))) * 0.05;
    final focusQuality = (1.0 -
            breakPenalty -
            interruptionPenalty -
            (fatigueIndex * 0.22) +
            pingBonus)
        .clamp(0.0, 1.0);
    final intensity = (1.0 - (breakPenalty * 0.7) - (interruptionPenalty * 0.9))
        .clamp(0.0, 1.0);

    final recommendedBreakMinutes = fatigueIndex >= 0.75
        ? 15
        : fatigueIndex >= 0.5
            ? 10
            : fatigueIndex >= 0.3
                ? 5
                : 0;

    return _SessionSignals(
      intensity: intensity,
      focusQuality: focusQuality,
      fatigueIndex: fatigueIndex,
      recommendedBreakMinutes: recommendedBreakMinutes,
    );
  }

  Future<void> _maybeNotifyIntensityDrop() async {
    final metrics = _ref.read(metricsProvider);
    final history = metrics.weekHistory;
    if (history.length < 3) return;
    final last3 = history.sublist(history.length - 3);
    final drop = last3.first.studyIntensity - last3.last.studyIntensity;
    if (drop < 0.12) return;

    final prefs = await SharedPreferences.getInstance();
    final lastAlert = prefs.getString('intensity_alert_date');
    final today = DateTime.now();
    final todayKey = '${today.year}-${today.month}-${today.day}';
    if (lastAlert == todayKey) return;

    final style = _ref.read(authStateProvider).user?.motivationStyle ??
        MotivationStyle.positiveReinforcement;
    final message = _buildMotivationMessage(style, drop);
    await NotificationService.show(
      id: 101,
      title: 'Axon Focus Alert',
      body: message,
    );
    await prefs.setString('intensity_alert_date', todayKey);
  }

  String _buildMotivationMessage(MotivationStyle style, double drop) {
    final pct = (drop * 100).round();
    switch (style) {
      case MotivationStyle.toughLove:
        return 'Your efficiency is down $pct%. At this rate, you miss your target. Reset and push now.';
      case MotivationStyle.logicBased:
        return 'Your focus trend dipped $pct% over 3 days. A 15-minute reset will lift your score.';
      case MotivationStyle.positiveReinforcement:
        return 'You are a bit tired. A short break can lift your focus by about $pct%.';
    }
  }

  Future<void> _clearSavedState() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_prefsKey);
  }

  @override
  void dispose() {
    _ticker?.cancel();
    _cloudSyncTimer?.cancel();
    super.dispose();
  }
}

// Helper to get Firebase auth (copied to avoid circular imports)
bool get _hasFirebaseApp {
  try {
    return Firebase.apps.isNotEmpty;
  } catch (_) {
    return false;
  }
}

FirebaseAuth? get _firebaseAuthOrNull {
  try {
    if (!_hasFirebaseApp) return null;
    return FirebaseAuth.instance;
  } catch (_) {
    return null;
  }
}
