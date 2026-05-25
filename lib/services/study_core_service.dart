import 'dart:convert';
import 'dart:math' as math;

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/models.dart';
import '../models/academic_engine_models.dart';
import 'app_state.dart';
import 'backend_health_service.dart';
import 'board_exam_service.dart';
import 'firestore_service.dart';
import 'spaced_repetition_service.dart';
import 'study_catalog.dart';
import 'syllabus_map_service.dart';

final studyCoreNotifierProvider =
    AsyncNotifierProvider<StudyCoreNotifier, StudyCoreSnapshot>(
  StudyCoreNotifier.new,
);

class StudyCoreNotifier extends AsyncNotifier<StudyCoreSnapshot> {
  @override
  Future<StudyCoreSnapshot> build() async {
    return _buildSnapshot();
  }

  Future<StudyCoreSnapshot> _buildSnapshot() async {
    final auth = ref.read(authStateProvider);
    final metrics = ref.read(metricsProvider);
    return StudyCoreService.instance.buildSnapshot(
      user: auth.user,
      metrics: metrics,
    );
  }

  Future<void> refresh() async {
    state = const AsyncValue.loading();
    state = await AsyncValue.guard(() => _buildSnapshot());
  }

  Future<void> refreshWithDelay() async {
    await Future.delayed(const Duration(milliseconds: 800));
    await refresh();
  }
}

class StudySessionTemplate {
  final String id;
  final String name;
  final String description;
  final int focusMinutes;
  final int breakMinutes;
  final int cycles;

  const StudySessionTemplate({
    required this.id,
    required this.name,
    required this.description,
    required this.focusMinutes,
    required this.breakMinutes,
    this.cycles = 1,
  });

  int get totalMinutes {
    final focus = focusMinutes * cycles;
    final breaks = breakMinutes * math.max(0, cycles - 1);
    return (focus + breaks).toInt();
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'description': description,
        'focusMinutes': focusMinutes,
        'breakMinutes': breakMinutes,
        'cycles': cycles,
      };

  factory StudySessionTemplate.fromJson(Map<String, dynamic> json) =>
      StudySessionTemplate(
        id: (json['id'] ?? '').toString(),
        name: (json['name'] ?? '').toString(),
        description: (json['description'] ?? '').toString(),
        focusMinutes: (json['focusMinutes'] as num?)?.toInt() ?? 25,
        breakMinutes: (json['breakMinutes'] as num?)?.toInt() ?? 5,
        cycles: (json['cycles'] as num?)?.toInt() ?? 1,
      );
}

class ChapterMastery {
  final String subject;
  final String chapter;
  final double score;
  final int minutesSpent;
  final int dueReviewCount;
  final DateTime? lastStudiedAt;
  final double lastMockScore;
  final double studyRatePerHour;

  const ChapterMastery({
    required this.subject,
    required this.chapter,
    required this.score,
    required this.minutesSpent,
    required this.dueReviewCount,
    required this.lastStudiedAt,
    this.lastMockScore = 0,
    this.studyRatePerHour = 0.05,
  });

  int? get daysSinceLastStudied {
    if (lastStudiedAt == null) return null;
    return DateTime.now().difference(lastStudiedAt!).inDays;
  }

  SyllabusStatus get status {
    final now = DateTime.now();
    final sevenDaysAgo = now.subtract(const Duration(days: 7));
    final fourteenDaysAgo = now.subtract(const Duration(days: 14));

    if (lastMockScore > 0 && lastMockScore < 0.4) {
      return SyllabusStatus.red;
    }

    final hasRecentHighScore = lastMockScore > 0.85 &&
        lastStudiedAt != null &&
        lastStudiedAt!.isAfter(sevenDaysAgo);

    if (hasRecentHighScore) {
      return SyllabusStatus.green;
    }

    final hasNotBeenReviewedRecently =
        lastStudiedAt == null || lastStudiedAt!.isBefore(fourteenDaysAgo);

    if (hasNotBeenReviewedRecently || score < 0.6) {
      return SyllabusStatus.yellow;
    }

    return SyllabusStatus.yellow;
  }

  double get hoursToGreen {
    if (score >= 0.85) return 0;
    final remaining = 0.85 - score;
    if (studyRatePerHour <= 0) return remaining / 0.05;
    return remaining / studyRatePerHour;
  }
}

class RevisionQueueItem {
  final String subject;
  final String chapter;
  final double priority;
  final int recommendedMinutes;
  final String reason;
  final ChapterMastery mastery;
  final String sourceType;
  final String syllabusPath;
  final List<String> trustReasons;
  final List<String> syllabusObjectiveIds;
  final String paper;

  const RevisionQueueItem({
    required this.subject,
    required this.chapter,
    required this.priority,
    required this.recommendedMinutes,
    required this.reason,
    required this.mastery,
    required this.sourceType,
    required this.syllabusPath,
    required this.trustReasons,
    required this.syllabusObjectiveIds,
    required this.paper,
  });
}

class AdaptiveStudyPlan {
  final String subject;
  final String chapter;
  final int recommendedMinutes;
  final String rationale;
  final List<RevisionQueueItem> tasks;
  final StudySessionMode mode;
  final String? templateId;
  final String sourceType;
  final List<String> trustReasons;
  final List<String> syllabusObjectiveIds;
  final String syllabusPath;

  const AdaptiveStudyPlan({
    required this.subject,
    required this.chapter,
    required this.recommendedMinutes,
    required this.rationale,
    required this.tasks,
    this.mode = StudySessionMode.pomodoro,
    this.templateId,
    required this.sourceType,
    required this.trustReasons,
    required this.syllabusObjectiveIds,
    required this.syllabusPath,
  });
}

class ResumeSessionState {
  final String subject;
  final String chapter;
  final Duration elapsed;
  final bool isRunning;
  final String? templateName;
  final Duration? targetDuration;

  const ResumeSessionState({
    required this.subject,
    required this.chapter,
    required this.elapsed,
    required this.isRunning,
    this.templateName,
    this.targetDuration,
  });
}

class StudyCoreSnapshot {
  final List<StudySessionTemplate> templates;
  final List<ChapterMastery> mastery;
  final List<RevisionQueueItem> revisionQueue;
  final AdaptiveStudyPlan? plan;
  final ResumeSessionState? resume;

  const StudyCoreSnapshot({
    required this.templates,
    required this.mastery,
    required this.revisionQueue,
    required this.plan,
    this.resume,
  });
}

class StudyCoreService {
  StudyCoreService._();

  static final StudyCoreService instance = StudyCoreService._();

  static const String _templatesKey = 'study_session_templates_v1';

  static const List<StudySessionTemplate> _defaultTemplates = [
    StudySessionTemplate(
      id: 'deep_work',
      name: 'Deep Work',
      description: 'Long focus block for one hard chapter.',
      focusMinutes: 50,
      breakMinutes: 10,
    ),
    StudySessionTemplate(
      id: 'recall_sprint',
      name: 'Recall Sprint',
      description: 'Fast recall and retrieval practice.',
      focusMinutes: 20,
      breakMinutes: 5,
      cycles: 2,
    ),
    StudySessionTemplate(
      id: 'past_paper',
      name: 'Past Paper',
      description: 'Single uninterrupted timed paper block.',
      focusMinutes: 60,
      breakMinutes: 0,
    ),
    StudySessionTemplate(
      id: 'exam_sim',
      name: 'Exam Sim',
      description: 'Longer mock conditions with minimal interruption.',
      focusMinutes: 90,
      breakMinutes: 0,
    ),
  ];

  Future<List<StudySessionTemplate>> loadTemplates() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_templatesKey);
    if (raw == null || raw.isEmpty) {
      return _defaultTemplates;
    }
    try {
      final decoded = jsonDecode(raw) as List<dynamic>;
      final templates = decoded
          .whereType<Map>()
          .map((item) =>
              StudySessionTemplate.fromJson(Map<String, dynamic>.from(item)))
          .where((template) => template.id.isNotEmpty)
          .toList();
      return templates.isEmpty ? _defaultTemplates : templates;
    } catch (_) {
      return _defaultTemplates;
    }
  }

  Future<List<ChapterMastery>> getChapterMastery({
    required Map<String, List<String>> catalog,
    required String board,
  }) async {
    final history = await _loadTimerHistory();
    final reviewCards = await SpacedRepetitionService.instance.getAllCards();
    final examDates = await _loadExamDates(board);
    final now = DateTime.now();
    final mastery = <ChapterMastery>[];

    for (final entry in catalog.entries) {
      final subject = entry.key;
      for (final chapter in entry.value) {
        final sessions = history.where((item) {
          return _norm(item['subject']) == _norm(subject) &&
              _norm(item['chapter']) == _norm(chapter);
        }).toList();
        sessions.sort(
          (a, b) => _parseDate(b['date']).compareTo(_parseDate(a['date'])),
        );
        final lastStudiedAt =
            sessions.isEmpty ? null : _parseDate(sessions.first['date']);
        final minutesSpent = sessions.fold<int>(0, (sum, item) {
          final duration = item['duration'];
          if (duration is num) return sum + (duration / 60).round();
          return sum;
        });
        final dueReviewCount = reviewCards.where((card) {
          return _norm(card.subject) == _norm(subject) &&
              _topicMatchesChapter(card.topic, chapter) &&
              !card.nextReview.isAfter(now);
        }).length;
        final score = _computeMasteryScore(
          minutesSpent: minutesSpent,
          lastStudiedAt: lastStudiedAt,
          dueReviewCount: dueReviewCount,
          nextExamAt: examDates[_norm(subject)],
        );
        mastery.add(
          ChapterMastery(
            subject: subject,
            chapter: chapter,
            score: score,
            minutesSpent: minutesSpent,
            dueReviewCount: dueReviewCount,
            lastStudiedAt: lastStudiedAt,
          ),
        );
      }
    }

    mastery.sort((a, b) {
      final gap = a.score.compareTo(b.score);
      if (gap != 0) return gap;
      return (b.daysSinceLastStudied ?? 999)
          .compareTo(a.daysSinceLastStudied ?? 999);
    });
    return mastery;
  }

  Future<StudyCoreSnapshot> buildSnapshot({
    required UserProfile? user,
    required MetricsState metrics,
  }) async {
    final catalog = user?.uid != null
        ? await StudyCatalog().syncWithFirestore(user!.uid)
        : await StudyCatalog().load();
    final templates = await loadTemplates();
    final mastery = await getChapterMastery(
      catalog: catalog,
      board: user?.board ?? '',
    );
    final revisionQueue = _buildRevisionQueue(
      mastery: mastery,
      primarySubject: metrics.primarySubject,
      activeStudyHours: metrics.activeStudyHours,
      targetStudyHours: metrics.targetStudyHours,
      board: user?.board ?? '',
    );
    final plan = revisionQueue.isEmpty
        ? null
        : AdaptiveStudyPlan(
            subject: revisionQueue.first.subject,
            chapter: revisionQueue.first.chapter,
            recommendedMinutes: math.max(
              revisionQueue.first.recommendedMinutes,
              ((metrics.targetStudyHours - metrics.activeStudyHours).clamp(
                          0.5,
                          metrics.targetStudyHours <= 0
                              ? 4.0
                              : metrics.targetStudyHours) *
                      60)
                  .round(),
            ),
            rationale: revisionQueue.first.reason,
            tasks: revisionQueue.take(3).toList(),
            mode: StudySessionMode.pomodoro,
            templateId: templates.isNotEmpty ? templates.first.id : null,
            sourceType: revisionQueue.first.sourceType,
            trustReasons: revisionQueue.first.trustReasons,
            syllabusObjectiveIds: revisionQueue.first.syllabusObjectiveIds,
            syllabusPath: revisionQueue.first.syllabusPath,
          );

    return StudyCoreSnapshot(
      templates: templates,
      mastery: mastery,
      revisionQueue: revisionQueue,
      plan: plan,
    );
  }

  List<RevisionQueueItem> _buildRevisionQueue({
    required List<ChapterMastery> mastery,
    required String primarySubject,
    required double activeStudyHours,
    required double targetStudyHours,
    required String board,
  }) {
    final remainingMinutes =
        (((targetStudyHours <= 0 ? 4.0 : targetStudyHours) - activeStudyHours)
                    .clamp(0.0, 8.0) *
                60)
            .round();

    final queue = mastery.map((item) {
      final subjectBoost =
          _norm(item.subject) == _norm(primarySubject) ? 10.0 : 0.0;
      final daysSince = item.daysSinceLastStudied ?? 14;
      final reviewBoost = item.dueReviewCount * 7.0;
      final masteryPenalty = (1 - item.score) * 75.0;
      final recencyBoost = math.min(daysSince * 2.4, 28).toDouble();
      final priority =
          masteryPenalty + recencyBoost + reviewBoost + subjectBoost;
      final evidence = SyllabusMapService.instance.buildEvidence(
        board: board,
        subject: item.subject,
        chapter: item.chapter,
        dueReviewCount: item.dueReviewCount,
        daysSinceLastStudied: daysSince,
        daysToExam: 999,
        isPrimarySubject: _norm(item.subject) == _norm(primarySubject),
      );
      final reason = item.dueReviewCount > 0
          ? '${item.dueReviewCount} review cards are due here for ${evidence.objectiveIds.first}.'
          : daysSince >= 5
              ? 'You have not touched ${evidence.objectiveIds.first} in $daysSince days.'
              : item.score < 0.45
                  ? 'Coverage is weak against ${evidence.paper} requirements.'
                  : 'This is the cleanest next chapter to keep momentum before ${evidence.paper}.';
      final minutes = remainingMinutes > 0
          ? math.min(
              remainingMinutes,
              item.score < 0.35
                  ? 45
                  : item.score < 0.6
                      ? 35
                      : 20,
            )
          : 20;
      return RevisionQueueItem(
        subject: item.subject,
        chapter: item.chapter,
        priority: priority,
        recommendedMinutes: minutes,
        reason: reason,
        mastery: item,
        sourceType: evidence.sourceType,
        syllabusPath: evidence.syllabusPath,
        trustReasons: evidence.trustReasons,
        syllabusObjectiveIds: evidence.objectiveIds,
        paper: evidence.paper,
      );
    }).toList()
      ..sort((a, b) => b.priority.compareTo(a.priority));

    return queue.take(6).toList();
  }

  double _computeMasteryScore({
    required int minutesSpent,
    required DateTime? lastStudiedAt,
    required int dueReviewCount,
    required DateTime? nextExamAt,
  }) {
    final coverage = (minutesSpent / 120).clamp(0.0, 1.0);
    final recency = () {
      if (lastStudiedAt == null) return 0.08;
      final days = DateTime.now().difference(lastStudiedAt).inDays;
      if (days <= 1) return 1.0;
      if (days <= 3) return 0.82;
      if (days <= 7) return 0.58;
      if (days <= 14) return 0.34;
      return 0.14;
    }();
    final examPressure = () {
      if (nextExamAt == null) return 0.0;
      final days = nextExamAt.difference(DateTime.now()).inDays;
      if (days <= 0) return 0.18;
      if (days <= 14) return 0.12;
      if (days <= 30) return 0.08;
      return 0.0;
    }();
    final reviewPenalty = math.min(dueReviewCount * 0.05, 0.22);
    return ((coverage * 0.56) + (recency * 0.44) - reviewPenalty - examPressure)
        .clamp(0.0, 1.0);
  }

  Future<List<Map<String, dynamic>>> _loadTimerHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('timer_history');
    final history = <Map<String, dynamic>>[];
    if (raw != null && raw.isNotEmpty) {
      try {
        final decoded = jsonDecode(raw) as List<dynamic>;
        history.addAll(decoded
            .whereType<Map>()
            .map((item) => Map<String, dynamic>.from(item)));
      } catch (_) {}
    }

    final user = FirebaseAuth.instance.currentUser;
    final firestoreAvailable =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (user != null && firestoreAvailable) {
      try {
        final snapshot =
            await AxonPaths.privateUserCollection(user.uid, 'sessions')
                .orderBy('date', descending: true)
                .limit(200)
                .get();
        history.addAll(snapshot.docs.map((doc) => doc.data()));
      } catch (_) {}
    }

    if (history.isEmpty) return const [];
    try {
      return history;
    } catch (_) {
      return const [];
    }
  }

  Future<Map<String, DateTime>> _loadExamDates(String board) async {
    if (board.trim().isEmpty) return const {};
    try {
      final result = await BoardExamService().fetchBoardDates(board);
      final next = <String, DateTime>{};
      for (final event in result.events) {
        final key = _norm(event.subject);
        final current = next[key];
        if (event.startDate.isBefore(DateTime.now())) continue;
        if (current == null || event.startDate.isBefore(current)) {
          next[key] = event.startDate;
        }
      }
      return next;
    } catch (_) {
      return const {};
    }
  }

  bool _topicMatchesChapter(String topic, String chapter) {
    final cleanTopic = _norm(topic);
    final cleanChapter = _norm(chapter);
    return cleanTopic.contains(cleanChapter) ||
        cleanChapter.contains(cleanTopic);
  }

  String _norm(Object? value) => value?.toString().trim().toLowerCase() ?? '';

  DateTime _parseDate(Object? value) =>
      DateTime.tryParse(value?.toString() ?? '') ??
      DateTime.fromMillisecondsSinceEpoch(0);
}

final studyCoreProvider = studyCoreNotifierProvider;
