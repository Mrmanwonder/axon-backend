import 'dart:convert';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class StudyTechnique {
  final String id;
  final String name;
  final String description;
  final String icon;
  final String color;
  final bool isActive;
  final Map<String, dynamic> settings;

  StudyTechnique({
    required this.id,
    required this.name,
    required this.description,
    required this.icon,
    required this.color,
    this.isActive = false,
    this.settings = const {},
  });

  StudyTechnique copyWith({
    String? id,
    String? name,
    String? description,
    String? icon,
    String? color,
    bool? isActive,
    Map<String, dynamic>? settings,
  }) {
    return StudyTechnique(
      id: id ?? this.id,
      name: name ?? this.name,
      description: description ?? this.description,
      icon: icon ?? this.icon,
      color: color ?? this.color,
      isActive: isActive ?? this.isActive,
      settings: settings ?? this.settings,
    );
  }
}

class TechniqueProgress {
  final String techniqueId;
  final int sessionsCompleted;
  final Duration totalTime;
  final double productivityScore;
  final DateTime lastSession;
  final Map<String, dynamic> metrics;

  TechniqueProgress({
    required this.techniqueId,
    this.sessionsCompleted = 0,
    this.totalTime = Duration.zero,
    this.productivityScore = 0.0,
    DateTime? lastSession,
    this.metrics = const {},
  }) : lastSession = lastSession ?? DateTime.now();

  TechniqueProgress copyWith({
    String? techniqueId,
    int? sessionsCompleted,
    Duration? totalTime,
    double? productivityScore,
    DateTime? lastSession,
    Map<String, dynamic>? metrics,
  }) {
    return TechniqueProgress(
      techniqueId: techniqueId ?? this.techniqueId,
      sessionsCompleted: sessionsCompleted ?? this.sessionsCompleted,
      totalTime: totalTime ?? this.totalTime,
      productivityScore: productivityScore ?? this.productivityScore,
      lastSession: lastSession ?? this.lastSession,
      metrics: metrics ?? this.metrics,
    );
  }
}

class FeynmanNote {
  final String id;
  final String topic;
  final String concept;
  final String explanation;
  final String simpleExplanation;
  final List<String> analogies;
  final int complexityLevel;
  final DateTime createdAt;
  final DateTime lastReviewed;

  FeynmanNote({
    required this.id,
    required this.topic,
    required this.concept,
    required this.explanation,
    required this.simpleExplanation,
    this.analogies = const [],
    this.complexityLevel = 1,
    DateTime? createdAt,
    DateTime? lastReviewed,
  })  : createdAt = createdAt ?? DateTime.now(),
        lastReviewed = lastReviewed ?? DateTime.now();

  FeynmanNote copyWith({
    String? id,
    String? topic,
    String? concept,
    String? explanation,
    String? simpleExplanation,
    List<String>? analogies,
    int? complexityLevel,
    DateTime? createdAt,
    DateTime? lastReviewed,
  }) {
    return FeynmanNote(
      id: id ?? this.id,
      topic: topic ?? this.topic,
      concept: concept ?? this.concept,
      explanation: explanation ?? this.explanation,
      simpleExplanation: simpleExplanation ?? this.simpleExplanation,
      analogies: analogies ?? this.analogies,
      complexityLevel: complexityLevel ?? this.complexityLevel,
      createdAt: createdAt ?? this.createdAt,
      lastReviewed: lastReviewed ?? this.lastReviewed,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'topic': topic,
        'concept': concept,
        'explanation': explanation,
        'simpleExplanation': simpleExplanation,
        'analogies': analogies,
        'complexityLevel': complexityLevel,
        'createdAt': createdAt.toIso8601String(),
        'lastReviewed': lastReviewed.toIso8601String(),
      };

  factory FeynmanNote.fromJson(Map<String, dynamic> json) {
    return FeynmanNote(
      id: json['id'] ?? '',
      topic: json['topic'] ?? '',
      concept: json['concept'] ?? '',
      explanation: json['explanation'] ?? '',
      simpleExplanation: json['simpleExplanation'] ?? '',
      analogies: List<String>.from(json['analogies'] ?? []),
      complexityLevel: json['complexityLevel'] ?? 1,
      createdAt: DateTime.tryParse(json['createdAt'] ?? '') ?? DateTime.now(),
      lastReviewed:
          DateTime.tryParse(json['lastReviewed'] ?? '') ?? DateTime.now(),
    );
  }
}

class LeitnerBox {
  final int boxNumber;
  final List<LeitnerCard> cards;
  final int reviewIntervalDays;

  LeitnerBox({
    required this.boxNumber,
    this.cards = const [],
  }) : reviewIntervalDays = pow(2, boxNumber - 1).toInt();
}

class LeitnerCard {
  final String id;
  final String front;
  final String back;
  final String subject;
  final String topic;
  final int boxNumber;
  final int correctCount;
  final int incorrectCount;
  final DateTime lastReviewed;
  final DateTime nextReview;
  final int totalReviews;

  LeitnerCard({
    required this.id,
    required this.front,
    required this.back,
    required this.subject,
    required this.topic,
    this.boxNumber = 1,
    this.correctCount = 0,
    this.incorrectCount = 0,
    DateTime? lastReviewed,
    DateTime? nextReview,
    this.totalReviews = 0,
  })  : lastReviewed = lastReviewed ?? DateTime.now(),
        nextReview = nextReview ?? DateTime.now();

  LeitnerCard copyWith({
    String? id,
    String? front,
    String? back,
    String? subject,
    String? topic,
    int? boxNumber,
    int? correctCount,
    int? incorrectCount,
    DateTime? lastReviewed,
    DateTime? nextReview,
    int? totalReviews,
  }) {
    return LeitnerCard(
      id: id ?? this.id,
      front: front ?? this.front,
      back: back ?? this.back,
      subject: subject ?? this.subject,
      topic: topic ?? this.topic,
      boxNumber: boxNumber ?? this.boxNumber,
      correctCount: correctCount ?? this.correctCount,
      incorrectCount: incorrectCount ?? this.incorrectCount,
      lastReviewed: lastReviewed ?? this.lastReviewed,
      nextReview: nextReview ?? this.nextReview,
      totalReviews: totalReviews ?? this.totalReviews,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'front': front,
        'back': back,
        'subject': subject,
        'topic': topic,
        'boxNumber': boxNumber,
        'correctCount': correctCount,
        'incorrectCount': incorrectCount,
        'lastReviewed': lastReviewed.toIso8601String(),
        'nextReview': nextReview.toIso8601String(),
        'totalReviews': totalReviews,
      };

  factory LeitnerCard.fromJson(Map<String, dynamic> json) {
    return LeitnerCard(
      id: json['id'] ?? '',
      front: json['front'] ?? '',
      back: json['back'] ?? '',
      subject: json['subject'] ?? '',
      topic: json['topic'] ?? '',
      boxNumber: json['boxNumber'] ?? 1,
      correctCount: json['correctCount'] ?? 0,
      incorrectCount: json['incorrectCount'] ?? 0,
      lastReviewed:
          DateTime.tryParse(json['lastReviewed'] ?? '') ?? DateTime.now(),
      nextReview: DateTime.tryParse(json['nextReview'] ?? '') ?? DateTime.now(),
      totalReviews: json['totalReviews'] ?? 0,
    );
  }
}

class InterleavedSession {
  final String id;
  final List<String> subjects;
  final List<InterleavedProblem> problems;
  final DateTime startedAt;
  final DateTime? completedAt;
  final int correctCount;
  final int totalProblems;

  InterleavedSession({
    required this.id,
    required this.subjects,
    required this.problems,
    required this.startedAt,
    this.completedAt,
    this.correctCount = 0,
    this.totalProblems = 0,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'subjects': subjects,
        'problems': problems.map((p) => p.toJson()).toList(),
        'startedAt': startedAt.toIso8601String(),
        'completedAt': completedAt?.toIso8601String(),
        'correctCount': correctCount,
        'totalProblems': totalProblems,
      };

  factory InterleavedSession.fromJson(Map<String, dynamic> json) {
    return InterleavedSession(
      id: json['id'] ?? '',
      subjects: List<String>.from(json['subjects'] ?? []),
      problems: (json['problems'] as List?)
              ?.map((p) =>
                  InterleavedProblem.fromJson(Map<String, dynamic>.from(p)))
              .toList() ??
          [],
      startedAt: DateTime.tryParse(json['startedAt'] ?? '') ?? DateTime.now(),
      completedAt: json['completedAt'] != null
          ? DateTime.tryParse(json['completedAt']!)
          : null,
      correctCount: json['correctCount'] ?? 0,
      totalProblems: json['totalProblems'] ?? 0,
    );
  }
}

class InterleavedProblem {
  final String id;
  final String question;
  final String answer;
  final String subject;
  final String topic;
  final String? hint;
  final String? explanation;
  final int difficulty;

  InterleavedProblem({
    required this.id,
    required this.question,
    required this.answer,
    required this.subject,
    required this.topic,
    this.hint,
    this.explanation,
    this.difficulty = 1,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'question': question,
        'answer': answer,
        'subject': subject,
        'topic': topic,
        'hint': hint,
        'explanation': explanation,
        'difficulty': difficulty,
      };

  factory InterleavedProblem.fromJson(Map<String, dynamic> json) {
    return InterleavedProblem(
      id: json['id'] ?? '',
      question: json['question'] ?? '',
      answer: json['answer'] ?? '',
      subject: json['subject'] ?? '',
      topic: json['topic'] ?? '',
      hint: json['hint'],
      explanation: json['explanation'],
      difficulty: json['difficulty'] ?? 1,
    );
  }
}

class StudyTechniquesService {
  static final StudyTechniquesService instance = StudyTechniquesService._();
  StudyTechniquesService._();

  static const String _progressKey = 'technique_progress';
  static const String _feynmanKey = 'feynman_notes';
  static const String _leitnerKey = 'leitner_cards';
  static const String _interleavedKey = 'interleaved_sessions';
  static const String _activeRecallKey = 'active_recall_items';

  List<StudyTechnique> getAllTechniques() {
    return [
      StudyTechnique(
        id: 'pomodoro',
        name: 'Pomodoro',
        description:
            '25 min work, 5 min break. Great for focused study sessions.',
        icon: 'timer',
        color: '#FF6B6B',
        settings: {
          'workDuration': 25,
          'shortBreak': 5,
          'longBreak': 15,
          'sessionsBeforeLongBreak': 4,
        },
      ),
      StudyTechnique(
        id: 'spaced_repetition',
        name: 'Spaced Repetition',
        description:
            'Review cards at increasing intervals. Optimal for long-term retention.',
        icon: 'repeat',
        color: '#4ECDC4',
        settings: {
          'minInterval': 1,
          'maxInterval': 365,
          'easeFactor': 2.5,
        },
      ),
      StudyTechnique(
        id: 'feynman',
        name: 'Feynman Technique',
        description: 'Explain concepts simply. Identify knowledge gaps.',
        icon: 'lightbulb',
        color: '#FFE66D',
        settings: {
          'maxComplexity': 5,
          'requireAnalogies': true,
        },
      ),
      StudyTechnique(
        id: 'leitner',
        name: 'Leitner System',
        description:
            'Box-based flashcards. Move cards up on success, down on failure.',
        icon: 'inbox',
        color: '#95E1D3',
        settings: {
          'boxCount': 5,
          'startingBox': 1,
        },
      ),
      StudyTechnique(
        id: 'interleaving',
        name: 'Interleaving',
        description:
            'Mix topics during practice. Improves discrimination skills.',
        icon: 'shuffle',
        color: '#A8E6CF',
        settings: {
          'minSubjects': 2,
          'maxProblems': 20,
          'shuffleMode': true,
        },
      ),
      StudyTechnique(
        id: 'active_recall',
        name: 'Active Recall',
        description:
            'Test yourself before looking at answers. Builds strong memory.',
        icon: 'brain',
        color: '#DDA0DD',
        settings: {
          'showHintAfter': 30,
          'requireExplanation': false,
        },
      ),
      StudyTechnique(
        id: 'blocking',
        name: 'Blocking',
        description: 'Focus on one topic at a time. Good for deep mastery.',
        icon: 'layers',
        color: '#98D8C8',
        settings: {
          'blockDuration': 45,
          'topicSwitchDelay': 300,
        },
      ),
      StudyTechnique(
        id: 'time_blocking',
        name: 'Time Blocking',
        description: 'Schedule specific time slots for different subjects.',
        icon: 'calendar_today',
        color: '#F7DC6F',
        settings: {
          'defaultBlockDuration': 60,
          'bufferTime': 10,
        },
      ),
    ];
  }

  Future<void> _saveProgress(
      String techniqueId, TechniqueProgress progress) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_progressKey);
    Map<String, dynamic> allProgress = {};

    if (data != null) {
      try {
        allProgress = Map<String, dynamic>.from(jsonDecode(data));
      } catch (e) {
        debugPrint(
            '[StudyTechniquesService] Failed to decode progress data: $e');
      }
    }

    allProgress[techniqueId] = {
      'sessionsCompleted': progress.sessionsCompleted,
      'totalTime': progress.totalTime.inSeconds,
      'productivityScore': progress.productivityScore,
      'lastSession': progress.lastSession.toIso8601String(),
      'metrics': progress.metrics,
    };

    await prefs.setString(_progressKey, jsonEncode(allProgress));
  }

  Future<TechniqueProgress> getProgress(String techniqueId) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_progressKey);

    if (data == null) {
      return TechniqueProgress(techniqueId: techniqueId);
    }

    try {
      final allProgress = Map<String, dynamic>.from(jsonDecode(data));
      final progress = allProgress[techniqueId];
      if (progress == null) {
        return TechniqueProgress(techniqueId: techniqueId);
      }

      return TechniqueProgress(
        techniqueId: techniqueId,
        sessionsCompleted: progress['sessionsCompleted'] ?? 0,
        totalTime: Duration(seconds: progress['totalTime'] ?? 0),
        productivityScore: (progress['productivityScore'] ?? 0.0).toDouble(),
        lastSession:
            DateTime.tryParse(progress['lastSession'] ?? '') ?? DateTime.now(),
        metrics: Map<String, dynamic>.from(progress['metrics'] ?? {}),
      );
    } catch (e) {
      debugPrint(
          '[StudyTechniquesService] Failed to get progress for technique: $e');
      return TechniqueProgress(techniqueId: techniqueId);
    }
  }

  Future<Map<String, TechniqueProgress>> getAllProgress() async {
    final techniques = getAllTechniques();
    final progressMap = <String, TechniqueProgress>{};

    for (final tech in techniques) {
      progressMap[tech.id] = await getProgress(tech.id);
    }

    return progressMap;
  }

  Future<void> recordSession({
    required String techniqueId,
    required Duration duration,
    required int problemsCompleted,
    required int correctAnswers,
  }) async {
    final progress = await getProgress(techniqueId);
    final productivity = problemsCompleted > 0
        ? (correctAnswers / problemsCompleted) * 100
        : 0.0;

    final newSessions = progress.sessionsCompleted + 1;
    final newTotalTime = progress.totalTime + duration;
    final newProductivity =
        ((progress.productivityScore * progress.sessionsCompleted) +
                productivity) /
            newSessions;

    await _saveProgress(
        techniqueId,
        progress.copyWith(
          sessionsCompleted: newSessions,
          totalTime: newTotalTime,
          productivityScore: newProductivity,
          lastSession: DateTime.now(),
        ));
  }

  Future<List<FeynmanNote>> getFeynmanNotes() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_feynmanKey);
    if (data == null) return [];

    try {
      final decoded = jsonDecode(data) as List;
      return decoded
          .map((n) => FeynmanNote.fromJson(Map<String, dynamic>.from(n)))
          .toList();
    } catch (e) {
      debugPrint('[StudyTechniquesService] Failed to decode Feynman notes: $e');
      return [];
    }
  }

  Future<void> addFeynmanNote(FeynmanNote note) async {
    final notes = await getFeynmanNotes();
    notes.add(note);
    await _saveFeynmanNotes(notes);
  }

  Future<void> updateFeynmanNote(FeynmanNote note) async {
    final notes = await getFeynmanNotes();
    final index = notes.indexWhere((n) => n.id == note.id);
    if (index != -1) {
      notes[index] = note;
      await _saveFeynmanNotes(notes);
    }
  }

  Future<void> deleteFeynmanNote(String noteId) async {
    final notes = await getFeynmanNotes();
    notes.removeWhere((n) => n.id == noteId);
    await _saveFeynmanNotes(notes);
  }

  Future<void> _saveFeynmanNotes(List<FeynmanNote> notes) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _feynmanKey,
      jsonEncode(notes.map((n) => n.toJson()).toList()),
    );
  }

  Future<List<LeitnerCard>> getLeitnerCards() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_leitnerKey);
    if (data == null) return [];

    try {
      final decoded = jsonDecode(data) as List;
      return decoded
          .map((c) => LeitnerCard.fromJson(Map<String, dynamic>.from(c)))
          .toList();
    } catch (e) {
      debugPrint('[StudyTechniquesService] Failed to decode Leitner cards: $e');
      return [];
    }
  }

  List<LeitnerBox> organizeCardsIntoBoxes(List<LeitnerCard> cards) {
    final boxes = <LeitnerBox>[];
    for (int i = 1; i <= 5; i++) {
      final boxCards = cards.where((c) => c.boxNumber == i).toList();
      boxes.add(LeitnerBox(boxNumber: i, cards: boxCards));
    }
    return boxes;
  }

  Future<List<LeitnerCard>> getDueCards() async {
    final cards = await getLeitnerCards();
    final now = DateTime.now();
    return cards
        .where((c) =>
            c.nextReview.isBefore(now) || c.nextReview.isAtSameMomentAs(now))
        .toList();
  }

  Future<void> addLeitnerCard(LeitnerCard card) async {
    final cards = await getLeitnerCards();
    cards.add(card);
    await _saveLeitnerCards(cards);
  }

  Future<void> processLeitnerReview(String cardId, bool wasCorrect) async {
    final cards = await getLeitnerCards();
    final index = cards.indexWhere((c) => c.id == cardId);
    if (index == -1) return;

    final card = cards[index];
    int newBox = card.boxNumber;
    int newCorrect = card.correctCount;
    int newIncorrect = card.incorrectCount;

    if (wasCorrect) {
      newCorrect++;
      if (newBox < 5) {
        newBox++;
      }
    } else {
      newIncorrect++;
      if (newBox > 1) {
        newBox = 1;
      }
    }

    final interval = pow(2, newBox - 1).toInt();
    final nextReview = DateTime.now().add(Duration(days: interval));

    cards[index] = card.copyWith(
      boxNumber: newBox,
      correctCount: newCorrect,
      incorrectCount: newIncorrect,
      lastReviewed: DateTime.now(),
      nextReview: nextReview,
      totalReviews: card.totalReviews + 1,
    );

    await _saveLeitnerCards(cards);
  }

  Future<void> _saveLeitnerCards(List<LeitnerCard> cards) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _leitnerKey,
      jsonEncode(cards.map((c) => c.toJson()).toList()),
    );
  }

  Future<List<InterleavedSession>> getInterleavedSessions(
      {int limit = 10}) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_interleavedKey);
    if (data == null) return [];

    try {
      final decoded = jsonDecode(data) as List;
      final sessions = decoded
          .map((s) => InterleavedSession.fromJson(Map<String, dynamic>.from(s)))
          .toList();
      sessions.sort((a, b) => b.startedAt.compareTo(a.startedAt));
      return sessions.take(limit).toList();
    } catch (e) {
      debugPrint(
          '[StudyTechniquesService] Failed to get interleaved sessions: $e');
      return [];
    }
  }

  List<InterleavedProblem> createInterleavedProblems({
    required List<String> subjects,
    required List<Map<String, dynamic>> questionBank,
    required int count,
  }) {
    final problems = <InterleavedProblem>[];
    final random = Random();

    final filtered =
        questionBank.where((q) => subjects.contains(q['subject'])).toList();

    if (filtered.isEmpty) {
      return problems;
    }

    final shuffled = List<Map<String, dynamic>>.from(filtered)..shuffle(random);

    for (int i = 0; i < count && i < shuffled.length; i++) {
      final q = shuffled[i];
      problems.add(InterleavedProblem(
        id: 'inter_${DateTime.now().millisecondsSinceEpoch}_$i',
        question: q['question'] ?? '',
        answer: q['answer'] ?? '',
        subject: q['subject'] ?? '',
        topic: q['topic'] ?? '',
        hint: q['hint'],
        explanation: q['explanation'],
        difficulty: q['difficulty'] ?? 1,
      ));
    }

    return problems;
  }

  Future<void> saveActiveRecallItem(Map<String, dynamic> item) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_activeRecallKey);
    List<Map<String, dynamic>> items = [];

    if (data != null) {
      try {
        items = List<Map<String, dynamic>>.from(jsonDecode(data));
      } catch (e) {
        debugPrint(
            '[StudyTechniquesService] Failed to decode active recall items: $e');
      }
    }

    items.add(item);
    await prefs.setString(_activeRecallKey, jsonEncode(items));
  }

  Future<List<Map<String, dynamic>>> getActiveRecallItems(
      {String? subject}) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_activeRecallKey);
    if (data == null) return [];

    try {
      final items = List<Map<String, dynamic>>.from(jsonDecode(data));
      if (subject != null) {
        return items.where((i) => i['subject'] == subject).toList();
      }
      return items;
    } catch (e) {
      debugPrint(
          '[StudyTechniquesService] Failed to get active recall items: $e');
      return [];
    }
  }

  Future<void> deleteActiveRecallItem(String itemId) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_activeRecallKey);
    if (data == null || data.isEmpty) return;

    try {
      final items = List<Map<String, dynamic>>.from(jsonDecode(data))
        ..removeWhere((item) => '${item['id']}' == itemId);
      await prefs.setString(_activeRecallKey, jsonEncode(items));
    } catch (e) {
      debugPrint(
          '[StudyTechniquesService] Failed to delete active recall item: $e');
    }
  }

  Future<Map<String, dynamic>> getTechniqueStats(String techniqueId) async {
    final progress = await getProgress(techniqueId);

    switch (techniqueId) {
      case 'pomodoro':
        return {
          'sessions': progress.sessionsCompleted,
          'totalTime': progress.totalTime.inMinutes,
          'avgSessionLength': progress.sessionsCompleted > 0
              ? progress.totalTime.inMinutes ~/ progress.sessionsCompleted
              : 0,
        };
      case 'spaced_repetition':
        return progress.metrics;
      case 'leitner':
        final cards = await getLeitnerCards();
        final boxes = organizeCardsIntoBoxes(cards);
        return {
          'totalCards': cards.length,
          'boxDistribution': {
            for (var b in boxes) 'box${b.boxNumber}': b.cards.length
          },
          'masteredCards': cards.where((c) => c.boxNumber >= 4).length,
        };
      case 'feynman':
        final notes = await getFeynmanNotes();
        return {
          'totalNotes': notes.length,
          'avgComplexity': notes.isEmpty
              ? 0
              : notes.map((n) => n.complexityLevel).reduce((a, b) => a + b) ~/
                  notes.length,
        };
      case 'interleaving':
        final sessions = await getInterleavedSessions();
        return {
          'sessions': sessions.length,
          'avgProblemsPerSession': sessions.isEmpty
              ? 0
              : sessions.map((s) => s.totalProblems).reduce((a, b) => a + b) ~/
                  sessions.length,
        };
      default:
        return {};
    }
  }

  StudyTechnique? getTechniqueById(String id) {
    final techniques = getAllTechniques();
    try {
      return techniques.firstWhere((t) => t.id == id);
    } catch (e) {
      debugPrint(
          '[StudyTechniquesService] Failed to find technique by ID $id: $e');
      return null;
    }
  }
}
