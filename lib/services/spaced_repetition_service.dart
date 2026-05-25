import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class SpacedRepetitionCard {
  final String id;
  final String subject;
  final String topic;
  final String question;
  final String answer;
  final String? hint;
  final String? explanation;
  final DateTime createdAt;
  final DateTime nextReview;
  final int interval;
  final double easeFactor;
  final int repetitions;
  final int lapses;

  SpacedRepetitionCard({
    required this.id,
    required this.subject,
    required this.topic,
    required this.question,
    required this.answer,
    this.hint,
    this.explanation,
    DateTime? createdAt,
    DateTime? nextReview,
    this.interval = 1,
    this.easeFactor = 2.5,
    this.repetitions = 0,
    this.lapses = 0,
  })  : createdAt = createdAt ?? DateTime.now(),
        nextReview = nextReview ?? DateTime.now();

  SpacedRepetitionCard copyWith({
    String? id,
    String? subject,
    String? topic,
    String? question,
    String? answer,
    String? hint,
    String? explanation,
    DateTime? createdAt,
    DateTime? nextReview,
    int? interval,
    double? easeFactor,
    int? repetitions,
    int? lapses,
  }) {
    return SpacedRepetitionCard(
      id: id ?? this.id,
      subject: subject ?? this.subject,
      topic: topic ?? this.topic,
      question: question ?? this.question,
      answer: answer ?? this.answer,
      hint: hint ?? this.hint,
      explanation: explanation ?? this.explanation,
      createdAt: createdAt ?? this.createdAt,
      nextReview: nextReview ?? this.nextReview,
      interval: interval ?? this.interval,
      easeFactor: easeFactor ?? this.easeFactor,
      repetitions: repetitions ?? this.repetitions,
      lapses: lapses ?? this.lapses,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'subject': subject,
        'topic': topic,
        'question': question,
        'answer': answer,
        'hint': hint,
        'explanation': explanation,
        'createdAt': createdAt.toIso8601String(),
        'nextReview': nextReview.toIso8601String(),
        'interval': interval,
        'easeFactor': easeFactor,
        'repetitions': repetitions,
        'lapses': lapses,
      };

  factory SpacedRepetitionCard.fromJson(Map<String, dynamic> json) {
    return SpacedRepetitionCard(
      id: json['id'] ?? '',
      subject: json['subject'] ?? '',
      topic: json['topic'] ?? '',
      question: json['question'] ?? '',
      answer: json['answer'] ?? '',
      hint: json['hint'],
      explanation: json['explanation'],
      createdAt: DateTime.tryParse(json['createdAt'] ?? '') ?? DateTime.now(),
      nextReview: DateTime.tryParse(json['nextReview'] ?? '') ?? DateTime.now(),
      interval: json['interval'] ?? 1,
      easeFactor: (json['easeFactor'] ?? 2.5).toDouble(),
      repetitions: json['repetitions'] ?? 0,
      lapses: json['lapses'] ?? 0,
    );
  }
}

enum ReviewQuality {
  again,
  hard,
  good,
  easy,
}

class ReviewSession {
  final String id;
  final DateTime date;
  final int cardsReviewed;
  final int correctCount;
  final int incorrectCount;
  final Duration timeSpent;
  final List<ReviewResult> reviews;

  ReviewSession({
    required this.id,
    required this.date,
    required this.cardsReviewed,
    required this.correctCount,
    required this.incorrectCount,
    required this.timeSpent,
    this.reviews = const [],
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'date': date.toIso8601String(),
        'cardsReviewed': cardsReviewed,
        'correctCount': correctCount,
        'incorrectCount': incorrectCount,
        'timeSpent': timeSpent.inSeconds,
        'reviews': reviews.map((r) => r.toJson()).toList(),
      };

  factory ReviewSession.fromJson(Map<String, dynamic> json) {
    return ReviewSession(
      id: json['id'] ?? '',
      date: DateTime.tryParse(json['date'] ?? '') ?? DateTime.now(),
      cardsReviewed: json['cardsReviewed'] ?? 0,
      correctCount: json['correctCount'] ?? 0,
      incorrectCount: json['incorrectCount'] ?? 0,
      timeSpent: Duration(seconds: json['timeSpent'] ?? 0),
      reviews: (json['reviews'] as List?)
              ?.map((r) => ReviewResult.fromJson(Map<String, dynamic>.from(r)))
              .toList() ??
          [],
    );
  }
}

class ReviewResult {
  final String cardId;
  final ReviewQuality quality;
  final DateTime reviewedAt;

  ReviewResult({
    required this.cardId,
    required this.quality,
    required this.reviewedAt,
  });

  Map<String, dynamic> toJson() => {
        'cardId': cardId,
        'quality': quality.index,
        'reviewedAt': reviewedAt.toIso8601String(),
      };

  factory ReviewResult.fromJson(Map<String, dynamic> json) {
    return ReviewResult(
      cardId: json['cardId'] ?? '',
      quality: ReviewQuality.values[json['quality'] ?? 2],
      reviewedAt: DateTime.tryParse(json['reviewedAt'] ?? '') ?? DateTime.now(),
    );
  }
}

class SpacedRepetitionService {
  static final SpacedRepetitionService instance = SpacedRepetitionService._();

  static const String _cardsKey = 'spaced_repetition_cards';
  static const String _sessionsKey = 'spaced_repetition_sessions';

  SpacedRepetitionService._();

  Future<List<SpacedRepetitionCard>> getCardsForReview() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_cardsKey);
    if (data == null) return [];

    try {
      final decoded = jsonDecode(data) as List;
      final now = DateTime.now();
      return decoded
          .map((c) =>
              SpacedRepetitionCard.fromJson(Map<String, dynamic>.from(c)))
          .where((card) =>
              card.nextReview.isBefore(now) ||
              card.nextReview.isAtSameMomentAs(now))
          .toList();
    } catch (e) {
      return [];
    }
  }

  Future<List<SpacedRepetitionCard>> getCardsBySubject(String subject) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_cardsKey);
    if (data == null) return [];

    try {
      final decoded = jsonDecode(data) as List;
      return decoded
          .map((c) =>
              SpacedRepetitionCard.fromJson(Map<String, dynamic>.from(c)))
          .where((card) => card.subject.toLowerCase() == subject.toLowerCase())
          .toList();
    } catch (e) {
      return [];
    }
  }

  Future<List<SpacedRepetitionCard>> getAllCards() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_cardsKey);
    if (data == null) return [];

    try {
      final decoded = jsonDecode(data) as List;
      return decoded
          .map((c) =>
              SpacedRepetitionCard.fromJson(Map<String, dynamic>.from(c)))
          .toList();
    } catch (e) {
      return [];
    }
  }

  Future<void> addCard(SpacedRepetitionCard card) async {
    final cards = await getAllCards();
    cards.add(card);
    await _saveCards(cards);
  }

  Future<void> addCards(List<SpacedRepetitionCard> newCards) async {
    final cards = await getAllCards();
    cards.addAll(newCards);
    await _saveCards(cards);
  }

  Future<void> upsertCards(List<SpacedRepetitionCard> newCards) async {
    final cards = await getAllCards();
    final byId = {for (final card in cards) card.id: card};
    for (final card in newCards) {
      byId[card.id] = card;
    }
    await _saveCards(byId.values.toList());
  }

  Future<void> updateCard(SpacedRepetitionCard updatedCard) async {
    final cards = await getAllCards();
    final index = cards.indexWhere((c) => c.id == updatedCard.id);
    if (index != -1) {
      cards[index] = updatedCard;
      await _saveCards(cards);
    }
  }

  Future<void> deleteCard(String cardId) async {
    final cards = await getAllCards();
    cards.removeWhere((c) => c.id == cardId);
    await _saveCards(cards);
  }

  Future<SpacedRepetitionCard> processReview(
    SpacedRepetitionCard card,
    ReviewQuality quality,
  ) async {
    final newCard = _calculateNextReview(card, quality);
    await updateCard(newCard);
    return newCard;
  }

  SpacedRepetitionCard _calculateNextReview(
    SpacedRepetitionCard card,
    ReviewQuality quality,
  ) {
    double newEaseFactor = card.easeFactor;
    int newInterval = card.interval;
    int newRepetitions = card.repetitions;
    int newLapses = card.lapses;

    switch (quality) {
      case ReviewQuality.again:
        newInterval = 1;
        newRepetitions = 0;
        newLapses++;
        newEaseFactor = (newEaseFactor - 0.2).clamp(1.3, 2.5);
        break;
      case ReviewQuality.hard:
        newInterval = (card.interval * 1.2).round();
        newEaseFactor = (newEaseFactor - 0.15).clamp(1.3, 2.5);
        break;
      case ReviewQuality.good:
        newRepetitions++;
        if (newRepetitions == 1) {
          newInterval = 1;
        } else if (newRepetitions == 2) {
          newInterval = 6;
        } else {
          newInterval = (card.interval * newEaseFactor).round();
        }
        break;
      case ReviewQuality.easy:
        newRepetitions++;
        newEaseFactor = (newEaseFactor + 0.15).clamp(1.3, 2.5);
        if (newRepetitions == 1) {
          newInterval = 4;
        } else {
          newInterval = (card.interval * newEaseFactor * 1.3).round();
        }
        break;
    }

    final nextReview = DateTime.now().add(Duration(days: newInterval));

    return card.copyWith(
      interval: newInterval,
      easeFactor: newEaseFactor,
      repetitions: newRepetitions,
      lapses: newLapses,
      nextReview: nextReview,
    );
  }

  Future<void> _saveCards(List<SpacedRepetitionCard> cards) async {
    final prefs = await SharedPreferences.getInstance();
    final data = jsonEncode(cards.map((c) => c.toJson()).toList());
    await prefs.setString(_cardsKey, data);
  }

  Future<Map<String, dynamic>> getStats() async {
    final cards = await getAllCards();
    final now = DateTime.now();

    int dueToday = 0;
    int mastered = 0;
    int learning = 0;
    int newCards = 0;

    for (final card in cards) {
      if (card.nextReview.isBefore(now) ||
          card.nextReview.isAtSameMomentAs(now)) {
        dueToday++;
      }
      if (card.repetitions >= 5 && card.easeFactor >= 2.3) {
        mastered++;
      } else if (card.repetitions > 0) {
        learning++;
      } else {
        newCards++;
      }
    }

    return {
      'totalCards': cards.length,
      'dueToday': dueToday,
      'mastered': mastered,
      'learning': learning,
      'newCards': newCards,
    };
  }

  Future<List<String>> getWeakTopics() async {
    final cards = await getAllCards();
    final topicPerformance = <String, List<bool>>{};

    for (final card in cards) {
      topicPerformance.putIfAbsent(card.topic, () => []);
      final wasCorrect = card.repetitions > 0 && card.lapses < 2;
      topicPerformance[card.topic]!.add(wasCorrect);
    }

    final weakTopics = <String>[];
    topicPerformance.forEach((topic, results) {
      if (results.isNotEmpty) {
        final correctRate = results.where((r) => r).length / results.length;
        if (correctRate < 0.7) {
          weakTopics.add(topic);
        }
      }
    });

    return weakTopics;
  }

  Future<void> saveReviewSession(ReviewSession session) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_sessionsKey);
    List<ReviewSession> sessions = [];

    if (data != null) {
      try {
        final decoded = jsonDecode(data) as List;
        sessions = decoded
            .map((s) => ReviewSession.fromJson(Map<String, dynamic>.from(s)))
            .toList();
      } catch (e) {
        // Ignore malformed cached review sessions.
      }
    }

    sessions.add(session);
    await prefs.setString(
      _sessionsKey,
      jsonEncode(sessions.map((s) => s.toJson()).toList()),
    );
  }

  Future<List<ReviewSession>> getReviewHistory({int limit = 30}) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_sessionsKey);
    if (data == null) return [];

    try {
      final decoded = jsonDecode(data) as List;
      final sessions = decoded
          .map((s) => ReviewSession.fromJson(Map<String, dynamic>.from(s)))
          .toList();
      sessions.sort((a, b) => b.date.compareTo(a.date));
      return sessions.take(limit).toList();
    } catch (e) {
      return [];
    }
  }
}
