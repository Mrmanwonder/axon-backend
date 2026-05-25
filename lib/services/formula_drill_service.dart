import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import 'exam_planner_service.dart';

class FormulaDrillCard {
  final String id;
  final String topicKey;
  final String topicLabel;
  final String subject;
  final String name;
  final String expression;
  final int box;
  final int reviewCount;
  final DateTime nextReviewAt;
  final DateTime? lastReviewedAt;

  const FormulaDrillCard({
    required this.id,
    required this.topicKey,
    required this.topicLabel,
    required this.subject,
    required this.name,
    required this.expression,
    required this.box,
    required this.reviewCount,
    required this.nextReviewAt,
    required this.lastReviewedAt,
  });

  bool get isDue => !nextReviewAt.isAfter(DateTime.now());

  FormulaDrillCard copyWith({
    int? box,
    int? reviewCount,
    DateTime? nextReviewAt,
    DateTime? lastReviewedAt,
  }) {
    return FormulaDrillCard(
      id: id,
      topicKey: topicKey,
      topicLabel: topicLabel,
      subject: subject,
      name: name,
      expression: expression,
      box: box ?? this.box,
      reviewCount: reviewCount ?? this.reviewCount,
      nextReviewAt: nextReviewAt ?? this.nextReviewAt,
      lastReviewedAt: lastReviewedAt ?? this.lastReviewedAt,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'topicKey': topicKey,
        'topicLabel': topicLabel,
        'subject': subject,
        'name': name,
        'expression': expression,
        'box': box,
        'reviewCount': reviewCount,
        'nextReviewAt': nextReviewAt.toIso8601String(),
        'lastReviewedAt': lastReviewedAt?.toIso8601String(),
      };

  factory FormulaDrillCard.fromJson(Map<String, dynamic> json) {
    return FormulaDrillCard(
      id: (json['id'] ?? '').toString(),
      topicKey: (json['topicKey'] ?? '').toString(),
      topicLabel: (json['topicLabel'] ?? '').toString(),
      subject: (json['subject'] ?? '').toString(),
      name: (json['name'] ?? '').toString(),
      expression: (json['expression'] ?? '').toString(),
      box: (json['box'] ?? 1) as int,
      reviewCount: (json['reviewCount'] ?? 0) as int,
      nextReviewAt:
          DateTime.tryParse((json['nextReviewAt'] ?? '').toString()) ??
              DateTime.now(),
      lastReviewedAt: DateTime.tryParse(
        (json['lastReviewedAt'] ?? '').toString(),
      ),
    );
  }
}

class FormulaDrillService {
  FormulaDrillService._();

  static final FormulaDrillService instance = FormulaDrillService._();

  static const String _storageKey = 'formula_drill_cards';

  Future<List<FormulaDrillCard>> _loadCards() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_storageKey);
    if (raw == null || raw.isEmpty) return [];
    try {
      final decoded = jsonDecode(raw) as List;
      return decoded
          .map((item) => FormulaDrillCard.fromJson(Map<String, dynamic>.from(item)))
          .toList();
    } catch (_) {
      return [];
    }
  }

  Future<void> _saveCards(List<FormulaDrillCard> cards) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _storageKey,
      jsonEncode(cards.map((card) => card.toJson()).toList()),
    );
  }

  Future<List<FormulaDrillCard>> syncFormulaCatalog(
    Map<String, FormulaSheet> sheets,
  ) async {
    final existing = await _loadCards();
    final byId = {for (final card in existing) card.id: card};
    final merged = <FormulaDrillCard>[];

    sheets.forEach((topicKey, sheet) {
      for (final formula in sheet.formulas) {
        final id = '$topicKey::${formula.name}';
        merged.add(
          byId[id] ??
              FormulaDrillCard(
                id: id,
                topicKey: topicKey,
                topicLabel: sheet.topic,
                subject: sheet.subject,
                name: formula.name,
                expression: formula.expression,
                box: 1,
                reviewCount: 0,
                nextReviewAt: DateTime.now(),
                lastReviewedAt: null,
              ),
        );
      }
    });

    await _saveCards(merged);
    return merged;
  }

  Future<List<FormulaDrillCard>> getDueCards(
    Map<String, FormulaSheet> sheets, {
    String? topicKey,
  }) async {
    final cards = await syncFormulaCatalog(sheets);
    final filtered = cards.where((card) {
      if (topicKey == null || topicKey.isEmpty) return true;
      return card.topicKey == topicKey;
    }).toList();

    final due = filtered.where((card) => card.isDue).toList()
      ..sort((a, b) => a.nextReviewAt.compareTo(b.nextReviewAt));
    if (due.isNotEmpty) return due;

    filtered.sort((a, b) => a.nextReviewAt.compareTo(b.nextReviewAt));
    return filtered.take(8).toList();
  }

  Future<void> reviewCard(String cardId, bool remembered) async {
    final cards = await _loadCards();
    final index = cards.indexWhere((card) => card.id == cardId);
    if (index == -1) return;

    final current = cards[index];
    final nextBox = remembered ? (current.box + 1).clamp(1, 5) : 1;
    final intervalDays = switch (nextBox) {
      1 => 0,
      2 => 1,
      3 => 3,
      4 => 7,
      _ => 14,
    };

    cards[index] = current.copyWith(
      box: nextBox,
      reviewCount: current.reviewCount + 1,
      lastReviewedAt: DateTime.now(),
      nextReviewAt: DateTime.now().add(Duration(days: intervalDays)),
    );
    await _saveCards(cards);
  }
}
