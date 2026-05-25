import 'dart:convert';
import 'dart:io';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/models.dart';
import 'study_catalog.dart';

class MistakeEntry {
  final String id;
  final String questionId;
  final String sourceTitle;
  final String sourcePath;
  final String subject;
  final String chapter;
  final String topic;
  final String difficulty;
  final String paperType;
  final String board;
  final String paperYear;
  final int questionNumber;
  final String questionText;
  final String userAnswer;
  final String? correctAnswer;
  final DateTime recordedAt;

  const MistakeEntry({
    required this.id,
    required this.questionId,
    required this.sourceTitle,
    required this.sourcePath,
    required this.subject,
    required this.chapter,
    required this.topic,
    required this.difficulty,
    required this.paperType,
    required this.board,
    required this.paperYear,
    required this.questionNumber,
    required this.questionText,
    required this.userAnswer,
    required this.correctAnswer,
    required this.recordedAt,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'questionId': questionId,
        'sourceTitle': sourceTitle,
        'sourcePath': sourcePath,
        'subject': subject,
        'chapter': chapter,
        'topic': topic,
        'difficulty': difficulty,
        'paperType': paperType,
        'board': board,
        'paperYear': paperYear,
        'questionNumber': questionNumber,
        'questionText': questionText,
        'userAnswer': userAnswer,
        'correctAnswer': correctAnswer,
        'recordedAt': recordedAt.toIso8601String(),
      };

  factory MistakeEntry.fromJson(Map<String, dynamic> json) => MistakeEntry(
        id: (json['id'] ?? '').toString(),
        questionId: (json['questionId'] ?? '').toString(),
        sourceTitle: (json['sourceTitle'] ?? '').toString(),
        sourcePath: (json['sourcePath'] ?? '').toString(),
        subject: (json['subject'] ?? '').toString(),
        chapter: (json['chapter'] ?? '').toString(),
        topic: (json['topic'] ?? '').toString(),
        difficulty: (json['difficulty'] ?? '').toString(),
        paperType: (json['paperType'] ?? '').toString(),
        board: (json['board'] ?? '').toString(),
        paperYear: (json['paperYear'] ?? '').toString(),
        questionNumber: (json['questionNumber'] as num?)?.toInt() ?? 0,
        questionText: (json['questionText'] ?? '').toString(),
        userAnswer: (json['userAnswer'] ?? '').toString(),
        correctAnswer: json['correctAnswer']?.toString(),
        recordedAt: DateTime.tryParse((json['recordedAt'] ?? '').toString()) ??
            DateTime.now(),
      );
}

class QuestionIntelligenceService {
  QuestionIntelligenceService._();

  static final QuestionIntelligenceService instance =
      QuestionIntelligenceService._();

  static const _mistakesKey = 'mistake_notebook_v1';
  final StudyCatalog _catalog = StudyCatalog();

  Future<List<PdfQuestion>> enrichQuestions(
    String sourcePath,
    List<PdfQuestion> questions, {
    String? sourceTitle,
    String? fallbackSubject,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final board = _normalizeBoard(
        (prefs.getString('userBoard') ?? '').trim(), sourceTitle);
    final title = (sourceTitle == null || sourceTitle.trim().isEmpty)
        ? _fileName(sourcePath)
        : sourceTitle.trim();
    final paperType = detectPaperType(title);
    final paperYear = detectPaperYear(title);
    final catalog = await _catalog.load();
    final subject = _pickSubject(
      title: title,
      fallbackSubject: fallbackSubject,
      catalog: catalog,
    );

    return questions.map((question) {
      final chapter = _matchChapter(question, subject, catalog);
      final topic = _inferTopic(question, chapter);
      final difficulty = _inferDifficulty(question);
      return question.copyWith(
        subjectTag: subject,
        chapterTag: chapter,
        topicTag: topic,
        difficultyTag: difficulty,
        paperType: paperType,
        boardTag: board,
        paperYear: paperYear,
      );
    }).toList();
  }

  String buildQuestionId(PdfQuestion question, {required String sourcePath}) {
    return '${sourcePath.hashCode}:${question.pageNumber}:${question.questionNumber}:${question.questionText.hashCode}';
  }

  Future<void> storeWrongQuestions({
    required String sourceTitle,
    required String sourcePath,
    required List<PdfQuestion> questions,
    required Map<String, String> answers,
    required bool Function(PdfQuestion question, String answer) isWrong,
  }) async {
    final mistakes = await loadMistakes();
    final byId = {for (final item in mistakes) item.id: item};

    for (final question in questions) {
      final keys = <String>[
        'q${question.questionNumber}',
        ...question.parts
            .map((part) => 'q${question.questionNumber}_${part.label}'),
      ];
      final answer = keys
          .map((key) => answers[key]?.trim() ?? '')
          .where((value) => value.isNotEmpty)
          .join(' | ');
      if (answer.isEmpty || !isWrong(question, answer)) {
        continue;
      }
      final questionId = buildQuestionId(question, sourcePath: sourcePath);
      final entry = MistakeEntry(
        id: questionId,
        questionId: questionId,
        sourceTitle: sourceTitle,
        sourcePath: sourcePath,
        subject: question.subjectTag,
        chapter: question.chapterTag,
        topic: question.topicTag,
        difficulty: question.difficultyTag,
        paperType: question.paperType,
        board: question.boardTag,
        paperYear: question.paperYear,
        questionNumber: question.questionNumber,
        questionText: question.fullText,
        userAnswer: answer,
        correctAnswer: question.correctAnswer,
        recordedAt: DateTime.now(),
      );
      byId[entry.id] = entry;
    }

    await _saveMistakes(byId.values.toList()
      ..sort((a, b) => b.recordedAt.compareTo(a.recordedAt)));
  }

  Future<List<MistakeEntry>> loadMistakes() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_mistakesKey);
    if (raw == null || raw.isEmpty) return const [];
    try {
      final decoded = jsonDecode(raw) as List<dynamic>;
      return decoded
          .whereType<Map>()
          .map((item) => MistakeEntry.fromJson(Map<String, dynamic>.from(item)))
          .toList();
    } catch (_) {
      return const [];
    }
  }

  Future<Set<String>> loadWrongQuestionIds() async {
    final mistakes = await loadMistakes();
    return mistakes.map((item) => item.questionId).toSet();
  }

  Future<List<String>> weakTopics({String? subject}) async {
    final mistakes = await loadMistakes();
    final counts = <String, int>{};
    for (final item in mistakes) {
      if (subject != null &&
          subject.trim().isNotEmpty &&
          item.subject.toLowerCase() != subject.trim().toLowerCase()) {
        continue;
      }
      final key = item.topic.isNotEmpty
          ? item.topic
          : (item.chapter.isNotEmpty ? item.chapter : item.subject);
      counts.update(key, (value) => value + 1, ifAbsent: () => 1);
    }
    final ranked = counts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return ranked.take(5).map((entry) => entry.key).toList();
  }

  Future<void> _saveMistakes(List<MistakeEntry> mistakes) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _mistakesKey,
      jsonEncode(mistakes.map((item) => item.toJson()).toList()),
    );
  }

  String detectPaperType(String text) {
    final lower = text.toLowerCase();
    if (lower.contains('mark scheme')) return 'Mark Scheme';
    if (lower.contains('specimen')) return 'Specimen';
    if (lower.contains('topical')) return 'Topical';
    if (lower.contains('worksheet')) return 'Worksheet';
    if (lower.contains('question paper')) return 'Question Paper';
    if (lower.contains('paper')) return 'Past Paper';
    return 'Practice';
  }

  String detectPaperYear(String text) {
    final match = RegExp(r'\b(20\d{2}|19\d{2})\b').firstMatch(text);
    if (match != null) return match.group(1)!;
    final short =
        RegExp(r'\b(\d{2})\b').allMatches(text).map((m) => m.group(1)!);
    for (final token in short) {
      final value = int.tryParse(token);
      if (value != null && value <= 30) {
        return '20${token.padLeft(2, '0')}';
      }
    }
    return '';
  }

  String _normalizeBoard(String board, String? sourceTitle) {
    if (board.isNotEmpty) return board;
    final text = (sourceTitle ?? '').toLowerCase();
    if (text.contains('cambridge') ||
        text.contains('caie') ||
        text.contains('cie') ||
        text.contains('igcse') ||
        text.contains('alevel') ||
        text.contains('as level')) {
      return 'Cambridge International';
    }
    if (text.contains('ib')) return 'IB Diploma';
    return '';
  }

  String _pickSubject({
    required String title,
    required String? fallbackSubject,
    required Map<String, List<String>> catalog,
  }) {
    if (fallbackSubject != null && fallbackSubject.trim().isNotEmpty) {
      return fallbackSubject.trim();
    }
    final lower = title.toLowerCase();
    for (final subject in catalog.keys) {
      if (lower.contains(subject.toLowerCase())) return subject;
    }
    return '';
  }

  String _matchChapter(
    PdfQuestion question,
    String subject,
    Map<String, List<String>> catalog,
  ) {
    final haystack = question.fullText.toLowerCase();
    final chapters = catalog[subject] ?? const <String>[];
    String best = '';
    int bestScore = 0;
    for (final chapter in chapters) {
      final score = _chapterScore(chapter, haystack);
      if (score > bestScore) {
        bestScore = score;
        best = chapter;
      }
    }
    return best;
  }

  int _chapterScore(String chapter, String haystack) {
    final chapterLower = chapter.toLowerCase();
    if (haystack.contains(chapterLower)) return 8;
    final tokens = chapterLower
        .split(RegExp(r'[^a-z0-9]+'))
        .where((token) => token.length >= 4)
        .toList();
    var score = 0;
    for (final token in tokens) {
      if (haystack.contains(token)) score += 2;
    }
    return score;
  }

  String _inferTopic(PdfQuestion question, String chapter) {
    if (chapter.isNotEmpty) return chapter;
    final source = question.fullText.toLowerCase();
    final candidates = <String>[
      'kinematics',
      'forces',
      'waves',
      'electricity',
      'calculus',
      'probability',
      'trigonometry',
      'algorithms',
      'programming',
      'data structures',
      'organic chemistry',
      'mechanics',
      'genetics',
    ];
    for (final candidate in candidates) {
      if (source.contains(candidate)) return candidate;
    }
    return '';
  }

  String _inferDifficulty(PdfQuestion question) {
    final marks = (question.marksAvailable ?? 0).toDouble();
    final parts = question.parts.length.toDouble();
    final lengthWeight = (question.questionText.length / 220).clamp(0.0, 1.4);
    final score = ((marks / 8) + (parts / 4) + lengthWeight).clamp(0.0, 1.0);
    if (score >= 0.72) return 'Hard';
    if (score >= 0.4) return 'Medium';
    return 'Easy';
  }

  String _fileName(String path) {
    try {
      return File(path).uri.pathSegments.isNotEmpty
          ? File(path).uri.pathSegments.last
          : path;
    } catch (_) {
      return path;
    }
  }
}
