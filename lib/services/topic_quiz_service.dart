// lib/services/topic_quiz_service.dart
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import '../models/models.dart';
import 'past_papers_service.dart';

class TopicQuizService {
  static final TopicQuizService _instance = TopicQuizService._internal();
  factory TopicQuizService() => _instance;
  TopicQuizService._internal();

  final _pastPapersService = PastPapersService();

  Future<String> get _questionsPath async {
    final appDir = await getApplicationDocumentsDirectory();
    return '${appDir.path}/../../extracted_questions';
  }

  Future<List<PdfQuestion>> getQuestionsForChapter({
    required String subject,
    String? chapter,
    String? topic,
    int minQuestions = 5,
    int maxQuestions = 10,
  }) async {
    final path = await _questionsPath;
    final dir = Directory(path);
    if (!await dir.exists()) return [];

    final subjectCode = _pastPapersService.getCodeForSubject(subject);
    if (subjectCode == null) return [];

    final questions = <PdfQuestion>[];
    final subjectLower = subject.toLowerCase();

    await for (final entity in dir.list()) {
      if (entity is File && entity.path.endsWith('.json')) {
        try {
          final content = await entity.readAsString();
          final data = jsonDecode(content);
          if (data is Map && data['questions'] is List) {
            for (final q in data['questions']) {
              if (q is Map) {
                final pdfQ =
                    _parseQuestion(Map<String, dynamic>.from(q), subjectLower);
                if (_matchesChapterOrTopic(pdfQ, chapter, topic)) {
                  questions.add(pdfQ);
                }
              }
            }
          }
        } catch (_) {}
      }
    }

    return _selectDiverseQuestions(questions, minQuestions, maxQuestions);
  }

  Future<List<PdfQuestion>> getQuestionsForSubject({
    required String subject,
    int minQuestions = 5,
    int maxQuestions = 10,
  }) async {
    final path = await _questionsPath;
    final dir = Directory(path);
    if (!await dir.exists()) return [];

    final questions = <PdfQuestion>[];
    final subjectLower = subject.toLowerCase();

    await for (final entity in dir.list()) {
      if (entity is File && entity.path.endsWith('.json')) {
        try {
          final content = await entity.readAsString();
          final data = jsonDecode(content);
          if (data is Map && data['questions'] is List) {
            for (final q in data['questions']) {
              if (q is Map) {
                final pdfQ =
                    _parseQuestion(Map<String, dynamic>.from(q), subjectLower);
                questions.add(pdfQ);
              }
            }
          }
        } catch (_) {}
      }
    }

    return _selectDiverseQuestions(questions, minQuestions, maxQuestions);
  }

  PdfQuestion _parseQuestion(Map<String, dynamic> q, String subjectLower) {
    final parts = <PdfPart>[];
    if (q['parts'] is List) {
      for (final p in q['parts']) {
        if (p is Map) {
          parts.add(PdfPart(
            label: p['label']?.toString() ?? '',
            text: p['text']?.toString() ?? '',
          ));
        }
      }
    }

    return PdfQuestion(
      questionNumber: int.tryParse(q['number']?.toString() ?? '1') ?? 1,
      questionText: q['text']?.toString() ?? '',
      parts: parts,
      yPosition: 0,
      xPosition: 0,
      width: 0,
      height: 0,
      pageNumber: int.tryParse(q['page']?.toString() ?? '1') ?? 1,
      chapterTag: q['chapter']?.toString() ?? '',
      topicTag: q['topic']?.toString() ?? '',
      difficultyTag: _estimateDifficulty(q),
      subjectTag: subjectLower,
    );
  }

  String _estimateDifficulty(Map<String, dynamic> q) {
    final text = q['text']?.toString() ?? '';
    final marks = q['marks'] as int? ?? 0;

    if (marks >= 8 || text.length > 300) return 'Hard';
    if (marks >= 4 || text.length > 150) return 'Medium';
    return 'Easy';
  }

  bool _matchesChapterOrTopic(PdfQuestion q, String? chapter, String? topic) {
    if (chapter != null && chapter.isNotEmpty) {
      if (!q.chapterTag.toLowerCase().contains(chapter.toLowerCase())) {
        return false;
      }
    }
    if (topic != null && topic.isNotEmpty) {
      if (!q.topicTag.toLowerCase().contains(topic.toLowerCase())) {
        return false;
      }
    }
    return true;
  }

  List<PdfQuestion> _selectDiverseQuestions(
    List<PdfQuestion> all,
    int minQuestions,
    int maxQuestions,
  ) {
    if (all.isEmpty) return [];

    final easy = all.where((q) => q.difficultyTag == 'Easy').toList();
    final medium = all.where((q) => q.difficultyTag == 'Medium').toList();
    final hard = all.where((q) => q.difficultyTag == 'Hard').toList();

    final target = all.length.clamp(minQuestions, maxQuestions);
    final result = <PdfQuestion>[];

    final easyCount = (target * 0.3).ceil();
    final mediumCount = (target * 0.4).ceil();
    final hardCount = target - easyCount - mediumCount;

    result.addAll(_pickRandom(easy, easyCount));
    result.addAll(_pickRandom(medium, mediumCount));
    result.addAll(_pickRandom(hard, hardCount));

    result.shuffle();
    return result.take(maxQuestions).toList();
  }

  List<PdfQuestion> _pickRandom(List<PdfQuestion> list, int count) {
    if (list.isEmpty) return [];
    final shuffled = List<PdfQuestion>.from(list)..shuffle();
    return shuffled.take(count).toList();
  }
}

class QuizResult {
  final int totalQuestions;
  final int correctAnswers;
  final int totalMarks;
  final int earnedMarks;
  final Map<String, int> chapterScores;
  final Map<String, int> chapterTotal;
  final List<TopicPerformance> topicPerformance;
  final String recommendedPlan;

  double get percentage =>
      totalQuestions > 0 ? (correctAnswers / totalQuestions * 100) : 0;

  List<String> get weakPoints => chapterScores.entries
      .where((e) =>
          chapterTotal[e.key]! > 0 && e.value / chapterTotal[e.key]! < 0.5)
      .map((e) => e.key)
      .toList();

  List<String> get strongPoints => chapterScores.entries
      .where((e) =>
          chapterTotal[e.key]! > 0 && e.value / chapterTotal[e.key]! >= 0.7)
      .map((e) => e.key)
      .toList();

  QuizResult({
    required this.totalQuestions,
    required this.correctAnswers,
    required this.totalMarks,
    required this.earnedMarks,
    required this.chapterScores,
    required this.chapterTotal,
    required this.topicPerformance,
    required this.recommendedPlan,
  });

  static QuizResult fromQuestions(List<PdfQuestion> questions) {
    final chapterScores = <String, int>{};
    final chapterTotal = <String, int>{};
    final topicPerf = <TopicPerformance>[];
    int totalMarks = 0;
    int earnedMarks = 0;
    int correct = 0;

    for (final q in questions) {
      final chapter = q.chapterTag.isNotEmpty ? q.chapterTag : 'General';
      final marks = q.marksAvailable ?? 1;
      totalMarks += marks;

      chapterTotal[chapter] = (chapterTotal[chapter] ?? 0) + marks;

      if (q.isCorrect == true) {
        correct++;
        earnedMarks += q.marksAwarded ?? marks;
        chapterScores[chapter] =
            (chapterScores[chapter] ?? 0) + (q.marksAwarded ?? marks);
      } else {
        chapterScores[chapter] = (chapterScores[chapter] ?? 0);
      }
    }

    for (final entry in chapterScores.entries) {
      final total = chapterTotal[entry.key] ?? 1;
      topicPerf.add(TopicPerformance(
        topic: entry.key,
        score: entry.value,
        total: total,
        percentage: (entry.value / total * 100).round(),
      ));
    }
    topicPerf.sort((a, b) => a.percentage.compareTo(b.percentage));

    final plan = _generatePlan(topicPerf);

    return QuizResult(
      totalQuestions: questions.length,
      correctAnswers: correct,
      totalMarks: totalMarks,
      earnedMarks: earnedMarks,
      chapterScores: chapterScores,
      chapterTotal: chapterTotal,
      topicPerformance: topicPerf,
      recommendedPlan: plan,
    );
  }

  static String _generatePlan(List<TopicPerformance> perf) {
    if (perf.isEmpty) return 'Keep practicing to improve your scores!';

    final weakest = perf.first;
    if (weakest.percentage >= 70) {
      return 'Great job! You\'ve mastered most topics. Focus on advanced practice in ${perf.last.topic}.';
    } else if (weakest.percentage >= 50) {
      return 'Good progress! Spend extra time on ${weakest.topic} (${weakest.percentage}% - needs improvement).';
    } else {
      return 'Focus on strengthening ${weakest.topic} (only ${weakest.percentage}%). Review fundamentals and practice 5+ questions daily.';
    }
  }
}

class TopicPerformance {
  final String topic;
  final int score;
  final int total;
  final int percentage;

  TopicPerformance({
    required this.topic,
    required this.score,
    required this.total,
    required this.percentage,
  });
}
