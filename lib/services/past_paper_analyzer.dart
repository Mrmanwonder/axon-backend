import 'dart:io';
import 'package:path/path.dart' as p;
import 'grok_service.dart';
import 'pdf_service.dart';
import 'study_catalog.dart';

class PastPaperAnalyzer {
  static final PastPaperAnalyzer _instance = PastPaperAnalyzer._internal();
  factory PastPaperAnalyzer() => _instance;
  PastPaperAnalyzer._internal();

  final PdfService _pdfService = PdfService();

  Future<List<PastPaperResult>> analyzeAllPastPapers({
    required String board,
    required String subject,
    String? paperType,
    bool verbose = true,
  }) async {
    final results = <PastPaperResult>[];

    if (verbose) {
      print('🔍 Scanning for past papers: $subject - $board');
    }

    final papers = await _findPastPapers(board: board, subject: subject);

    if (papers.isEmpty) {
      if (verbose) {
        print('⚠️ No past papers found for $subject ($board)');
      }
      return results;
    }

    if (verbose) {
      print('📄 Found ${papers.length} past papers');
    }

    for (final paper in papers) {
      if (verbose) {
        print('\n📝 Processing: ${paper.name}');
      }

      try {
        final result = await _analyzePastPaper(
          paper: paper,
          verbose: verbose,
        );
        results.add(result);

        if (verbose) {
          print('✅ Score: ${result.totalScore.toStringAsFixed(1)}%');
        }
      } catch (e) {
        if (verbose) {
          print('❌ Error processing ${paper.name}: $e');
        }
        results.add(PastPaperResult(
          paperPath: paper.path,
          paperName: paper.name,
          totalScore: 0,
          questionsAnalyzed: 0,
          errors: [e.toString()],
        ));
      }
    }

    if (verbose) {
      _printSummary(results);
    }

    return results;
  }

  Future<List<PastPaperFile>> _findPastPapers({
    required String board,
    required String subject,
  }) async {
    final papers = <PastPaperFile>[];

    try {
      final directory = Directory('/storage/emulated/0/Axon/PastPapers');
      if (await directory.exists()) {
        await for (final entity in directory.list(recursive: true)) {
          if (entity is File && entity.path.toLowerCase().endsWith('.pdf')) {
            final fileName = p.basename(entity.path).toLowerCase();
            final normalizedSubject =
                StudyCatalog.normalizeSubject(subject).toLowerCase();
            final normalizedBoard = board.toLowerCase();

            if (fileName.contains(normalizedSubject) ||
                fileName.contains(normalizedBoard) ||
                fileName.contains('paper') ||
                fileName.contains('question')) {
              papers.add(PastPaperFile(
                path: entity.path,
                name: p.basename(entity.path),
              ));
            }
          }
        }
      }
    } catch (e) {
      print('Error scanning directory: $e');
    }

    try {
      final altDirectory = Directory('/storage/emulated/0/Download');
      if (await altDirectory.exists()) {
        await for (final entity in altDirectory.list(recursive: true)) {
          if (entity is File && entity.path.toLowerCase().endsWith('.pdf')) {
            final fileName = p.basename(entity.path).toLowerCase();
            if (fileName.contains('paper') ||
                fileName.contains('question') ||
                fileName.contains('exam')) {
              final exists = papers.any((p) => p.path == entity.path);
              if (!exists) {
                papers.add(PastPaperFile(
                  path: entity.path,
                  name: p.basename(entity.path),
                ));
              }
            }
          }
        }
      }
    } catch (e) {
      print('Error scanning download directory: $e');
    }

    return papers;
  }

  Future<PastPaperResult> _analyzePastPaper({
    required PastPaperFile paper,
    required bool verbose,
  }) async {
    final questionResults = <QuestionResult>[];
    double totalScore = 0;
    double totalPossible = 0;

    if (verbose) {
      print('  📖 Extracting text from PDF...');
    }

    final text = await _pdfService.extractText(paper.path);
    if (text.isEmpty) {
      throw Exception('Could not extract text from PDF');
    }

    if (verbose) {
      print('  🔍 Extracting questions...');
    }

    final questions = _extractQuestions(text);

    if (questions.isEmpty) {
      throw Exception('No questions found in paper');
    }

    if (verbose) {
      print('  📋 Found ${questions.length} questions');
    }

    for (int i = 0; i < questions.length; i++) {
      final question = questions[i];

      if (verbose) {
        print('  ${i + 1}/${questions.length}: Analyzing Q${i + 1}...');
      }

      final markingScheme = await _findMarkingScheme(paper.path, i + 1);

      final result = await _analyzeQuestion(
        question: question,
        questionNumber: i + 1,
        markingScheme: markingScheme,
        paperContext: text,
        verbose: verbose,
      );

      questionResults.add(result);
      totalScore += result.obtainedMarks;
      totalPossible += result.possibleMarks;
    }

    final percentage =
        totalPossible > 0 ? (totalScore / totalPossible) * 100 : 0.0;

    return PastPaperResult(
      paperPath: paper.path,
      paperName: paper.name,
      totalScore: percentage,
      questionsAnalyzed: questionResults.length,
      questionResults: questionResults,
    );
  }

  List<String> _extractQuestions(String text) {
    final questions = <String>[];
    final lines = text.split('\n');
    String currentQuestion = '';
    bool inQuestion = false;

    for (final line in lines) {
      final trimmed = line.trim();

      if (RegExp(r'^\d+[\.)]', caseSensitive: false).hasMatch(trimmed)) {
        if (currentQuestion.isNotEmpty) {
          questions.add(currentQuestion.trim());
        }
        currentQuestion = line;
        inQuestion = true;
      } else if (inQuestion && trimmed.isNotEmpty) {
        currentQuestion += ' $line';
      }
    }

    if (currentQuestion.isNotEmpty) {
      questions.add(currentQuestion.trim());
    }

    return questions;
  }

  Future<String?> _findMarkingScheme(
      String paperPath, int questionNumber) async {
    final basePath = paperPath.replaceAll(
        RegExp(
          r'\.pdf\$',
        ),
        '');
    final possiblePaths = [
      '${basePath}_ms.pdf',
      '${basePath}_marking.pdf',
      '${basePath}_answers.pdf',
      '${basePath.replaceAll('paper', 'ms')}.pdf',
      '${basePath.replaceAll('question', 'ms')}.pdf',
    ];

    for (final path in possiblePaths) {
      final file = File(path);
      if (await file.exists()) {
        try {
          final text = await _pdfService.extractText(path);
          return text;
        } catch (_) {
          continue;
        }
      }
    }

    return null;
  }

  Future<QuestionResult> _analyzeQuestion({
    required String question,
    required int questionNumber,
    String? markingScheme,
    required String paperContext,
    required bool verbose,
  }) async {
    final prompt = _buildQuestionAnalysisPrompt(
      question: question,
      questionNumber: questionNumber,
      markingScheme: markingScheme,
      paperContext: paperContext,
    );

    final response = await GrokService().chat(prompt);

    return _parseGeminiResponse(
      response: response,
      question: question,
      questionNumber: questionNumber,
      markingScheme: markingScheme,
    );
  }

  String _buildQuestionAnalysisPrompt({
    required String question,
    required int questionNumber,
    String? markingScheme,
    required String paperContext,
  }) {
    final buffer = StringBuffer();

    buffer.writeln(
        'You are an expert examiner evaluating a student\'s answer to a past paper question.');
    buffer.writeln('');
    buffer.writeln('QUESTION $questionNumber:');
    buffer.writeln(question);
    buffer.writeln('');

    if (markingScheme != null && markingScheme.isNotEmpty) {
      buffer.writeln('MARKING SCHEME (reference answer):');
      buffer.writeln(markingScheme);
      buffer.writeln('');
    } else {
      buffer.writeln(
          'No marking scheme available. Provide estimated marks based on question difficulty.');
      buffer.writeln('');
    }

    buffer.writeln('INSTRUCTIONS:');
    buffer.writeln(
        '1. Evaluate the student\'s answer (simulated) against the marking scheme');
    buffer.writeln(
        '2. If no marking scheme, analyze what a good answer should contain');
    buffer.writeln(
        '3. Provide marks out of typical marks for this question type');
    buffer.writeln('4. Provide feedback on areas to improve');
    buffer.writeln('');
    buffer.writeln('OUTPUT in this exact JSON format:');
    buffer.writeln('''{
  "obtained_marks": <number>,
  "possible_marks": <number>,
  "feedback": "<brief feedback>",
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<weakness1>", "<weakness2>"],
  "improvement_tips": ["<tip1>", "<tip2>"]
}''');

    return buffer.toString();
  }

  QuestionResult _parseGeminiResponse({
    required String response,
    required String question,
    required int questionNumber,
    String? markingScheme,
  }) {
    try {
      final jsonMatch = RegExp(r'\{[\s\S]*\}').firstMatch(response);
      if (jsonMatch != null) {
        final json = jsonMatch.group(0)!;
        final data = _parseJsonSimple(json);

        return QuestionResult(
          questionNumber: questionNumber,
          question: question,
          obtainedMarks: (data['obtained_marks'] ?? 0).toDouble(),
          possibleMarks: (data['possible_marks'] ?? 10).toDouble(),
          feedback: data['feedback'] ?? 'Analyzed',
          strengths: List<String>.from(data['strengths'] ?? []),
          weaknesses: List<String>.from(data['weaknesses'] ?? []),
          improvementTips: List<String>.from(data['improvement_tips'] ?? []),
        );
      }
    } catch (e) {
      print('Error parsing response: $e');
    }

    return QuestionResult(
      questionNumber: questionNumber,
      question: question,
      obtainedMarks: 0,
      possibleMarks: 10,
      feedback: 'Could not analyze',
      strengths: [],
      weaknesses: [],
      improvementTips: ['Review question content'],
    );
  }

  Map<String, dynamic> _parseJsonSimple(String json) {
    final result = <String, dynamic>{};

    final obtainedMatch =
        RegExp(r'"obtained_marks"\s*:\s*([\d.]+)').firstMatch(json);
    if (obtainedMatch != null) {
      result['obtained_marks'] = double.tryParse(obtainedMatch.group(1)!) ?? 0;
    }

    final possibleMatch =
        RegExp(r'"possible_marks"\s*:\s*([\d.]+)').firstMatch(json);
    if (possibleMatch != null) {
      result['possible_marks'] = double.tryParse(possibleMatch.group(1)!) ?? 10;
    }

    final feedbackMatch =
        RegExp(r'"feedback"\s*:\s*"([^"]*)"').firstMatch(json);
    if (feedbackMatch != null) {
      result['feedback'] = feedbackMatch.group(1);
    }

    final strengthsMatch =
        RegExp(r'"strengths"\s*:\s*\[([\s\S]*?)\]').firstMatch(json);
    if (strengthsMatch != null) {
      final strengths = <String>[];
      final matches = RegExp(r'"([^"]*)"').allMatches(strengthsMatch.group(1)!);
      for (final match in matches) {
        strengths.add(match.group(1)!);
      }
      result['strengths'] = strengths;
    }

    final weaknessesMatch =
        RegExp(r'"weaknesses"\s*:\s*\[([\s\S]*?)\]').firstMatch(json);
    if (weaknessesMatch != null) {
      final weaknesses = <String>[];
      final matches =
          RegExp(r'"([^"]*)"').allMatches(weaknessesMatch.group(1)!);
      for (final match in matches) {
        weaknesses.add(match.group(1)!);
      }
      result['weaknesses'] = weaknesses;
    }

    final tipsMatch =
        RegExp(r'"improvement_tips"\s*:\s*\[([\s\S]*?)\]').firstMatch(json);
    if (tipsMatch != null) {
      final tips = <String>[];
      final matches = RegExp(r'"([^"]*)"').allMatches(tipsMatch.group(1)!);
      for (final match in matches) {
        tips.add(match.group(1)!);
      }
      result['improvement_tips'] = tips;
    }

    return result;
  }

  void _printSummary(List<PastPaperResult> results) {
    if (results.isEmpty) {
      print('\n📊 No results to display');
      return;
    }

    final totalScore = results.fold<double>(0, (sum, r) => sum + r.totalScore);
    final avgScore = totalScore / results.length;

    print('\n${'=' * 50}');
    print('📊 PAST PAPER ANALYSIS SUMMARY');
    print('=' * 50);
    print('Total Papers Analyzed: ${results.length}');
    print('Average Score: ${avgScore.toStringAsFixed(1)}%');
    print('');

    for (final result in results) {
      print('  ${result.paperName}: ${result.totalScore.toStringAsFixed(1)}%');
    }

    print('=' * 50);

    final allStrengths = <String, int>{};
    final allWeaknesses = <String, int>{};

    for (final result in results) {
      for (final q in result.questionResults) {
        for (final s in q.strengths) {
          allStrengths[s] = (allStrengths[s] ?? 0) + 1;
        }
        for (final w in q.weaknesses) {
          allWeaknesses[w] = (allWeaknesses[w] ?? 0) + 1;
        }
      }
    }

    if (allStrengths.isNotEmpty) {
      print('\n🎯 TOP STRENGTHS:');
      final sortedStrengths = allStrengths.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      for (final entry in sortedStrengths.take(3)) {
        print('  • ${entry.key} (${entry.value} times)');
      }
    }

    if (allWeaknesses.isNotEmpty) {
      print('\n⚠️ TOP AREAS TO IMPROVE:');
      final sortedWeaknesses = allWeaknesses.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      for (final entry in sortedWeaknesses.take(3)) {
        print('  • ${entry.key} (${entry.value} times)');
      }
    }

    print('\n📚 TRAINING RECOMMENDATIONS:');
    print('  1. Focus on weak areas identified above');
    print('  2. Review marking schemes for exam technique');
    print('  3. Practice timing for each question type');
    print('  4. Use improvement tips from each question');
  }

  Future<void> trainFromResults(List<PastPaperResult> results) async {
    if (results.isEmpty) {
      print('No results to train from');
      return;
    }

    print('\n🧠 Training from ${results.length} past papers...');

    final trainingData = <Map<String, dynamic>>[];

    for (final result in results) {
      for (final question in result.questionResults) {
        trainingData.add({
          'question': question.question,
          'obtained_marks': question.obtainedMarks,
          'possible_marks': question.possibleMarks,
          'feedback': question.feedback,
          'strengths': question.strengths,
          'weaknesses': question.weaknesses,
          'tips': question.improvementTips,
        });
      }
    }

    final prompt =
        '''You are an AI tutor trained on past paper performance data.
Analyze the following training data and provide insights for improvement:

TRAINING DATA:
${trainingData.map((d) => '''
Q: ${d['question']}
Score: ${d['obtained_marks']}/${d['possible_marks']}
Feedback: ${d['feedback']}
Weaknesses: ${d['weaknesses']}
Tips: ${d['tips']}
''').join('\n')}

Provide a comprehensive learning plan in JSON format:
{
  "weakness_patterns": ["pattern1", "pattern2"],
  "recommended_topics": ["topic1", "topic2"],
  "study_strategy": "strategy description",
  "practice_recommendations": ["rec1", "rec2"]
}''';

    final response = await GrokService().chat(prompt);
    print('\n📋 PERSONALIZED LEARNING PLAN:');
    print(response);
  }
}

class PastPaperFile {
  final String path;
  final String name;

  PastPaperFile({required this.path, required this.name});
}

class PastPaperResult {
  final String paperPath;
  final String paperName;
  final double totalScore;
  final int questionsAnalyzed;
  final List<QuestionResult> questionResults;
  final List<String> errors;

  PastPaperResult({
    required this.paperPath,
    required this.paperName,
    required this.totalScore,
    required this.questionsAnalyzed,
    this.questionResults = const [],
    this.errors = const [],
  });
}

class QuestionResult {
  final int questionNumber;
  final String question;
  final double obtainedMarks;
  final double possibleMarks;
  final String feedback;
  final List<String> strengths;
  final List<String> weaknesses;
  final List<String> improvementTips;

  QuestionResult({
    required this.questionNumber,
    required this.question,
    required this.obtainedMarks,
    required this.possibleMarks,
    required this.feedback,
    this.strengths = const [],
    this.weaknesses = const [],
    this.improvementTips = const [],
  });
}
