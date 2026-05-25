import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';
import 'study_catalog.dart';
import 'grok_service.dart';
import 'pdf_service.dart';

class OneTimePastPaperTrainer {
  static const String _trainedConfigKey = 'axon_trained_exam_config';
  static const String _trainingStatusKey = 'axon_training_completed';

  static final OneTimePastPaperTrainer _instance =
      OneTimePastPaperTrainer._internal();
  factory OneTimePastPaperTrainer() => _instance;
  OneTimePastPaperTrainer._internal();

  bool _isTraining = false;
  Map<String, dynamic>? _cachedConfig;
  final PdfService _pdfService = PdfService();

  bool get isTraining => _isTraining;
  bool get isTrainingComplete => _cachedConfig != null;

  Future<bool> _getTrainingComplete() async {
    if (_cachedConfig != null) return true;
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_trainingStatusKey) ?? false;
  }

  Future<Map<String, dynamic>?> getTrainedConfig() async {
    if (await _getTrainingComplete()) {
      final prefs = await SharedPreferences.getInstance();
      final configJson = prefs.getString(_trainedConfigKey);
      if (configJson != null) {
        return jsonDecode(configJson) as Map<String, dynamic>;
      }
    }
    return null;
  }

  Future<bool> trainIfNeeded({
    required String board,
    required String subject,
    Function(String)? onProgress,
  }) async {
    if (_isTraining) return false;
    if (await _getTrainingComplete()) return true;

    _isTraining = true;
    onProgress?.call('Starting training...');

    try {
      final papers = await _findAllPastPapersWithMarkingSchemes(
        board: board,
        subject: subject,
        onProgress: onProgress,
      );

      if (papers.isEmpty) {
        onProgress?.call('No past papers with marking schemes found');
        _isTraining = false;
        return false;
      }

      onProgress?.call('Found ${papers.length} paper-marking scheme pairs');

      final allQuestions = <Map<String, dynamic>>[];
      final allCriteria = <String, dynamic>{};

      for (int i = 0; i < papers.length; i++) {
        final paper = papers[i];
        onProgress
            ?.call('Processing ${i + 1}/${papers.length}: ${paper.paperName}');

        try {
          final data = await _processPaper(
            paperPath: paper.paperPath,
            markingSchemePath: paper.markingSchemePath,
          );

          if (data != null) {
            allQuestions
                .addAll(data['questions'] as List<Map<String, dynamic>>);
            allCriteria.addAll(data['criteria'] as Map<String, dynamic>);
          }
        } catch (e) {
          onProgress?.call('Error: $e');
        }
      }

      if (allQuestions.isEmpty) {
        onProgress?.call('No questions extracted');
        _isTraining = false;
        return false;
      }

      onProgress?.call('Building trained model config...');

      final config = _buildModelConfig(
        board: board,
        subject: subject,
        questions: allQuestions,
        criteria: allCriteria,
        totalPapers: papers.length,
      );

      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_trainedConfigKey, jsonEncode(config));
      await prefs.setBool(_trainingStatusKey, true);

      onProgress
          ?.call('Training complete! ${allQuestions.length} questions learned');

      _isTraining = false;
      return true;
    } catch (e) {
      onProgress?.call('Training failed: $e');
      _isTraining = false;
      return false;
    }
  }

  Future<List<_PastPaperData>> _findAllPastPapersWithMarkingSchemes({
    required String board,
    required String subject,
    Function(String)? onProgress,
  }) async {
    final papers = <_PastPaperData>[];

    final searchDirs = [
      '/storage/emulated/0/Axon/PastPapers',
      '/storage/emulated/0/Download',
      '/storage/emulated/0/Documents',
    ];

    for (final dirPath in searchDirs) {
      try {
        final dir = Directory(dirPath);
        if (!await dir.exists()) continue;

        await for (final entity in dir.list(recursive: true)) {
          if (entity is! File || !entity.path.toLowerCase().endsWith('.pdf')) {
            continue;
          }

          final fileName = p.basename(entity.path).toLowerCase();
          final normalizedSubject =
              StudyCatalog.normalizeSubject(subject).toLowerCase();

          bool matches = fileName.contains(normalizedSubject) ||
              fileName.contains('paper') ||
              fileName.contains('question') ||
              fileName.contains('exam');

          if (!matches) continue;

          final msPath = _findMarkingScheme(entity.path);

          if (msPath != null) {
            papers.add(_PastPaperData(
              paperPath: entity.path,
              paperName: p.basename(entity.path),
              markingSchemePath: msPath,
            ));
          }
        }
      } catch (e) {
        // Skip inaccessible directories
      }
    }

    return papers;
  }

  String? _findMarkingScheme(String paperPath) {
    final baseName = p
        .basename(paperPath)
        .replaceAll(
            RegExp(
              r'\.pdf\$',
            ),
            '')
        .toLowerCase();

    final dir = Directory(p.dirname(paperPath));
    if (!dir.existsSync()) return null;

    final patterns = [
      '${baseName}_ms.pdf',
      '${baseName}_marking.pdf',
      '${baseName}_answers.pdf',
      '${baseName}_solution.pdf',
      '${baseName}_ms',
      '${baseName}_solution',
    ];

    for (final pattern in patterns) {
      final msPath = p.join(p.dirname(paperPath), pattern);
      if (File(msPath).existsSync()) return msPath;
    }

    final allFiles = dir.listSync();
    for (final f in allFiles) {
      if (f is File) {
        final name = p.basename(f.path).toLowerCase();
        if (name.contains(baseName) &&
            (name.contains('ms') ||
                name.contains('marking') ||
                name.contains('solution') ||
                name.contains('answer'))) {
          return f.path;
        }
      }
    }

    return null;
  }

  Future<Map<String, dynamic>?> _processPaper({
    required String paperPath,
    required String markingSchemePath,
  }) async {
    final paperText = await _pdfService.extractText(paperPath);
    final msText = await _pdfService.extractText(markingSchemePath);

    if (paperText.isEmpty || msText.isEmpty) return null;

    final questions = _extractQuestions(paperText);
    final answers = _extractAnswers(msText);

    final questionsData = <Map<String, dynamic>>[];
    final criteria = <String, dynamic>{};

    for (int i = 0; i < questions.length && i < answers.length; i++) {
      final qData = _parseQuestion(questions[i]);
      final aData = _parseAnswer(answers[i]);

      questionsData.add({
        'question_number': i + 1,
        'question_text': qData['text'],
        'question_type': qData['type'],
        'marks': qData['marks'],
        'answer_text': aData['text'],
        'key_points': aData['keyPoints'],
      });

      criteria['q${i + 1}'] = {
        'total_marks': qData['marks'],
        'key_points': aData['keyPoints'],
      };
    }

    return {
      'questions': questionsData,
      'criteria': criteria,
    };
  }

  List<String> _extractQuestions(String text) {
    final questions = <String>[];
    final lines = text.split('\n');
    String current = '';
    bool inQ = false;

    for (final line in lines) {
      final trimmed = line.trim();
      if (RegExp(r'^\d+[\.)]', caseSensitive: false).hasMatch(trimmed)) {
        if (current.isNotEmpty) {
          questions.add(current.trim());
        }
        current = line;
        inQ = true;
      } else if (inQ && trimmed.isNotEmpty) {
        current += ' $line';
      }
    }
    if (current.isNotEmpty) {
      questions.add(current.trim());
    }
    return questions;
  }

  List<String> _extractAnswers(String text) {
    final answers = <String>[];
    final lines = text.split('\n');
    String current = '';
    bool inA = false;

    for (final line in lines) {
      final trimmed = line.trim();
      if (RegExp(r'^\d+[\.)]', caseSensitive: false).hasMatch(trimmed)) {
        if (current.isNotEmpty) {
          answers.add(current.trim());
        }
        current = '';
        inA = true;
      } else if (inA && trimmed.isNotEmpty) {
        current += ' $line';
      }
    }
    if (current.isNotEmpty) {
      answers.add(current.trim());
    }
    return answers;
  }

  Map<String, dynamic> _parseQuestion(String question) {
    final marksMatch = RegExp(r'\((\d+)\s*m|(\d+)\s*m', caseSensitive: false)
        .firstMatch(question);
    final marks = marksMatch != null
        ? int.tryParse(marksMatch.group(1) ?? marksMatch.group(2) ?? '0') ?? 0
        : 5;

    String type = 'short_answer';
    if (question.length > 200 || marks > 10) {
      type = 'long_answer';
    } else if (question.contains('explain') || question.contains('describe')) {
      type = 'explanation';
    }

    return {'text': question, 'marks': marks, 'type': type};
  }

  Map<String, dynamic> _parseAnswer(String answer) {
    final sentences = answer
        .split(RegExp(r'[.!?]'))
        .where((s) => s.trim().length > 10)
        .take(5)
        .map((s) => s.trim())
        .toList();

    return {
      'text': answer,
      'keyPoints': sentences,
    };
  }

  Map<String, dynamic> _buildModelConfig({
    required String board,
    required String subject,
    required List<Map<String, dynamic>> questions,
    required Map<String, dynamic> criteria,
    required int totalPapers,
  }) {
    final systemPrompt = '''
You are an expert examiner for $board $subject.
You have been trained on $totalPapers past exam papers and their official marking schemes.

Your expertise:
1. Evaluate answers according to official marking schemes
2. Identify key points that earn marks
3. Provide constructive feedback
4. Suggest exam technique improvements

When answering student questions:
- Provide accurate, exam-focused answers
- Highlight key marking points
- Be encouraging but precise about improvements
''';

    final trainingExamples = questions
        .take(100)
        .map((q) => {
              'question': q['question_text'],
              'answer': q['answer_text'],
              'key_points': q['key_points'],
              'marks': q['marks'],
            })
        .toList();

    return {
      'version': '1.0',
      'board': board,
      'subject': subject,
      'trained_at': DateTime.now().toIso8601String(),
      'training_samples': questions.length,
      'total_papers': totalPapers,
      'system_prompt': systemPrompt,
      'training_examples': trainingExamples,
      'marking_criteria': criteria,
    };
  }

  String getPromptForQuestion(String userQuestion) {
    final config = _cachedConfig;
    final subject = config?['subject'] ?? 'your subject';
    final board = config?['board'] ?? 'your exam';

    return '''
You are an expert $subject examiner for $board exams.
You have been trained on past paper marking schemes.

USER QUESTION:
$userQuestion

Provide:
1. A clear, accurate answer
2. Key points that earn marks
3. Common mistakes to avoid
4. Exam tip for this question type
''';
  }

  Future<String> askTrainedQuestion(String question) async {
    final prompt = getPromptForQuestion(question);
    return await GrokService().chat(prompt);
  }
}

class _PastPaperData {
  final String paperPath;
  final String paperName;
  final String markingSchemePath;

  _PastPaperData({
    required this.paperPath,
    required this.paperName,
    required this.markingSchemePath,
  });
}
