import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';
import 'study_catalog.dart';

class PastPaperTrainer {
  static const String _trainingProgressKey = 'axon_training_progress';

  Future<void> trainFromPastPapers({
    required String board,
    required String subject,
    required String outputPath,
    bool verbose = true,
  }) async {
    if (verbose) {
      print('📚 PAST PAPER MODEL TRAINER');
      print('=' * 50);
      print('Board: $board | Subject: $subject');
      print('=' * 50);
    }

    final allTrainingData = <Map<String, dynamic>>[];
    final allMarkingCriteria = <String, dynamic>{};
    final questionPatterns = <Map<String, dynamic>>[];

    final papers = await _findPastPapersWithMarkingSchemes(
      board: board,
      subject: subject,
      verbose: verbose,
    );

    if (papers.isEmpty) {
      print('❌ No past papers with marking schemes found!');
      return;
    }

    if (verbose) {
      print('\n🔬 Processing ${papers.length} paper-marking scheme pairs...\n');
    }

    for (int i = 0; i < papers.length; i++) {
      final paper = papers[i];
      if (verbose) {
        print('📄 [${i + 1}/${papers.length}] ${paper.paperName}');
      }

      try {
        final data = await _processPaperAndMarkingScheme(
          paper: paper.paperPath,
          markingScheme: paper.markingSchemePath!,
          paperName: paper.paperName,
          verbose: verbose,
        );

        if (data != null) {
          allTrainingData
              .addAll(data['questions'] as List<Map<String, dynamic>>);
          allMarkingCriteria.addAll(data['criteria'] as Map<String, dynamic>);
          questionPatterns
              .addAll(data['patterns'] as List<Map<String, dynamic>>);

          if (verbose) {
            print('  ✅ Extracted ${data['questions'].length} Q&A pairs');
          }
        }
      } catch (e) {
        if (verbose) {
          print('  ❌ Error: $e');
        }
      }
    }

    if (allTrainingData.isEmpty) {
      print('❌ No training data extracted!');
      return;
    }

    if (verbose) {
      print('\n📊 Training Data Summary:');
      print('  • Total Q&A pairs: ${allTrainingData.length}');
      print('  • Marking criteria: ${allMarkingCriteria.length}');
      print('  • Question patterns: ${questionPatterns.length}');
    }

    if (verbose) {
      print('\n🧠 Training Gemini model...');
    }

    final modelConfig = await _generateModelConfig(
      trainingData: allTrainingData,
      markingCriteria: allMarkingCriteria,
      questionPatterns: questionPatterns,
      board: board,
      subject: subject,
      verbose: verbose,
    );

    await _saveModelConfig(modelConfig, outputPath);

    await _saveTrainingProgress(
      papersProcessed: papers.length,
      questionsExtracted: allTrainingData.length,
      board: board,
      subject: subject,
    );

    if (verbose) {
      print('\n✅ TRAINING COMPLETE!');
      print('📁 Model config saved to: $outputPath');
      print('\nTo use in app:');
      print('  1. Load the trained config');
      print('  2. Use it in GeminiService for exam responses');
    }
  }

  Future<List<_PaperWithMarkingScheme>> _findPastPapersWithMarkingSchemes({
    required String board,
    required String subject,
    required bool verbose,
  }) async {
    final papers = <_PaperWithMarkingScheme>[];

    final searchDirs = [
      '/storage/emulated/0/Axon/PastPapers',
      '/storage/emulated/0/Download',
    ];

    for (final dirPath in searchDirs) {
      try {
        final dir = Directory(dirPath);
        if (!await dir.exists()) continue;

        await for (final entity in dir.list(recursive: true)) {
          if (entity is! File) continue;
          if (!entity.path.toLowerCase().endsWith('.pdf')) continue;

          final fileName = p.basename(entity.path).toLowerCase();
          final normalizedSubject =
              StudyCatalog.normalizeSubject(subject).toLowerCase();

          bool matches = fileName.contains(normalizedSubject) ||
              fileName.contains('paper') ||
              fileName.contains('question');

          if (!matches) continue;

          final markingScheme = _findCorrespondingMarkingScheme(entity.path);

          papers.add(_PaperWithMarkingScheme(
            paperPath: entity.path,
            paperName: p.basename(entity.path),
            markingSchemePath: markingScheme,
          ));
        }
      } catch (e) {
        if (verbose) {
          print('⚠️ Error scanning $dirPath: $e');
        }
      }
    }

    return papers.where((p) => p.markingSchemePath != null).toList();
  }

  String? _findCorrespondingMarkingScheme(String paperPath) {
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

    final possibleMarkingSchemes = [
      '${baseName}_ms.pdf',
      '${baseName}_marking.pdf',
      '${baseName}_answers.pdf',
      '${baseName}_solution.pdf',
      baseName.replaceAll('paper', 'ms'),
      baseName.replaceAll('question', 'ms'),
      baseName.replaceAll('paper', 'solution'),
      baseName.replaceAll('question', 'solution'),
    ];

    for (final msName in possibleMarkingSchemes) {
      final msPath = p.join(p.dirname(paperPath), msName);
      if (File(msPath).existsSync()) {
        return msPath;
      }
    }

    return null;
  }

  Future<Map<String, dynamic>?> _processPaperAndMarkingScheme({
    required String paper,
    required String markingScheme,
    required String paperName,
    required bool verbose,
  }) async {
    // PDF text extraction moved to native - stub for now
    final paperText = '';
    final msText = '';

    if (paperText.isEmpty || msText.isEmpty) {
      debugPrint(
          'PastPaperTrainer: PDF extraction not available yet. Use the PDF viewer instead.');
      return null;
    }

    final questions = _extractQuestions(paperText);
    final answers = _extractAnswers(msText);

    final questionsData = <Map<String, dynamic>>[];
    final criteria = <String, dynamic>{};
    final patterns = <Map<String, dynamic>>[];

    for (int i = 0; i < questions.length && i < answers.length; i++) {
      final question = questions[i];
      final answer = answers[i];

      final qData = _extractQuestionData(question);
      final aData = _extractAnswerData(answer);

      questionsData.add({
        'question_number': i + 1,
        'question_text': qData['text'],
        'question_type': qData['type'],
        'marks': qData['marks'],
        'answer_text': aData['text'],
        'key_points': aData['keyPoints'],
        'marking_guidance': aData['guidance'],
      });

      criteria['q${i + 1}'] = {
        'total_marks': qData['marks'],
        'key_points': aData['keyPoints'],
        'common_mistakes': aData['commonMistakes'],
      };

      patterns.add({
        'type': qData['type'],
        'marks': qData['marks'],
        'difficulty': _estimateDifficulty(qData['marks'], qData['text']),
      });
    }

    return {
      'questions': questionsData,
      'criteria': criteria,
      'patterns': patterns,
    };
  }

  List<String> _extractQuestions(String text) {
    final questions = <String>[];
    final lines = text.split('\n');
    String current = '';
    bool inQuestion = false;

    for (final line in lines) {
      final trimmed = line.trim();
      if (RegExp(r'^\d+[\.)]', caseSensitive: false).hasMatch(trimmed)) {
        if (current.isNotEmpty) {
          questions.add(current.trim());
        }
        current = line;
        inQuestion = true;
      } else if (inQuestion && trimmed.isNotEmpty) {
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
    bool inAnswer = false;

    for (final line in lines) {
      final trimmed = line.trim();
      if (RegExp(r'^\d+[\.)]', caseSensitive: false).hasMatch(trimmed)) {
        if (current.isNotEmpty) {
          answers.add(current.trim());
        }
        current = '';
        inAnswer = true;
      } else if (inAnswer && trimmed.isNotEmpty) {
        current += ' $line';
      }
    }
    if (current.isNotEmpty) {
      answers.add(current.trim());
    }
    return answers;
  }

  Map<String, dynamic> _extractQuestionData(String question) {
    final marksMatch =
        RegExp(r'\((\d+)\s*marks?\)|(\d+)\s*marks?', caseSensitive: false)
            .firstMatch(question);
    final marks = marksMatch != null
        ? int.tryParse(marksMatch.group(1) ?? marksMatch.group(2) ?? '0') ?? 0
        : 0;

    String type = 'short_answer';
    if (question.length > 200 || marks > 10) {
      type = 'long_answer';
    } else if (question.contains('explain') || question.contains('describe')) {
      type = 'explanation';
    } else if (question.contains('define')) {
      type = 'definition';
    }

    return {
      'text': question,
      'marks': marks,
      'type': type,
    };
  }

  Map<String, dynamic> _extractAnswerData(String answer) {
    final sentences = answer
        .split(RegExp(r'[.!?]'))
        .where((s) => s.trim().isNotEmpty)
        .toList();

    return {
      'text': answer,
      'keyPoints': sentences.take(5).map((s) => s.trim()).toList(),
      'guidance': 'Follow the marking scheme key points',
      'commonMistakes': [
        'Missing key points',
        'Incomplete explanations',
        'Incorrect formula usage',
      ],
    };
  }

  String _estimateDifficulty(int marks, String question) {
    if (marks >= 10) return 'hard';
    if (marks >= 5) return 'medium';
    return 'easy';
  }

  Future<Map<String, dynamic>> _generateModelConfig({
    required List<Map<String, dynamic>> trainingData,
    required Map<String, dynamic> markingCriteria,
    required List<Map<String, dynamic>> questionPatterns,
    required String board,
    required String subject,
    required bool verbose,
  }) async {
    final systemPrompt = _buildSystemPrompt(
      board: board,
      subject: subject,
      markingCriteria: markingCriteria,
      questionPatterns: questionPatterns,
    );

    final trainingExamples = trainingData
        .take(50)
        .map((d) => {
              'question': d['question_text'],
              'answer': d['answer_text'],
              'key_points': d['key_points'],
              'marks': d['marks'],
            })
        .toList();

    return {
      'version': '1.0',
      'board': board,
      'subject': subject,
      'trained_at': DateTime.now().toIso8601String(),
      'training_samples': trainingData.length,
      'system_prompt': systemPrompt,
      'training_examples': trainingExamples,
      'marking_criteria': markingCriteria,
      'question_patterns': questionPatterns,
    };
  }

  String _buildSystemPrompt({
    required String board,
    required String subject,
    required Map<String, dynamic> markingCriteria,
    required List<Map<String, dynamic>> questionPatterns,
  }) {
    final buffer = StringBuffer();

    buffer.writeln('You are an expert examiner for $board $subject.');
    buffer.writeln(
        'You have been trained on past exam papers and marking schemes.');
    buffer.writeln('');
    buffer.writeln('Your expertise includes:');
    buffer
        .writeln('1. Evaluating answers according to official marking schemes');
    buffer.writeln('2. Identifying key points that earn marks');
    buffer
        .writeln('3. Providing constructive feedback based on common mistakes');
    buffer.writeln('4. Suggesting improvements for better scores');
    buffer.writeln('');

    final avgMarks = questionPatterns.isEmpty
        ? 5
        : questionPatterns
                .map((p) => p['marks'] as int)
                .reduce((a, b) => a + b) ~/
            questionPatterns.length;

    buffer.writeln('Question patterns in this exam:');
    buffer.writeln('• Average marks per question: $avgMarks');
    buffer.writeln(
        '• Question types: ${questionPatterns.map((p) => p['type']).toSet().join(", ")}');
    buffer.writeln('');

    buffer.writeln('When responding to student questions:');
    buffer.writeln('1. Provide accurate, exam-focused answers');
    buffer.writeln('2. Highlight key marking points');
    buffer.writeln('3. Suggest exam technique improvements');
    buffer.writeln('4. Be encouraging but precise about areas needing work');
    buffer.writeln('');

    buffer.writeln('Use this marking criteria as reference:');
    for (final entry in markingCriteria.entries.take(10)) {
      buffer.writeln('  - Q${entry.key}: ${entry.value['total_marks']} marks');
    }

    return buffer.toString();
  }

  Future<void> _saveModelConfig(
      Map<String, dynamic> config, String outputPath) async {
    final file = File(outputPath);
    await file.parent.create(recursive: true);
    await file.writeAsString(jsonEncode(config));
  }

  Future<void> _saveTrainingProgress({
    required int papersProcessed,
    required int questionsExtracted,
    required String board,
    required String subject,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        _trainingProgressKey,
        jsonEncode({
          'papers_processed': papersProcessed,
          'questions_extracted': questionsExtracted,
          'board': board,
          'subject': subject,
          'trained_at': DateTime.now().toIso8601String(),
        }));
  }

  Future<TrainedModelConfig?> loadTrainedConfig(String path) async {
    try {
      final file = File(path);
      if (!await file.exists()) return null;

      final content = await file.readAsString();
      final data = jsonDecode(content) as Map<String, dynamic>;

      return TrainedModelConfig(
        version: data['version'] ?? '1.0',
        board: data['board'] ?? '',
        subject: data['subject'] ?? '',
        trainedAt:
            DateTime.tryParse(data['trained_at'] ?? '') ?? DateTime.now(),
        trainingSamples: data['training_samples'] ?? 0,
        systemPrompt: data['system_prompt'] ?? '',
        trainingExamples:
            List<Map<String, dynamic>>.from(data['training_examples'] ?? []),
        markingCriteria:
            Map<String, dynamic>.from(data['marking_criteria'] ?? {}),
        questionPatterns:
            List<Map<String, dynamic>>.from(data['question_patterns'] ?? []),
      );
    } catch (e) {
      print('Error loading config: $e');
      return null;
    }
  }

  String getSystemPromptForExam(TrainedModelConfig config, String question) {
    final buffer = StringBuffer();

    buffer.writeln(config.systemPrompt);
    buffer.writeln('');
    buffer.writeln('CURRENT QUESTION:');
    buffer.writeln(question);
    buffer.writeln('');
    buffer.writeln('Provide an answer as if you are an examiner. Include:');
    buffer.writeln('1. The answer');
    buffer.writeln('2. Key points that would earn marks');
    buffer.writeln('3. Common mistakes to avoid');

    return buffer.toString();
  }

  Future<void> continueTraining({
    required String board,
    required String subject,
    required String configPath,
    bool verbose = true,
  }) async {
    final existingConfig = await loadTrainedConfig(configPath);
    if (existingConfig == null) {
      print('❌ No existing config found. Run initial training first.');
      return;
    }

    if (verbose) {
      print('📚 Continuing training from existing config...');
      print('Current samples: ${existingConfig.trainingSamples}');
    }

    final newPapers = await _findPastPapersWithMarkingSchemes(
      board: board,
      subject: subject,
      verbose: verbose,
    );

    final newData = <Map<String, dynamic>>[];
    for (final paper in newPapers) {
      final data = await _processPaperAndMarkingScheme(
        paper: paper.paperPath,
        markingScheme: paper.markingSchemePath!,
        paperName: paper.paperName,
        verbose: verbose,
      );
      if (data != null) {
        newData.addAll(data['questions'] as List<Map<String, dynamic>>);
      }
    }

    if (newData.isEmpty) {
      print('⚠️ No new data found to add.');
      return;
    }

    final updatedExamples = [
      ...existingConfig.trainingExamples,
      ...newData.take(30).map((d) => {
            'question': d['question_text'],
            'answer': d['answer_text'],
            'key_points': d['key_points'],
            'marks': d['marks'],
          }),
    ];

    final updatedConfig = {
      ...existingConfig.toJson(),
      'training_samples': existingConfig.trainingSamples + newData.length,
      'training_examples': updatedExamples,
      'last_updated': DateTime.now().toIso8601String(),
    };

    await _saveModelConfig(updatedConfig, configPath);

    if (verbose) {
      print('✅ Training continued!');
      print('Total samples: ${updatedConfig['training_samples']}');
    }
  }
}

class _PaperWithMarkingScheme {
  final String paperPath;
  final String paperName;
  final String? markingSchemePath;

  _PaperWithMarkingScheme({
    required this.paperPath,
    required this.paperName,
    this.markingSchemePath,
  });
}

class TrainedModelConfig {
  final String version;
  final String board;
  final String subject;
  final DateTime trainedAt;
  final int trainingSamples;
  final String systemPrompt;
  final List<Map<String, dynamic>> trainingExamples;
  final Map<String, dynamic> markingCriteria;
  final List<Map<String, dynamic>> questionPatterns;

  TrainedModelConfig({
    required this.version,
    required this.board,
    required this.subject,
    required this.trainedAt,
    required this.trainingSamples,
    required this.systemPrompt,
    required this.trainingExamples,
    required this.markingCriteria,
    required this.questionPatterns,
  });

  Map<String, dynamic> toJson() => {
        'version': version,
        'board': board,
        'subject': subject,
        'trained_at': trainedAt.toIso8601String(),
        'training_samples': trainingSamples,
        'system_prompt': systemPrompt,
        'training_examples': trainingExamples,
        'marking_criteria': markingCriteria,
        'question_patterns': questionPatterns,
      };
}
