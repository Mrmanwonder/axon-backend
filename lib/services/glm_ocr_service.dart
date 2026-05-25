import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

class GLMOcrService {
  GLMOcrService._();
  static final GLMOcrService instance = GLMOcrService._();

  static const String _modelDirName = 'glm_ocr_models';
  String? _modelPath;

  Future<String> get _modelDirectory async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.parent.path}/$_modelDirName');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir.path;
  }

  Future<void> initialize() async {
    _modelPath = await _modelDirectory;
    debugPrint('GLM OCR initialized at: $_modelPath');
  }

  Future<String> extractTextFromPdf(String pdfPath) async {
    if (_modelPath == null) await initialize();

    try {
      final result = await Process.run(
        'python',
        [
          '-c',
          '''
import fitz
doc = fitz.open(r"$pdfPath")
text = "\\n".join([page.get_text() for page in doc])
print(text[:50000])
doc.close()
''',
        ],
        runInShell: true,
      );

      if (result.exitCode == 0 && result.stdout.toString().isNotEmpty) {
        return result.stdout.toString();
      }
    } catch (e) {
      debugPrint('PDF extraction failed: $e');
    }

    return '';
  }

  Future<List<Map<String, dynamic>>> extractQuestions(String pdfPath) async {
    final text = await extractTextFromPdf(pdfPath);
    if (text.isEmpty) return [];

    return _parseQuestionsFromText(text);
  }

  List<Map<String, dynamic>> _parseQuestionsFromText(String text) {
    final questions = <Map<String, dynamic>>[];
    final lines = text.split('\n');

    final questionPattern = RegExp(r'^(\d+)\.?\s*(.+)$');
    final partPattern = RegExp(r'^\(([a-z])\)\s*(.+)$');

    int? currentQuestionNum;
    String? currentQuestionText;
    List<Map<String, String>> currentParts = [];

    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;

      final qMatch = questionPattern.firstMatch(trimmed);
      if (qMatch != null) {
        if (currentQuestionNum != null) {
          questions.add({
            'question_number': currentQuestionNum,
            'question_text': currentQuestionText ?? '',
            'parts': currentParts,
          });
        }

        currentQuestionNum = int.parse(qMatch.group(1)!);
        currentQuestionText = qMatch.group(2);
        currentParts = [];
        continue;
      }

      final partMatch = partPattern.firstMatch(trimmed);
      if (partMatch != null && currentQuestionNum != null) {
        currentParts.add({
          'label': partMatch.group(1)!,
          'text': partMatch.group(2) ?? '',
        });
      }
    }

    if (currentQuestionNum != null) {
      questions.add({
        'question_number': currentQuestionNum,
        'question_text': currentQuestionText ?? '',
        'parts': currentParts,
      });
    }

    return questions;
  }

  Future<List<Map<String, dynamic>>> extractDatesheetEvents(
      String pdfPath) async {
    final text = await extractTextFromPdf(pdfPath);
    if (text.isEmpty) return [];

    return _parseDatesheetFromText(text);
  }

  List<Map<String, dynamic>> _parseDatesheetFromText(String text) {
    final events = <Map<String, dynamic>>[];
    final lines = text.split('\n');

    final pattern = RegExp(
      r'^([A-Za-z\s]+(?:Primary|Lower Secondary)?)\s+(\d{4}/\d{2})\s+(.+?)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\s*(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
      caseSensitive: false,
    );

    for (final line in lines) {
      final match = pattern.firstMatch(line.trim());
      if (match != null) {
        final subject = match.group(1)?.trim() ?? '';
        final component = match.group(2) ?? '';
        final duration = match.group(3) ?? '';
        final day = int.tryParse(match.group(4) ?? '') ?? 1;
        final monthStr = match.group(5) ?? 'January';
        final year = int.tryParse(match.group(6) ?? '') ?? DateTime.now().year;

        final monthMap = {
          'january': 1,
          'february': 2,
          'march': 3,
          'april': 4,
          'may': 5,
          'june': 6,
          'july': 7,
          'august': 8,
          'september': 9,
          'october': 10,
          'november': 11,
          'december': 12,
        };

        final month = monthMap[monthStr.toLowerCase()] ?? 1;
        final date = DateTime(year, month, day);
        final times = _inferTimes(duration);

        events.add({
          'subject': subject,
          'component': component,
          'date': date.toIso8601String().split('T').first,
          'start_time': times['start'],
          'end_time': times['end'],
          'board': 'Cambridge',
        });
      }
    }

    return events;
  }

  Map<String, String> _inferTimes(String duration) {
    final hourMatch =
        RegExp(r'(\d+)\s*hour').firstMatch(duration.toLowerCase());
    final minMatch =
        RegExp(r'(\d+)\s*minute').firstMatch(duration.toLowerCase());

    int totalMinutes = 0;
    if (hourMatch != null) {
      totalMinutes += int.parse(hourMatch.group(1)!) * 60;
    }
    if (minMatch != null) {
      totalMinutes += int.parse(minMatch.group(1)!);
    }

    if (totalMinutes == 0) totalMinutes = 60;

    final endHour = 8 + totalMinutes ~/ 60;
    final endMin = totalMinutes % 60;

    return {
      'start': '08:00',
      'end':
          '$endHour:${endMin == 0 ? '00' : endMin.toString().padLeft(2, '0')}',
    };
  }

  Future<bool> isModelAvailable() async {
    final dir = await _modelDirectory;
    final modelDir = Directory('$dir/finetuned');
    return await modelDir.exists();
  }

  Future<String?> downloadModel() async {
    debugPrint('Model download not implemented - using fallback extraction');
    return null;
  }
}
