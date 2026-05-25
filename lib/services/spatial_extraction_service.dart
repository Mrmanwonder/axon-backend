import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/models.dart';
import 'advanced_pdf_service.dart';

class SpatialExtractionService {
  SpatialExtractionService({
    http.Client? client,
    AdvancedPdfService? advancedPdfService,
    String? endpoint,
  })  : _client = client ?? http.Client(),
        _advancedPdfService = advancedPdfService ?? AdvancedPdfService(),
        _endpoint = endpoint ??
            const String.fromEnvironment(
              'DEEPSEEK_VL2_ENDPOINT',
              defaultValue: '',
            );

  static const String deepSeekExamPrompt =
      'You are an Exam Parsing Expert. Analyze this image. Output a valid JSON array. '
      'Each object must include: q_num, text (Raw LaTeX), sub_parts (nested array), '
      'marks, and visual_bounds. Identify and crop any figures/graphs associated with '
      'the question. If text is scanned/blurry, use visual context to reconstruct '
      'mathematical symbols perfectly.';

  final http.Client _client;
  final AdvancedPdfService _advancedPdfService;
  final String _endpoint;

  Future<List<PdfQuestion>> extractQuestions(
    String filePath, {
    int maxPages = 12,
  }) async {
    final ocrPages = await _advancedPdfService.extractTextWithMlKit(
      filePath,
      maxPages: maxPages,
    );
    final pages = await _advancedPdfService.renderPagesAtDpi(
      filePath,
      maxPages: maxPages,
      dpi: 300,
    );
    final questions = <PdfQuestion>[];
    var runningNumber = 1;

    for (final page in pages) {
      final pageHint = ocrPages[page.pageNumber] ?? '';
      final expectedMarks = _estimateTotalMarksFromOcr(pageHint);
      final initialPayload = await _inferPage(
        imageBytes: page.bytes,
        pageNumber: page.pageNumber,
        prompt: deepSeekExamPrompt,
        ocrHint: pageHint,
      );
      final validatedPayload = await _validateAndRefine(
        page: page,
        ocrHint: pageHint,
        expectedMarks: expectedMarks,
        payload: initialPayload,
      );
      final pageQuestions = _toQuestions(
        validatedPayload,
        pageNumber: page.pageNumber,
        fallbackQuestionNumber: runningNumber,
      );
      if (pageQuestions.isEmpty) {
        continue;
      }
      runningNumber = pageQuestions.last.questionNumber + 1;
      questions.addAll(pageQuestions);
    }

    return questions;
  }

  Future<List<Map<String, dynamic>>> _validateAndRefine({
    required RenderedPdfPage page,
    required String ocrHint,
    required int expectedMarks,
    required List<Map<String, dynamic>> payload,
  }) async {
    final extractedMarks = _sumMarks(payload);
    if (expectedMarks <= 0 || extractedMarks == expectedMarks) {
      return payload;
    }

    final missingRegions = _findMissingRegions(payload);
    if (missingRegions.isEmpty) {
      return payload;
    }

    final focusPrompt = StringBuffer()
      ..writeln(deepSeekExamPrompt)
      ..writeln(
        'Focus Prompt: extracted marks = $extractedMarks, total marks = $expectedMarks.',
      )
      ..writeln(
        'Re-read only these missing coordinate bands and recover omitted marks or sub-parts:',
      )
      ..writeln(jsonEncode(missingRegions))
      ..writeln('Return a full corrected JSON array, not a diff.');

    final refined = await _inferPage(
      imageBytes: page.bytes,
      pageNumber: page.pageNumber,
      prompt: focusPrompt.toString(),
      ocrHint: ocrHint,
    );
    return _sumMarks(refined) >= extractedMarks ? refined : payload;
  }

  Future<List<Map<String, dynamic>>> _inferPage({
    required List<int> imageBytes,
    required int pageNumber,
    required String prompt,
    required String ocrHint,
  }) async {
    final response = await _client.post(
      Uri.parse(_endpoint),
      headers: const {'Content-Type': 'application/json'},
      body: jsonEncode({
        'page_number': pageNumber,
        'prompt': prompt,
        'ocr_hint': ocrHint,
        'image_base64': base64Encode(imageBytes),
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(
        'DeepSeek-VL2 request failed (${response.statusCode}): ${response.body}',
      );
    }

    final decoded = _decodeResponse(response.body);
    if (decoded is List) {
      return decoded.whereType<Map>().map(_normalizeMap).toList();
    }
    if (decoded is Map && decoded['questions'] is List) {
      return (decoded['questions'] as List)
          .whereType<Map>()
          .map(_normalizeMap)
          .toList();
    }
    return const [];
  }

  dynamic _decodeResponse(String body) {
    final trimmed = body.trim();
    if (trimmed.isEmpty) {
      return const [];
    }
    final fenced = RegExp(r'```(?:json)?\s*([\s\S]*?)```', multiLine: true)
        .firstMatch(trimmed);
    final candidate = fenced?.group(1)?.trim() ?? trimmed;
    return jsonDecode(candidate);
  }

  Map<String, dynamic> _normalizeMap(Map input) {
    return Map<String, dynamic>.from(
      input.map((key, value) => MapEntry(key.toString(), value)),
    );
  }

  int _estimateTotalMarksFromOcr(String text) {
    final matches = RegExp(r'\[(\d+)\]|\((\d+)\)\s*marks?', caseSensitive: false)
        .allMatches(text);
    return matches.fold<int>(0, (sum, match) {
      final value = int.tryParse(match.group(1) ?? match.group(2) ?? '');
      return sum + (value ?? 0);
    });
  }

  int _sumMarks(List<Map<String, dynamic>> payload) {
    return payload.fold<int>(0, (sum, item) => sum + _itemMarks(item));
  }

  int _itemMarks(Map<String, dynamic> item) {
    final marks = item['marks'];
    if (marks is num) {
      return marks.toInt();
    }
    if (marks is String) {
      return int.tryParse(marks.replaceAll(RegExp(r'[^0-9]'), '')) ?? 0;
    }
    final subParts = item['sub_parts'];
    if (subParts is List) {
      return subParts.whereType<Map>().fold<int>(
            0,
            (sum, part) => sum + _itemMarks(_normalizeMap(part)),
          );
    }
    return 0;
  }

  List<Map<String, dynamic>> _findMissingRegions(List<Map<String, dynamic>> data) {
    final boxes = data
        .map((item) => _boundsFrom(item['visual_bounds']))
        .whereType<List<double>>()
        .toList()
      ..sort((a, b) => a.first.compareTo(b.first));
    if (boxes.length < 2) {
      return const [];
    }

    final missing = <Map<String, dynamic>>[];
    for (var index = 0; index < boxes.length - 1; index++) {
      final current = boxes[index];
      final next = boxes[index + 1];
      final gap = next[0] - current[2];
      if (gap < 64) {
        continue;
      }
      missing.add({
        'ymin': current[2],
        'xmin': 0,
        'ymax': next[0],
        'xmax': 9999,
      });
    }
    return missing;
  }

  List<double>? _boundsFrom(dynamic raw) {
    if (raw is List && raw.length == 4) {
      return raw
          .map((value) => value is num ? value.toDouble() : 0.0)
          .toList(growable: false);
    }
    if (raw is Map) {
      return [
        _asDouble(raw['ymin']),
        _asDouble(raw['xmin']),
        _asDouble(raw['ymax']),
        _asDouble(raw['xmax']),
      ];
    }
    return null;
  }

  List<PdfQuestion> _toQuestions(
    List<Map<String, dynamic>> payload, {
    required int pageNumber,
    required int fallbackQuestionNumber,
  }) {
    var runningQuestionNumber = fallbackQuestionNumber;
    return payload.map((item) {
      final questionNumber =
          _asInt(item['q_num']) > 0 ? _asInt(item['q_num']) : runningQuestionNumber;
      runningQuestionNumber = questionNumber + 1;
      final parts = _toParts(item['sub_parts']);
      final marks = _itemMarks(item);
      final visualBounds = _boundsFrom(item['visual_bounds']) ?? const [0, 0, 0, 0];
      final figurePaths = (item['figure_crops'] as List? ?? const [])
          .map((value) => value.toString())
          .where((value) => value.isNotEmpty)
          .toList();
      return PdfQuestion(
        questionNumber: questionNumber,
        questionText: (item['text'] ?? '').toString().trim(),
        parts: parts,
        yPosition: visualBounds[0],
        xPosition: visualBounds[1],
        width: (visualBounds[3] - visualBounds[1]).clamp(0, double.infinity),
        height: (visualBounds[2] - visualBounds[0]).clamp(0, double.infinity),
        pageNumber: pageNumber,
        marksAvailable: marks > 0 ? marks : null,
        figurePaths: figurePaths,
        spatialMetadata: {
          'visual_bounds': {
            'ymin': visualBounds[0],
            'xmin': visualBounds[1],
            'ymax': visualBounds[2],
            'xmax': visualBounds[3],
          },
          'page_number': pageNumber,
        },
      );
    }).toList();
  }

  List<PdfPart> _toParts(dynamic raw) {
    if (raw is! List) {
      return const [];
    }
    return raw.whereType<Map>().map((part) {
      final normalized = _normalizeMap(part);
      return PdfPart(
        label: (normalized['label'] ?? normalized['q_num'] ?? '').toString(),
        text: (normalized['text'] ?? '').toString().trim(),
      );
    }).toList();
  }

  int _asInt(dynamic value) {
    if (value is int) {
      return value;
    }
    if (value is num) {
      return value.toInt();
    }
    return int.tryParse(value?.toString() ?? '') ?? 0;
  }

  double _asDouble(dynamic value) {
    if (value is double) {
      return value;
    }
    if (value is num) {
      return value.toDouble();
    }
    return double.tryParse(value?.toString() ?? '') ?? 0.0;
  }
}
