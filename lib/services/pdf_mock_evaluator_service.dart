import 'dart:io';
import 'dart:convert';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;

import '../models/models.dart';

class PdfMockGradeOutcome {
  final String itemId;
  final bool isCompleted;
  final double awardedMarks;
  final double availableMarks;
  final String feedback;
  final String errorType;
  final List<String> marksAwarded;
  final List<String> marksMissed;
  final Map<String, dynamic> commandWordDepth;
  final Map<String, dynamic> semanticMatch;
  final String? error;

  const PdfMockGradeOutcome({
    required this.itemId,
    required this.isCompleted,
    required this.awardedMarks,
    required this.availableMarks,
    required this.feedback,
    required this.errorType,
    this.marksAwarded = const [],
    this.marksMissed = const [],
    this.commandWordDepth = const {},
    this.semanticMatch = const {},
    this.error,
  });
}

class PdfMockEvaluationRequest {
  final String itemId;
  final PdfQuestion question;
  final String studentAnswer;
  final String markingSchemeText;
  final String objective;
  final List<String> learningObjectiveIds;
  final String commandWord;
  final double availableMarks;

  const PdfMockEvaluationRequest({
    required this.itemId,
    required this.question,
    required this.studentAnswer,
    required this.markingSchemeText,
    required this.objective,
    required this.learningObjectiveIds,
    required this.commandWord,
    required this.availableMarks,
  });
}

class MarkSchemeLayoutPage {
  final int pageNumber;
  final double width;
  final double height;
  final List<Map<String, dynamic>> lines;

  const MarkSchemeLayoutPage({
    required this.pageNumber,
    required this.width,
    required this.height,
    required this.lines,
  });

  factory MarkSchemeLayoutPage.fromJson(Map<String, dynamic> json) {
    return MarkSchemeLayoutPage(
      pageNumber: (json['page_number'] as num?)?.toInt() ?? 0,
      width: (json['width'] as num?)?.toDouble() ?? 0,
      height: (json['height'] as num?)?.toDouble() ?? 0,
      lines: (json['lines'] as List? ?? const [])
          .whereType<Map>()
          .map((item) => Map<String, dynamic>.from(item))
          .toList(),
    );
  }

  Map<String, dynamic> toJson() => {
        'page_number': pageNumber,
        'width': width,
        'height': height,
        'lines': lines,
      };
}

class PdfMockEvaluatorService {
  PdfMockEvaluatorService({http.Client? client})
      : _client = client ?? http.Client();

  static const String _backendUrl = 'https://bhavu.up.railway.app';
  final http.Client _client;

  Future<List<MarkSchemeLayoutPage>> analyzeMarkSchemePdf(File file) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null || !await file.exists()) return const [];
    final token = await user.getIdToken();
    if (token == null || token.isEmpty) return const [];

    final response = await _client.post(
      Uri.parse('$_backendUrl/analyze-mark-scheme-pdf'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode({
        'filename': file.uri.pathSegments.isEmpty
            ? 'mark_scheme.pdf'
            : file.uri.pathSegments.last,
        'pdf_base64': base64Encode(await file.readAsBytes()),
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      return const [];
    }

    final decoded = jsonDecode(response.body);
    if (decoded is! Map<String, dynamic>) return const [];
    final pages = decoded['page_layouts'];
    if (pages is! List) return const [];
    return pages
        .whereType<Map>()
        .map((item) => MarkSchemeLayoutPage.fromJson(Map<String, dynamic>.from(item)))
        .toList();
  }

  Future<Map<String, String>> segmentMarkSchemeBatch({
    required String markSchemeText,
    required List<MarkSchemeLayoutPage> pageLayouts,
    required List<PdfQuestion> questions,
  }) async {
    if (questions.isEmpty) return const {};
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return const {};
    final token = await user.getIdToken();
    if (token == null || token.isEmpty) return const {};

    final response = await _client.post(
      Uri.parse('$_backendUrl/generate/segment-mark-scheme-batch'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode({
        'mark_scheme_text': markSchemeText,
        'page_layouts': pageLayouts.map((page) => page.toJson()).toList(),
        'questions': questions.map((question) {
          final partLabels = question.parts
              .map((part) => part.label.replaceAll(RegExp(r'[^a-zA-Z]'), '').toLowerCase())
              .where((label) => label.isNotEmpty)
              .toList();
          return {
            'item_id': '${question.pageNumber}:${question.questionNumber}',
            'question_number': question.questionNumber,
            'question_text': question.questionText,
            'page_number': question.pageNumber,
            'parts': partLabels,
            'spatial_metadata': question.spatialMetadata,
          };
        }).toList(),
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      return const {};
    }

    final decoded = jsonDecode(response.body);
    if (decoded is! Map<String, dynamic>) return const {};
    final results = decoded['results'];
    if (results is! List) return const {};
    return {
      for (final item in results.whereType<Map>())
        (item['item_id'] ?? '').toString(): (item['snippet'] ?? '').toString(),
    };
  }

  Future<List<PdfMockGradeOutcome>> gradeBatch(
    List<PdfMockEvaluationRequest> requests,
  ) async {
    if (requests.isEmpty) return const [];
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return const [];
    final token = await user.getIdToken();
    if (token == null || token.isEmpty) return const [];

    final response = await _client.post(
      Uri.parse('$_backendUrl/generate/grade-pdf-mock'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode({
        'items': requests
            .map(
              (request) => {
                'item_id': request.itemId,
                'question_prompt': request.question.fullText,
                'student_answer': request.studentAnswer,
                'marking_scheme_text': request.markingSchemeText,
                'objective': request.objective,
                'learning_objective_ids': request.learningObjectiveIds,
                'command_word': request.commandWord,
                'available_marks': request.availableMarks,
              },
            )
            .toList(),
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      return const [];
    }

    final decoded = jsonDecode(response.body);
    if (decoded is! Map<String, dynamic>) {
      return const [];
    }
    final results = decoded['results'];
    if (results is! List) {
      return const [];
    }

    return results.whereType<Map>().map((entry) {
      final map = Map<String, dynamic>.from(entry);
      final result = Map<String, dynamic>.from(map['result'] as Map? ?? const {});
      return PdfMockGradeOutcome(
        itemId: (map['item_id'] ?? '').toString(),
        isCompleted: (map['status'] ?? '') == 'completed',
        awardedMarks: (result['awarded_marks'] as num?)?.toDouble() ?? 0,
        availableMarks: (result['available_marks'] as num?)?.toDouble() ?? 0,
        feedback: (result['feedback'] ?? '').toString(),
        errorType: (result['error_type'] ?? 'none').toString(),
        marksAwarded: (result['marks_awarded'] as List? ?? const [])
            .map((value) => value.toString())
            .toList(),
        marksMissed: (result['marks_missed'] as List? ?? const [])
            .map((value) => value.toString())
            .toList(),
        commandWordDepth:
            Map<String, dynamic>.from(result['command_word_depth'] as Map? ?? const {}),
        semanticMatch:
            Map<String, dynamic>.from(result['semantic_match'] as Map? ?? const {}),
        error: map['error']?.toString(),
      );
    }).toList();
  }
}
