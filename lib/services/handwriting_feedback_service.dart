import 'dart:convert';
import 'dart:io';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;

import '../models/models.dart';
import 'cloudinary_service.dart';

class HandwritingGradeOutcome {
  final double score;
  final double availableMarks;
  final String feedback;
  final List<Map<String, dynamic>> marksAwarded;
  final List<Map<String, dynamic>> marksMissed;
  final List<Map<String, dynamic>> learningObjectiveGaps;
  final Map<String, dynamic> commandWordDepth;
  final Map<String, dynamic> semanticMatch;
  final String sourceType;
  final String imageUrl;

  const HandwritingGradeOutcome({
    required this.score,
    required this.availableMarks,
    required this.feedback,
    required this.marksAwarded,
    required this.marksMissed,
    required this.learningObjectiveGaps,
    required this.commandWordDepth,
    required this.semanticMatch,
    required this.sourceType,
    required this.imageUrl,
  });
}

class HandwritingFeedbackService {
  HandwritingFeedbackService({http.Client? client})
      : _client = client ?? http.Client();

  static const String _backendUrl = 'https://axon-ml.onrender.com';
  final http.Client _client;
  final CloudinaryService _cloudinary = CloudinaryService();

  Future<HandwritingGradeOutcome> gradeQuestion({
    required PdfQuestion question,
    required File imageFile,
    required String objective,
    required String commandWord,
    required List<String> learningObjectiveIds,
    required String markingSchemeText,
    bool archiveAfterGrading = false,
  }) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      throw StateError('No authenticated user');
    }

    final token = await user.getIdToken();
    if (token == null || token.isEmpty) {
      throw StateError('Authentication token unavailable');
    }

    final upload = await _cloudinary.uploadStudyArtifact(imageFile);
    if (upload == null) {
      throw StateError('Image upload failed');
    }

    final response = await _client.post(
      Uri.parse('$_backendUrl/generate/grade-handwriting'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode({
        'image_url': upload['secure_url'],
        'cloudinary_public_id': upload['public_id'],
        'marking_scheme_text': markingSchemeText,
        'marking_scheme': {
          'raw_text': markingSchemeText,
          'question_number': question.questionNumber,
          'available_marks': question.marksAvailable ?? 0,
        },
        'objective': objective,
        'question_prompt': question.fullText,
        'learning_objective_ids': learningObjectiveIds,
        'command_word': commandWord,
        'spatial_layout': question.spatialMetadata,
        'archive_after_grading': archiveAfterGrading,
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
        'Handwriting grading failed (${response.statusCode})',
      );
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    final result = Map<String, dynamic>.from(decoded['result'] as Map? ?? const {});

    return HandwritingGradeOutcome(
      score: (result['score'] as num?)?.toDouble() ?? 0,
      availableMarks: (result['available_marks'] as num?)?.toDouble() ??
          (question.marksAvailable ?? 0).toDouble(),
      feedback: (result['feedback'] ?? '').toString(),
      marksAwarded: (result['marks_awarded'] as List? ?? const [])
          .whereType<Map>()
          .map((item) => Map<String, dynamic>.from(item))
          .toList(),
      marksMissed: (result['marks_missed'] as List? ?? const [])
          .whereType<Map>()
          .map((item) => Map<String, dynamic>.from(item))
          .toList(),
      learningObjectiveGaps:
          (result['learning_objective_gaps'] as List? ?? const [])
              .whereType<Map>()
              .map((item) => Map<String, dynamic>.from(item))
              .toList(),
      commandWordDepth:
          Map<String, dynamic>.from(result['command_word_depth'] as Map? ?? const {}),
      semanticMatch:
          Map<String, dynamic>.from(result['semantic_match'] as Map? ?? const {}),
      sourceType: (result['source_type'] ?? '').toString(),
      imageUrl: (upload['secure_url'] ?? '').toString(),
    );
  }
}
