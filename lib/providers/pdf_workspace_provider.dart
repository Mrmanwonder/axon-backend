// lib/providers/pdf_workspace_provider.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/models.dart';

class PdfWorkspaceState {
  final String title;
  final List<PdfQuestion> questions;
  final Map<String, String> answers;
  final String markingSchemeText;
  final String filePath;

  const PdfWorkspaceState({
    required this.title,
    required this.questions,
    required this.answers,
    required this.markingSchemeText,
    required this.filePath,
  });

  PdfWorkspaceState copyWith({
    String? title,
    List<PdfQuestion>? questions,
    Map<String, String>? answers,
    String? markingSchemeText,
    String? filePath,
  }) {
    return PdfWorkspaceState(
      title: title ?? this.title,
      questions: questions ?? this.questions,
      answers: answers ?? this.answers,
      markingSchemeText: markingSchemeText ?? this.markingSchemeText,
      filePath: filePath ?? this.filePath,
    );
  }
}

final pdfWorkspaceProvider = AsyncNotifierProvider.family<
    PdfWorkspaceNotifier, PdfWorkspaceState, String>(
  PdfWorkspaceNotifier.new,
);

class PdfWorkspaceNotifier
    extends FamilyAsyncNotifier<PdfWorkspaceState, String> {
  @override
  PdfWorkspaceState build(String arg) {
    return PdfWorkspaceState(
      title: '',
      questions: const [],
      answers: const {},
      markingSchemeText: '',
      filePath: arg,
    );
  }

  void initialize({
    required String title,
    required List<PdfQuestion> questions,
    required Map<String, String> answers,
    required String? markingSchemeText,
  }) {
    final current = state.valueOrNull;
    final next = PdfWorkspaceState(
      title: title,
      questions: List<PdfQuestion>.from(questions),
      answers: Map<String, String>.from(answers),
      markingSchemeText: markingSchemeText ?? '',
      filePath: arg,
    );
    if (current != null &&
        current.title == next.title &&
        current.filePath == next.filePath &&
        current.questions.length == next.questions.length &&
        current.answers.length == next.answers.length &&
        current.markingSchemeText == next.markingSchemeText) {
      return;
    }
    state = AsyncData(next);
  }

  void updateAnswer(String key, String value) {
    final current = state.valueOrNull;
    if (current == null) return;
    final updatedAnswers = Map<String, String>.from(current.answers)
      ..[key] = value;
    state = AsyncData(current.copyWith(answers: updatedAnswers));
  }

  void updateMarkingScheme(String text) {
    final current = state.valueOrNull;
    if (current == null) return;
    state = AsyncData(current.copyWith(markingSchemeText: text));
  }
}
