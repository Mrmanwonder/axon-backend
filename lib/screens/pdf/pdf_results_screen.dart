import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

import '../../models/models.dart';
import '../../services/app_state.dart';
import '../../services/handwriting_feedback_service.dart';
import '../../services/pdf_service_stub.dart';
import '../../services/question_intelligence_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/pdf/pristine_question_card.dart';
import '../study/solution_wall_screen.dart';

class PdfResultsScreen extends ConsumerStatefulWidget {
  final String title;
  final List<PdfQuestion> questions;
  final Map<String, String> answers;
  final String? markingSchemeText;
  final String filePath;

  const PdfResultsScreen({
    super.key,
    required this.title,
    required this.questions,
    required this.answers,
    this.markingSchemeText,
    required this.filePath,
  });

  @override
  ConsumerState<PdfResultsScreen> createState() => _PdfResultsScreenState();
}

class _PdfResultsScreenState extends ConsumerState<PdfResultsScreen> {
  final PdfService _service = PdfService();
  final HandwritingFeedbackService _handwritingService =
      HandwritingFeedbackService();
  late final PageController _pageController;
  final Map<String, TextEditingController> _controllers = {};
  bool _capturedMistakes = false;
  int _currentPage = 0;
  bool _gradingHandwriting = false;

  String get _providerKey => 'pdf_workspace_${widget.filePath.hashCode}';

  @override
  void initState() {
    super.initState();
    _pageController = PageController();
    SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
      systemNavigationBarColor: Colors.transparent,
      systemNavigationBarDividerColor: Colors.transparent,
      systemNavigationBarIconBrightness: Brightness.light,
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
    ));
    ref.read(pdfWorkspaceProvider(_providerKey).notifier).initialize(
          title: widget.title,
          questions: widget.questions,
          answers: widget.answers,
          markingSchemeText: widget.markingSchemeText,
        );
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await _loadSchemeIfNeeded();
      await _captureMistakes();
    });
  }

  @override
  void dispose() {
    for (final controller in _controllers.values) {
      controller.dispose();
    }
    _pageController.dispose();
    super.dispose();
  }

  Future<void> _loadSchemeIfNeeded() async {
    final current = ref.read(pdfWorkspaceProvider(_providerKey)).valueOrNull;
    if (current == null || current.markingSchemeText.trim().isNotEmpty) {
      return;
    }
    if (widget.filePath.isEmpty) {
      return;
    }
    final response = await _service.autoFindMarkingScheme(widget.filePath);
    if (!mounted || response == null || response['error'] != null) {
      return;
    }
    var text = response['text']?.toString() ?? '';
    final path = response['path']?.toString();
    if (text.trim().isEmpty && path != null && path.isNotEmpty) {
      text = await _service.extractText(path);
    }
    if (!mounted) {
      return;
    }
    ref
        .read(pdfWorkspaceProvider(_providerKey).notifier)
        .updateMarkingScheme(text);
  }

  Future<void> _captureMistakes() async {
    if (_capturedMistakes) {
      return;
    }
    final current = ref.read(pdfWorkspaceProvider(_providerKey)).valueOrNull;
    if (current == null || current.questions.isEmpty) {
      return;
    }
    _capturedMistakes = true;
    await QuestionIntelligenceService.instance.storeWrongQuestions(
      sourceTitle: current.title,
      sourcePath: current.filePath,
      questions: current.questions,
      answers: current.answers,
      isWrong: (question, answer) => answer.trim().isEmpty,
    );
  }

  TextEditingController _controllerFor(String key) {
    final workspace = ref.read(pdfWorkspaceProvider(_providerKey)).valueOrNull;
    return _controllers.putIfAbsent(
      key,
      () => TextEditingController(text: workspace?.answers[key] ?? ''),
    );
  }

  void _handleAnswerChanged(String key, String value) {
    ref
        .read(pdfWorkspaceProvider(_providerKey).notifier)
        .updateAnswer(key, value);
  }

  PdfQuestion? _currentQuestion(PdfWorkspaceState workspace) {
    if (workspace.questions.isEmpty) return null;
    final index = _currentPage.clamp(0, workspace.questions.length - 1);
    return workspace.questions[index];
  }

  String _questionId(PdfQuestion question) {
    return '${widget.filePath.hashCode}:${question.pageNumber}:${question.questionNumber}:${question.questionText.hashCode}';
  }

  String _objectiveFor(PdfQuestion question) {
    final segments = [
      question.boardTag,
      question.subjectTag,
      question.chapterTag,
      question.topicTag,
    ].where((item) => item.trim().isNotEmpty);
    return segments.isEmpty ? 'General' : segments.join(' / ');
  }

  String _commandWordFor(PdfQuestion question) {
    final prompt = question.fullText.toLowerCase();
    const words = [
      'define',
      'describe',
      'explain',
      'compare',
      'evaluate',
      'calculate',
      'state',
      'identify',
      'outline',
      'justify',
      'discuss',
    ];
    for (final word in words) {
      if (prompt.startsWith(word) || prompt.contains(' $word ')) {
        return word;
      }
    }
    return 'respond';
  }

  Future<void> _openSolutionWall(PdfWorkspaceState workspace) async {
    final question = _currentQuestion(workspace);
    if (question == null) return;
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => SolutionWallScreen(
          questionId: _questionId(question),
          question: question,
          sourceTitle: widget.title,
        ),
      ),
    );
  }

  Future<void> _runHandwritingReview(PdfWorkspaceState workspace) async {
    final question = _currentQuestion(workspace);
    if (question == null || _gradingHandwriting) return;

    final picker = ImagePicker();
    final source = await showModalBottomSheet<ImageSource>(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt_outlined),
              title: const Text('Camera'),
              onTap: () => Navigator.of(context).pop(ImageSource.camera),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library_outlined),
              title: const Text('Gallery'),
              onTap: () => Navigator.of(context).pop(ImageSource.gallery),
            ),
          ],
        ),
      ),
    );
    if (source == null) return;

    final picked = await picker.pickImage(source: source, imageQuality: 90);
    if (picked == null) return;

    setState(() => _gradingHandwriting = true);
    try {
      final outcome = await _handwritingService.gradeQuestion(
        question: question,
        imageFile: File(picked.path),
        objective: _objectiveFor(question),
        commandWord: _commandWordFor(question),
        learningObjectiveIds: [
          question.topicTag,
          question.chapterTag,
        ].where((item) => item.trim().isNotEmpty).toList(),
        markingSchemeText: question.correctAnswer ??
            widget.markingSchemeText ??
            'No marking scheme available.',
      );
      if (!mounted) return;
      await showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        builder: (context) => DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.75,
          builder: (context, scrollController) {
            return Container(
              padding: const EdgeInsets.all(20),
              child: ListView(
                controller: scrollController,
                children: [
                  Text(
                    'Handwritten Feedback',
                    style: GoogleFonts.googleSans(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    '${outcome.score.toStringAsFixed(1)} / ${outcome.availableMarks.toStringAsFixed(1)}',
                    style: GoogleFonts.robotoMono(
                      fontSize: 24,
                      color: const Color(0xFF3A86FF),
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(outcome.feedback),
                  if (outcome.marksAwarded.isNotEmpty) ...[
                    const SizedBox(height: 20),
                    const Text(
                      'Marks Awarded',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 8),
                    for (final item in outcome.marksAwarded)
                      ListTile(
                        dense: true,
                        contentPadding: EdgeInsets.zero,
                        leading: const Icon(Icons.check_circle_outline_rounded),
                        title: Text((item['point'] ?? '').toString()),
                        subtitle: Text((item['evidence'] ?? '').toString()),
                      ),
                  ],
                  if (outcome.marksMissed.isNotEmpty) ...[
                    const SizedBox(height: 20),
                    const Text(
                      'Marks Missed',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 8),
                    for (final item in outcome.marksMissed)
                      ListTile(
                        dense: true,
                        contentPadding: EdgeInsets.zero,
                        leading: const Icon(Icons.cancel_outlined),
                        title: Text((item['point'] ?? '').toString()),
                        subtitle: Text((item['evidence'] ?? '').toString()),
                      ),
                  ],
                ],
              ),
            );
          },
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _gradingHandwriting = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = AxonThemeMode.isDark;
    final workspaceAsync = ref.watch(pdfWorkspaceProvider(_providerKey));
    return Scaffold(
      backgroundColor:
          isDark ? const Color(0xFF000000) : const Color(0xFFFFFFFF),
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.6),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.arrow_back, color: Colors.white),
          ),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          widget.title,
          style: GoogleFonts.googleSans(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      body: SafeArea(
        child: workspaceAsync.when(
          loading: () => Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(
                  4,
                  (i) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 8),
                        child: Container(
                          width: 200 - (i * 30).toDouble(),
                          height: 20,
                          decoration: BoxDecoration(
                            color: const Color(0xFF141414),
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                      )),
            ),
          ),
          error: (error, _) => Center(
            child: Text(
              error.toString(),
              style: const TextStyle(color: Colors.white70),
            ),
          ),
          data: (workspace) => PageView.builder(
            controller: _pageController,
            itemCount: workspace.questions.length,
            onPageChanged: (index) {
              setState(() => _currentPage = index);
              HapticFeedback.selectionClick();
            },
            itemBuilder: (context, index) {
              final question = workspace.questions[index];
              return PristineQuestionCard(
                question: question,
                controllerFor: _controllerFor,
                onAnswerChanged: _handleAnswerChanged,
                currentIndex: index,
                totalQuestions: workspace.questions.length,
              );
            },
          ),
        ),
      ),
      bottomNavigationBar: workspaceAsync.maybeWhen(
        data: (workspace) {
          if (_currentQuestion(workspace) == null) return null;
          return SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _openSolutionWall(workspace),
                      icon: const Icon(Icons.hub_outlined),
                      label: const Text('Solution Wall'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: _gradingHandwriting
                          ? null
                          : () => _runHandwritingReview(workspace),
                      icon: Icon(_gradingHandwriting
                          ? Icons.hourglass_top_rounded
                          : Icons.document_scanner_outlined),
                      label: Text(
                        _gradingHandwriting
                            ? 'Grading...'
                            : 'Handwriting Review',
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
        orElse: () => null,
      ),
    );
  }
}
