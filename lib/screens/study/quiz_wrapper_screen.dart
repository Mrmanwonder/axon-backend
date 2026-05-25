// lib/screens/study/quiz_wrapper_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../models/models.dart';
import '../../theme/app_theme.dart';
import 'quiz_results_screen.dart';

class QuizWrapperScreen extends ConsumerStatefulWidget {
  final String subject;
  final String chapter;
  final List<PdfQuestion> questions;

  const QuizWrapperScreen({
    super.key,
    required this.subject,
    required this.chapter,
    required this.questions,
  });

  @override
  ConsumerState<QuizWrapperScreen> createState() => _QuizWrapperScreenState();
}

class _QuizWrapperScreenState extends ConsumerState<QuizWrapperScreen> {
  final Map<String, String> _answers = {};
  bool _submitted = false;

  void _handleAnswerChanged(String key, String value) {
    if (!_submitted) {
      setState(() {
        _answers[key] = value;
      });
    }
  }

  void _submitQuiz() {
    setState(() => _submitted = true);

    final evaluatedQuestions = widget.questions.map((q) {
      final answer = _answers['q${q.questionNumber}'] ?? '';
      final hasAnswer = answer.trim().isNotEmpty;

      return q.copyWith(
        userAnswer: answer,
        isCorrect: hasAnswer,
        marksAwarded: hasAnswer ? (q.marksAvailable ?? 1) : 0,
      );
    }).toList();

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => QuizResultsScreen(
          subject: widget.subject,
          chapter: widget.chapter,
          questions: evaluatedQuestions,
          answers: Map<String, String>.from(_answers),
          filePath: 'quiz_${widget.subject}_${widget.chapter}',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF090A0B),
      appBar: AppBar(
        backgroundColor: AxonColors.surface,
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.white),
          onPressed: () {
            if (_answers.isNotEmpty) {
              showDialog(
                context: context,
                builder: (ctx) => AlertDialog(
                  backgroundColor: AxonColors.surface,
                  title: Text('Exit Quiz?',
                      style: GoogleFonts.googleSans(color: Colors.white)),
                  content: Text('Your progress will be lost.',
                      style: GoogleFonts.googleSans(color: Colors.white70)),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(ctx),
                      child: const Text('Cancel'),
                    ),
                    TextButton(
                      onPressed: () {
                        Navigator.pop(ctx);
                        Navigator.pop(context);
                      },
                      child: Text('Exit',
                          style: GoogleFonts.googleSans(color: Colors.red)),
                    ),
                  ],
                ),
              );
            } else {
              Navigator.pop(context);
            }
          },
        ),
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Quiz: ${widget.chapter}',
              style: GoogleFonts.googleSans(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600),
            ),
            Text(
              '${widget.questions.length} questions',
              style:
                  GoogleFonts.googleSans(color: Colors.white70, fontSize: 11),
            ),
          ],
        ),
        actions: [
          if (!_submitted)
            TextButton(
              onPressed: _submitQuiz,
              child: Text('Submit',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.electricCyan,
                      fontWeight: FontWeight.w600)),
            ),
        ],
      ),
      body: _QuizQuestionsView(
        questions: widget.questions,
        answers: _answers,
        onAnswerChanged: _handleAnswerChanged,
        submitted: _submitted,
      ),
    );
  }
}

class _QuizQuestionsView extends StatefulWidget {
  final List<PdfQuestion> questions;
  final Map<String, String> answers;
  final void Function(String, String) onAnswerChanged;
  final bool submitted;

  const _QuizQuestionsView({
    required this.questions,
    required this.answers,
    required this.onAnswerChanged,
    required this.submitted,
  });

  @override
  State<_QuizQuestionsView> createState() => _QuizQuestionsViewState();
}

class _QuizQuestionsViewState extends State<_QuizQuestionsView> {
  late final PageController _pageController;
  final Map<String, TextEditingController> _controllers = {};
  int _currentPage = 0;

  @override
  void initState() {
    super.initState();
    _pageController = PageController();
  }

  @override
  void dispose() {
    _pageController.dispose();
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  TextEditingController _getController(String key) {
    return _controllers.putIfAbsent(
      key,
      () => TextEditingController(text: widget.answers[key] ?? ''),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          color: AxonColors.surface,
          child: Row(
            children: [
              Text(
                'Question ${_currentPage + 1} of ${widget.questions.length}',
                style:
                    GoogleFonts.googleSans(color: Colors.white70, fontSize: 12),
              ),
              const Spacer(),
              ...List.generate(
                widget.questions.length,
                (i) => Container(
                  width: 8,
                  height: 8,
                  margin: const EdgeInsets.only(left: 4),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: widget.answers.containsKey(
                            'q${widget.questions[i].questionNumber}')
                        ? AxonColors.electricCyan
                        : Colors.white24,
                  ),
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: PageView.builder(
            controller: _pageController,
            itemCount: widget.questions.length,
            onPageChanged: (i) => setState(() => _currentPage = i),
            itemBuilder: (context, index) {
              final q = widget.questions[index];
              return _QuizQuestionCard(
                question: q,
                controller: _getController('q${q.questionNumber}'),
                onChanged: (v) =>
                    widget.onAnswerChanged('q${q.questionNumber}', v),
              );
            },
          ),
        ),
        Container(
          padding: const EdgeInsets.all(16),
          color: AxonColors.surface,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              TextButton.icon(
                onPressed: _currentPage > 0
                    ? () => _pageController.previousPage(
                        duration: const Duration(milliseconds: 300),
                        curve: Curves.easeInOut)
                    : null,
                icon: const Icon(Icons.arrow_back, size: 18),
                label: const Text('Previous'),
                style: TextButton.styleFrom(foregroundColor: Colors.white70),
              ),
              if (_currentPage == widget.questions.length - 1)
                ElevatedButton(
                  onPressed: () {
                    final allAnswered = widget.questions.every((q) =>
                        widget.answers['q${q.questionNumber}']
                            ?.trim()
                            .isNotEmpty ??
                        false);
                    if (!allAnswered) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                            content: Text('Answer all questions to submit',
                                style: GoogleFonts.googleSans())),
                      );
                      return;
                    }
                    (context.findAncestorStateOfType<_QuizWrapperScreenState>())
                        ?._submitQuiz();
                  },
                  style: ElevatedButton.styleFrom(
                      backgroundColor: AxonColors.electricCyan,
                      foregroundColor: Colors.black),
                  child: Text('Submit Quiz',
                      style:
                          GoogleFonts.googleSans(fontWeight: FontWeight.w700)),
                )
              else
                TextButton.icon(
                  onPressed: () => _pageController.nextPage(
                      duration: const Duration(milliseconds: 300),
                      curve: Curves.easeInOut),
                  icon: const Icon(Icons.arrow_forward, size: 18),
                  label: const Text('Next'),
                  style: TextButton.styleFrom(
                      foregroundColor: AxonColors.electricCyan),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

class _QuizQuestionCard extends StatelessWidget {
  final PdfQuestion question;
  final TextEditingController controller;
  final ValueChanged<String> onChanged;

  const _QuizQuestionCard({
    required this.question,
    required this.controller,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: AxonColors.electricCyan.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Q${question.questionNumber}',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.electricCyan,
                      fontWeight: FontWeight.w700),
                ),
              ),
              const SizedBox(width: 8),
              if (question.marksAvailable != null)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.orange.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    '${question.marksAvailable} marks',
                    style: GoogleFonts.googleSans(
                        color: Colors.orange, fontSize: 11),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 16),
          if (question.contextText != null &&
              question.contextText!.isNotEmpty) ...[
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AxonColors.surfaceElevated,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: AxonColors.divider),
              ),
              child: Text(
                question.contextText!,
                style: GoogleFonts.googleSans(
                    color: Colors.white70,
                    fontSize: 12,
                    fontStyle: FontStyle.italic),
              ),
            ),
            const SizedBox(height: 12),
          ],
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AxonColors.surface,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AxonColors.divider),
            ),
            child: Text(
              question.questionText,
              style: GoogleFonts.googleSans(
                  color: Colors.white, fontSize: 14, height: 1.6),
            ),
          ),
          if (question.parts.isNotEmpty) ...[
            const SizedBox(height: 16),
            ...question.parts.map((part) => Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: AxonColors.accent.withValues(alpha: 0.2),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              part.label,
                              style: GoogleFonts.googleSans(
                                  color: AxonColors.accent,
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              part.text,
                              style: GoogleFonts.googleSans(
                                  color: Colors.white70, fontSize: 13),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                )),
          ],
          const SizedBox(height: 16),
          Container(
            decoration: BoxDecoration(
              color: AxonColors.surfaceElevated,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                  color: AxonColors.electricCyan.withValues(alpha: 0.3)),
            ),
            child: TextField(
              controller: controller,
              onChanged: onChanged,
              maxLines: 5,
              minLines: 3,
              style: GoogleFonts.googleSans(color: Colors.white),
              decoration: InputDecoration(
                hintText: 'Write your answer here...',
                hintStyle: GoogleFonts.googleSans(color: Colors.white38),
                border: InputBorder.none,
                contentPadding: const EdgeInsets.all(16),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
