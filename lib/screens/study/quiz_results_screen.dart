// lib/screens/study/quiz_results_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../models/models.dart';
import '../../services/topic_quiz_service.dart';
import '../../theme/app_theme.dart';

class QuizResultsScreen extends ConsumerStatefulWidget {
  final String subject;
  final String? chapter;
  final List<PdfQuestion> questions;
  final Map<String, String> answers;
  final String? markingSchemeText;
  final String filePath;

  const QuizResultsScreen({
    super.key,
    required this.subject,
    this.chapter,
    required this.questions,
    required this.answers,
    this.markingSchemeText,
    required this.filePath,
  });

  @override
  ConsumerState<QuizResultsScreen> createState() => _QuizResultsScreenState();
}

class _QuizResultsScreenState extends ConsumerState<QuizResultsScreen> {
  late QuizResult _result;

  @override
  void initState() {
    super.initState();
    _calculateResults();
  }

  void _calculateResults() {
    final evaluatedQuestions = widget.questions.map((q) {
      final answer = widget.answers['q${q.questionNumber}'] ?? '';
      final correctAnswer = q.correctAnswer ?? '';

      bool isCorrect = false;
      int awardedMarks = 0;

      if (answer.isNotEmpty && correctAnswer.isNotEmpty) {
        final normalizedAnswer = answer.trim().toLowerCase();
        final normalizedCorrect = correctAnswer.trim().toLowerCase();
        isCorrect =
            normalizedAnswer.contains(normalizedCorrect.split(' ').first) ||
                normalizedCorrect.contains(normalizedAnswer.split(' ').first);
        if (isCorrect) {
          awardedMarks = q.marksAvailable ?? 1;
        }
      }

      return q.copyWith(
        userAnswer: answer,
        isCorrect: isCorrect,
        marksAwarded: awardedMarks,
      );
    }).toList();

    setState(() {
      _result = QuizResult.fromQuestions(evaluatedQuestions);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            backgroundColor: AxonColors.surface,
            expandedHeight: 200,
            pinned: true,
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      AxonColors.electricCyan.withValues(alpha: 0.3),
                      AxonColors.surface,
                    ],
                  ),
                ),
                child: SafeArea(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const SizedBox(height: 40),
                      Text(
                        widget.subject,
                        style: GoogleFonts.googleSans(
                          color: Colors.white70,
                          fontSize: 14,
                        ),
                      ),
                      if (widget.chapter != null)
                        Text(
                          widget.chapter!,
                          style: GoogleFonts.googleSans(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
            leading: IconButton(
              icon: const Icon(Icons.close, color: Colors.white),
              onPressed: () => Navigator.pop(context),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildScoreCard(),
                  const SizedBox(height: 20),
                  _buildPerformanceBreakdown(),
                  const SizedBox(height: 20),
                  _buildWeakStrongPoints(),
                  const SizedBox(height: 20),
                  _buildQuestionReview(),
                  const SizedBox(height: 20),
                  _buildRecommendedPlan(),
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScoreCard() {
    final percentage = _result.percentage;
    Color scoreColor;
    String emoji;
    String message;

    if (percentage >= 80) {
      scoreColor = Colors.green;
      emoji = '🎉';
      message = 'Excellent!';
    } else if (percentage >= 60) {
      scoreColor = Colors.orange;
      emoji = '👍';
      message = 'Good job!';
    } else if (percentage >= 40) {
      scoreColor = Colors.amber;
      emoji = '💪';
      message = 'Keep practicing!';
    } else {
      scoreColor = Colors.red;
      emoji = '📚';
      message = 'Need more work';
    }

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: scoreColor.withValues(alpha: 0.5)),
      ),
      child: Column(
        children: [
          Text(emoji, style: const TextStyle(fontSize: 40)),
          const SizedBox(height: 8),
          Text(
            message,
            style: GoogleFonts.googleSans(
              color: scoreColor,
              fontSize: 20,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _StatBox(
                label: 'Score',
                value: '${percentage.round()}%',
                color: scoreColor,
              ),
              const SizedBox(width: 24),
              _StatBox(
                label: 'Correct',
                value: '${_result.correctAnswers}/${_result.totalQuestions}',
                color: AxonColors.electricCyan,
              ),
              const SizedBox(width: 24),
              _StatBox(
                label: 'Marks',
                value: '${_result.earnedMarks}/${_result.totalMarks}',
                color: Colors.orange,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPerformanceBreakdown() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Performance by Difficulty',
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _DifficultyBar(
                  label: 'Easy',
                  correct: _result.topicPerformance
                      .where((t) => t.percentage >= 70)
                      .length,
                  total: _result.topicPerformance.length,
                  color: Colors.green,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _DifficultyBar(
                  label: 'Medium',
                  correct: _result.topicPerformance
                      .where((t) => t.percentage >= 40 && t.percentage < 70)
                      .length,
                  total: _result.topicPerformance.length,
                  color: Colors.orange,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _DifficultyBar(
                  label: 'Hard',
                  correct: _result.topicPerformance
                      .where((t) => t.percentage < 40)
                      .length,
                  total: _result.topicPerformance.length,
                  color: Colors.red,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildWeakStrongPoints() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.red.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.trending_down, color: Colors.red, size: 18),
                    const SizedBox(width: 6),
                    Text(
                      'Weak Points',
                      style: GoogleFonts.googleSans(
                        color: Colors.red,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                if (_result.weakPoints.isEmpty)
                  Text(
                    'None - great job!',
                    style: GoogleFonts.googleSans(
                      color: Colors.white70,
                      fontSize: 12,
                    ),
                  )
                else
                  ...(_result.weakPoints.map((p) => Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Text(
                          '• $p',
                          style: GoogleFonts.googleSans(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                        ),
                      ))),
              ],
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.green.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.green.withValues(alpha: 0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.trending_up, color: Colors.green, size: 18),
                    const SizedBox(width: 6),
                    Text(
                      'Strong Points',
                      style: GoogleFonts.googleSans(
                        color: Colors.green,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                if (_result.strongPoints.isEmpty)
                  Text(
                    'Keep practicing',
                    style: GoogleFonts.googleSans(
                      color: Colors.white70,
                      fontSize: 12,
                    ),
                  )
                else
                  ...(_result.strongPoints.map((p) => Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Text(
                          '• $p',
                          style: GoogleFonts.googleSans(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                        ),
                      ))),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildQuestionReview() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Question Review',
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          ...widget.questions.asMap().entries.map((entry) {
            final index = entry.key;
            final q = entry.value;
            final isCorrect = q.isCorrect ?? false;

            return Container(
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: (isCorrect ? Colors.green : Colors.red)
                    .withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: (isCorrect ? Colors.green : Colors.red)
                      .withValues(alpha: 0.3),
                ),
              ),
              child: Row(
                children: [
                  Container(
                    width: 28,
                    height: 28,
                    decoration: BoxDecoration(
                      color: isCorrect ? Colors.green : Colors.red,
                      shape: BoxShape.circle,
                    ),
                    child: Center(
                      child: Icon(
                        isCorrect ? Icons.check : Icons.close,
                        color: Colors.white,
                        size: 16,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Q${index + 1}: ${q.questionText.length > 50 ? '${q.questionText.substring(0, 50)}...' : q.questionText}',
                          style: GoogleFonts.googleSans(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 4),
                        Text(
                          isCorrect
                              ? 'Correct!'
                              : 'Answer: ${q.correctAnswer ?? "N/A"}',
                          style: GoogleFonts.googleSans(
                            color: isCorrect ? Colors.green : Colors.orange,
                            fontSize: 10,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildRecommendedPlan() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.electricCyan.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16),
        border:
            Border.all(color: AxonColors.electricCyan.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.lightbulb, color: AxonColors.electricCyan, size: 20),
              const SizedBox(width: 8),
              Text(
                'Recommended Study Plan',
                style: GoogleFonts.googleSans(
                  color: AxonColors.electricCyan,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            _result.recommendedPlan,
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 14,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AxonColors.surface,
                    foregroundColor: Colors.white,
                  ),
                  child: const Text('Back to Study'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AxonColors.electricCyan,
                    foregroundColor: Colors.black,
                  ),
                  child: const Text('Try Again'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatBox extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _StatBox(
      {required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          value,
          style: GoogleFonts.googleSans(
            color: color,
            fontSize: 24,
            fontWeight: FontWeight.w700,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.googleSans(
            color: Colors.white70,
            fontSize: 12,
          ),
        ),
      ],
    );
  }
}

class _DifficultyBar extends StatelessWidget {
  final String label;
  final int correct;
  final int total;
  final Color color;

  const _DifficultyBar({
    required this.label,
    required this.correct,
    required this.total,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final percentage = total > 0 ? correct / total : 0.0;
    return Column(
      children: [
        Container(
          height: 60,
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              Container(
                height: 60 * percentage,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: GoogleFonts.googleSans(
            color: Colors.white70,
            fontSize: 10,
          ),
        ),
        Text(
          '$correct/$total',
          style: GoogleFonts.googleSans(
            color: color,
            fontSize: 10,
          ),
        ),
      ],
    );
  }
}
