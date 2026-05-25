import 'package:flutter/material.dart';
import 'package:flutter_math_fork/flutter_math.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../models/models.dart';

class PristineQuestionCard extends StatelessWidget {
  const PristineQuestionCard({
    super.key,
    required this.question,
    required this.controllerFor,
    required this.onAnswerChanged,
    required this.currentIndex,
    required this.totalQuestions,
  });

  final PdfQuestion question;
  final TextEditingController Function(String) controllerFor;
  final void Function(String, String) onAnswerChanged;
  final int currentIndex;
  final int totalQuestions;

  @override
  Widget build(BuildContext context) {
    final key = 'q${question.questionNumber}';
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: IntrinsicHeight(
        child: Container(
          margin: const EdgeInsets.all(24),
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.02),
            borderRadius: BorderRadius.circular(28),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
              width: 0.5,
            ),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF3A86FF).withValues(alpha: 0.08),
                blurRadius: 36,
                spreadRadius: 2,
                offset: const Offset(0, 18),
              ),
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.24),
                blurRadius: 48,
                offset: const Offset(0, 24),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Text(
                      'Question ${question.questionNumber}',
                      style: GoogleFonts.inter(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  if (question.marksAvailable != null)
                    Text(
                      '${question.marksAvailable} marks',
                      style: GoogleFonts.inter(
                        color: Colors.white70,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                '${currentIndex + 1} / $totalQuestions',
                style: GoogleFonts.inter(
                  color: Colors.white54,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 24),
              MathView(
                question.questionText,
                style: GoogleFonts.inter(
                  color: Colors.white,
                  fontSize: 16,
                  height: 1.6,
                ),
              ),
              if (question.parts.isNotEmpty) ...[
                const SizedBox(height: 24),
                _ThreadedParts(
                  parts: question.parts,
                  controllerFor: controllerFor,
                  questionNumber: question.questionNumber,
                  onAnswerChanged: onAnswerChanged,
                ),
              ] else ...[
                const SizedBox(height: 24),
                _AnswerField(
                  controller: controllerFor(key),
                  onChanged: (value) => onAnswerChanged(key, value),
                ),
              ],
              if (question.feedback != null &&
                  question.feedback!.trim().isNotEmpty) ...[
                const SizedBox(height: 20),
                _EvaluationPanel(question: question),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _EvaluationPanel extends StatelessWidget {
  const _EvaluationPanel({required this.question});

  final PdfQuestion question;

  @override
  Widget build(BuildContext context) {
    final bool graded = question.marksAwarded != null || question.marksAvailable != null;
    final Color statusColor = question.isCorrect == true
        ? const Color(0xFF3DDC97)
        : const Color(0xFFFF7A59);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: statusColor.withValues(alpha: 0.35),
          width: 0.8,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  question.isCorrect == true ? 'Strong Answer' : 'Needs Work',
                  style: GoogleFonts.inter(
                    color: statusColor,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              const Spacer(),
              if (graded)
                Text(
                  '${question.marksAwarded ?? 0}/${question.marksAvailable ?? 0}',
                  style: GoogleFonts.inter(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            question.feedback!.trim(),
            style: GoogleFonts.inter(
              color: Colors.white70,
              fontSize: 13,
              height: 1.5,
            ),
          ),
          if ((question.correctAnswer ?? '').trim().isNotEmpty) ...[
            const SizedBox(height: 12),
            Text(
              'Mark Scheme',
              style: GoogleFonts.inter(
                color: Colors.white54,
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              question.correctAnswer!.trim(),
              maxLines: 6,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.inter(
                color: Colors.white60,
                fontSize: 12,
                height: 1.45,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class MathView extends StatelessWidget {
  const MathView(
    this.content, {
    super.key,
    required this.style,
  });

  final String content;
  final TextStyle style;

  @override
  Widget build(BuildContext context) {
    final normalized = content.trim();
    if (normalized.isEmpty) {
      return const SizedBox.shrink();
    }
    if (!_looksLikeLatex(normalized)) {
      return SelectableText(normalized, style: style);
    }
    final normalizedLatex = normalized
        .replaceAll('\n', r' \\ ')
        .replaceAll('%', r'\%');
    try {
      return SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Math.tex(
          normalizedLatex,
          mathStyle: MathStyle.text,
          textStyle: style,
        ),
      );
    } catch (_) {
      return SelectableText(normalized, style: style);
    }
  }

  bool _looksLikeLatex(String value) {
    return value.contains(r'\') ||
        value.contains(r'$') ||
        value.contains('^') ||
        value.contains('_') ||
        value.contains('{') ||
        value.contains('}') ||
        value.contains('%') ||
        RegExp(r'[=+\-*/<>]').hasMatch(value);
  }
}

class _ThreadedParts extends StatelessWidget {
  const _ThreadedParts({
    required this.parts,
    required this.controllerFor,
    required this.questionNumber,
    required this.onAnswerChanged,
  });

  final List<PdfPart> parts;
  final TextEditingController Function(String) controllerFor;
  final int questionNumber;
  final void Function(String, String) onAnswerChanged;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 1,
          margin: const EdgeInsets.only(left: 8, top: 4),
          color: Colors.white.withValues(alpha: 0.12),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: parts.map((part) {
              final key = 'q${questionNumber}_${part.label}';
              return Padding(
                padding: const EdgeInsets.only(bottom: 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      part.label,
                      style: GoogleFonts.inter(
                        color: Colors.white70,
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    MathView(
                      part.text,
                      style: GoogleFonts.inter(
                        color: Colors.white,
                        fontSize: 15,
                        height: 1.6,
                      ),
                    ),
                    const SizedBox(height: 16),
                    _AnswerField(
                      controller: controllerFor(key),
                      onChanged: (value) => onAnswerChanged(key, value),
                    ),
                  ],
                ),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }
}

class _AnswerField extends StatelessWidget {
  const _AnswerField({
    required this.controller,
    required this.onChanged,
  });

  final TextEditingController controller;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      minLines: 3,
      maxLines: 10,
      onChanged: onChanged,
      style: GoogleFonts.inter(
        color: Colors.white,
        fontSize: 14,
        height: 1.5,
      ),
      decoration: InputDecoration(
        hintText: 'Write your answer',
        hintStyle: GoogleFonts.inter(
          color: Colors.white38,
          fontSize: 13,
        ),
        filled: true,
        fillColor: Colors.white.withValues(alpha: 0.03),
        contentPadding: const EdgeInsets.all(16),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: BorderSide(
            color: Colors.white.withValues(alpha: 0.08),
            width: 0.5,
          ),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: BorderSide(
            color: Colors.white.withValues(alpha: 0.08),
            width: 0.5,
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: BorderSide(
            color: const Color(0xFF3A86FF).withValues(alpha: 0.8),
            width: 1,
          ),
        ),
      ),
    );
  }
}
