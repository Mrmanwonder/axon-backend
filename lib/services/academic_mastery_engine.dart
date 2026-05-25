import 'dart:math' as math;

import '../models/academic_engine_models.dart';

abstract class HandwritingGradingGateway {
  Future<Map<String, dynamic>> gradeHandwrittenAnswer({
    required String answerImageUrl,
    required String questionPrompt,
    required String markSchemeJson,
    required String commandWord,
    required List<String> learningObjectiveIds,
  });
}

class ReadinessScoreAlgorithm {
  const ReadinessScoreAlgorithm();

  double compute({
    required double mockScore,
    required DateTime? lastTested,
    required double decayFactor,
    DateTime? now,
  }) {
    final anchor = now ?? DateTime.now();
    final testedAt = lastTested ?? anchor.subtract(const Duration(days: 30));
    final days = math.max(0, anchor.difference(testedAt).inDays);
    final recency = 1 / (1 + (days / 14));
    final boundedDecay = decayFactor <= 0 ? 0.1 : decayFactor;
    return ((mockScore * recency) / boundedDecay).clamp(0, 100);
  }
}

class CommandWordValidator {
  static const Map<String, List<String>> _signals = {
    'define': ['is', 'means', 'refers to', 'called'],
    'describe': ['first', 'then', 'next', 'finally', 'observed'],
    'explain': ['because', 'therefore', 'so that', 'as a result', 'due to'],
    'compare': ['whereas', 'however', 'both', 'similar', 'different'],
    'evaluate': ['advantage', 'disadvantage', 'overall', 'best', 'limitation'],
    'calculate': ['=', 'working', 'substitute', 'answer'],
  };

  const CommandWordValidator();

  CommandWordValidation validate({
    required String expectedCommandWord,
    required String answer,
  }) {
    final normalizedCommand = expectedCommandWord.trim().toLowerCase();
    final normalizedAnswer = answer.trim().toLowerCase();
    final expectedSignals = _signals[normalizedCommand] ?? const <String>[];
    final matched = expectedSignals
        .where((signal) => normalizedAnswer.contains(signal))
        .toList();
    final missing = expectedSignals.where((signal) => !matched.contains(signal)).toList();
    final requiredSignalCount = expectedSignals.isEmpty
        ? 1
        : math.min(2, math.max(1, (expectedSignals.length / 2).ceil()));
    final isSatisfied = expectedSignals.isEmpty
        ? normalizedAnswer.isNotEmpty
        : matched.length >= requiredSignalCount;

    final rationale = isSatisfied
        ? 'The answer shows the response pattern expected for "$expectedCommandWord".'
        : 'The answer does not yet demonstrate enough "$expectedCommandWord" signals.';

    return CommandWordValidation(
      expectedCommandWord: expectedCommandWord,
      isSatisfied: isSatisfied,
      missingSignals: missing,
      rationale: rationale,
    );
  }
}

class WeightedMockGenerator {
  const WeightedMockGenerator();

  GeneratedMockPaper generate({
    required String board,
    required String subject,
    required String paper,
    required List<TopicWeighting> weightings,
    required List<QuestionBankEntry> questionBank,
    required int totalMarks,
  }) {
    final pool = questionBank
        .where((entry) =>
            entry.board == board &&
            entry.subject == subject &&
            entry.paper == paper)
        .toList();

    final selected = <QuestionBankEntry>[];
    final marksByTopic = <String, int>{};
    final usedIds = <String>{};

    for (final weighting in weightings) {
      final targetMarks = math.max(1, (totalMarks * weighting.weight).round());
      var currentMarks = 0;
      final topicPool = pool
          .where((entry) =>
              entry.topic.toLowerCase() == weighting.topic.toLowerCase())
          .toList()
        ..sort((a, b) => a.marks.compareTo(b.marks));

      for (final entry in topicPool) {
        if (usedIds.contains(entry.id)) continue;
        selected.add(entry);
        usedIds.add(entry.id);
        currentMarks += entry.marks;
        marksByTopic[weighting.topic] =
            (marksByTopic[weighting.topic] ?? 0) + entry.marks;
        if (currentMarks >= targetMarks) break;
      }
    }

    if (selected.isEmpty) {
      final fallback = [...pool]..sort((a, b) => b.marks.compareTo(a.marks));
      for (final entry in fallback) {
        if (usedIds.contains(entry.id)) continue;
        selected.add(entry);
        marksByTopic[entry.topic] = (marksByTopic[entry.topic] ?? 0) + entry.marks;
        if (selected.fold<int>(0, (sum, item) => sum + item.marks) >= totalMarks) {
          break;
        }
      }
    }

    return GeneratedMockPaper(
      board: board,
      subject: subject,
      paper: paper,
      questions: selected,
      marksByTopic: marksByTopic,
    );
  }
}

class AcademicMasteryEngine {
  AcademicMasteryEngine({
    required HandwritingGradingGateway gradingGateway,
    ReadinessScoreAlgorithm? readinessScoreAlgorithm,
    CommandWordValidator? commandWordValidator,
    WeightedMockGenerator? weightedMockGenerator,
  })  : _gradingGateway = gradingGateway,
        _readiness = readinessScoreAlgorithm ?? const ReadinessScoreAlgorithm(),
        _commandWords = commandWordValidator ?? const CommandWordValidator(),
        _mockGenerator = weightedMockGenerator ?? const WeightedMockGenerator();

  final HandwritingGradingGateway _gradingGateway;
  final ReadinessScoreAlgorithm _readiness;
  final CommandWordValidator _commandWords;
  final WeightedMockGenerator _mockGenerator;

  String _deriveErrorType({
    required CommandWordValidation validation,
    required List<GradedAnswerGap> gaps,
    required double awardedMarks,
    required double availableMarks,
  }) {
    if (gaps.isNotEmpty) {
      return 'learning_objective_gap';
    }
    if (!validation.isSatisfied) {
      return 'command_word_miss';
    }
    if (availableMarks > 0 && awardedMarks < availableMarks) {
      return 'marking_point_miss';
    }
    return 'none';
  }

  GeneratedMockPaper generateWeightedMock({
    required String board,
    required String subject,
    required String paper,
    required List<TopicWeighting> weightings,
    required List<QuestionBankEntry> questionBank,
    required int totalMarks,
  }) {
    return _mockGenerator.generate(
      board: board,
      subject: subject,
      paper: paper,
      weightings: weightings,
      questionBank: questionBank,
      totalMarks: totalMarks,
    );
  }

  CommandWordValidation validateCommandWord({
    required String expectedCommandWord,
    required String answer,
  }) {
    return _commandWords.validate(
      expectedCommandWord: expectedCommandWord,
      answer: answer,
    );
  }

  double computeReadinessScore({
    required MasteryRecord masteryRecord,
    DateTime? now,
  }) {
    final effectiveMastery = masteryRecord.decayedMastery(now: now) * 100;
    return _readiness.compute(
      mockScore: math.max(masteryRecord.lastMockScore, effectiveMastery),
      lastTested: masteryRecord.lastTested,
      decayFactor: masteryRecord.decayFactor,
      now: now,
    );
  }

  Future<GradedAnswerResult> gradeHandwrittenAnswer({
    required String answerImageUrl,
    required QuestionBankEntry question,
    required String expectedCommandWord,
  }) async {
    final validation = validateCommandWord(
      expectedCommandWord: expectedCommandWord,
      answer: question.prompt,
    );
    final raw = await _gradingGateway.gradeHandwrittenAnswer(
      answerImageUrl: answerImageUrl,
      questionPrompt: question.prompt,
      markSchemeJson: question.markSchemeJson,
      commandWord: expectedCommandWord,
      learningObjectiveIds: [question.learningObjectiveId],
    );

    final gaps = (raw['learning_objective_gaps'] as List? ?? const [])
        .map((item) => item is Map<String, dynamic>
            ? item
            : Map<String, dynamic>.from(item as Map))
        .map((gap) => GradedAnswerGap(
              learningObjectiveId:
                  (gap['learning_objective_id'] ?? question.learningObjectiveId)
                      .toString(),
              reason: (gap['reason'] ?? '').toString(),
            ))
        .toList();

    return GradedAnswerResult(
      awardedMarks: (raw['awarded_marks'] as num?)?.toDouble() ?? 0,
      availableMarks: (raw['available_marks'] as num?)?.toDouble() ??
          question.marks.toDouble(),
      feedback:
          '${(raw['feedback'] ?? '').toString()} ${validation.rationale}'.trim(),
      gaps: gaps,
      rawModelPayload: {
        ...raw,
        'command_word': expectedCommandWord,
        'error_type': _deriveErrorType(
          validation: validation,
          gaps: gaps,
          awardedMarks: (raw['awarded_marks'] as num?)?.toDouble() ?? 0,
          availableMarks: (raw['available_marks'] as num?)?.toDouble() ??
              question.marks.toDouble(),
        ),
      },
      commandWord: expectedCommandWord,
      errorType: _deriveErrorType(
        validation: validation,
        gaps: gaps,
        awardedMarks: (raw['awarded_marks'] as num?)?.toDouble() ?? 0,
        availableMarks: (raw['available_marks'] as num?)?.toDouble() ??
            question.marks.toDouble(),
      ),
    );
  }

  MockResultRecord buildMockResultRecord({
    required String id,
    required QuestionBankEntry question,
    required GradedAnswerResult gradedResult,
    int? durationSeconds,
    DateTime? recordedAt,
  }) {
    return MockResultRecord(
      id: id,
      objectiveId: question.learningObjectiveId,
      subject: question.subject,
      commandWord: gradedResult.commandWord,
      awardedMarks: gradedResult.awardedMarks,
      availableMarks: gradedResult.availableMarks,
      durationSeconds: durationSeconds,
      errorType: gradedResult.errorType,
      commandWordDepth:
          gradedResult.rawModelPayload['command_word_depth'] is Map
              ? Map<String, dynamic>.from(
                  gradedResult.rawModelPayload['command_word_depth'] as Map,
                )
              : null,
      semanticMatch:
          gradedResult.rawModelPayload['semantic_match'] is Map
              ? Map<String, dynamic>.from(
                  gradedResult.rawModelPayload['semantic_match'] as Map,
                )
              : null,
      recordedAt: recordedAt ?? DateTime.now(),
    );
  }
}
