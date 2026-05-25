import 'dart:math' as math;

class SyllabusLearningObjective {
  final String id;
  final String objectiveId;
  final String board;
  final String subject;
  final String paper;
  final String topic;
  final String subTopic;
  final String code;
  final String title;
  final String description;
  final double paperWeight;
  final List<String> commandWords;

  const SyllabusLearningObjective({
    required this.id,
    required this.objectiveId,
    required this.board,
    required this.subject,
    required this.paper,
    required this.topic,
    this.subTopic = '',
    required this.code,
    required this.title,
    required this.description,
    this.paperWeight = 0,
    this.commandWords = const [],
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'objective_id': objectiveId,
        'board': board,
        'subject': subject,
        'paper': paper,
        'topic': topic,
        'sub_topic': subTopic,
        'code': code,
        'title': title,
        'description': description,
        'paper_weight': paperWeight,
        'command_words': commandWords,
      };

  factory SyllabusLearningObjective.fromJson(Map<String, dynamic> json) =>
      SyllabusLearningObjective(
        id: (json['id'] ?? '').toString(),
        objectiveId: (json['objective_id'] ?? json['code'] ?? json['id'] ?? '')
            .toString(),
        board: (json['board'] ?? '').toString(),
        subject: (json['subject'] ?? '').toString(),
        paper: (json['paper'] ?? '').toString(),
        topic: (json['topic'] ?? '').toString(),
        subTopic: (json['sub_topic'] ?? '').toString(),
        code: (json['code'] ?? '').toString(),
        title: (json['title'] ?? '').toString(),
        description: (json['description'] ?? '').toString(),
        paperWeight: (json['paper_weight'] as num?)?.toDouble() ?? 0,
        commandWords: (json['command_words'] as List? ?? const [])
            .map((item) => item.toString())
            .toList(),
      );
}

class MasteryRecord {
  final String learningObjectiveId;
  final double confidenceScore;
  final double decayFactor;
  final DateTime? lastTested;
  final double lastMockScore;
  final double masteryScore;
  final double studyRatePerHour;

  const MasteryRecord({
    required this.learningObjectiveId,
    required this.confidenceScore,
    required this.decayFactor,
    this.lastTested,
    this.lastMockScore = 0,
    this.masteryScore = 0,
    this.studyRatePerHour = 0.05,
  });

  double decayedMastery({DateTime? now}) {
    final anchor = now ?? DateTime.now();
    final testedAt = lastTested ?? anchor.subtract(const Duration(days: 60));
    final daysSinceReview = anchor.difference(testedAt).inDays.clamp(0, 3650);
    final boundedDecay = decayFactor <= 0 ? 0.012 : decayFactor;
    final baseline = masteryScore > 0 ? masteryScore : confidenceScore;
    final retention = math.exp(-boundedDecay * daysSinceReview);
    return (baseline * retention).clamp(0, 1);
  }

  SyllabusStatus get status {
    final now = DateTime.now();
    final sevenDaysAgo = now.subtract(const Duration(days: 7));
    final fourteenDaysAgo = now.subtract(const Duration(days: 14));

    final currentMastery = decayedMastery();
    final hasRecentHighScore = lastMockScore > 0.85 &&
        lastTested != null &&
        lastTested!.isAfter(sevenDaysAgo);

    if (lastMockScore > 0 && lastMockScore < 0.4) {
      return SyllabusStatus.red;
    }

    if (hasRecentHighScore) {
      return SyllabusStatus.green;
    }

    final hasNotBeenReviewedRecently =
        lastTested == null || lastTested!.isBefore(fourteenDaysAgo);

    if (hasNotBeenReviewedRecently || currentMastery < 0.6) {
      return SyllabusStatus.yellow;
    }

    return SyllabusStatus.yellow;
  }

  double get hoursToGreen {
    final current = decayedMastery();
    if (current >= 0.85) return 0;

    final remaining = 0.85 - current;
    if (studyRatePerHour <= 0) return remaining / 0.05;
    return remaining / studyRatePerHour;
  }

  bool get needsReview {
    final daysUntilDecay =
        (math.log(0.6 / (decayedMastery() + 0.01)) / -decayFactor)
            .clamp(0, 90)
            .toInt();
    return daysUntilDecay <= 2;
  }

  Map<String, dynamic> toJson() => {
        'learning_objective_id': learningObjectiveId,
        'confidence_score': confidenceScore,
        'decay_factor': decayFactor,
        'last_tested': lastTested?.toIso8601String(),
        'last_mock_score': lastMockScore,
        'mastery_score': masteryScore > 0 ? masteryScore : confidenceScore,
        'study_rate_per_hour': studyRatePerHour,
      };

  factory MasteryRecord.fromJson(Map<String, dynamic> json) => MasteryRecord(
        learningObjectiveId: (json['learning_objective_id'] ?? '').toString(),
        confidenceScore: (json['confidence_score'] as num?)?.toDouble() ?? 0,
        decayFactor: (json['decay_factor'] as num?)?.toDouble() ?? 1,
        lastTested: DateTime.tryParse((json['last_tested'] ?? '').toString()),
        lastMockScore: (json['last_mock_score'] as num?)?.toDouble() ?? 0,
        masteryScore: (json['mastery_score'] as num?)?.toDouble() ??
            (json['confidence_score'] as num?)?.toDouble() ??
            0,
        studyRatePerHour:
            (json['study_rate_per_hour'] as num?)?.toDouble() ?? 0.05,
      );
}

enum SyllabusStatus {
  red,
  yellow,
  green,
}

class StudyEventLog {
  final String id;
  final String type;
  final int durationMinutes;
  final String learningObjectiveId;
  final double intensity;
  final DateTime occurredAt;
  final String subject;

  const StudyEventLog({
    required this.id,
    required this.type,
    required this.durationMinutes,
    required this.learningObjectiveId,
    required this.intensity,
    required this.occurredAt,
    required this.subject,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'type': type,
        'duration_minutes': durationMinutes,
        'learning_objective_id': learningObjectiveId,
        'intensity': intensity,
        'occurred_at': occurredAt.toIso8601String(),
        'subject': subject,
      };
}

class QuestionBankEntry {
  final String id;
  final String board;
  final String subject;
  final String paper;
  final String topic;
  final String subTopic;
  final String learningObjectiveId;
  final int marks;
  final List<String> commandWords;
  final String prompt;
  final String markSchemeJson;

  const QuestionBankEntry({
    required this.id,
    required this.board,
    required this.subject,
    required this.paper,
    required this.topic,
    this.subTopic = '',
    required this.learningObjectiveId,
    required this.marks,
    required this.commandWords,
    required this.prompt,
    required this.markSchemeJson,
  });
}

class TopicWeighting {
  final String topic;
  final double weight;

  const TopicWeighting({
    required this.topic,
    required this.weight,
  });
}

class GeneratedMockPaper {
  final String board;
  final String subject;
  final String paper;
  final List<QuestionBankEntry> questions;
  final Map<String, int> marksByTopic;

  const GeneratedMockPaper({
    required this.board,
    required this.subject,
    required this.paper,
    required this.questions,
    required this.marksByTopic,
  });
}

class CommandWordValidation {
  final String expectedCommandWord;
  final bool isSatisfied;
  final List<String> missingSignals;
  final String rationale;

  const CommandWordValidation({
    required this.expectedCommandWord,
    required this.isSatisfied,
    required this.missingSignals,
    required this.rationale,
  });
}

class GradedAnswerGap {
  final String learningObjectiveId;
  final String reason;

  const GradedAnswerGap({
    required this.learningObjectiveId,
    required this.reason,
  });
}

class GradedAnswerResult {
  final double awardedMarks;
  final double availableMarks;
  final String feedback;
  final List<GradedAnswerGap> gaps;
  final Map<String, dynamic> rawModelPayload;
  final String commandWord;
  final String errorType;

  const GradedAnswerResult({
    required this.awardedMarks,
    required this.availableMarks,
    required this.feedback,
    required this.gaps,
    required this.rawModelPayload,
    required this.commandWord,
    required this.errorType,
  });
}

class MockResultRecord {
  final String id;
  final String objectiveId;
  final String subject;
  final String commandWord;
  final double awardedMarks;
  final double availableMarks;
  final int? durationSeconds;
  final String? errorType;
  final Map<String, dynamic>? commandWordDepth;
  final Map<String, dynamic>? semanticMatch;
  final DateTime recordedAt;

  const MockResultRecord({
    required this.id,
    required this.objectiveId,
    required this.subject,
    required this.commandWord,
    required this.awardedMarks,
    required this.availableMarks,
    this.durationSeconds,
    this.errorType,
    this.commandWordDepth,
    this.semanticMatch,
    required this.recordedAt,
  });

  Map<String, dynamic> toJson() => {
        'objective_id': objectiveId,
        'topic_id': objectiveId,
        'subject': subject,
        'command_word': commandWord,
        'awarded_marks': awardedMarks,
        'available_marks': availableMarks,
        if (durationSeconds != null) 'duration_seconds': durationSeconds,
        if (errorType != null && errorType!.trim().isNotEmpty)
          'error_type': errorType,
        if (commandWordDepth != null) 'command_word_depth': commandWordDepth,
        if (semanticMatch != null) 'semantic_match': semanticMatch,
        'recorded_at': recordedAt.toIso8601String(),
      };
}

class NoStudyZone {
  final DateTime start;
  final DateTime end;
  final String label;

  const NoStudyZone({
    required this.start,
    required this.end,
    this.label = '',
  });

  bool contains(DateTime day) {
    final normalized = DateTime(day.year, day.month, day.day);
    final zoneStart = DateTime(start.year, start.month, start.day);
    final zoneEnd = DateTime(end.year, end.month, end.day);
    return !normalized.isBefore(zoneStart) && !normalized.isAfter(zoneEnd);
  }
}

class PlannerTaskAllocation {
  final String learningObjectiveId;
  final String title;
  final double loadUnits;

  const PlannerTaskAllocation({
    required this.learningObjectiveId,
    required this.title,
    required this.loadUnits,
  });
}

enum StudyPlanPhase {
  foundationBuild,
  firstPassCompletion,
  hardTopicDeepDive,
  timedMockSprint,
}

enum IntensityBand {
  blue,
  orange,
  red,
}

class PlannedStudyDay {
  final DateTime date;
  final bool blocked;
  final List<PlannerTaskAllocation> tasks;
  final double dailyLoad;
  final double intensityScore;
  final double adjustedDailyLoad;
  final double bufferLoadShift;
  final StudyPlanPhase phase;
  final IntensityBand intensityBand;

  const PlannedStudyDay({
    required this.date,
    required this.blocked,
    required this.tasks,
    required this.dailyLoad,
    required this.intensityScore,
    required this.adjustedDailyLoad,
    required this.bufferLoadShift,
    required this.phase,
    required this.intensityBand,
  });
}

class PlannerResult {
  final List<PlannedStudyDay> days;
  final double baselineDailyLoad;
  final double adjustedDailyLoad;
  final double peakIntensity;
  final double redistributedLoad;

  const PlannerResult({
    required this.days,
    required this.baselineDailyLoad,
    required this.adjustedDailyLoad,
    required this.peakIntensity,
    required this.redistributedLoad,
  });

  double get dailyLoad => adjustedDailyLoad;
}

class CalendarSyncEvent {
  final String title;
  final DateTime start;
  final DateTime end;
  final String description;

  const CalendarSyncEvent({
    required this.title,
    required this.start,
    required this.end,
    required this.description,
  });
}
