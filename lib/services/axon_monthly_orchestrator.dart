import 'dart:math' as math;

import '../models/academic_engine_models.dart';
import '../models/models.dart';

enum MonthlyPlanningStage {
  gapClosure,
  interleaving,
  simulation,
}

class PersonalGoal {
  final String id;
  final String title;
  final DateTime targetDate;
  final double priority;
  final String category;
  final String subjectId;
  final String targetGrade;
  final String strategy;
  final List<String> linkedObjectiveIds;

  const PersonalGoal({
    required this.id,
    required this.title,
    required this.targetDate,
    required this.priority,
    required this.category,
    this.subjectId = '',
    this.targetGrade = '',
    this.strategy = '',
    this.linkedObjectiveIds = const [],
  });
}

class SyllabusNode {
  final String id;
  final String subjectId;
  final String subjectName;
  final String moduleName;
  final String topicName;
  final double masteryScore;
  final double subjectDifficulty;
  final DateTime? lastReviewed;
  final bool supportsPastPaperSimulation;
  final bool highVolatility;
  final String pdfQuestionId;
  final String pdfPath;
  final int? pdfPage;

  const SyllabusNode({
    required this.id,
    required this.subjectId,
    required this.subjectName,
    required this.moduleName,
    required this.topicName,
    this.masteryScore = 0,
    this.subjectDifficulty = 0.5,
    this.lastReviewed,
    this.supportsPastPaperSimulation = true,
    this.highVolatility = false,
    this.pdfQuestionId = '',
    this.pdfPath = '',
    this.pdfPage,
  });

  factory SyllabusNode.fromLearningObjective({
    required SyllabusLearningObjective objective,
    MasteryRecord? mastery,
    double subjectDifficulty = 0.5,
  }) {
    return SyllabusNode(
      id: objective.objectiveId,
      subjectId: objective.subject,
      subjectName: objective.subject,
      moduleName: objective.code.isEmpty ? objective.title : objective.code,
      topicName: objective.topic.isEmpty ? objective.title : objective.topic,
      masteryScore: mastery?.decayedMastery() ?? 0,
      subjectDifficulty: subjectDifficulty,
      lastReviewed: mastery?.lastTested,
      supportsPastPaperSimulation: objective.paper.trim().isNotEmpty,
      highVolatility:
          _isHighVolatilityTopic(objective.subject, objective.topic),
    );
  }

  bool get isExcellentRecently {
    if (masteryScore < 0.85 || lastReviewed == null) return false;
    return DateTime.now().difference(lastReviewed!).inDays < 4;
  }

  static bool _isHighVolatilityTopic(String subject, String topic) {
    final text = '$subject $topic'.toLowerCase();
    return text.contains('derivation') ||
        text.contains('mechanics') ||
        text.contains('integration') ||
        text.contains('matrices') ||
        text.contains('differential') ||
        text.contains('paper 3') ||
        text.contains('paper 4');
  }
}

class CognitiveLoad {
  final double currentStress;
  final double burnoutRisk;
  final double sleepDebt;
  final double sleepHours;
  final double screenTimeHours;

  const CognitiveLoad({
    this.currentStress = 0,
    this.burnoutRisk = 0,
    this.sleepDebt = 0,
    this.sleepHours = 7,
    this.screenTimeHours = 0,
  });

  bool get needsRecovery =>
      currentStress >= 7 || burnoutRisk >= 0.7 || sleepDebt >= 2.5;

  double get readinessMultiplier {
    if (needsRecovery) return 0.55;
    if (currentStress >= 5 || sleepDebt >= 1.5) return 0.8;
    return 1.0;
  }

  String get loadState {
    if (needsRecovery) return 'recovery';
    if (readinessMultiplier < 1) return 'balanced';
    return 'intensive';
  }
}

class AcademicVault {
  final List<SyllabusNode> syllabus;
  final Map<String, double> masteryHeatmap;
  final List<ExamEvent> upcomingExams;

  const AcademicVault({
    required this.syllabus,
    required this.masteryHeatmap,
    required this.upcomingExams,
  });
}

class BiometricVault {
  final CognitiveLoad cognitiveLoad;
  final int goldenHour;

  const BiometricVault({
    required this.cognitiveLoad,
    this.goldenHour = 16,
  });
}

class PsychologicalVault {
  final MotivationStyle motivationStyle;

  const PsychologicalVault({
    required this.motivationStyle,
  });
}

class RankedSyllabusNode {
  final SyllabusNode node;
  final double priorityScore;
  final ExamEvent? nearestExam;

  const RankedSyllabusNode({
    required this.node,
    required this.priorityScore,
    this.nearestExam,
  });
}

class StudyOrchestrator {
  final UserProfile profile;
  final List<DailyMetrics> history;
  final List<ExamEvent> upcomingExams;
  final List<PersonalGoal> activeGoals;
  final List<SyllabusNode> syllabus;
  final List<StudySession> sessionHistory;
  final CognitiveLoad cognitiveLoad;

  final double consistencyWeight;
  final double urgencyWeight;
  final double weaknessWeight;

  const StudyOrchestrator({
    required this.profile,
    required this.history,
    required this.upcomingExams,
    required this.activeGoals,
    required this.syllabus,
    this.sessionHistory = const [],
    this.cognitiveLoad = const CognitiveLoad(),
    this.consistencyWeight = 0.4,
    this.urgencyWeight = 0.35,
    this.weaknessWeight = 0.25,
  });

  List<AxonDailyPlan> generateMonthlyRoadmap({DateTime? startDate}) {
    final planner = AxonMonthlyOrchestrator(
      syllabus: syllabus,
      user: profile,
      history: history,
      upcomingExams: upcomingExams,
      activeGoals: activeGoals,
      sessionHistory: sessionHistory,
      cognitiveLoad: cognitiveLoad,
      consistencyWeight: consistencyWeight,
      urgencyWeight: urgencyWeight,
      weaknessWeight: weaknessWeight,
    );
    return planner.generateMonthlyRoadmap(startDate: startDate);
  }
}

class AxonMonthlyOrchestrator {
  final List<SyllabusNode> syllabus;
  final UserProfile user;
  final List<DailyMetrics> history;
  final List<ExamEvent> upcomingExams;
  final List<PersonalGoal> activeGoals;
  final List<StudySession> sessionHistory;
  final CognitiveLoad cognitiveLoad;
  final double consistencyWeight;
  final double urgencyWeight;
  final double weaknessWeight;

  const AxonMonthlyOrchestrator({
    required this.syllabus,
    required this.user,
    this.history = const [],
    this.upcomingExams = const [],
    this.activeGoals = const [],
    this.sessionHistory = const [],
    this.cognitiveLoad = const CognitiveLoad(),
    this.consistencyWeight = 0.4,
    this.urgencyWeight = 0.35,
    this.weaknessWeight = 0.25,
  });

  AcademicVault get academicVault => AcademicVault(
        syllabus: syllabus,
        masteryHeatmap: {
          for (final node in syllabus) node.id: node.masteryScore.clamp(0, 1),
        },
        upcomingExams: upcomingExams,
      );

  BiometricVault get biometricVault => BiometricVault(
        cognitiveLoad: cognitiveLoad,
        goldenHour: _goldenHour(),
      );

  PsychologicalVault get psychologicalVault => PsychologicalVault(
        motivationStyle: user.motivationStyle,
      );

  List<AxonDailyPlan> generateMonthlyRoadmap({DateTime? startDate}) {
    final start = _dateOnly(startDate ?? DateTime.now());
    return List<AxonDailyPlan>.generate(30, (index) {
      return calculateOptimalDay(start.add(Duration(days: index)));
    });
  }

  AxonDailyPlan calculateOptimalDay(DateTime targetDate) {
    final day = _dateOnly(targetDate);
    final stage = _stageForOffset(day);
    final ranked = _rankNodes(day, stage);
    final availableNodes = _filterForReadiness(ranked, stage);
    final selected = _selectRotatedNodes(availableNodes, stage);
    final nearestExam = _nearestExamForPlan(day, selected);
    final isExamDay = nearestExam != null &&
        !_dateOnly(nearestExam.startDate).isAfter(day) &&
        !_dateOnly(nearestExam.endDate).isBefore(day);
    final units = _recommendedUnits(day, selected.length);
    final tasks = _buildExecutionTasks(day, selected, stage, units);
    final triggers = _buildTriggers(day, tasks, selected);
    final subject = selected.isNotEmpty
        ? selected.first.node.subjectId
        : nearestExam?.subject ??
            (user.subjects.isEmpty ? 'General' : user.subjects.first);

    return AxonDailyPlan(
      date: day,
      subject: subject,
      board: nearestExam?.board ?? user.board,
      examLabel: nearestExam?.label ?? '',
      isExamDay: isExamDay,
      intensity: _intensityFor(units, selected.length),
      loadState: cognitiveLoad.loadState,
      recommendedUnits: units,
      receipt: _receiptFor(day, selected, stage, units),
      executionTasks: tasks,
      triggers: triggers,
      focusMantra: _focusMantra(selected, stage),
      stage: _stageLabel(stage),
      cognitiveLoadReceipt: _cognitiveLoadReceipt(),
    );
  }

  List<AxonDailyPlan> applySessionFeedback({
    required List<AxonDailyPlan> currentRoadmap,
    required StudySession latestSession,
  }) {
    if (currentRoadmap.isEmpty) return currentRoadmap;
    final affectedDates = {
      _dateOnly(latestSession.date.add(const Duration(days: 1))),
      _dateOnly(latestSession.date.add(const Duration(days: 2))),
    };
    return currentRoadmap.map((plan) {
      if (!affectedDates.contains(_dateOnly(plan.date))) return plan;
      if (latestSession.intensityIndex >= 0.45) return plan;
      final recoveryTask = DailyExecutionTask(
        taskId: '${_ymd(plan.date)}-recovery',
        moduleName: 'Recovery Session',
        estimatedTime: const Duration(minutes: 45),
        focusTechnique: 'Light Review',
        isHighPriority: true,
        subjectId: plan.subject,
        topic: 'Low-intensity consolidation',
        startTime: '17:00',
        recoveryNote:
            'Previous session intensity was low; next 48 hours are softened.',
      );
      return AxonDailyPlan(
        date: plan.date,
        subject: plan.subject,
        board: plan.board,
        examLabel: plan.examLabel,
        isExamDay: plan.isExamDay,
        intensity: math.min(plan.intensity, 0.45),
        loadState: 'recovery',
        recommendedUnits: math.min(plan.recommendedUnits, 2),
        receipt:
            '${plan.receipt} Feedback loop: low intensity triggered recovery load.',
        executionTasks: [recoveryTask, ...plan.executionTasks.take(1)],
        triggers: plan.triggers,
        focusMantra: plan.focusMantra,
        stage: plan.stage,
        cognitiveLoadReceipt:
            'Recovery mode because intensityIndex was ${latestSession.intensityIndex.toStringAsFixed(2)}.',
      );
    }).toList();
  }

  List<RankedSyllabusNode> _rankNodes(
    DateTime day,
    MonthlyPlanningStage stage,
  ) {
    final goalsByObjective = <String, PersonalGoal>{
      for (final goal in activeGoals)
        for (final id in goal.linkedObjectiveIds) id: goal,
    };
    final consistency = _consistencyScore();
    final ranked =
        syllabus.where((node) => !node.isExcellentRecently).map((node) {
      final exam = _nearestExam(node.subjectId, day);
      final days = exam == null
          ? 120
          : math.max(1, _dateOnly(exam.startDate).difference(day).inDays);
      final mastery = node.masteryScore.clamp(0, 1);
      final difficulty = _subjectDifficulty(node);
      final neuralPriority = ((1 - mastery) * difficulty) / days;
      final urgency = (1 / days).clamp(0, 1).toDouble();
      final weakness = (1 - mastery).clamp(0, 1).toDouble();
      final goalBoost = goalsByObjective[node.id]?.priority ?? 0;
      final stageBoost = switch (stage) {
        MonthlyPlanningStage.gapClosure => mastery < 0.5 ? 0.25 : 0,
        MonthlyPlanningStage.interleaving => 0.08,
        MonthlyPlanningStage.simulation =>
          node.supportsPastPaperSimulation ? 0.2 : -0.05,
      };
      final score = neuralPriority +
          (urgency * urgencyWeight) +
          (weakness * weaknessWeight) +
          ((1 - consistency) * consistencyWeight) +
          goalBoost +
          stageBoost;
      return RankedSyllabusNode(
        node: node,
        priorityScore: score,
        nearestExam: exam,
      );
    }).toList()
          ..sort((a, b) => b.priorityScore.compareTo(a.priorityScore));
    return ranked;
  }

  List<RankedSyllabusNode> _filterForReadiness(
    List<RankedSyllabusNode> ranked,
    MonthlyPlanningStage stage,
  ) {
    if (!cognitiveLoad.needsRecovery) return ranked;
    final stable = ranked.where((item) => !item.node.highVolatility).toList();
    if (stable.isNotEmpty) return stable;
    return ranked;
  }

  List<RankedSyllabusNode> _selectRotatedNodes(
    List<RankedSyllabusNode> ranked,
    MonthlyPlanningStage stage,
  ) {
    final limit = switch (stage) {
      MonthlyPlanningStage.gapClosure => 3,
      MonthlyPlanningStage.interleaving => 4,
      MonthlyPlanningStage.simulation => 3,
    };
    final selected = <RankedSyllabusNode>[];
    for (final item in ranked) {
      if (selected.length >= limit) break;
      if (selected.isNotEmpty &&
          _hasMathInterference(
              selected.last.node.subjectId, item.node.subjectId)) {
        continue;
      }
      selected.add(item);
    }
    if (selected.isEmpty && ranked.isNotEmpty) {
      selected.add(ranked.first);
    }
    return selected;
  }

  List<DailyExecutionTask> _buildExecutionTasks(
    DateTime day,
    List<RankedSyllabusNode> selected,
    MonthlyPlanningStage stage,
    double units,
  ) {
    final goldenHour = _goldenHour();
    final totalMinutes = math.max(45, (units * 60).round());
    final minutesByTask = selected.isEmpty
        ? totalMinutes
        : math.max(35, (totalMinutes / selected.length).round());
    final tasks = <DailyExecutionTask>[];

    for (var index = 0; index < selected.length; index++) {
      final item = selected[index];
      final cappedMinutes = math.min(90, minutesByTask);
      final startHour = goldenHour + index;
      tasks.add(
        DailyExecutionTask(
          taskId: '${_ymd(day)}-${item.node.id}-$index',
          moduleName: item.node.moduleName,
          estimatedTime: Duration(minutes: cappedMinutes),
          focusTechnique: _techniqueFor(item.node, stage),
          isHighPriority: item.priorityScore >= 0.55 || index == 0,
          subjectId: item.node.subjectId,
          objectiveId: item.node.id,
          topic: item.node.topicName,
          startTime: '${startHour.toString().padLeft(2, '0')}:00',
        ),
      );
      if (index < selected.length - 1 && cappedMinutes >= 75) {
        final restStart = DateTime(day.year, day.month, day.day, startHour)
            .add(Duration(minutes: cappedMinutes));
        tasks.add(
          DailyExecutionTask(
            taskId: '${_ymd(day)}-visual-rest-$index',
            moduleName: 'Visual Rest',
            estimatedTime: const Duration(minutes: 15),
            focusTechnique: 'Visual Rest',
            isHighPriority: false,
            subjectId: 'recovery',
            topic: 'Protect cognitive load before the next subject',
            startTime:
                '${restStart.hour.toString().padLeft(2, '0')}:${restStart.minute.toString().padLeft(2, '0')}',
          ),
        );
      }
    }

    if (tasks.isEmpty) {
      tasks.add(
        DailyExecutionTask(
          taskId: '${_ymd(day)}-maintenance',
          moduleName: 'Maintenance Review',
          estimatedTime: const Duration(minutes: 45),
          focusTechnique: 'Active Recall',
          isHighPriority: false,
          subjectId: 'General',
          topic: 'Light review and planning',
          startTime: '${goldenHour.toString().padLeft(2, '0')}:00',
        ),
      );
    }
    return tasks;
  }

  List<PlannerTrigger> _buildTriggers(
    DateTime day,
    List<DailyExecutionTask> tasks,
    List<RankedSyllabusNode> selected,
  ) {
    if (tasks.isEmpty || selected.isEmpty) return const [];
    final node = selected.first.node;
    if (node.pdfPath.isEmpty && node.pdfQuestionId.isEmpty) return const [];
    return [
      PlannerTrigger(
        id: '${_ymd(day)}-jarvis-${node.id}',
        fireAt: DateTime(day.year, day.month, day.day, 16),
        action: 'jarvis.open_pdf_question',
        payload: {
          'objective_id': node.id,
          'subject_id': node.subjectId,
          if (node.pdfQuestionId.isNotEmpty)
            'pdf_question_id': node.pdfQuestionId,
          if (node.pdfPath.isNotEmpty) 'pdf_path': node.pdfPath,
          if (node.pdfPage != null) 'page': node.pdfPage,
        },
      ),
    ];
  }

  ExamEvent? _nearestExam(String subjectId, DateTime day) {
    final matches = upcomingExams
        .where((exam) =>
            exam.subject.toLowerCase() == subjectId.toLowerCase() ||
            exam.label.toLowerCase().contains(subjectId.toLowerCase()))
        .where((exam) => !_dateOnly(exam.startDate).isBefore(day))
        .toList()
      ..sort((a, b) => a.startDate.compareTo(b.startDate));
    return matches.isEmpty ? null : matches.first;
  }

  ExamEvent? _nearestExamForPlan(
    DateTime day,
    List<RankedSyllabusNode> selected,
  ) {
    if (selected.isNotEmpty && selected.first.nearestExam != null) {
      return selected.first.nearestExam;
    }
    final future = upcomingExams
        .where((exam) => !_dateOnly(exam.endDate).isBefore(day))
        .toList()
      ..sort((a, b) => a.startDate.compareTo(b.startDate));
    return future.isEmpty ? null : future.first;
  }

  double _recommendedUnits(DateTime day, int selectedCount) {
    final base = user.targetStudyHours <= 0 ? 4.0 : user.targetStudyHours;
    final exam = _nearestExamForPlan(day, const []);
    final daysToExam = exam == null
        ? 60
        : math.max(1, _dateOnly(exam.startDate).difference(day).inDays);
    final urgencyMultiplier = daysToExam <= 7
        ? 1.25
        : daysToExam <= 21
            ? 1.1
            : 1.0;
    final buffered = base * urgencyMultiplier * 0.8;
    final ready = buffered * cognitiveLoad.readinessMultiplier;
    final taskFloor = selectedCount <= 1 ? 1.0 : selectedCount * 0.75;
    return math.max(taskFloor, ready).clamp(1.0, 6.0).toDouble();
  }

  double _intensityFor(double units, int selectedCount) {
    final density = selectedCount <= 0 ? 0.2 : selectedCount / 4;
    return ((units / 6) * 0.7 + density * 0.3).clamp(0.0, 1.0).toDouble();
  }

  MonthlyPlanningStage _stageForOffset(DateTime day) {
    final today = _dateOnly(DateTime.now());
    final offset = day.difference(today).inDays;
    if (offset < 14) return MonthlyPlanningStage.gapClosure;
    if (offset < 21) return MonthlyPlanningStage.interleaving;
    return MonthlyPlanningStage.simulation;
  }

  String _techniqueFor(SyllabusNode node, MonthlyPlanningStage stage) {
    if (stage == MonthlyPlanningStage.simulation) return 'Timed Past Paper';
    final text = '${node.subjectId} ${node.topicName}'.toLowerCase();
    if (text.contains('physics') || text.contains('9702')) return 'Feynman';
    if (text.contains('computer') || text.contains('9618')) {
      return 'Active Recall';
    }
    if (text.contains('math') ||
        text.contains('9709') ||
        text.contains('9231')) {
      return 'Pomodoro';
    }
    return stage == MonthlyPlanningStage.interleaving
        ? 'Interleaved Practice'
        : 'Active Recall';
  }

  String _focusMantra(
    List<RankedSyllabusNode> selected,
    MonthlyPlanningStage stage,
  ) {
    final topic =
        selected.isEmpty ? 'maintenance review' : selected.first.node.topicName;
    return switch (user.motivationStyle) {
      MotivationStyle.toughLove =>
        'No passive review. Execute $topic and close the gap.',
      MotivationStyle.logicBased =>
        'Priority is $topic because mastery, urgency, and exam proximity align.',
      MotivationStyle.positiveReinforcement =>
        'Today is built around $topic. One precise block moves the grade.',
    };
  }

  String _receiptFor(
    DateTime day,
    List<RankedSyllabusNode> selected,
    MonthlyPlanningStage stage,
    double units,
  ) {
    if (selected.isEmpty) {
      return 'Logic: no urgent weak topic found, so the day is set to maintenance.';
    }
    final top = selected.first;
    final examPart = top.nearestExam == null
        ? 'no fixed exam date'
        : '${top.nearestExam!.label} in ${math.max(0, _dateOnly(top.nearestExam!.startDate).difference(day).inDays)} days';
    return 'Logic: focus on ${top.node.subjectId}: ${top.node.topicName} '
        'because mastery is ${(top.node.masteryScore * 100).round()}%, '
        '$examPart, stage is ${_stageLabel(stage)}, and load is '
        '${units.toStringAsFixed(1)} units after buffer.';
  }

  String _cognitiveLoadReceipt() {
    if (cognitiveLoad.needsRecovery) {
      return 'Recovery applied: stress ${cognitiveLoad.currentStress.toStringAsFixed(1)}, sleep debt ${cognitiveLoad.sleepDebt.toStringAsFixed(1)}h.';
    }
    return '20% cognitive buffer applied before scheduling.';
  }

  String _stageLabel(MonthlyPlanningStage stage) => switch (stage) {
        MonthlyPlanningStage.gapClosure => 'Gap Closure',
        MonthlyPlanningStage.interleaving => 'Interleaving',
        MonthlyPlanningStage.simulation => 'Past Paper Simulation',
      };

  double _subjectDifficulty(SyllabusNode node) {
    if (node.subjectDifficulty > 0) return node.subjectDifficulty.clamp(0.1, 1);
    final subject = node.subjectId.toLowerCase();
    if (subject.contains('9231')) return 1.0;
    if (subject.contains('9709')) return 0.85;
    if (subject.contains('9702')) return 0.8;
    if (subject.contains('9618')) return 0.75;
    return 0.55;
  }

  double _consistencyScore() {
    if (history.isEmpty) return 0.4;
    final recent =
        history.length > 7 ? history.sublist(history.length - 7) : history;
    final completed = recent.where((item) => item.activeStudyHours > 0).length;
    return (completed / 7).clamp(0.0, 1.0).toDouble();
  }

  int _goldenHour() {
    if (sessionHistory.isEmpty) return 16;
    final buckets = <int, List<double>>{};
    for (final session in sessionHistory) {
      buckets
          .putIfAbsent(session.date.hour, () => [])
          .add(session.intensityIndex);
    }
    var bestHour = 16;
    var bestScore = -1.0;
    for (final entry in buckets.entries) {
      final average = entry.value.reduce((a, b) => a + b) / entry.value.length;
      if (average > bestScore) {
        bestScore = average;
        bestHour = entry.key;
      }
    }
    return bestHour.clamp(6, 21);
  }

  bool _hasMathInterference(String a, String b) {
    final first = a.toLowerCase();
    final second = b.toLowerCase();
    final firstMath = first.contains('9231') || first.contains('9709');
    final secondMath = second.contains('9231') || second.contains('9709');
    return firstMath && secondMath && first != second;
  }

  DateTime _dateOnly(DateTime value) =>
      DateTime(value.year, value.month, value.day);

  String _ymd(DateTime value) => value.toIso8601String().split('T').first;
}
