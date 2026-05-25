import 'dart:math' as math;

import '../models/academic_engine_models.dart';

abstract class SyncfusionCalendarBridge {
  Future<void> syncStudyPlan(List<CalendarSyncEvent> events);
}

abstract class GoogleCalendarBridge {
  Future<void> syncStudyPlan(List<CalendarSyncEvent> events);
}

class PlannerObjective {
  final String learningObjectiveId;
  final String title;
  final double estimatedLoad;
  final double masteryScore;
  final double highYieldWeight;
  final List<String> prerequisiteIds;
  final bool supportsTimedMock;

  const PlannerObjective({
    required this.learningObjectiveId,
    required this.title,
    this.estimatedLoad = 1,
    this.masteryScore = 0,
    this.highYieldWeight = 0,
    this.prerequisiteIds = const [],
    this.supportsTimedMock = true,
  });
}

class BackwardsInductionPlanner {
  const BackwardsInductionPlanner({
    this.syncfusionBridge,
    this.googleCalendarBridge,
  });

  final SyncfusionCalendarBridge? syncfusionBridge;
  final GoogleCalendarBridge? googleCalendarBridge;

  PlannerResult plan({
    required DateTime examDate,
    required List<PlannerObjective> remainingObjectives,
    required List<NoStudyZone> noStudyZones,
    DateTime? startDate,
  }) {
    final origin = _normalize(startDate ?? DateTime.now());
    final anchor = _normalize(examDate);
    if (anchor.isBefore(origin)) {
      return const PlannerResult(
        days: [],
        baselineDailyLoad: 0,
        adjustedDailyLoad: 0,
        peakIntensity: 0,
        redistributedLoad: 0,
      );
    }

    final allDays = <DateTime>[];
    for (var cursor = origin;
        !cursor.isAfter(anchor.subtract(const Duration(days: 1)));
        cursor = cursor.add(const Duration(days: 1))) {
      allDays.add(cursor);
    }

    final activeDays = allDays
        .where((day) => noStudyZones.every((zone) => !zone.contains(day)))
        .toList();
    final blockedDays = allDays.length - activeDays.length;

    final totalLoad = remainingObjectives.fold<double>(
      0,
      (sum, item) => sum + item.estimatedLoad,
    );
    final totalDays = allDays.isEmpty ? 0 : allDays.length;
    final baselineDailyLoad = totalDays == 0 ? 0 : totalLoad / totalDays;
    final adjustedDailyLoad =
        activeDays.isEmpty ? 0 : totalLoad / activeDays.length;
    final redistributedLoad =
        math.max(0, adjustedDailyLoad - baselineDailyLoad);

    final queue = _topologicalObjectives(remainingObjectives);
    final allocationsByDate = <DateTime, List<PlannerTaskAllocation>>{};
    final scheduledObjectives = <String>{};
    for (final day in activeDays) {
      allocationsByDate[day] = [];
      var assigned = 0.0;
      final phase = _phaseFor(day: day, examDate: anchor);
      final targetLoad = _targetLoadForPhase(
        phase: phase,
        adjustedDailyLoad: adjustedDailyLoad.toDouble(),
      );
      while (queue.isNotEmpty && assigned < targetLoad + 0.01) {
        final index = _nextObjectiveIndex(queue, phase, scheduledObjectives);
        final objective = queue.removeAt(index);
        allocationsByDate[day]!.add(
          PlannerTaskAllocation(
            learningObjectiveId: objective.learningObjectiveId,
            title: _taskTitleForPhase(phase, objective.title),
            loadUnits: objective.estimatedLoad,
          ),
        );
        scheduledObjectives.add(objective.learningObjectiveId);
        assigned += objective.estimatedLoad;
      }
    }

    final plannedDays = allDays.map((day) {
      final blocked = noStudyZones.any((zone) => zone.contains(day));
      final tasks = allocationsByDate[day] ?? const <PlannerTaskAllocation>[];
      final currentLoad =
          tasks.fold<double>(0, (sum, item) => sum + item.loadUnits);
      final phase = _phaseFor(day: day, examDate: anchor);
      final intensity = blocked || adjustedDailyLoad == 0
          ? 0.0
          : (currentLoad / math.max(adjustedDailyLoad, 0.1))
              .clamp(0.0, 3.0)
              .toDouble();
      return PlannedStudyDay(
        date: day,
        blocked: blocked,
        tasks: tasks,
        dailyLoad: currentLoad,
        intensityScore: intensity,
        adjustedDailyLoad: adjustedDailyLoad.toDouble(),
        bufferLoadShift: redistributedLoad.toDouble(),
        phase: phase,
        intensityBand: _intensityBandFor(intensity),
      );
    }).toList();

    final peakIntensity = plannedDays.fold<double>(
      0,
      (peak, day) => math.max(peak, day.intensityScore),
    );

    return PlannerResult(
      days: plannedDays,
      baselineDailyLoad: baselineDailyLoad.toDouble(),
      adjustedDailyLoad: adjustedDailyLoad.toDouble(),
      peakIntensity: peakIntensity,
      redistributedLoad: blockedDays == 0 ? 0 : redistributedLoad.toDouble(),
    );
  }

  Future<void> sync({
    required PlannerResult result,
    required String subject,
  }) async {
    final events = buildCalendarEvents(result: result, subject: subject);

    if (syncfusionBridge != null) {
      await syncfusionBridge!.syncStudyPlan(events);
    }
    if (googleCalendarBridge != null) {
      await googleCalendarBridge!.syncStudyPlan(events);
    }
  }

  List<CalendarSyncEvent> buildCalendarEvents({
    required PlannerResult result,
    required String subject,
  }) {
    return result.days
        .where((day) => day.tasks.isNotEmpty)
        .map(
          (day) => CalendarSyncEvent(
            title: '${_phaseLabel(day.phase)}: $subject',
            start: DateTime(day.date.year, day.date.month, day.date.day, 17),
            end: DateTime(day.date.year, day.date.month, day.date.day, 18),
            description: '${_intensityLabel(day.intensityBand)} intensity | '
                'load ${day.dailyLoad.toStringAsFixed(1)} / '
                '${day.adjustedDailyLoad.toStringAsFixed(1)}\n'
                '${day.tasks.map((task) => '${task.title} (${task.learningObjectiveId})').join(', ')}',
          ),
        )
        .toList();
  }

  DateTime _normalize(DateTime value) =>
      DateTime(value.year, value.month, value.day);

  List<PlannerObjective> _topologicalObjectives(
      List<PlannerObjective> objectives) {
    final byId = {
      for (final item in objectives) item.learningObjectiveId: item
    };
    final visited = <String>{};
    final visiting = <String>{};
    final ordered = <PlannerObjective>[];

    void visit(String id) {
      if (visited.contains(id) || visiting.contains(id)) return;
      final objective = byId[id];
      if (objective == null) return;
      visiting.add(id);
      for (final prereq in objective.prerequisiteIds) {
        visit(prereq);
      }
      visiting.remove(id);
      visited.add(id);
      ordered.add(objective);
    }

    final ranked = [...objectives]
      ..sort((a, b) => b.highYieldWeight.compareTo(a.highYieldWeight));
    for (final objective in ranked) {
      visit(objective.learningObjectiveId);
    }
    return ordered
        .where((item) => byId.containsKey(item.learningObjectiveId))
        .toList();
  }

  int _nextObjectiveIndex(
    List<PlannerObjective> queue,
    StudyPlanPhase phase,
    Set<String> scheduledObjectives,
  ) {
    if (queue.length <= 1) return 0;
    var bestIndex = 0;
    var bestScore = double.negativeInfinity;
    for (var index = 0; index < queue.length; index++) {
      final objective = queue[index];
      final prereqsSatisfied = objective.prerequisiteIds.every(
        scheduledObjectives.contains,
      );
      if (!prereqsSatisfied) {
        continue;
      }
      final score = switch (phase) {
        StudyPlanPhase.timedMockSprint =>
          (objective.supportsTimedMock ? 40 : 0) +
              (objective.highYieldWeight * 10) +
              (objective.masteryScore * 5),
        StudyPlanPhase.hardTopicDeepDive =>
          ((1 - objective.masteryScore) * 40) + (objective.highYieldWeight * 6),
        StudyPlanPhase.firstPassCompletion =>
          ((1 - objective.masteryScore) * 18) + (objective.highYieldWeight * 4),
        StudyPlanPhase.foundationBuild =>
          (objective.highYieldWeight * 5) - objective.prerequisiteIds.length,
      };
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }
    if (bestScore == double.negativeInfinity) {
      return 0;
    }
    return bestIndex;
  }

  StudyPlanPhase _phaseFor({
    required DateTime day,
    required DateTime examDate,
  }) {
    final daysRemaining = examDate.difference(day).inDays;
    if (daysRemaining <= 7) return StudyPlanPhase.timedMockSprint;
    if (daysRemaining <= 14) return StudyPlanPhase.hardTopicDeepDive;
    if (daysRemaining <= 30) return StudyPlanPhase.firstPassCompletion;
    return StudyPlanPhase.foundationBuild;
  }

  double _targetLoadForPhase({
    required StudyPlanPhase phase,
    required double adjustedDailyLoad,
  }) {
    final multiplier = switch (phase) {
      StudyPlanPhase.timedMockSprint => 1.2,
      StudyPlanPhase.hardTopicDeepDive => 1.1,
      StudyPlanPhase.firstPassCompletion => 1.0,
      StudyPlanPhase.foundationBuild => 0.9,
    };
    return adjustedDailyLoad * multiplier;
  }

  String _taskTitleForPhase(StudyPlanPhase phase, String title) {
    final prefix = switch (phase) {
      StudyPlanPhase.timedMockSprint => 'Timed Mock',
      StudyPlanPhase.hardTopicDeepDive => 'Hard-Topic Deep Dive',
      StudyPlanPhase.firstPassCompletion => 'First Pass',
      StudyPlanPhase.foundationBuild => 'Foundation Build',
    };
    return '$prefix: $title';
  }

  IntensityBand _intensityBandFor(double intensity) {
    if (intensity >= 2.2) return IntensityBand.red;
    if (intensity >= 1.15) return IntensityBand.orange;
    return IntensityBand.blue;
  }

  String _phaseLabel(StudyPlanPhase phase) {
    return switch (phase) {
      StudyPlanPhase.timedMockSprint => 'T-7 Mock Sprint',
      StudyPlanPhase.hardTopicDeepDive => 'T-14 Deep Dive',
      StudyPlanPhase.firstPassCompletion => 'T-30 Completion',
      StudyPlanPhase.foundationBuild => 'Foundation Build',
    };
  }

  String _intensityLabel(IntensityBand band) {
    return switch (band) {
      IntensityBand.blue => 'Blue',
      IntensityBand.orange => 'Orange',
      IntensityBand.red => 'Red',
    };
  }
}
