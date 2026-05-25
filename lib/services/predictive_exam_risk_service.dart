import 'dart:math' as math;

class PredictiveExamRiskSnapshot {
  final double coverage;
  final double weightedMockAccuracy;
  final int daysRemaining;
  final double timeProximityScore;
  final double readinessScore;
  final String projectedGrade;
  final bool crisisMode;

  const PredictiveExamRiskSnapshot({
    required this.coverage,
    required this.weightedMockAccuracy,
    required this.daysRemaining,
    required this.timeProximityScore,
    required this.readinessScore,
    required this.projectedGrade,
    required this.crisisMode,
  });
}

class PredictiveExamRiskService {
  const PredictiveExamRiskService();

  PredictiveExamRiskSnapshot calculate({
    required double syllabusCoverage,
    required double weightedMockAccuracy,
    required DateTime anchorDate,
    DateTime? now,
  }) {
    final current = now ?? DateTime.now();
    final normalizedCoverage = syllabusCoverage.clamp(0.0, 1.0);
    final normalizedAccuracy = weightedMockAccuracy.clamp(0.0, 1.0);
    final daysRemaining =
        math.max(0, anchorDate.difference(current).inDays);

    // 90 days is the healthy planning runway. As that runway shrinks, risk rises.
    final timeProximityScore = (daysRemaining / 90.0).clamp(0.0, 1.0);
    final readinessScore = (((normalizedCoverage * 0.35) +
                (normalizedAccuracy * 0.45) +
                (timeProximityScore * 0.20)) *
            100.0)
        .clamp(0.0, 100.0);
    final projectedGrade = _projectGrade(readinessScore);

    return PredictiveExamRiskSnapshot(
      coverage: normalizedCoverage,
      weightedMockAccuracy: normalizedAccuracy,
      daysRemaining: daysRemaining,
      timeProximityScore: timeProximityScore,
      readinessScore: readinessScore,
      projectedGrade: projectedGrade,
      crisisMode: readinessScore < 50.0,
    );
  }

  String _projectGrade(double readinessScore) {
    if (readinessScore >= 85) return 'A*/7';
    if (readinessScore >= 72) return 'A/6';
    if (readinessScore >= 60) return 'B/5';
    if (readinessScore >= 50) return 'C/4';
    return 'D or below';
  }
}
