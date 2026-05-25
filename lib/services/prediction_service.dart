class PredictionService {
  static final PredictionService _instance = PredictionService._internal();
  factory PredictionService() => _instance;
  PredictionService._internal();

  double predict({
    double? mockScore,
    double? studyHours,
    double? targetHours,
    double? focusRatio,
    double? sleepHours,
    double? screenTimeHours,
    double? syllabusCoverage,
    int? consistencyStreak,
    double? stressLevel,
  }) {
    final mock = (mockScore ?? 0.0).clamp(0.0, 100.0);
    final hours = studyHours ?? 0.0;
    final target = targetHours ?? 6.0;
    final focus = (focusRatio ?? 0.5).clamp(0.0, 1.0);
    final sleep = (sleepHours ?? 7.0).clamp(0.0, 12.0);
    final screen = (screenTimeHours ?? 4.0).clamp(0.0, 16.0);
    final coverage = (syllabusCoverage ?? 0.0).clamp(0.0, 1.0);
    final streak = (consistencyStreak ?? 0).clamp(0, 30);
    final stress = (stressLevel ?? 0.3).clamp(0.0, 1.0);

    // Study ratio: how close to target hours (0-1)
    final studyRatio = target > 0 ? (hours / target).clamp(0.0, 1.0) : 0.0;

    // Sleep score: optimal 7-9 hours, penalize extremes
    double sleepScore;
    if (sleep >= 7.0 && sleep <= 9.0) {
      sleepScore = 1.0;
    } else if (sleep < 7.0) {
      sleepScore = (sleep / 7.0).clamp(0.0, 1.0);
    } else {
      sleepScore = (1.0 - ((sleep - 9.0) * 0.15)).clamp(0.0, 1.0);
    }

    // Screen time penalty: optimal under 4h, heavy penalty above 8h
    double screenScore;
    if (screen <= 4.0) {
      screenScore = 1.0;
    } else if (screen <= 8.0) {
      screenScore = 1.0 - ((screen - 4.0) / 4.0) * 0.3;
    } else {
      screenScore = (0.7 - ((screen - 8.0) / 8.0) * 0.5).clamp(0.1, 1.0);
    }

    // Streak bonus: 0-10 days = 0-0.1, 10-20 = 0.1-0.15, 20-30 = 0.15-0.2
    double streakBonus;
    if (streak <= 10) {
      streakBonus = streak * 0.01;
    } else if (streak <= 20) {
      streakBonus = 0.1 + (streak - 10) * 0.005;
    } else {
      streakBonus = 0.15 + (streak - 20) * 0.005;
    }
    streakBonus = streakBonus.clamp(0.0, 0.2);

    // Stress penalty
    final stressPenalty = stress * 0.15;

    // Weighted combination
    // If mock score exists, it dominates; otherwise base on study habits
    double baseScore;
    if (mock > 0) {
      // Balanced prediction: 30% mock, 25% study ratio, 20% focus, 15% sleep, 10% consistency
      final consistencyScore = (streak / 30).clamp(0.0, 1.0);
      baseScore = ((mock / 100).clamp(0, 1) * 0.30 +
              studyRatio * 0.25 +
              focus * 0.20 +
              sleepScore * 0.15 +
              consistencyScore * 0.10 -
              stressPenalty)
          .clamp(0.0, 1.0) * 100.0;
    } else {
      // Habit-based prediction when no mock score
      baseScore = (studyRatio * 25.0 +
              focus * 25.0 +
              sleepScore * 20.0 +
              screenScore * 10.0 +
              coverage * 10.0 +
              streakBonus * 10.0 -
              stressPenalty * 100.0)
          .clamp(0.0, 100.0);
    }

    return baseScore.clamp(0.0, 100.0);
  }

  String explain({
    required double currentScore,
    required double previousScore,
    required double sleepHours,
    required double sevenDayAvgSleep,
    required double screenTimeHours,
  }) {
    final delta = currentScore - previousScore;
    final sleep = sleepHours;
    final avgSleep = sevenDayAvgSleep;
    final screen = screenTimeHours;

    final parts = <String>[];

    // Score direction
    if (delta > 3) {
      parts.add('Strong upward trend');
    } else if (delta > 0.5) {
      parts.add('Gradual improvement');
    } else if (delta < -3) {
      parts.add('Sharp decline detected');
    } else if (delta < -0.5) {
      parts.add('Slight dip');
    } else {
      parts.add('Stable performance');
    }

    // Sleep feedback
    if (sleep < 6.0) {
      parts.add('sleep deficit hurting focus');
    } else if (avgSleep > 0 && sleep < avgSleep - 1.5) {
      parts.add('sleep below your average');
    } else if (sleep >= 7.0 && sleep <= 9.0) {
      parts.add('good sleep rhythm');
    }

    // Screen time feedback
    if (screen > 8.0) {
      parts.add('high screen time detected');
    } else if (screen > 5.0) {
      parts.add('moderate screen time');
    }

    return parts.join(' · ');
  }
}
