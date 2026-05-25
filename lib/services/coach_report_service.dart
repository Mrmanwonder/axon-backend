import 'package:shared_preferences/shared_preferences.dart';

import '../models/models.dart';
import 'notification_service.dart';
import 'app_state.dart';

class CoachReport {
  final String title;
  final String message;
  final List<String> wins;
  final List<String> misses;
  final List<String> nextPriorities;
  final bool streakAtRisk;
  final bool perfectDay;
  final bool examPressure;
  final Duration recoveryWindow;
  final List<String> rewards;

  const CoachReport({
    required this.title,
    required this.message,
    required this.wins,
    required this.misses,
    required this.nextPriorities,
    required this.streakAtRisk,
    required this.perfectDay,
    required this.examPressure,
    required this.recoveryWindow,
    required this.rewards,
  });
}

class CoachReportService {
  CoachReportService._();

  static final CoachReportService instance = CoachReportService._();

  Future<CoachReport> buildReport({
    required MetricsState metrics,
    required MotivationStyle style,
    required int currentStreak,
    String board = '',
    String subject = '',
  }) async {
    final wins = <String>[];
    final misses = <String>[];
    final nextPriorities = <String>[];
    final rewards = <String>[];
    final targetHours =
        metrics.targetStudyHours <= 0 ? 4.0 : metrics.targetStudyHours;
    final progress =
        targetHours <= 0 ? 0.0 : metrics.activeStudyHours / targetHours;
    final weeklyHours = metrics.weekHistory.fold<double>(
      0.0,
      (sum, day) => sum + day.activeStudyHours,
    );
    final weeklyFocus = metrics.weekHistory.isEmpty
        ? metrics.focusRatio
        : metrics.weekHistory.fold<double>(
              0.0,
              (sum, day) => sum + day.focusRatio,
            ) /
            metrics.weekHistory.length;
    final studiedToday = metrics.activeStudyHours >= 0.08;
    final streakAtRisk = !studiedToday && DateTime.now().hour >= 16;
    final perfectDay = metrics.activeStudyHours >= targetHours &&
        metrics.focusRatio >= 0.78 &&
        metrics.screenTimeHours <= 5.5;
    final examDays = _estimateDaysUntilExam(board);
    final examPressure = examDays <= 21;
    final recoveryWindow = streakAtRisk
        ? Duration(hours: 24 - DateTime.now().hour)
        : Duration.zero;

    if (metrics.activeStudyHours > 0) {
      wins.add('${metrics.activeStudyHours.toStringAsFixed(1)}h logged today');
    }
    if (weeklyHours > 0) {
      wins.add('${weeklyHours.toStringAsFixed(1)}h accumulated this week');
    }
    if (metrics.focusRatio >= 0.8) {
      wins.add('Focus rhythm stayed above 80%');
    }
    if (currentStreak > 0) {
      wins.add('$currentStreak-day streak still alive');
    }
    if (perfectDay) {
      rewards.add('Perfect Day');
      wins.add('Daily goal cleared cleanly');
    }

    if (!studiedToday) {
      misses.add('No meaningful study logged yet today');
      nextPriorities.add('Start one focused session immediately');
    }
    if (metrics.screenTimeHours >= 6) {
      misses.add('Screen time is crowding out revision');
      nextPriorities.add('Cut 60 minutes of passive screen time tonight');
    }
    if (metrics.focusRatio < 0.6 && metrics.activeStudyHours > 0) {
      misses.add('Focus quality dipped below stable range');
      nextPriorities.add('Use a shorter mode and take cleaner breaks');
    }
    if (weeklyFocus < 0.62 && metrics.weekHistory.length >= 3) {
      misses.add('Weekly focus average is slipping');
      nextPriorities
          .add('Protect the first session of the day from interruptions');
    }
    if (examPressure) {
      misses.add('Exam window is closing fast');
      nextPriorities.add('Push weak-topic recall before fresh content');
    }
    if (currentStreak >= 7) {
      rewards.add('Week Warrior');
    }
    if (progress >= 1.0) {
      rewards.add('Goal Unlocked');
    }
    if (currentStreak > 0 && streakAtRisk) {
      rewards.add('Recovery Window');
    }

    final title = _titleForState(
      style: style,
      streakAtRisk: streakAtRisk,
      perfectDay: perfectDay,
      examPressure: examPressure,
      progress: progress,
    );
    final message = _messageForState(
      style: style,
      streakAtRisk: streakAtRisk,
      perfectDay: perfectDay,
      examPressure: examPressure,
      subject: subject,
      recoveryWindow: recoveryWindow,
    );

    return CoachReport(
      title: title,
      message: message,
      wins: wins,
      misses: misses,
      nextPriorities: nextPriorities,
      streakAtRisk: streakAtRisk,
      perfectDay: perfectDay,
      examPressure: examPressure,
      recoveryWindow: recoveryWindow,
      rewards: rewards,
    );
  }

  Future<void> maybeSendLossAversionReminder({
    required MetricsState metrics,
    required MotivationStyle style,
    required int currentStreak,
    String board = '',
    String subject = '',
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final now = DateTime.now();
    final dayKey = '${now.year}-${now.month}-${now.day}';
    if (prefs.getString('coach_loss_alert_day') == dayKey) return;

    final studiedToday = metrics.activeStudyHours >= 0.08;
    final streakAtRisk = !studiedToday && now.hour >= 18 && currentStreak > 0;
    final examDays = _estimateDaysUntilExam(board);
    final examPressure = examDays <= 14 && !studiedToday;
    if (!streakAtRisk && !examPressure) return;

    final body = streakAtRisk
        ? _streakRiskMessage(style, currentStreak)
        : _examRiskMessage(style, subject, examDays);
    await NotificationService.show(
      id: 4103,
      title: streakAtRisk ? 'Protect Your Streak' : 'Exam Window Closing',
      body: body,
      channel: AxonChannel.studyReminder,
    );
    await prefs.setString('coach_loss_alert_day', dayKey);
  }

  int _estimateDaysUntilExam(String board) {
    final now = DateTime.now();
    final base = board.toLowerCase().contains('a-level') ? 54 : 38;
    return base + ((now.month % 2) * 7);
  }

  String _titleForState({
    required MotivationStyle style,
    required bool streakAtRisk,
    required bool perfectDay,
    required bool examPressure,
    required double progress,
  }) {
    if (perfectDay) return 'Perfect day secured.';
    if (streakAtRisk) {
      return switch (style) {
        MotivationStyle.toughLove => 'Your streak is bleeding.',
        MotivationStyle.positiveReinforcement =>
          'Your streak still can be saved.',
        MotivationStyle.logicBased => 'Streak loss probability is rising.',
      };
    }
    if (examPressure) {
      return switch (style) {
        MotivationStyle.toughLove => 'The exam clock is not waiting.',
        MotivationStyle.positiveReinforcement =>
          'This is the moment to tighten up.',
        MotivationStyle.logicBased => 'Exam urgency crossed the warning line.',
      };
    }
    if (progress >= 1.0) return 'Goal unlocked.';
    return switch (style) {
      MotivationStyle.toughLove => 'The day is still open. Use it.',
      MotivationStyle.positiveReinforcement => 'Momentum is building.',
      MotivationStyle.logicBased => 'Daily execution snapshot.',
    };
  }

  String _messageForState({
    required MotivationStyle style,
    required bool streakAtRisk,
    required bool perfectDay,
    required bool examPressure,
    required String subject,
    required Duration recoveryWindow,
  }) {
    if (perfectDay) {
      return 'You hit your goal with clean focus. Bank the win and come back tomorrow before the streak cools.';
    }
    if (streakAtRisk) {
      final hours = recoveryWindow.inHours.clamp(1, 24);
      return switch (style) {
        MotivationStyle.toughLove =>
          'You have about $hours hours to keep this streak alive. One real session now is cheaper than rebuilding from zero.',
        MotivationStyle.positiveReinforcement =>
          'A single focused session in the next $hours hours keeps your streak safe. Protect the work you already earned.',
        MotivationStyle.logicBased =>
          'Recovery window: ~$hours hours. Starting one session now preserves your streak and stabilizes tomorrow\'s prediction.',
      };
    }
    if (examPressure) {
      final label = subject.isEmpty ? 'your weakest subject' : subject;
      return switch (style) {
        MotivationStyle.toughLove =>
          'Exams are close. Stop leaking days and attack $label tonight.',
        MotivationStyle.positiveReinforcement =>
          'Exams are getting close, but a focused push on $label still moves the curve.',
        MotivationStyle.logicBased =>
          'Exam pressure is up. The highest-return move now is recall work on $label.',
      };
    }
    return switch (style) {
      MotivationStyle.toughLove =>
        'The system is live. Keep your sessions clean and stop handing time away.',
      MotivationStyle.positiveReinforcement =>
        'You do not need a perfect day, just the next clean block.',
      MotivationStyle.logicBased =>
        'Current guidance: protect focus ratio, preserve the streak, and push the weakest subject first.',
    };
  }

  String _streakRiskMessage(MotivationStyle style, int streak) {
    return switch (style) {
      MotivationStyle.toughLove =>
        '$streak days of work are on the line. Start now or accept the reset.',
      MotivationStyle.positiveReinforcement =>
        'Your $streak-day streak is still recoverable. One focused session protects it.',
      MotivationStyle.logicBased =>
        'Streak risk detected: $streak days active. One completed session prevents a reset.',
    };
  }

  String _examRiskMessage(MotivationStyle style, String subject, int examDays) {
    final label = subject.isEmpty ? 'priority subject' : subject;
    return switch (style) {
      MotivationStyle.toughLove =>
        '$examDays days left. $label needs work today, not tomorrow.',
      MotivationStyle.positiveReinforcement =>
        '$examDays days left. A focused session on $label today keeps you ahead of the panic curve.',
      MotivationStyle.logicBased =>
        '$examDays days to exam. Highest-value next step: recall and timed practice in $label.',
    };
  }
}
