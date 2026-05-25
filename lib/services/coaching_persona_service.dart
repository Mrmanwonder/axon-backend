import 'personalization_service.dart';

class CoachingPersonaService {
  static final CoachingPersonaService _instance =
      CoachingPersonaService._internal();
  factory CoachingPersonaService() => _instance;
  CoachingPersonaService._internal();

  CoachingPersona _activePersona = CoachingPersona.mentor;

  Future<void> initialize() async {}

  String setActivePersona(CoachingPersona persona) {
    _activePersona = persona;
    return 'ok';
  }

  CoachingPersona get activePersona => _activePersona;

  String getNotificationTitle(String type) {
    switch (_activePersona) {
      case CoachingPersona.mentor:
        return 'Axon Coach';
      case CoachingPersona.drillSergeant:
        return 'Axon Command';
      case CoachingPersona.cheerleader:
        return 'Axon Spirit';
      case CoachingPersona.scientist:
        return 'Axon Analytics';
      case CoachingPersona.buddy:
        return 'Axon Buddy';
    }
  }

  String getNeglectReminder(String subject, String chapter, int daysSince) {
    switch (_activePersona) {
      case CoachingPersona.mentor:
        return 'You have not studied $chapter in $daysSince days. A quick session now will keep your progress on track.';
      case CoachingPersona.drillSergeant:
        return '$daysSince days since $chapter. Get back to work now.';
      case CoachingPersona.cheerleader:
        return '$chapter is waiting for you. It has been $daysSince days, and you can handle this.';
      case CoachingPersona.scientist:
        return 'Data point: $daysSince days since last $chapter session. Retention probability is declining. Recommend immediate review.';
      case CoachingPersona.buddy:
        return 'It has been $daysSince days since you looked at $chapter. Want to knock it out together?';
    }
  }

  String getExamWarningMessage(int days, List<String> subjects) {
    final subjStr = subjects.join(', ');
    switch (_activePersona) {
      case CoachingPersona.mentor:
        return '$days days until your exam. Focus on: $subjStr. Let us make a plan.';
      case CoachingPersona.drillSergeant:
        return '$days days left. Subjects: $subjStr. Stop procrastinating and start grinding now.';
      case CoachingPersona.cheerleader:
        return '$days days to go. You can absolutely crush $subjStr.';
      case CoachingPersona.scientist:
        return 'Time remaining: $days days. Estimated coverage gap in $subjStr. Recommended: ${(subjects.length * 2).toStringAsFixed(0)}h/day for optimal readiness.';
      case CoachingPersona.buddy:
        return 'Only $days days left for $subjStr. Let us chip away at it together.';
    }
  }

  String getWinMessage(String subject, String chapter, double improvement) {
    switch (_activePersona) {
      case CoachingPersona.mentor:
        return 'Solid progress in $chapter ($subject). Keep this momentum going.';
      case CoachingPersona.drillSergeant:
        return 'Good work on $chapter. The next target starts now.';
      case CoachingPersona.cheerleader:
        return 'Amazing job on $chapter. You improved by ${improvement.toStringAsFixed(0)}%.';
      case CoachingPersona.scientist:
        return 'Improvement detected in $chapter: +${improvement.toStringAsFixed(0)}% ($subject). This positive trend correlates with increased study frequency.';
      case CoachingPersona.buddy:
        return 'You are making real progress in $chapter: ${improvement.toStringAsFixed(0)}% better.';
    }
  }

  String getDailyBriefing({
    required double studiedYesterday,
    required double targetHours,
    required double todayHours,
    required double focusQuality,
    required int streak,
    List<String> weakSubjects = const [],
  }) {
    final studiedPct =
        targetHours > 0 ? (studiedYesterday / targetHours * 100).round() : 0;

    String subjectNote = '';
    if (weakSubjects.isNotEmpty) {
      subjectNote =
          ' Give extra attention to ${weakSubjects.take(2).join(' and ')}.';
    }

    final hourStr = studiedYesterday.toStringAsFixed(1);

    switch (_activePersona) {
      case CoachingPersona.mentor:
        return 'You studied $hourStr hours yesterday ($studiedPct% of target). Focus quality: ${(focusQuality * 100).round()}%.$subjectNote';
      case CoachingPersona.drillSergeant:
        if (studiedPct < 50) {
          return 'Only $hourStr hours yesterday? That is $studiedPct%. Today target: ${targetHours.toStringAsFixed(0)}h. Move.';
        }
        return '$hourStr hours yesterday ($studiedPct%). $streak-day streak. Maintain discipline.$subjectNote';
      case CoachingPersona.cheerleader:
        return '$hourStr hours yesterday. You are $studiedPct% of the way. $streak-day streak. Keep going.$subjectNote';
      case CoachingPersona.scientist:
        final focusPct = (focusQuality * 100).round();
        return 'Report: $hourStr h / ${targetHours.toStringAsFixed(0)}h target ($studiedPct%). Focus: $focusPct%. Streak: $streak days.$subjectNote';
      case CoachingPersona.buddy:
        return 'You did $hourStr hours yesterday. That is $studiedPct% of your goal. Keep it up today.$subjectNote';
    }
  }
}
