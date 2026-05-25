import 'paper.dart';

// ─── Session ───────────────────────────────────────────────────────────────

enum Session { march, summer, winter }

extension SessionExt on Session {
  String get displayName {
    switch (this) {
      case Session.march:  return 'March';
      case Session.summer: return 'May / Jun';
      case Session.winter: return 'Oct / Nov';
    }
  }

  String get shortCode {
    switch (this) {
      case Session.march:  return 'm';
      case Session.summer: return 's';
      case Session.winter: return 'w';
    }
  }
}

// ─── PaperSeries ───────────────────────────────────────────────────────────

class PaperSeries {
  final String id;
  final int year;
  final Session session;
  final List<Paper> papers;

  const PaperSeries({
    required this.id,
    required this.year,
    required this.session,
    required this.papers,
  });

  String get displayLabel => '${session.displayName}  ·  $year';

  String get shortLabel {
    final y = (year % 100).toString().padLeft(2, '0');
    return "${session.shortCode.toUpperCase()}'$y";
  }

  int get paperCount => papers.length;

  Set<SubjectGroup> get subjectGroups =>
      papers.map((p) => p.subjectGroup).toSet();
}
