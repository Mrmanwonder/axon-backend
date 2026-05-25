// lib/models/study_activity.dart

class StudyActivity {
  final DateTime date;
  final int chaptersCompleted;

  StudyActivity({
    required this.date,
    required this.chaptersCompleted,
  });

  int get level => _calculateLevel(chaptersCompleted);

  int _calculateLevel(int count) {
    if (count == 0) return 0;
    if (count <= 2) return 1;
    if (count <= 4) return 2;
    return 3;
  }

  Map<String, dynamic> toJson() => {
        "date": date.toIso8601String().substring(0, 10),
        "level": level,
      };

  static StudyActivity empty(DateTime date) =>
      StudyActivity(date: date, chaptersCompleted: 0);

  static List<StudyActivity> generateLast35Days(List<int> levels) {
    final now = DateTime.now();
    final activities = <StudyActivity>[];

    for (int i = 34; i >= 0; i--) {
      final date = now.subtract(Duration(days: i));
      final level = i < levels.length ? levels[i] : 0;
      int chapters = 0;
      if (level == 1) {
        chapters = 1;
      } else if (level == 2) {
        chapters = 3;
      } else if (level == 3) {
        chapters = 5;
      }

      activities.add(StudyActivity(date: date, chaptersCompleted: chapters));
    }

    return activities;
  }
}

class StudyHeatmapSession {
  final DateTime date;
  final int minutes;
  final double quality;

  const StudyHeatmapSession({
    required this.date,
    required this.minutes,
    required this.quality,
  });

  factory StudyHeatmapSession.fromJson(Map<String, dynamic> json) {
    return StudyHeatmapSession(
      date: DateTime.tryParse(json['date'] as String? ?? '') ?? DateTime.now(),
      minutes: (json['minutes'] as num? ?? 0).toInt(),
      quality: ((json['quality'] as num? ?? 0).toDouble())
          .clamp(0.0, 1.0)
          .toDouble(),
    );
  }

  Map<String, dynamic> toJson() => {
        'date': date.toIso8601String().substring(0, 10),
        'minutes': minutes,
        'quality': quality.clamp(0.0, 1.0),
      };

  StudyHeatmapSession copyWith({
    DateTime? date,
    int? minutes,
    double? quality,
  }) {
    return StudyHeatmapSession(
      date: date ?? this.date,
      minutes: minutes ?? this.minutes,
      quality: quality ?? this.quality,
    );
  }

  static List<StudyHeatmapSession> generateLastNDaysFromLevels(
    List<int> levels, {
    int days = 28,
  }) {
    final now = DateTime.now();
    final padded = levels.length >= days
        ? levels.sublist(levels.length - days)
        : <int>[...List<int>.filled(days - levels.length, 0), ...levels];

    return List<StudyHeatmapSession>.generate(days, (index) {
      final level = padded[index].clamp(0, 3);
      final date = DateTime(now.year, now.month, now.day)
          .subtract(Duration(days: days - index - 1));
      final minutes = switch (level) {
        0 => 0,
        1 => 60,
        2 => 180,
        _ => 300,
      };
      final quality = switch (level) {
        0 => 0.0,
        1 => 0.35,
        2 => 0.65,
        _ => 1.0,
      };
      return StudyHeatmapSession(
        date: date,
        minutes: minutes,
        quality: quality,
      );
    });
  }
}

class ActivityColors {
  static const int empty = 0;
  static const int low = 1;
  static const int medium = 2;
  static const int high = 3;

  static int fromChapters(int chapters) {
    if (chapters == 0) return empty;
    if (chapters <= 2) return low;
    if (chapters <= 4) return medium;
    return high;
  }

  static List<int> colorList = [
    0xFF161B22,
    0xFF0E4429,
    0xFF006D32,
    0xFF39D353,
  ];
}
