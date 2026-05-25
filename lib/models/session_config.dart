// lib/models/session_config.dart
// SessionConfig model - defines timer session configuration

class SessionConfig {
  final String subjectId;
  final String subjectName;
  final String chapterId;
  final String chapterName;
  final Duration duration;
  final bool isPomodoro;
  final bool isStrictExamMode;
  final bool isStopwatch;
  final PomodoroConfig? pomodoroConfig;

  const SessionConfig({
    required this.subjectId,
    required this.subjectName,
    required this.chapterId,
    required this.chapterName,
    this.duration = const Duration(minutes: 25),
    this.isPomodoro = false,
    this.isStrictExamMode = false,
    this.isStopwatch = false,
    this.pomodoroConfig,
  });

  SessionConfig copyWith({
    String? subjectId,
    String? subjectName,
    String? chapterId,
    String? chapterName,
    Duration? duration,
    bool? isPomodoro,
    bool? isStrictExamMode,
    bool? isStopwatch,
    PomodoroConfig? pomodoroConfig,
  }) {
    return SessionConfig(
      subjectId: subjectId ?? this.subjectId,
      subjectName: subjectName ?? this.subjectName,
      chapterId: chapterId ?? this.chapterId,
      chapterName: chapterName ?? this.chapterName,
      duration: duration ?? this.duration,
      isPomodoro: isPomodoro ?? this.isPomodoro,
      isStrictExamMode: isStrictExamMode ?? this.isStrictExamMode,
      isStopwatch: isStopwatch ?? this.isStopwatch,
      pomodoroConfig: pomodoroConfig ?? this.pomodoroConfig,
    );
  }

  Map<String, dynamic> toJson() => {
        'subjectId': subjectId,
        'subjectName': subjectName,
        'chapterId': chapterId,
        'chapterName': chapterName,
        'durationMinutes': duration.inMinutes,
        'isPomodoro': isPomodoro,
        'isStrictExamMode': isStrictExamMode,
        'isStopwatch': isStopwatch,
        'pomodoroConfig': pomodoroConfig?.toJson(),
      };
}

class PomodoroConfig {
  final Duration focusDuration;
  final Duration breakDuration;
  final int longBreakInterval;
  final Duration longBreakDuration;

  const PomodoroConfig({
    this.focusDuration = const Duration(minutes: 25),
    this.breakDuration = const Duration(minutes: 5),
    this.longBreakInterval = 4,
    this.longBreakDuration = const Duration(minutes: 15),
  });

  factory PomodoroConfig.autoCalculate(Duration totalTime) {
    final minutes = totalTime.inMinutes;
    if (minutes <= 30) {
      return const PomodoroConfig(
        focusDuration: Duration(minutes: 25),
        breakDuration: Duration(minutes: 5),
      );
    } else if (minutes <= 60) {
      return const PomodoroConfig(
        focusDuration: Duration(minutes: 50),
        breakDuration: Duration(minutes: 10),
      );
    } else {
      return const PomodoroConfig(
        focusDuration: Duration(minutes: 90),
        breakDuration: Duration(minutes: 15),
      );
    }
  }

  Map<String, dynamic> toJson() => {
        'focusDurationMinutes': focusDuration.inMinutes,
        'breakDurationMinutes': breakDuration.inMinutes,
        'longBreakInterval': longBreakInterval,
        'longBreakDurationMinutes': longBreakDuration.inMinutes,
      };
}

class TimerPreset {
  final String id;
  final String name;
  final String? subjectId;
  final Duration duration;
  final bool isUserFavorite;
  final bool isBoardPreset;
  final DateTime? createdAt;

  const TimerPreset({
    required this.id,
    required this.name,
    this.subjectId,
    required this.duration,
    this.isUserFavorite = false,
    this.isBoardPreset = false,
    this.createdAt,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'subjectId': subjectId,
        'durationMinutes': duration.inMinutes,
        'isUserFavorite': isUserFavorite,
        'isBoardPreset': isBoardPreset,
        'createdAt': createdAt?.toIso8601String(),
      };

  factory TimerPreset.fromJson(Map<String, dynamic> json) {
    return TimerPreset(
      id: json['id']?.toString() ?? '',
      name: json['name']?.toString() ?? '',
      subjectId: json['subjectId']?.toString(),
      duration:
          Duration(minutes: (json['durationMinutes'] as num?)?.toInt() ?? 25),
      isUserFavorite: json['isUserFavorite'] == true,
      isBoardPreset: json['isBoardPreset'] == true,
      createdAt: json['createdAt'] != null
          ? DateTime.tryParse(json['createdAt'].toString())
          : null,
    );
  }

  String get formattedDuration {
    final h = duration.inHours;
    final m = duration.inMinutes % 60;
    if (h > 0) {
      return '${h}h ${m}m';
    }
    return '${m}m';
  }
}

class SessionEvent {
  final String id;
  final String subjectId;
  final String chapterId;
  final DateTime startTime;
  final DateTime? endTime;
  final Duration duration;
  final SessionEventType eventType;
  final bool isCompleted;
  final Map<String, dynamic>? metadata;

  const SessionEvent({
    required this.id,
    required this.subjectId,
    required this.chapterId,
    required this.startTime,
    this.endTime,
    required this.duration,
    required this.eventType,
    this.isCompleted = false,
    this.metadata,
  });
}

enum SessionEventType {
  pomodoro,
  stopwatch,
  examSimulation,
  freeStudy,
}
