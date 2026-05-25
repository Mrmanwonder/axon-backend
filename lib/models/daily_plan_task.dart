enum TaskType {
  deepWork('deep_work'),
  practice('practice'),
  review('review'),
  pastPaper('past_paper'),
  flashcards('flashcards'),
  mockExam('mock_exam'),
  commandWordDrill('command_word_drill');

  final String value;
  const TaskType(this.value);

  static TaskType fromString(String s) {
    if (s == 'revision') return TaskType.review;
    return TaskType.values.firstWhere((e) => e.value == s, orElse: () => TaskType.deepWork);
  }
}

enum TaskStatus {
  pending('pending'),
  completed('completed'),
  rescheduled('rescheduled'),
  trimmed('trimmed');

  final String value;
  const TaskStatus(this.value);

  static TaskStatus fromString(String s) =>
      TaskStatus.values.firstWhere((e) => e.value == s, orElse: () => TaskStatus.pending);
}

enum IntensityLevel {
  blue('blue'),
  orange('orange'),
  red('red');

  final String value;
  const IntensityLevel(this.value);

  static IntensityLevel fromString(String s) =>
      IntensityLevel.values.firstWhere((e) => e.value == s, orElse: () => IntensityLevel.blue);
}

enum StudyPhase {
  foundation('Foundation Build'),
  t30Completion('T-30 Completion'),
  t14DeepDive('T-14 Deep Dive'),
  t7MockSprint('T-7 Mock Sprint');

  final String label;
  const StudyPhase(this.label);

  static StudyPhase fromDays(int daysUntilExam) {
    if (daysUntilExam <= 7) return StudyPhase.t7MockSprint;
    if (daysUntilExam <= 14) return StudyPhase.t14DeepDive;
    if (daysUntilExam <= 30) return StudyPhase.t30Completion;
    return StudyPhase.foundation;
  }

  static StudyPhase fromString(String s) =>
      StudyPhase.values.firstWhere(
        (e) => e.label == s || e.name == s,
        orElse: () => StudyPhase.foundation,
      );
}

enum ScheduledWindow {
  morning('morning'),
  afternoon('afternoon'),
  evening('evening'),
  peakFocusMorning('peak_focus_morning'),
  reviewEvening('review_evening');

  final String value;
  const ScheduledWindow(this.value);

  static ScheduledWindow fromString(String s) =>
      ScheduledWindow.values.firstWhere((e) => e.value == s, orElse: () => ScheduledWindow.morning);
}

enum Priority { none, low, medium, high }

extension TaskTypeDisplay on TaskType {
  String get label {
    switch (this) {
      case TaskType.deepWork: return 'Deep Work';
      case TaskType.practice: return 'Practice';
      case TaskType.review: return 'Review';
      case TaskType.flashcards: return 'Flashcards';
      case TaskType.pastPaper: return 'Past Paper';
      case TaskType.mockExam: return 'Mock Exam';
      case TaskType.commandWordDrill: return 'Command Drill';
    }
  }

  String get routePath {
    switch (this) {
      case TaskType.pastPaper:
      case TaskType.commandWordDrill:
        return '/study/quiz';
      case TaskType.flashcards:
        return '/study/flashcards';
      case TaskType.mockExam:
      case TaskType.deepWork:
      case TaskType.practice:
      case TaskType.review:
        return '/timer';
    }
  }

  bool get needsSubjectParam => this == TaskType.pastPaper ||
      this == TaskType.flashcards ||
      this == TaskType.commandWordDrill;

  String buildRoute(String subject) =>
      needsSubjectParam ? '$routePath?subject=${Uri.encodeComponent(subject)}' : routePath;
}

class DailyPlanTask {
  final String id;
  final String title;
  final String subject;
  final String description;
  final String paper;
  final String objectiveId;
  final DateTime startTime;
  final DateTime endTime;
  final TaskStatus status;
  final String date;
  final String reason;
  final bool isSyncToGoogle;
  final bool isCompleted;
  final double intensityScore;
  final IntensityLevel intensityLabel;
  final StudyPhase phase;
  final String anchorDate;
  final TaskType taskType;
  final ScheduledWindow scheduledWindow;
  final Priority priority;

  DailyPlanTask({
    required this.id,
    required this.title,
    required this.subject,
    this.description = '',
    this.paper = '',
    this.objectiveId = '',
    required this.startTime,
    required this.endTime,
    this.status = TaskStatus.pending,
    required this.date,
    this.reason = '',
    this.isSyncToGoogle = false,
    this.isCompleted = false,
    this.intensityScore = 0.0,
    this.intensityLabel = IntensityLevel.blue,
    this.phase = StudyPhase.foundation,
    this.anchorDate = '',
    this.taskType = TaskType.deepWork,
    this.scheduledWindow = ScheduledWindow.morning,
    this.priority = Priority.none,
  });

  factory DailyPlanTask.fromJson(String id, Map<String, dynamic> json) {
    final startTimeStr = json['start_time']?.toString();
    final endTimeStr = json['end_time']?.toString();
    final now = DateTime.now();

    return DailyPlanTask(
      id: id,
      title: (json['title'] ?? '').toString(),
      subject: (json['subject'] ?? '').toString(),
      description: (json['description'] ?? '').toString(),
      paper: (json['paper'] ?? '').toString(),
      objectiveId: (json['objective_id'] ?? '').toString(),
      startTime: startTimeStr != null && startTimeStr.isNotEmpty
          ? DateTime.parse(startTimeStr).toLocal()
          : now,
      endTime: endTimeStr != null && endTimeStr.isNotEmpty
          ? DateTime.parse(endTimeStr).toLocal()
          : now.add(const Duration(hours: 1)),
      status: TaskStatus.fromString((json['status'] ?? 'pending').toString()),
      date: (json['date'] ?? '').toString(),
      reason: (json['reason'] ?? '').toString(),
      isSyncToGoogle: json['is_sync_to_google'] == true,
      isCompleted: json['is_completed'] == true,
      intensityScore: (json['intensity_score'] as num?)?.toDouble() ?? 0,
      intensityLabel: IntensityLevel.fromString((json['intensity_label'] ?? 'blue').toString()),
      phase: StudyPhase.fromString((json['phase'] ?? '').toString()),
      anchorDate: (json['anchor_date'] ?? '').toString(),
      taskType: TaskType.fromString((json['task_type'] ?? 'deep_work').toString()),
      scheduledWindow: ScheduledWindow.fromString((json['scheduled_window'] ?? 'morning').toString()),
      priority: Priority.values.length > (json['priority'] as int? ?? 0)
          ? Priority.values[json['priority'] as int? ?? 0]
          : Priority.none,
    );
  }

  Map<String, dynamic> toJson() => {
    'title': title,
    'subject': subject,
    'description': description,
    'date': date,
    'start_time': startTime.toIso8601String(),
    'end_time': endTime.toIso8601String(),
    'status': status.value,
    'reason': reason,
    'is_sync_to_google': isSyncToGoogle,
    'is_completed': isCompleted,
    'intensity_score': intensityScore,
    'intensity_label': intensityLabel.value,
    'phase': phase.label,
    'anchor_date': anchorDate,
    'task_type': taskType.value,
    'scheduled_window': scheduledWindow.value,
    'priority': priority.index,
    'paper': paper,
    'objective_id': objectiveId,
  };

  DailyPlanTask copyWith({
    DateTime? startTime,
    DateTime? endTime,
    bool? isCompleted,
    TaskStatus? status,
  }) {
    return DailyPlanTask(
      id: id,
      title: title,
      subject: subject,
      description: description,
      paper: paper,
      objectiveId: objectiveId,
      startTime: startTime ?? this.startTime,
      endTime: endTime ?? this.endTime,
      status: status ?? this.status,
      date: date,
      reason: reason,
      isSyncToGoogle: isSyncToGoogle,
      isCompleted: isCompleted ?? this.isCompleted,
      intensityScore: intensityScore,
      intensityLabel: intensityLabel,
      phase: phase,
      anchorDate: anchorDate,
      taskType: taskType,
      scheduledWindow: scheduledWindow,
      priority: priority,
    );
  }
}
