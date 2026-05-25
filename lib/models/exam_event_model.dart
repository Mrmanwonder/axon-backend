class ExamEventModel {
  final String id;
  final String board;
  final String subject;
  final String code;
  final String component;
  final DateTime date;
  final String time;
  final String duration;
  final String type;

  const ExamEventModel({
    required this.id,
    required this.board,
    required this.subject,
    required this.code,
    required this.component,
    required this.date,
    required this.time,
    this.duration = '',
    this.type = 'Exam',
  });

  DateTime? get examDate => date;

  bool get isUpcoming => date.isAfter(DateTime.now().subtract(const Duration(days: 1)));

  bool get isToday {
    final now = DateTime.now();
    return date.year == now.year && date.month == now.month && date.day == now.day;
  }

  int get daysRemaining {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final examDay = DateTime(date.year, date.month, date.day);
    return examDay.difference(today).inDays;
  }

  factory ExamEventModel.fromMap(Map<String, dynamic> map) {
    final examCode = map['exam_code']?.toString() ?? '';
    final subjectCode = map['subject_code']?.toString() ??
        RegExp(r'^\d{4}').firstMatch(examCode)?.group(0) ?? '';
    return ExamEventModel(
      id: map['id']?.toString() ?? '',
      board: map['board'] ?? 'Cambridge',
      subject: map['subject_name'] ?? subjectCode,
      code: subjectCode,
      component: map['component'] ?? '',
      date: DateTime.fromMillisecondsSinceEpoch(map['exam_date'] as int),
      time: map['exam_time'] ?? '09:00',
      duration: map['duration'] ?? '2h',
      type: map['exam_type'] ?? 'IGCSE',
    );
  }

  Map<String, dynamic> toMap(String userId) {
    final subjectCode = RegExp(r'^\d{4}').firstMatch(code)?.group(0) ?? code;
    final componentKey = component.trim().isEmpty ? 'exam' : component.trim();
    return {
      'user_id': userId,
      'subject_code': subjectCode,
      'subject_name': subject,
      'exam_code': '$subjectCode/$componentKey/${date.millisecondsSinceEpoch}',
      'component': component,
      'exam_date': date.millisecondsSinceEpoch,
      'exam_time': time,
      'duration': duration,
      'exam_type': type,
    };
  }

  ExamEventModel copyWith({
    String? id,
    String? board,
    String? subject,
    String? code,
    String? component,
    DateTime? date,
    String? time,
    String? duration,
    String? type,
  }) {
    return ExamEventModel(
      id: id ?? this.id,
      board: board ?? this.board,
      subject: subject ?? this.subject,
      code: code ?? this.code,
      component: component ?? this.component,
      date: date ?? this.date,
      time: time ?? this.time,
      duration: duration ?? this.duration,
      type: type ?? this.type,
    );
  }

  factory ExamEventModel.fromJson(Map<String, dynamic> json) {
    return ExamEventModel(
      id: (json['id'] ?? '').toString(),
      board: (json['board'] ?? '').toString(),
      subject: (json['subject'] ?? '').toString(),
      code: (json['code'] ?? '').toString(),
      component: (json['component'] ?? '').toString(),
      date: DateTime.tryParse((json['date'] ?? '').toString()) ?? DateTime.now(),
      time: (json['time'] ?? '').toString(),
      duration: (json['duration'] ?? '').toString(),
      type: (json['type'] ?? 'Exam').toString(),
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'board': board,
        'subject': subject,
        'code': code,
        'component': component,
        'date': date.toIso8601String(),
        'time': time,
        'duration': duration,
        'type': type,
      };
}
