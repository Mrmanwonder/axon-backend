import 'package:shared_preferences/shared_preferences.dart';
import '../models/exam_event_model.dart';
import 'exam_repository.dart';
import 'curriculum_catalog_service.dart';

class ExamScheduleGenerator {
  static final ExamScheduleGenerator instance =
      ExamScheduleGenerator._internal();
  ExamScheduleGenerator._internal();

  static const String _userSubjectsKey = 'userSubjects';
  static const String _examSessionKey =
      'examSession'; // 'june_2026', 'november_2026'

  // Standard CAIE exam dates by session
  static const Map<String, Map<String, dynamic>> _examSessions = {
    'june_2026': {
      'name': 'June 2026',
      'year': 2026,
      'month': 5, // Starts in May usually
      'startDay': 1,
      'endDay': 31,
      'type': 'IGCSE/AS/A Level',
    },
    'november_2026': {
      'name': 'November 2026',
      'year': 2026,
      'month': 10, // Starts in October
      'startDay': 1,
      'endDay': 30,
      'type': 'IGCSE/AS/A Level',
    },
  };

  // Standard paper schedule patterns (time slots)
  static const Map<String, List<String>> _paperPatterns = {
    'P1': ['09:00', '2h'],
    'P2': ['09:00', '1h 45m'],
    'P3': ['09:00', '1h 45m'],
    'M1': ['13:00', '1h 15m'],
    'M2': ['13:00', '1h 15m'],
    'S1': ['13:00', '1h 15m'],
    'S2': ['13:00', '1h 15m'],
    'FP1': ['09:00', '1h 30m'],
    'FP2': ['09:00', '1h 30m'],
    'FM': ['13:00', '1h 30m'],
    'FPS': ['13:00', '1h 30m'],
    'AS': ['09:00', '1h 30m'],
    'AL': ['09:00', '2h'],
    'Paper1': ['09:00', '1h 30m'],
    'Paper2': ['09:00', '1h 30m'],
    'Paper3': ['13:00', '1h'],
    'Paper4': ['09:00', '2h'],
  };

  // Subject to papers mapping (from curriculum)
  static const Map<String, List<String>> _subjectPapers = {
    '0580': ['Paper2', 'Paper4'], // IGCSE Math
    '0625': ['Paper2', 'Paper3'], // IGCSE Physics
    '0620': ['Paper2', 'Paper3'], // IGCSE Chemistry
    '0610': ['Paper2', 'Paper3'], // IGCSE Biology
    '0478': ['Paper1', 'Paper2'], // IGCSE CS
    '0455': ['Paper1', 'Paper2'], // IGCSE Economics
    '0450': ['Paper1', 'Paper2'], // IGCSE Business
    '0500': ['Paper1', 'Paper2'], // IGCSE English Lang
    '4024': ['Paper1', 'Paper2'], // O Level Math
    '5054': ['Paper1', 'Paper2'], // O Level Physics
    '5070': ['Paper1', 'Paper2'], // O Level Chemistry
    '5090': ['Paper1', 'Paper2'], // O Level Biology
    '2210': ['Paper1', 'Paper2'], // O Level CS
    '9709': ['P1', 'P2', 'M1', 'S1'], // A Level Math (AS papers)
    '9231': ['FP1', 'FP2', 'FM', 'FPS'], // Further Math
    '9702': ['AS', 'AL'], // Physics
    '9701': ['AS', 'AL'], // Chemistry
    '9700': ['AS', 'AL'], // Biology
    '9618': ['AS', 'AL'], // Computer Science
    '9708': ['AS', 'AL'], // Economics
    '9609': ['AS', 'AL'], // Business
  };

  Future<String?> getExamSession() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_examSessionKey) ?? 'june_2026';
  }

  Future<void> setExamSession(String session) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_examSessionKey, session);
  }

  Future<List<String>> getAvailableSessions() async {
    return _examSessions.keys.toList();
  }

  Future<Map<String, String>> getSessionInfo(String session) async {
    final info = _examSessions[session];
    if (info == null) return {};
    return {
      'name': info['name'],
      'type': info['type'],
      'year': info['year'].toString(),
      'month': info['month'].toString(),
    };
  }

  Future<List<String>> getUserSubjects() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getStringList(_userSubjectsKey) ?? [];
  }

  Future<void> setUserSubjects(List<String> subjects) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_userSubjectsKey, subjects);
  }

  Future<List<ExamEventModel>> generateExamSchedule() async {
    final rawSubjects = await getUserSubjects();
    String? session = await getExamSession();

    if (rawSubjects.isEmpty) {
      return [];
    }

    // Auto-detect session if null (already handled in getExamSession but be safe)
    session ??= 'june_2026';

    final sessionInfo = _examSessions[session];
    if (sessionInfo == null) return [];

    final year = sessionInfo['year'] as int;
    final month = sessionInfo['month'] as int;
    final exams = <ExamEventModel>[];

    int dayOffset = 1;

    for (final rawSubject in rawSubjects) {
      // Map name to code if possible
      String code = rawSubject;
      String name = rawSubject;

      // Check if it's already a code (4 digits)
      final isCode = RegExp(r'^\d{4}$').hasMatch(rawSubject);
      if (!isCode) {
        // Try mapping name to code
        code = await CurriculumCatalogService.instance
                .getSubjectCode(rawSubject) ??
            rawSubject;
      } else {
        name = _getSubjectName(code);
      }

      final papers = _subjectPapers[code] ?? ['Paper1', 'Paper2'];

      for (final paper in papers) {
        final pattern = _paperPatterns[paper] ?? ['09:00', '1h 30m'];

        // Assign days sequentially (simplified - real schedule has conflicts)
        final exam = ExamEventModel(
          id: '${code}_${paper}_$dayOffset',
          board: 'Cambridge',
          subject: name,
          code: '$code/${paper.replaceAll('Paper', 'P')}',
          component: paper,
          date: DateTime(year, month, (dayOffset % 28) + 1),
          time: pattern[0],
          duration: pattern[1],
          type: _getExamType(code),
        );

        exams.add(exam);
        dayOffset += 2; // Offset by 2 days to spread them out
      }
    }

    // Sort by date
    exams.sort((a, b) => a.date.compareTo(b.date));
    return exams;
  }

  String _getSubjectName(String code) {
    const names = {
      '0580': 'Mathematics',
      '0625': 'Physics',
      '0620': 'Chemistry',
      '0610': 'Biology',
      '0478': 'Computer Science',
      '0455': 'Economics',
      '0450': 'Business Studies',
      '0500': 'English Language',
      '4024': 'Mathematics',
      '5054': 'Physics',
      '5070': 'Chemistry',
      '5090': 'Biology',
      '2210': 'Computer Science',
      '9709': 'Mathematics',
      '9231': 'Further Mathematics',
      '9702': 'Physics',
      '9701': 'Chemistry',
      '9700': 'Biology',
      '9618': 'Computer Science',
      '9708': 'Economics',
      '9609': 'Business',
    };
    return names[code] ?? 'Subject $code';
  }

  String _getExamType(String code) {
    if (code.startsWith('9')) return 'A Level';
    if (code.startsWith('5') || code.startsWith('4')) return 'O Level';
    return 'IGCSE';
  }

  Future<void> saveGeneratedExams() async {
    final exams = await generateExamSchedule();
    await ExamRepository.instance.saveExamDates(exams);
  }

  Future<void> updateExamDate(String examId, DateTime newDate) async {
    final exams = ExamRepository.instance.allExams;
    final index = exams.indexWhere((e) => e.id == examId);

    if (index >= 0) {
      final updated = exams[index].copyWith(date: newDate);
      await ExamRepository.instance.saveExamDatesForSubject(
        exams[index].subject,
        [updated],
      );
    }
  }
}
