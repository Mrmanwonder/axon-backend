// lib/services/exam_repository.dart
import 'package:shared_preferences/shared_preferences.dart';
import '../models/models.dart' as models_exam;
import '../models/exam_event_model.dart';
import 'local_database_service.dart';
import 'board_exam_service.dart';
import 'exam_data_service.dart' as exam_data;
import 'exam_zone_service.dart';
import 'supabase_exam_dates_service.dart';
import 'local_exam_dates_service.dart';

import '../services/curriculum_catalog_service.dart';

class ExamRepository {
  static final ExamRepository instance = ExamRepository._internal();
  ExamRepository._internal();

  static const String _userSubjectsKey = 'userSubjects';

  static final BoardExamService _boardExamService = BoardExamService();
  static final exam_data.ExamDataService _examDataService =
      exam_data.ExamDataService();

  final LocalDatabaseService _db = LocalDatabaseService();
  List<ExamEventModel> _memoryCache = [];
  bool _isLoaded = false;
  String? _userId;

  List<ExamEventModel> get allExams => _memoryCache;
  bool get isLoaded => _isLoaded;

  Future<void> initialize({String? userId}) async {
    print(
        'ExamRepository: Initializing for $userId (current: $_userId, isLoaded: $_isLoaded)');
    if (_isLoaded && _userId == userId) {
      print('ExamRepository: Already initialized');
      return;
    }

    _userId = userId ?? await getUserId();
    print('ExamRepository: Loading from database for $_userId...');
    await _loadFromDatabase();
    _isLoaded = true;
    print('ExamRepository: Initialization complete');
  }

  Future<String> getUserId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('userId') ?? 'default_user';
  }

  Future<List<String>> _getUserSubjects() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getStringList(_userSubjectsKey) ?? [];
  }

  Future<void> _loadFromDatabase() async {
    if (_userId == null) return;

    final userSubjects = await _getUserSubjects();
    if (userSubjects.isEmpty) {
      _memoryCache = [];
      return;
    }

    final catalog = CurriculumCatalogService.instance;
    await catalog.initializeLocalData();
    final allSubjects = await catalog.getAllSubjects();

    // Map user subject names to codes and vice-versa for robust matching
    final Map<String, String> nameToCode = {};
    final Map<String, String> codeToName = {};

    for (final s in allSubjects) {
      nameToCode[s.name.toLowerCase()] = s.code;
      codeToName[s.code] = s.name;
    }

    final Set<String> targetCodes = {};
    final Set<String> targetNames = {};
    for (final subject in userSubjects) {
      final normalized = subject.trim().toLowerCase();
      final label = _cleanSubjectLabel(subject).toLowerCase();
      if (normalized.isNotEmpty) targetNames.add(normalized);
      if (label.isNotEmpty) targetNames.add(label);

      final embeddedCode = RegExp(r'\b\d{4}\b').firstMatch(subject)?.group(0);
      if (embeddedCode != null) targetCodes.add(embeddedCode);

      if (nameToCode.containsKey(normalized)) {
        targetCodes.add(nameToCode[normalized]!);
      }
      if (nameToCode.containsKey(label)) {
        targetCodes.add(nameToCode[label]!);
      }
      if (RegExp(r'^\d{4}$').hasMatch(subject.trim())) {
        targetCodes.add(subject.trim());
      }
    }

    final results = await _db.getUserExamDates(_userId!);
    _memoryCache = results.where((r) {
      final dbCode = r['subject_code']?.toString() ?? '';
      final dbName = r['subject_name']?.toString();
      final cleanDbName =
          dbName == null ? null : _cleanSubjectLabel(dbName).toLowerCase();

      return targetCodes.contains(dbCode) ||
          targetCodes.any((code) => dbCode.startsWith('$code/')) ||
          (dbName != null &&
              (targetNames.contains(dbName.toLowerCase()) ||
                  (cleanDbName != null && targetNames.contains(cleanDbName))));
    }).map((r) {
      // Enrich with human name if missing from DB record
      final map = Map<String, dynamic>.from(r);
      if (map['subject_name'] == null) {
        final code = map['subject_code'] as String;
        map['subject_name'] = codeToName[code] ?? code;
      }
      return ExamEventModel.fromMap(map);
    }).toList()
      ..sort((a, b) => a.date.compareTo(b.date));
  }

  Future<void> saveExamDates(List<ExamEventModel> exams) async {
    if (_userId == null) return;

    final maps = exams.map((e) => e.toMap(_userId!)).toList();
    await _db.insertExamDates(maps, _userId!);
    await _loadFromDatabase();
  }

  Future<void> saveExamDatesForSubject(
      String subjectCode, List<ExamEventModel> exams) async {
    if (_userId == null) return;

    // Clear existing for this subject first
    await _db.deleteExamDatesForSubject(_userId!, subjectCode);

    final maps = exams.map((e) => e.toMap(_userId!)).toList();
    await _db.insertExamDates(maps, _userId!);
    await _loadFromDatabase();
  }

  Future<void> refreshExams() async {
    await _loadFromDatabase();
  }

  Future<List<ExamEventModel>> syncSupabaseExamDates({
    String? curriculum,
    List<String>? subjects,
    bool persist = true,
  }) async {
    print('ExamRepository: syncSupabaseExamDates starting (userId: $_userId)');
    if (_userId == null) {
      print('ExamRepository: userId is null, cannot sync');
      return const [];
    }

    final targetSubjects = subjects ?? await _getUserSubjects();
    final targetCurriculum = curriculum ?? await _getUserBoard();

    // Resolve subject names/input to numeric codes for consistent filtering
    final catalog = CurriculumCatalogService.instance;
    await catalog.initializeLocalData();

    // Determine user's curriculum level for code disambiguation
    final isALevel = targetCurriculum.contains('a_level');

    final Set<String> resolvedCodes = {};
    for (final s in targetSubjects) {
      final trimmed = s.trim();
      // 1. Pure 4-digit code
      if (RegExp(r'^\d{4}$').hasMatch(trimmed)) {
        resolvedCodes.add(trimmed);
        continue;
      }
      // 2. Embedded code in name (e.g. "Physics (9702)")
      final embedded = RegExp(r'\b\d{4}\b').firstMatch(trimmed)?.group(0);
      if (embedded != null) {
        resolvedCodes.add(embedded);
        continue;
      }
      // 3. Catalog lookup — prefer match matching user's curriculum level
      final allSubjects = await catalog.getAllSubjects();
      String? bestCode;
      for (final subj in allSubjects) {
        if (subj.name.toLowerCase() == trimmed.toLowerCase()) {
          final codeNum = int.tryParse(subj.code);
          if (codeNum != null) {
            final isSubjALevel = codeNum >= 9000;
            if (isALevel == isSubjALevel) {
              bestCode = subj.code;
              break;
            }
            bestCode ??= subj.code;
          }
        }
      }
      if (bestCode != null) {
        resolvedCodes.add(bestCode);
      } else {
        // 4. Partial match fallback
        final code = await catalog.getSubjectCode(s);
        if (code != null && code.isNotEmpty) resolvedCodes.add(code);
      }
    }
    final targetCodes = resolvedCodes.toList();

    print(
        'ExamRepository: Calling SupabaseExamDatesService (curriculum: $targetCurriculum, subjects: $targetSubjects)...');
    var exams = await SupabaseExamDatesService.instance.fetchUserExamDates(
      userId: _userId!,
      curriculum: targetCurriculum,
      subjects: targetSubjects,
    );

    if (exams.isEmpty && targetCodes.isNotEmpty) {
      print(
          'ExamRepository: Supabase returned no exams, trying local CSV fallback...');
      try {
        exams = await LocalExamDatesService.instance.getUpcomingExamDates(
          subjectCodes: targetCodes,
        );
        print('ExamRepository: Local CSV returned ${exams.length} exams');
      } catch (e) {
        print('ExamRepository: Local CSV fallback failed: $e');
      }
    }

    if (exams.isEmpty && targetCodes.isNotEmpty) {
      print('ExamRepository: No exams from CSV, trying Firestore fallback...');
      try {
        final firestoreData = await _boardExamService.getDeadlines(_userId!);
        if (firestoreData.isNotEmpty) {
          print(
              'ExamRepository: Found ${firestoreData.length} exams in Firestore');
          exams = firestoreData
              .where((data) => targetCodes
                  .any((c) => (data['subject_code'] ?? '').toString().startsWith(c)))
              .map((data) {
            final dateStr = data['exam_date'] ?? data['date'] ?? '';
            final date =
                DateTime.tryParse(dateStr.toString()) ?? DateTime.now();

            return ExamEventModel(
              id: data['id']?.toString() ?? 'fs_${date.millisecondsSinceEpoch}',
              board: data['board']?.toString() ?? targetCurriculum,
              subject: data['subject_name'] ?? data['subject'] ?? 'Unknown',
              code: data['subject_code'] ?? '',
              component: data['component'] ?? data['label'] ?? '',
              date: date,
              time: data['exam_time'] ?? data['time'] ?? '09:00',
              duration: data['duration']?.toString() ?? '2h',
              type: data['exam_type'] ?? data['type'] ?? 'Exam',
            );
          }).toList();
        }
      } catch (e) {
        print('ExamRepository: Firestore fallback failed: $e');
      }
    }

    if (exams.isEmpty) {
      print(
          'ExamRepository: No exams found in any source (Supabase, CSV, Firestore)');
      return const [];
    }

    print('ExamRepository: Total ${exams.length} exams loaded');

    if (persist) {
      await saveExamDates(exams);
    } else {
      _memoryCache = exams;
    }
    return exams;
  }

  Future<String> _getUserBoard() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('userBoard') ?? '';
  }

  List<ExamEventModel> getUpcomingExams(
      {String? filterSubject, String? filterType}) {
    var exams = _memoryCache.where((e) => e.isUpcoming).toList();

    if (filterSubject != null && filterSubject != 'All') {
      exams = exams
          .where((e) =>
              e.subject.toLowerCase().contains(filterSubject.toLowerCase()) ||
              e.code.startsWith(filterSubject))
          .toList();
    }

    if (filterType != null && filterType != 'All') {
      exams = exams.where((e) => e.type == filterType).toList();
    }

    return exams;
  }

  List<ExamEventModel> getExamsForSubject(String subjectCode) {
    return _memoryCache
        .where((e) =>
            e.subject.toLowerCase() == subjectCode.toLowerCase() ||
            e.code.startsWith(subjectCode))
        .toList();
  }

  List<ExamEventModel> getTodayExams() {
    return _memoryCache.where((e) => e.isToday).toList();
  }

  List<String> getAvailableSubjects() {
    final subjects = _memoryCache.map((e) => e.subject).toSet().toList();
    subjects.sort();
    return subjects;
  }

  List<String> getAvailableTypes() {
    return _memoryCache.map((e) => e.type).toSet().toList()..sort();
  }

  int get totalUpcomingCount => _memoryCache.where((e) => e.isUpcoming).length;

  Map<String, int> get countByType {
    final map = <String, int>{};
    for (final exam in _memoryCache.where((e) => e.isUpcoming)) {
      map[exam.type] = (map[exam.type] ?? 0) + 1;
    }
    return map;
  }

  Future<void> clearAllExams() async {
    if (_userId == null) return;
    await _db.clearAllExamDates(_userId!);
    _memoryCache.clear();
  }

  // ==================== Legacy Service Wrappers ====================

  Future<BoardFetchResult> fetchBoardDates({
    required String board,
    List<String> subjects = const [],
    int? year,
    String? series,
    String? administrativeZone,
    bool persist = false,
  }) async {
    return _boardExamService.fetchBoardDates(
      board,
      subjects: subjects,
      year: year,
      series: series,
      administrativeZone: administrativeZone,
      persist: persist,
    );
  }

  Future<AxonScheduleResult> generateLegacySchedule({
    required List<models_exam.ExamEvent> events,
    required Map<String, dynamic> metrics,
  }) async {
    return _boardExamService.generateSchedule(
      events: events,
      metrics: metrics,
    );
  }

  Future<BoardFetchResult> syncOfficialDeadlines({
    required String board,
    required List<String> subjects,
    int? year,
    String? series,
    String? administrativeZone,
  }) async {
    return _boardExamService.syncOfficialDeadlines(
      board: board,
      subjects: subjects,
      year: year,
      series: series,
      administrativeZone: administrativeZone,
    );
  }

  exam_data.ExamEvent getTargetExamLegacy(String? manualSubject) {
    return _examDataService.getTargetExam(manualSubject);
  }

  exam_data.ExamEvent getNearestExamLegacy() {
    return _examDataService.getNearestExam();
  }

  List<exam_data.ExamEvent> getUpcomingExamsLegacy({int limit = 5}) {
    return _examDataService.getUpcomingExams(limit: limit);
  }

  static Map<String, dynamic>? getZone(String country) {
    return ExamZoneService.getZone(country);
  }

  static List<String> get countryList => ExamZoneService.countryList;

  static String getTimezone(String country) {
    return ExamZoneService.getTimezone(country);
  }

  static List<String> getSeries(String country) {
    return ExamZoneService.getSeries(country);
  }

  static String _cleanSubjectLabel(String value) {
    return value.replaceAll(RegExp(r'\s*\(\d{4}\)\s*'), '').trim();
  }
}
