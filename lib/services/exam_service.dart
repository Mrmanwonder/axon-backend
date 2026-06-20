// lib/services/exam_service.dart
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../models/models.dart';
import '../models/exam_event_model.dart';
import 'curriculum_catalog_service.dart';
import 'firestore_service.dart';
import 'supabase_proxy_service.dart';

// ── Curriculum normalization ───────────────────────────────────────────
String _canonicalCurriculum(String value) {
  final normalized = value.trim().toLowerCase().replaceAll('-', ' ').replaceAll('_', ' ');
  if (normalized.isEmpty) return '';
  String? boardId;
  if (normalized.contains('caie') || normalized.contains('cambridge') || normalized.contains('cie')) {
    boardId = 'caie';
  }
  String level;
  if (normalized.contains('as level') && !normalized.contains('a level')) {
    level = 'as_level';
  } else if (normalized.contains('a level') || normalized.contains('alevel') || normalized.contains('ial') || normalized.contains('international advanced')) {
    level = 'a_level';
  } else if (normalized.contains('igcse') || normalized.contains('international gcse')) {
    level = 'igcse';
  } else if (normalized.contains('o level') || normalized.contains('olevel')) {
    level = 'igcse';
  } else if (normalized.contains('gcse')) {
    level = 'igcse';
  } else {
    level = 'igcse';
  }
  if (boardId != null) return '${boardId}_$level';
  if (normalized.contains('a level') || normalized.contains('alevel')) return 'caie_a_level';
  if (normalized.contains('igcse') || normalized.contains('cambridge') || normalized.contains('cie') || normalized.contains('caie')) return 'caie_igcse';
  return value.trim();
}

class UserExamDateProfile {
  final String curriculum;
  final List<String> subjectCodes;
  final Map<String, String> subjectNamesByCode;

  UserExamDateProfile({
    required this.curriculum,
    required this.subjectCodes,
    required this.subjectNamesByCode,
  });
}

class BoardFetchResult {
  final String board;
  final String sourceUrl;
  final List<Map<String, String>> sources;
  final List<ExamEvent> events;
  final String? error;

  const BoardFetchResult({
    required this.board,
    required this.sourceUrl,
    required this.sources,
    required this.events,
    this.error,
  });
}

class AxonScheduleResult {
  final double efficiencyIndex;
  final List<AxonDailyPlan> dailyPlan;
  final Map<String, dynamic> receiptByDate;

  const AxonScheduleResult({
    required this.efficiencyIndex,
    required this.dailyPlan,
    required this.receiptByDate,
  });
}

class ExamService {
  static ExamService? _instance;
  ExamService._();
  factory ExamService() => _instance ??= ExamService._();

  final http.Client _client = http.Client();
  final SupabaseProxyService _proxy = SupabaseProxyService.instance;
  final Map<String, List<ExamEventModel>> _cache = {};
  final Map<String, Future<List<ExamEventModel>>> _inFlight = {};

  static String _backendUrl = 'https://axon-ml.onrender.com';
  static bool _initialized = false;

  // ── Backend URL management ──────────────────────────────────────────
  static String get backendUrl => _backendUrl;

  static Future<void> setBackendUrl(String url) async {
    _backendUrl = url;
    _initialized = true;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('backendUrl', url);
    } catch (_) {}
  }

  static Future<void> initFromPrefs() async {
    if (_initialized) return;
    try {
      final prefs = await SharedPreferences.getInstance();
      final stored = prefs.getString('backendUrl');
      if (stored != null && stored.isNotEmpty) {
        _backendUrl = stored;
      }
    } catch (_) {}
    _initialized = true;
  }

  // ── Supabase exam dates ────────────────────────────────────────────
  static const String _examDatesTable = 'Datesheet';
  static const _codeCols = ['subject_code', 'code', 'Code', 'SubjectCode', 'Subject_Code', 'subject code', 'Subject Code', 'Subject code'];
  static const _boardCols = ['Curriculum', 'curriculum', 'board', 'Board', 'Level', 'level'];
  static const _subjectCols = ['Subject', 'subject', 'subject_name', 'SubjectName', 'Subject Name'];
  static const _componentCols = ['component', 'Component', 'Paper', 'paper', 'PaperNo', 'ComponentCode'];
  static const _dateCols = ['exam_date', 'Date', 'date', 'ExamDate', 'Exam_Date', 'StartDate'];
  static const _startCols = ['start_time', 'Start', 'time', 'Time', 'Session', 'session'];

  Future<List<ExamEventModel>> fetchUserExamDates({
    required String userId,
    String? curriculum,
    List<String>? subjects,
    bool upcomingOnly = true,
  }) async {
    final key = '$userId-${curriculum ?? ""}-${subjects?.join(",") ?? ""}';
    final existing = _inFlight[key];
    if (existing != null) return existing;

    final future = _fetchExamDatesImpl(userId, curriculum, subjects, upcomingOnly);
    _inFlight[key] = future;
    try {
      final results = await future;
      _cache[key] = results;
      return results;
    } finally {
      _inFlight.remove(key);
    }
  }

  Future<List<ExamEventModel>> _fetchExamDatesImpl(
    String userId,
    String? curriculum,
    List<String>? subjects,
    bool upcomingOnly,
  ) async {
    try {
      final profile = await resolveUserExamProfile(
        userId: userId,
        curriculum: curriculum,
        subjects: subjects,
      );
      if (profile.subjectCodes.isEmpty || profile.curriculum.isEmpty) {
        return const [];
      }
      final filterString = profile.subjectCodes
          .map((c) => 'subject_code.like.$c%')
          .join(',');
      final response = await _proxy.query(_examDatesTable, params: {
        'or': filterString,
      });
      if (response.isEmpty) return const [];
      final columns = response.first.keys.toList();
      return _processRows(response, profile, upcomingOnly, columns);
    } catch (e) {
      return const [];
    }
  }

  Future<UserExamDateProfile> resolveUserExamProfile({
    required String userId,
    String? curriculum,
    List<String>? subjects,
  }) async {
    final namesByCode = <String, String>{};
    final codes = <String>{};

    String resolvedCurriculum = _canonicalCurriculum(curriculum ?? '');
    final requestedSubjects = subjects ?? await _subjectsFromPrefs();

    if (requestedSubjects.isEmpty) {
      final catalog = CurriculumCatalogService.instance;
      await catalog.initializeLocalData();
      final allSubjects = await catalog.getAllSubjects();
      for (final s in allSubjects) {
        requestedSubjects.add(s.name);
      }
    }

    final catalog = CurriculumCatalogService.instance;
    await catalog.initializeLocalData();

    for (final rawSubject in requestedSubjects) {
      final code = await _subjectCodeFor(catalog, rawSubject);
      if (code.isEmpty) continue;
      codes.add(code);
      namesByCode.putIfAbsent(code, () => _subjectLabel(rawSubject));
    }

    if (resolvedCurriculum.isEmpty) {
      final prefs = await SharedPreferences.getInstance();
      resolvedCurriculum = _canonicalCurriculum(prefs.getString('userBoard') ?? '');
    }

    if (resolvedCurriculum.isEmpty && codes.isNotEmpty) {
      final allSubjects = await catalog.getAllSubjects();
      for (final s in allSubjects) {
        if (codes.contains(s.code)) {
          resolvedCurriculum = _canonicalCurriculum('');
          if (resolvedCurriculum.isNotEmpty) break;
        }
      }
    }

    return UserExamDateProfile(
      curriculum: resolvedCurriculum,
      subjectCodes: codes.toList()..sort(),
      subjectNamesByCode: namesByCode,
    );
  }

  List<ExamEventModel> _processRows(
    List<dynamic> rows,
    UserExamDateProfile profile,
    bool upcomingOnly,
    List<String> columns,
  ) {
    final results = <ExamEventModel>[];
    final now = DateTime.now();

    final codeCol = columns.firstWhere((c) => _codeCols.contains(c), orElse: () => 'code');
    final boardCol = columns.firstWhere((c) => _boardCols.contains(c), orElse: () => 'curriculum');
    final subjectCol = columns.firstWhere((c) => _subjectCols.contains(c), orElse: () => 'subject');
    final componentCol = columns.firstWhere((c) => _componentCols.contains(c), orElse: () => 'component');
    final dateCol = columns.firstWhere((c) => _dateCols.contains(c), orElse: () => 'date');
    final startCol = columns.firstWhere((c) => _startCols.contains(c), orElse: () => 'start_time');

    for (final rowData in rows) {
      try {
        final row = rowData as Map<String, dynamic>;
        final rowCode = _string(row[codeCol]);
        final rowBoard = _string(row[boardCol]);

        if (!_curriculumMatches(rowBoard, profile.curriculum)) continue;
        if (!_subjectCodeMatches(rowCode, profile.subjectCodes)) continue;

        final date = _parseDate(_string(row[dateCol]));
        if (upcomingOnly && date.isBefore(now) && !date.isAtSameMomentAs(now)) continue;

        results.add(ExamEventModel(
          id: _string(row['id'] ?? rowCode + date.millisecondsSinceEpoch.toString()),
          subject: profile.subjectNamesByCode[rowCode] ?? _string(row[subjectCol]),
          code: rowCode,
          component: _string(row[componentCol]),
          date: date,
          time: _string(row[startCol]),
          duration: '2h',
          board: profile.curriculum,
        ));
      } catch (_) {}
    }
    return results;
  }

  bool _curriculumMatches(String rowValue, String targetValue) {
    final row = _canonicalCurriculum(rowValue).toLowerCase();
    final target = _canonicalCurriculum(targetValue).toLowerCase();
    if (row.isEmpty || target.isEmpty) return true;
    if (row == target) return true;
    final rowParts = row.split('_');
    final targetParts = target.split('_');
    final rowBoard = rowParts.isNotEmpty ? rowParts[0] : '';
    final rowLevel = rowParts.length > 1 ? rowParts.sublist(1).join('_') : '';
    final targetBoard = targetParts.isNotEmpty ? targetParts[0] : '';
    final targetLevel = targetParts.length > 1 ? targetParts.sublist(1).join('_') : '';
    if (rowBoard == targetBoard) {
      final caieLevels = {'igcse', 'as_level', 'a_level', 'o_level'};
      if (caieLevels.contains(rowLevel) && caieLevels.contains(targetLevel)) return true;
    }
    final legacyLevels = {'igcse', 'a_level', 'as_level', 'o_level', 'gcse'};
    if (legacyLevels.contains(row) && legacyLevels.contains(target)) return true;
    return false;
  }

  DateTime _parseDate(String raw) {
    if (raw.isEmpty) return DateTime.now();
    final iso = DateTime.tryParse(raw);
    if (iso != null) return iso;
    try {
      final cleaned = raw.replaceAll(RegExp(r'^[A-Za-z]+,\s*'), '').trim();
      final parts = cleaned.split(RegExp(r'\s+'));
      if (parts.length >= 3) {
        final day = int.parse(parts[0]);
        const months = {
          'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
          'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
          'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
          'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        };
        final monthStr = parts[1].toLowerCase();
        final month = months[monthStr] ?? 1;
        final year = int.parse(parts[2]);
        if (year >= 2020 && year <= 2100) return DateTime(year, month, day);
      }
    } catch (_) {}
    return DateTime.now();
  }

  bool _subjectCodeMatches(String rowCode, List<String> profileCodes) {
    for (final code in profileCodes) {
      if (rowCode == code) return true;
      if (rowCode.startsWith('$code/')) return true;
      if (rowCode.startsWith('$code ')) return true;
    }
    return false;
  }

  bool _supportsLocalCambridgeFallback(String board) {
    final normalized = board.trim().toLowerCase();
    return normalized.contains('caie') ||
        normalized.contains('cambridge') ||
        normalized.contains('cie') ||
        normalized.contains('igcse') ||
        normalized.contains('o level') ||
        normalized.contains('olevel') ||
        normalized.contains('a level') ||
        normalized.contains('alevel') ||
        normalized.contains('as level');
  }

  CollectionReference<Map<String, dynamic>> _deadlines(String uid) =>
      AxonPaths.privateUserCollection(uid, 'deadlines');

  List<ExamEvent> _loadLocalDateSheet() {
    final currentYear = DateTime.now().year;
    final possibleFiles = [
      'datesheet_cambridge_${currentYear}_June.json',
      'datesheet_cambridge_${currentYear}_Nov.json',
      'datesheet_cambridge_${currentYear - 1}_June.json',
      'datesheet_cambridge_${currentYear - 1}_Nov.json',
      'datesheet_cambridge_2026_June.json',
    ];
    for (final fileName in possibleFiles) {
      final file = File(fileName);
      if (file.existsSync()) {
        try {
          final content = file.readAsStringSync();
          final data = jsonDecode(content) as Map<String, dynamic>;
          final events = data['events'] as List<dynamic>? ?? [];
          return events.map((e) {
            final event = Map<String, dynamic>.from(e);
            return ExamEvent(
              board: (event['board'] ?? 'Cambridge').toString(),
              subject: (event['subject'] ?? '').toString(),
              label: (event['component'] ?? '').toString(),
              startDate: DateTime.tryParse((event['date'] ?? '').toString()) ?? DateTime.now(),
              endDate: DateTime.tryParse((event['date'] ?? '').toString())?.add(const Duration(hours: 2)) ?? DateTime.now().add(const Duration(hours: 2)),
              source: (event['source'] ?? '').toString(),
            );
          }).toList();
        } catch (e) {
          continue;
        }
      }
    }
    return [];
  }

  // ── Backend exam dates ──────────────────────────────────────────────
  Future<BoardFetchResult> fetchBoardDates(
    String board, {
    List<String> subjects = const [],
    int? year,
    String? series,
    String? administrativeZone,
    bool persist = false,
  }) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      return BoardFetchResult(
        board: board, sourceUrl: '', sources: const [], events: const [],
        error: 'User is not signed in.',
      );
    }
    final token = await user.getIdToken();
    if (token == null || token.isEmpty) {
      return BoardFetchResult(
        board: board, sourceUrl: '', sources: const [], events: const [],
        error: 'Missing auth token.',
      );
    }

    try {
      final response = await _client
          .post(
            Uri.parse('$_backendUrl/sync-exam-dates'),
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $token',
            },
            body: jsonEncode({
              'board': board,
              'subjects': subjects,
              if (year != null) 'year': year,
              if (series != null && series.isNotEmpty) 'series': series,
              if (administrativeZone != null && administrativeZone.isNotEmpty)
                'administrative_zone': administrativeZone,
              'persist': persist,
            }),
          )
          .timeout(const Duration(seconds: 45));

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;
        final events = (decoded['events'] as List? ?? const [])
            .whereType<Map>()
            .map((entry) {
          final data = Map<String, dynamic>.from(entry);
          return ExamEvent(
            board: (data['board'] ?? board).toString(),
            subject: (data['subject'] ?? '').toString(),
            label: (data['label'] ?? '').toString(),
            startDate: DateTime.tryParse((data['exam_date'] ?? '').toString()) ?? DateTime.now(),
            endDate: DateTime.tryParse((data['exam_date'] ?? '').toString()) ?? DateTime.now(),
            source: (data['source_url'] ?? '').toString(),
          );
        }).toList();

        return BoardFetchResult(
          board: (decoded['board'] ?? board).toString(),
          sourceUrl: '', sources: [], events: events,
        );
      }
    } catch (e) {
      debugPrint('ExamService: Backend unavailable — using local datesheet as fallback');
    }

    final localEvents = _loadLocalDateSheet();
    if (_supportsLocalCambridgeFallback(board) && localEvents.isNotEmpty) {
      return BoardFetchResult(
        board: board, sourceUrl: 'Local datesheet (fallback)', sources: const [], events: localEvents,
      );
    }

    return BoardFetchResult(
      board: board, sourceUrl: '', sources: const [], events: const [],
      error: 'Could not fetch datesheet. Using local fallback.',
    );
  }

  Future<BoardFetchResult> syncOfficialDeadlines({
    required String board,
    required List<String> subjects,
    int? year,
    String? series,
    String? administrativeZone,
  }) {
    return fetchBoardDates(
      board,
      subjects: subjects,
      year: year,
      series: series,
      administrativeZone: administrativeZone,
      persist: true,
    );
  }

  Future<List<Map<String, dynamic>>> getDeadlines(String uid) async {
    try {
      final snapshot = await _deadlines(uid).orderBy('exam_date').get();
      return snapshot.docs.map((doc) => doc.data()).toList();
    } catch (e) {
      debugPrint('Error fetching deadlines from Firestore: $e');
      return [];
    }
  }

  Stream<List<Map<String, dynamic>>> watchDeadlines(String uid) {
    return _deadlines(uid)
        .orderBy('exam_date')
        .snapshots()
        .map((snapshot) => snapshot.docs.map((doc) => doc.data()).toList());
  }

  Future<AxonScheduleResult> generateSchedule({
    required List<ExamEvent> events,
    required Map<String, dynamic> metrics,
  }) async {
    final sorted = [...events]..sort((a, b) => a.startDate.compareTo(b.startDate));
    if (sorted.isEmpty) {
      return const AxonScheduleResult(efficiencyIndex: 0.0, dailyPlan: [], receiptByDate: {});
    }
    final now = DateTime.now();
    final exam = sorted.first.startDate;
    final daysToExam = exam.difference(DateTime(now.year, now.month, now.day)).inDays;
    final availableDays = daysToExam <= 0 ? 1 : daysToExam;
    final recommendedUnits = (metrics['remaining_objectives'] is num)
        ? ((metrics['remaining_objectives'] as num) / availableDays).toDouble()
        : 0.0;

    return AxonScheduleResult(
      efficiencyIndex: recommendedUnits,
      dailyPlan: sorted
          .map((event) => AxonDailyPlan(
                date: event.startDate,
                subject: event.subject,
                board: event.board,
                examLabel: event.label,
                isExamDay: true,
                intensity: 1.0,
                loadState: 'exam',
                recommendedUnits: recommendedUnits,
                receipt: 'Official datesheet synced from ${event.source}',
              ))
          .toList(),
      receiptByDate: {
        for (final event in sorted)
          event.startDate.toIso8601String().split('T').first: event.source,
      },
    );
  }

  // ── Helpers ─────────────────────────────────────────────────────────
  Future<List<String>> _subjectsFromPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final fromPrefs = prefs.getStringList('userSubjects') ?? [];
    if (fromPrefs.isNotEmpty) return fromPrefs;
    final catalog = CurriculumCatalogService.instance;
    await catalog.initializeLocalData();
    final allSubjects = await catalog.getAllSubjects();
    return allSubjects.map((s) => s.name).toList();
  }

  Future<String> _subjectCodeFor(CurriculumCatalogService catalog, String rawSubject) async {
    final trimmed = rawSubject.trim();
    if (RegExp(r'^\d{4}$').hasMatch(trimmed)) return trimmed;
    final codeInName = RegExp(r'\b\d{4}\b').firstMatch(trimmed)?.group(0);
    if (codeInName != null) return codeInName;
    final fromCatalog = await catalog.getSubjectCode(trimmed);
    if (fromCatalog != null && fromCatalog.isNotEmpty) return fromCatalog;
    final stripped = trimmed.replaceAll(RegExp(r'\s*\(.*?\)\s*'), '').trim();
    if (stripped != trimmed) {
      return await catalog.getSubjectCode(stripped) ?? '';
    }
    return '';
  }

  String _subjectLabel(String rawSubject) {
    return rawSubject.replaceAll(RegExp(r'\s*\(\d{4}\)\s*'), '').trim();
  }

  String _string(dynamic value) => (value ?? '').toString().trim();
}
