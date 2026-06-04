import 'package:shared_preferences/shared_preferences.dart';

import '../models/exam_event_model.dart';
import 'curriculum_catalog_service.dart';
import 'supabase_service.dart';

class SupabaseExamDatesService {
  final _supabase = SupabaseService.instance;
  final Map<String, List<ExamEventModel>> _cache = {};
  final Map<String, Future<List<ExamEventModel>>> _inFlight = {};

  Future<List<ExamEventModel>> fetchUserExamDates({
    required String userId,
    String? curriculum,
    List<String>? subjects,
    bool upcomingOnly = true,
  }) async {
    print('SupabaseExamDates: fetchUserExamDates called for $userId');

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
      print('SupabaseExamDates: Resolving profile...');
      final profile = await resolveUserExamProfile(
        userId: userId,
        curriculum: curriculum,
        subjects: subjects,
      );

      if (profile.subjectCodes.isEmpty || profile.curriculum.isEmpty) {
        print('SupabaseExamDates: Profile incomplete (codes=${profile.subjectCodes.length}, curriculum="${profile.curriculum}"), skipping fetch');
        return const [];
      }

      // 1. Fetch rows with OR filter for subject codes
      print('SupabaseExamDates: Querying $_examDatesTable for codes: ${profile.subjectCodes}');
      final filterString = profile.subjectCodes
          .map((c) => 'subject_code.like.$c%')
          .join(',');

      final response = await _supabase.query(_examDatesTable, params: {
        'or': filterString,
      });

      if (response.isEmpty) {
        print('SupabaseExamDates: No rows returned');
        return const [];
      }

      print('SupabaseExamDates: Received ${response.length} rows');
      final columns = response.first.keys.toList();
      return _processRows(response, profile, upcomingOnly, columns);
    } catch (e) {
      print('SupabaseExamDates: Fetch failed: $e');
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
      final catalog = CurriculumCatalogService.instance;
      await catalog.initializeLocalData();
      final allSubjects = await catalog.getAllSubjects();
      for (final s in allSubjects) {
        if (codes.contains(s.code)) {
          resolvedCurriculum = _canonicalCurriculum('');
          if (resolvedCurriculum.isNotEmpty) break;
        }
      }
    }

    print('SupabaseExamDates: Resolved curriculum="$resolvedCurriculum" codes=$codes');
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

    // Identify all columns with extended detection
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

        // Fix #2: Match curriculum with expanded detection
        if (!_curriculumMatches(rowBoard, profile.curriculum)) continue;

        // Fix #4: Match subject code with paper variant support
        if (!_subjectCodeMatches(rowCode, profile.subjectCodes)) continue;

        // Fix #5: Parse date with support for "Thursday, 07 May 2026" format
        final date = _parseDate(_string(row[dateCol]));

        // Fix #5: Only filter if date is genuinely in the past
        if (upcomingOnly && date.isBefore(now) && !date.isAtSameMomentAs(now)) continue;

        final event = ExamEventModel(
          id: _string(row['id'] ?? rowCode + date.millisecondsSinceEpoch.toString()),
          subject: profile.subjectNamesByCode[rowCode] ?? _string(row[subjectCol]),
          code: rowCode,
          component: _string(row[componentCol]),
          date: date,
          time: _string(row[startCol]),
          duration: '2h',
          board: profile.curriculum,
        );

        results.add(event);
      } catch (_) {}
    }

    print('SupabaseExamDates: Processed ${results.length}/${rows.length} valid rows (upcomingOnly=$upcomingOnly)');
    return results;
  }

  bool _curriculumMatches(String rowValue, String targetValue) {
    final row = _canonicalCurriculum(rowValue).toLowerCase();
    final target = _canonicalCurriculum(targetValue).toLowerCase();
    if (row.isEmpty || target.isEmpty) return true;
    if (row == target) return true;

    // Parse board_id + level from both
    final rowParts = row.split('_');
    final targetParts = target.split('_');
    final rowBoard = rowParts.isNotEmpty ? rowParts[0] : '';
    final rowLevel = rowParts.length > 1 ? rowParts.sublist(1).join('_') : '';
    final targetBoard = targetParts.isNotEmpty ? targetParts[0] : '';
    final targetLevel = targetParts.length > 1 ? targetParts.sublist(1).join('_') : '';

    // Same board — CAIE cross-level matching (AS/A Level share subjects)
    if (rowBoard == targetBoard) {
      final caieLevels = {'igcse', 'as_level', 'a_level', 'o_level'};
      if (caieLevels.contains(rowLevel) && caieLevels.contains(targetLevel)) return true;
    }

    // Legacy: IGCSE/A Level/O Level without board prefix
    final legacyLevels = {'igcse', 'a_level', 'as_level', 'o_level', 'gcse'};
    if (legacyLevels.contains(row) && legacyLevels.contains(target)) return true;

    return false;
  }

  DateTime _parseDate(String raw) {
    if (raw.isEmpty) return DateTime.now();
    // Try ISO first (e.g. "2026-05-07")
    final iso = DateTime.tryParse(raw);
    if (iso != null) return iso;
    // Try "Thursday, 07 May 2026" format
    try {
      final cleaned = raw.replaceAll(RegExp(r'^[A-Za-z]+,\s*'), '');
      final parts = cleaned.trim().split(' ');
      if (parts.length >= 3) {
        final day = int.parse(parts[0]);
        const months = {
          'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
          'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12,
        };
        final month = months[parts[1]] ?? 1;
        final year = int.parse(parts[2]);
        if (year >= 2020 && year <= 2100) return DateTime(year, month, day);
      }
    } catch (_) {}
    return DateTime.now();
  }

  bool _subjectCodeMatches(String rowCode, List<String> profileCodes) {
    for (final code in profileCodes) {
      if (rowCode == code) return true;
      if (rowCode.startsWith('$code/')) return true;  // e.g. "9709/11" matches "9709"
      if (rowCode.startsWith('$code ')) return true;  // e.g. "9709 11" matches "9709"
    }
    return false;
  }

  Future<List<String>> _subjectsFromPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final fromPrefs = prefs.getStringList('userSubjects') ?? [];
    if (fromPrefs.isNotEmpty) return fromPrefs;
    // Also try the curriculum catalog subjects
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
    // Try with parenthesis stripping
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

// ── Curriculum normalization (global for shared use) ──────────────────
String _canonicalCurriculum(String value) {
  final normalized = value.trim().toLowerCase().replaceAll('-', ' ').replaceAll('_', ' ');
  if (normalized.isEmpty) return '';

  // Step 1: Identify board — only CAIE
  String? boardId;
  if (normalized.contains('caie') || normalized.contains('cambridge') || normalized.contains('cie')) {
    boardId = 'caie';
  }

  // Step 2: Identify level
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

  // Fallbacks — all unrecognized input defaults to CAIE
  if (normalized.contains('a level') || normalized.contains('alevel')) return 'caie_a_level';
  if (normalized.contains('igcse') || normalized.contains('cambridge') || normalized.contains('cie') || normalized.contains('caie')) return 'caie_igcse';

  return value.trim();
}
