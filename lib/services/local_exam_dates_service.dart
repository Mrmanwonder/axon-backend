import 'package:flutter/foundation.dart';
import 'local_csv_service.dart';
import '../models/exam_event_model.dart';

class LocalExamDatesService {
  static final LocalExamDatesService _instance = LocalExamDatesService._();
  static LocalExamDatesService get instance => _instance;
  LocalExamDatesService._();

  final _csv = LocalCsvService.instance;
  static const String _assetPath = 'assets/data/Datesheet_rows.csv';

  Future<List<ExamEventModel>> getAllExamDates() async {
    try {
      final rows = await _csv.loadCsv(_assetPath);
      return _parseToEvents(rows);
    } catch (e) {
      debugPrint('LocalExamDatesService: Failed to load dates: $e');
      return [];
    }
  }

  Future<List<ExamEventModel>> getExamDatesForSubject(String subjectCode) async {
    final rows = await _csv.query(_assetPath, where: {'Code': subjectCode});
    return _parseToEvents(rows);
  }

  Future<List<ExamEventModel>> getExamDatesForSubjects(List<String> subjectCodes) async {
    final all = await _csv.loadCsv(_assetPath);
    final codes = subjectCodes.map((c) => c.toLowerCase()).toSet();
    final filtered = all.where((r) =>
      codes.contains((r['Code'] ?? '').toLowerCase()) ||
      codes.any((c) => (r['Code'] ?? '').toLowerCase().startsWith(c))
    ).toList();
    return _parseToEvents(filtered);
  }

  Future<List<ExamEventModel>> getExamDatesByZone(String zone) async {
    final rows = await _csv.search(_assetPath, 'Zone', zone);
    return _parseToEvents(rows);
  }

  Future<List<ExamEventModel>> getUpcomingExamDates({
    String? zone,
    List<String>? subjectCodes,
  }) async {
    var rows = await _csv.loadCsv(_assetPath);
    final now = DateTime.now();

    if (zone != null && zone.isNotEmpty) {
      rows = rows.where((r) =>
        (r['Zone'] ?? '').toLowerCase() == zone.toLowerCase()).toList();
    }

    if (subjectCodes != null && subjectCodes.isNotEmpty) {
      final codes = subjectCodes.map((c) => c.toLowerCase()).toSet();
      rows = rows.where((r) {
        final rc = (r['Code'] ?? '').toLowerCase();
        return codes.contains(rc) || codes.any((c) => rc.startsWith(c));
      }).toList();
    }

    final events = _parseToEvents(rows);
    events.sort((a, b) => a.date.compareTo(b.date));
    return events.where((e) => e.date.isAfter(now.subtract(const Duration(days: 1)))).toList();
  }

  List<ExamEventModel> _parseToEvents(List<Map<String, String>> rows) {
    final events = <ExamEventModel>[];
    int idCounter = 0;

    for (final row in rows) {
      try {
        final date = _parseDate(row['Date'] ?? '');
        if (date.year < 2020 || date.year > 2100) continue;

        final code = row['Code'] ?? '';
        final codePrefix = code.contains('/') ? code.split('/')[0] : code;

        events.add(ExamEventModel(
          id: 'local_${++idCounter}_${date.millisecondsSinceEpoch}',
          board: _canonicalCurriculum(row['Level'] ?? ''),
          subject: row['Subject'] ?? code,
          code: codePrefix,
          component: row['Component'] ?? code,
          date: date,
          time: row['Session'] ?? 'AM',
          duration: row['Duration'] ?? '2h',
          type: row['Level'] ?? 'IGCSE',
        ));
      } catch (_) {}
    }

    return events;
  }

  DateTime _parseDate(String raw) {
    if (raw.isEmpty) return DateTime.now();
    
    final cleaned = raw.replaceAll(RegExp(r'^[A-Za-z]+,\s*'), '').trim();
    
    final iso = DateTime.tryParse(cleaned);
    if (iso != null) return iso;
    
    final isoRaw = DateTime.tryParse(raw);
    if (isoRaw != null) return isoRaw;
    
    try {
      final parts = cleaned.split(' ');
      if (parts.length >= 3) {
        final day = int.parse(parts[0]);
        const months = {
          'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
          'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
          'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
          'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        };
        final month = months[parts[1].toLowerCase()] ?? 1;
        final year = int.parse(parts[2]);
        if (year >= 2020 && year <= 2100) return DateTime(year, month, day);
      }
    } catch (_) {}
    
    return DateTime.now();
  }

  String _canonicalCurriculum(String value) {
    final normalized = value.trim().toLowerCase().replaceAll('-', ' ').replaceAll('_', ' ');
    if (normalized.isEmpty) return value.trim();

    // Step 1: Identify board — only CAIE
    String? boardId;
    if (normalized.contains('caie') || normalized.contains('cambridge') || normalized.contains('cie')) {
      boardId = 'caie';
    }

    // Step 2: Identify level
    String level;
    if (normalized.contains('as level') && !normalized.contains('a level')) {
      level = 'as_level';
    } else if (normalized.contains('a level') || normalized.contains('alevel') || normalized.contains('ial')) {
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
}
