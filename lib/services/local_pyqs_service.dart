import 'package:flutter/foundation.dart';
import 'local_csv_service.dart';

class PastPaperRecord {
  final String curriculum;
  final String subject;
  final String subjectCode;
  final String year;
  final String series;
  final String paperType;
  final String paperNumber;
  final String filename;
  final String url;

  PastPaperRecord({
    required this.curriculum,
    required this.subject,
    required this.subjectCode,
    required this.year,
    required this.series,
    required this.paperType,
    required this.paperNumber,
    required this.filename,
    required this.url,
  });

  factory PastPaperRecord.fromMap(Map<String, String> map) => PastPaperRecord(
        curriculum: map['curriculum'] ?? '',
        subject: map['subject'] ?? '',
        subjectCode: map['subject_code'] ?? '',
        year: map['year'] ?? '',
        series: map['series'] ?? '',
        paperType: map['paper_type'] ?? '',
        paperNumber: map['paper_number'] ?? '',
        filename: map['filename'] ?? '',
        url: map['url'] ?? '',
      );

  bool matchesSubject(String subjectCodeOrName) {
    final requested = LocalPyqsService.normalizeLookup(subjectCodeOrName);
    final requestedCode =
        LocalPyqsService.normalizeSubjectCode(subjectCodeOrName);
    return LocalPyqsService.normalizeSubjectCode(subjectCode) ==
            requestedCode ||
        LocalPyqsService.normalizeLookup(subject) == requested ||
        LocalPyqsService.normalizeLookup(subject).contains(requested) ||
        requested.contains(LocalPyqsService.normalizeLookup(subject));
  }

  bool get isUsable =>
      subjectCode.trim().isNotEmpty &&
      year.trim().isNotEmpty &&
      filename.trim().isNotEmpty &&
      url.trim().isNotEmpty;
}

class LocalPyqsService {
  static final LocalPyqsService _instance = LocalPyqsService._();
  static LocalPyqsService get instance => _instance;
  LocalPyqsService._();

  final _csv = LocalCsvService.instance;
  static const String _assetPath = 'assets/data/PYQs_rows.csv';

  Future<List<PastPaperRecord>> getAllPapers() async {
    try {
      final rows = await _csv.loadCsv(_assetPath);
      return _normalizeRecords(rows.map((r) => PastPaperRecord.fromMap(r)));
    } catch (e) {
      debugPrint('LocalPyqsService: Failed to load papers: $e');
      return [];
    }
  }

  Future<List<PastPaperRecord>> getPapersBySubject(String subjectCode) async {
    final records = await getAllPapers();
    return records.where((r) => r.matchesSubject(subjectCode)).toList();
  }

  Future<List<PastPaperRecord>> getPapersBySubjectAndYear(
      String subjectCode, String year) async {
    final records = await getPapersBySubject(subjectCode);
    return records
        .where((r) => normalizeLookup(r.year) == normalizeLookup(year))
        .toList();
  }

  Future<List<PastPaperRecord>> getPapersByType(
      String subjectCode, String paperType) async {
    final records = await getPapersBySubject(subjectCode);
    return records
        .where(
            (r) => normalizeLookup(r.paperType) == normalizeLookup(paperType))
        .toList();
  }

  Future<List<String>> getAvailableYears(String subjectCode) async {
    final papers = await getPapersBySubject(subjectCode);
    final years = papers
        .map((r) => r.year)
        .where((y) => y.isNotEmpty)
        .toSet()
        .toList()
      ..sort((a, b) => b.compareTo(a));
    return years;
  }

  Future<List<String>> getAvailableSubjects() async {
    final papers = await getAllPapers();
    final subjects = papers
        .map((r) => '${r.subjectCode}|${r.subject}')
        .where((s) => !s.startsWith('|') && !s.endsWith('|'))
        .toSet()
        .toList()
      ..sort();
    return subjects;
  }

  Future<List<PastPaperRecord>> searchByCurriculum(String curriculum) async {
    final rows = await _csv.search(_assetPath, 'curriculum', curriculum);
    return _normalizeRecords(rows.map((r) => PastPaperRecord.fromMap(r)));
  }

  static String normalizeLookup(String value) =>
      value.trim().toLowerCase().replaceAll(RegExp(r'\s+'), ' ');

  static String normalizeSubjectCode(String value) =>
      value.trim().toUpperCase().replaceAll(RegExp(r'[^A-Z0-9]'), '');

  List<PastPaperRecord> _normalizeRecords(Iterable<PastPaperRecord> records) {
    final seen = <String>{};
    final normalized = <PastPaperRecord>[];

    for (final record in records) {
      if (!record.isUsable) continue;
      final key = record.url.trim().isNotEmpty
          ? record.url.trim()
          : '${record.subjectCode}:${record.year}:${record.series}:${record.paperType}:${record.paperNumber}:${record.filename}';
      if (seen.add(key)) {
        normalized.add(record);
      }
    }

    normalized.sort((a, b) {
      final yearCompare = b.year.compareTo(a.year);
      if (yearCompare != 0) return yearCompare;
      final seriesCompare = a.series.compareTo(b.series);
      if (seriesCompare != 0) return seriesCompare;
      return a.paperNumber.compareTo(b.paperNumber);
    });
    return normalized;
  }
}
