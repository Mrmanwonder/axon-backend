import 'package:flutter/services.dart';

class LocalCsvService {
  static final LocalCsvService _instance = LocalCsvService._();
  static LocalCsvService get instance => _instance;
  LocalCsvService._();

  final Map<String, List<Map<String, String>>> _cache = {};

  Future<List<Map<String, String>>> loadCsv(String assetPath) async {
    if (_cache.containsKey(assetPath)) {
      return _cache[assetPath]!;
    }

    final raw = await rootBundle.loadString(assetPath);
    final lines = raw
        .replaceAll('\r\n', '\n')
        .replaceAll('\r', '\n')
        .split('\n')
        .where((l) => l.trim().isNotEmpty)
        .toList();
    if (lines.isEmpty) return [];

    final headers = _parseCsvLine(lines[0])
        .map((h) => h.replaceFirst('\uFEFF', '').trim())
        .toList();
    final rows = <Map<String, String>>[];

    for (int i = 1; i < lines.length; i++) {
      final values = _parseCsvLine(lines[i]);
      if (values.isEmpty) continue;
      final row = <String, String>{};
      for (int j = 0; j < headers.length; j++) {
        row[headers[j]] = j < values.length ? values[j].trim() : '';
      }
      rows.add(row);
    }

    _cache[assetPath] = rows;
    return rows;
  }

  List<String> _parseCsvLine(String line) {
    final result = <String>[];
    var current = StringBuffer();
    var inQuotes = false;

    for (int i = 0; i < line.length; i++) {
      final char = line[i];
      if (char == '"') {
        if (inQuotes && i + 1 < line.length && line[i + 1] == '"') {
          current.write('"');
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char == ',' && !inQuotes) {
        result.add(current.toString());
        current = StringBuffer();
      } else {
        current.write(char);
      }
    }
    result.add(current.toString());
    return result;
  }

  Future<List<Map<String, String>>> query(
    String assetPath, {
    Map<String, String>? where,
    String? orderBy,
    int? limit,
  }) async {
    var rows = await loadCsv(assetPath);

    if (where != null) {
      rows = rows.where((row) {
        return where.entries.every(
            (e) => (row[e.key] ?? '').toLowerCase() == e.value.toLowerCase());
      }).toList();
    }

    if (orderBy != null) {
      rows.sort((a, b) => (a[orderBy] ?? '').compareTo(b[orderBy] ?? ''));
    }

    if (limit != null && rows.length > limit) {
      rows = rows.take(limit).toList();
    }

    return rows;
  }

  Future<List<Map<String, String>>> search(
      String assetPath, String column, String query) async {
    final rows = await loadCsv(assetPath);
    final q = query.toLowerCase();
    return rows
        .where((row) => (row[column] ?? '').toLowerCase().contains(q))
        .toList();
  }

  void clearCache() => _cache.clear();
}
