import 'dart:async';

import 'package:flutter/foundation.dart';
import 'supabase_service.dart';

class PaperService {
  final _supabase = SupabaseService.instance;
  final Map<String, List<PaperModel>> _cache = {};
  final Map<String, Future<List<PaperModel>>> _inFlightRequests = {};

  bool get hasCachedPapers => _cache.isNotEmpty;

  Future<List<PaperModel>> fetchPapers({
    required String subjectCode,
    String? year,
    String? series,
  }) async {
    final normalizedSubjectCode = subjectCode.trim();
    final normalizedYear = _normalizeOptional(year);
    final normalizedSeries = _normalizeOptional(series)?.toUpperCase();

    if (normalizedSubjectCode.isEmpty) {
      debugPrint('PaperService.fetchPapers skipped: empty subjectCode');
      return const [];
    }

    final cacheKey = _cacheKey(
      subjectCode: normalizedSubjectCode,
      year: normalizedYear,
      series: normalizedSeries,
    );

    final cached = _cache[cacheKey];
    if (cached != null) return cached;

    final existingRequest = _inFlightRequests[cacheKey];
    if (existingRequest != null) return existingRequest;

    final request = _fetchFromSupabase(
      subjectCode: normalizedSubjectCode,
      year: normalizedYear,
      series: normalizedSeries,
      cacheKey: cacheKey,
    );

    _inFlightRequests[cacheKey] = request;
    return request;
  }

  void warmUpSubjectCache({
    required String subjectCode,
    int numberOfYears = 3,
    List<String> series = _defaultWarmupSeries,
    int? startYear,
  }) {
    final normalizedSubjectCode = subjectCode.trim();
    if (normalizedSubjectCode.isEmpty || numberOfYears <= 0) return;

    final firstYear = startYear ?? DateTime.now().year;
    final years = List.generate(numberOfYears, (index) => firstYear - index);

    for (final year in years) {
      for (final rawSeries in series) {
        final normalizedSeries = rawSeries.trim().toUpperCase();
        if (normalizedSeries.isEmpty) continue;

        unawaited(
          fetchPapers(
            subjectCode: normalizedSubjectCode,
            year: year.toString(),
            series: normalizedSeries,
          ),
        );
      }
    }
  }

  List<PaperModel> getCachedPapers({
    required String subjectCode,
    String? year,
    String? series,
  }) {
    final normalizedSubjectCode = subjectCode.trim();
    if (normalizedSubjectCode.isEmpty) return const [];

    return _cache[_cacheKey(
          subjectCode: normalizedSubjectCode,
          year: _normalizeOptional(year),
          series: _normalizeOptional(series)?.toUpperCase(),
        )] ??
        const [];
  }

  void clearCache({String? subjectCode}) {
    final normalizedSubjectCode = _normalizeOptional(subjectCode);
    if (normalizedSubjectCode == null) {
      _cache.clear();
      _inFlightRequests.clear();
      return;
    }

    final prefix = '${normalizedSubjectCode.toLowerCase()}|';
    _cache.removeWhere((key, _) => key.startsWith(prefix));
    _inFlightRequests.removeWhere((key, _) => key.startsWith(prefix));
  }

  Future<List<PaperModel>> _fetchFromSupabase({
    required String subjectCode,
    required String? year,
    required String? series,
    required String cacheKey,
  }) async {
    try {
      final params = <String, String>{
        'select': _selectColumns,
        'subject_code': 'eq.${int.tryParse(subjectCode) ?? subjectCode}',
      };
      if (year != null) {
        params['year'] = 'eq.${int.tryParse(year) ?? year}';
      }
      if (series != null) {
        params['series'] = 'eq.$series';
      }

      final response = await _supabase.query(_table, params: params);
      final papers = response.map((data) {
        return PaperModel.fromMap(Map<String, dynamic>.from(data));
      }).toList();

      if (papers.isNotEmpty) {
        _cache[cacheKey] = papers;
      }
      _inFlightRequests.remove(cacheKey);
      return papers;
    } catch (e) {
      _inFlightRequests.remove(cacheKey);
      return const [];
    }
  }

  static String? _normalizeOptional(String? value) {
    final normalized = value?.trim();
    if (normalized == null || normalized.isEmpty) return null;
    return normalized;
  }

  static String _cacheKey({
    required String subjectCode,
    required String? year,
    required String? series,
  }) {
    return [
      subjectCode.toLowerCase(),
      year ?? '*',
      series?.toUpperCase() ?? '*',
    ].join('|');
  }
}
