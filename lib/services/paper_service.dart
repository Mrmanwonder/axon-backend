import 'dart:async';

import 'package:flutter/foundation.dart';
import 'supabase_proxy_service.dart';

@immutable
class PaperModel {
  final String curriculum;
  final String subject;
  final String subjectCode;
  final String year;
  final String series;
  final String paperType;
  final String paperNumber;
  final String filename;
  final String url;

  const PaperModel({
    required this.curriculum,
    required this.subject,
    required this.subjectCode,
    required this.year,
    required this.series,
    required this.paperType,
    this.paperNumber = '',
    this.filename = '',
    required this.url,
  });

  factory PaperModel.fromMap(Map<String, dynamic> map) {
    return PaperModel(
      curriculum: (map['curriculum'] ?? '').toString().trim(),
      subject: (map['subject'] ?? '').toString().trim(),
      subjectCode: (map['subject_code'] ?? '').toString().trim(),
      year: (map['year'] ?? '').toString().trim(),
      series: (map['series'] ?? '').toString().trim().toUpperCase(),
      paperType: (map['paper_type'] ?? '').toString().trim(),
      paperNumber: (map['paper_number'] ?? '').toString().trim(),
      filename: (map['filename'] ?? '').toString().trim(),
      url: (map['url'] ?? '').toString().trim(),
    );
  }

  bool get hasPdfUrl => url.isNotEmpty;

  Map<String, dynamic> toMap() {
    return {
      'curriculum': curriculum,
      'subject': subject,
      'subject_code': subjectCode,
      'year': year,
      'series': series,
      'paper_type': paperType,
      'paper_number': paperNumber,
      'filename': filename,
      'url': url,
    };
  }
}

class PaperService {
  PaperService._();

  static final PaperService _instance = PaperService._();
  static PaperService get instance => _instance;

  static const String _table = 'PYQs';
  static const String _selectColumns =
      'curriculum, subject, subject_code, year, series, paper_type, paper_number, filename, url';
  static const List<String> _defaultWarmupSeries = ['MJ', 'ON', 'FM'];

  final _proxy = SupabaseProxyService.instance;
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

      final response = await _proxy.query(_table, params: params);
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
