import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../models/models.dart';
import 'firestore_service.dart';

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

class BoardExamService {
  BoardExamService({http.Client? client}) : _client = client ?? http.Client();

  static String _backendUrl = 'https://bhavu.up.railway.app';
  static bool _initialized = false;

  static Future<void> setBackendUrl(String url) async {
    _backendUrl = url;
    _initialized = true;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('backendUrl', url);
    } catch (_) {}
  }

  static String get backendUrl {
    return _backendUrl;
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

  final http.Client _client;

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

    // Try multiple possible filenames
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
              startDate: DateTime.tryParse((event['date'] ?? '').toString()) ??
                  DateTime.now(),
              endDate: DateTime.tryParse((event['date'] ?? '').toString())
                      ?.add(const Duration(hours: 2)) ??
                  DateTime.now().add(const Duration(hours: 2)),
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
        board: board,
        sourceUrl: '',
        sources: const [],
        events: const [],
        error: 'User is not signed in.',
      );
    }

    final token = await user.getIdToken();
    if (token == null || token.isEmpty) {
      return BoardFetchResult(
        board: board,
        sourceUrl: '',
        sources: const [],
        events: const [],
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
            startDate:
                DateTime.tryParse((data['exam_date'] ?? '').toString()) ??
                    DateTime.now(),
            endDate: DateTime.tryParse((data['exam_date'] ?? '').toString()) ??
                DateTime.now(),
            source: (data['source_url'] ?? '').toString(),
          );
        }).toList();

        return BoardFetchResult(
          board: (decoded['board'] ?? board).toString(),
          sourceUrl: '',
          sources: [],
          events: events,
        );
      }
    } catch (e) {
      debugPrint('BoardExamService: Backend unavailable at $backendUrl — using local datesheet as fallback');
    }

    // Fallback to local datesheet
    final localEvents = _loadLocalDateSheet();
    if (_supportsLocalCambridgeFallback(board) && localEvents.isNotEmpty) {
      return BoardFetchResult(
        board: board,
        sourceUrl: 'Local datesheet (fallback)',
        sources: const [],
        events: localEvents,
      );
    }

    return BoardFetchResult(
      board: board,
      sourceUrl: '',
      sources: const [],
      events: const [],
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
    final sorted = [...events]
      ..sort((a, b) => a.startDate.compareTo(b.startDate));
    if (sorted.isEmpty) {
      return const AxonScheduleResult(
        efficiencyIndex: 0.0,
        dailyPlan: [],
        receiptByDate: {},
      );
    }

    final now = DateTime.now();
    final exam = sorted.first.startDate;
    final daysToExam =
        exam.difference(DateTime(now.year, now.month, now.day)).inDays;
    final availableDays = daysToExam <= 0 ? 1 : daysToExam;
    final recommendedUnits = (metrics['remaining_objectives'] is num)
        ? ((metrics['remaining_objectives'] as num) / availableDays).toDouble()
        : 0.0;

    return AxonScheduleResult(
      efficiencyIndex: recommendedUnits,
      dailyPlan: sorted
          .map(
            (event) => AxonDailyPlan(
              date: event.startDate,
              subject: event.subject,
              board: event.board,
              examLabel: event.label,
              isExamDay: true,
              intensity: 1.0,
              loadState: 'exam',
              recommendedUnits: recommendedUnits,
              receipt: 'Official datesheet synced from ${event.source}',
            ),
          )
          .toList(),
      receiptByDate: {
        for (final event in sorted)
          event.startDate.toIso8601String().split('T').first: event.source,
      },
    );
  }
}
