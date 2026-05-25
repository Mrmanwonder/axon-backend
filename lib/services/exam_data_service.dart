import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:home_widget/home_widget.dart';

class ExamDataService {
  static final ExamDataService _instance = ExamDataService._internal();
  factory ExamDataService() => _instance;
  ExamDataService._internal();

  static const String _manualSubjectKey = 'flip_clock_manual_subject';
  static const String _colorModeKey = 'flip_clock_color_mode';

  final List<ExamEvent> _cachedExams = [];
  bool _isLoaded = false;

  Future<void> initialize() async {
    if (_isLoaded) return;
    try {
      await _loadExamsFromCache();
    } catch (e) {
      print('Error initializing ExamDataService: $e');
    }
    _isLoaded = true;
  }

  Future<void> _loadExamsFromCache() async {
    try {
      final cacheDir = await getApplicationDocumentsDirectory();
      final datesheetDir = Directory('${cacheDir.parent.path}/datesheet_cache');

      if (!await datesheetDir.exists()) return;

      final files = datesheetDir.listSync().where(
            (f) => f.path.endsWith('.json'),
          );

      for (final file in files) {
        try {
          final content = await File(file.path).readAsString();
          final data = jsonDecode(content) as Map<String, dynamic>;
          final events = data['events'] as List? ?? [];

          for (final event in events) {
            try {
              _cachedExams.add(
                ExamEvent(
                  board: event['board'] ?? '',
                  subject: event['subject'] ?? '',
                  component: event['component'] ?? '',
                  date:
                      DateTime.tryParse(event['date'] ?? '') ?? DateTime(2026),
                  startTime: event['start_time'] ?? '',
                  endTime: event['end_time'] ?? '',
                ),
              );
            } catch (_) {}
          }
        } catch (_) {}
      }

      _cachedExams.sort((a, b) => a.date.compareTo(b.date));
    } catch (e) {
      print('Error loading exam cache: $e');
    }
  }

  Future<void> reloadCache() async {
    _cachedExams.clear();
    _isLoaded = false;
    await _loadExamsFromCache();
    _isLoaded = true;
  }

  List<ExamEvent> get allExams => List.unmodifiable(_cachedExams);

  ExamEvent getTargetExam(String? manualSubject) {
    if (manualSubject != null && manualSubject.isNotEmpty) {
      try {
        return _cachedExams.firstWhere(
          (e) =>
              e.subject.toUpperCase().contains(manualSubject.toUpperCase()) ||
              e.component.contains(manualSubject),
        );
      } catch (_) {}
    }
    return getNearestExam();
  }

  ExamEvent getNearestExam() {
    if (_cachedExams.isEmpty) {
      return ExamEvent(
        board: 'Cambridge A-Level',
        subject: 'No Exams Found',
        component: '---',
        date: DateTime.now().add(const Duration(days: 365)),
        startTime: '09:00',
        endTime: '12:00',
      );
    }

    final now = DateTime.now();
    final upcoming = _cachedExams.where((e) => e.date.isAfter(now)).toList();
    if (upcoming.isEmpty) return _cachedExams.last;
    return upcoming.first;
  }

  List<ExamEvent> getUpcomingExams({int limit = 5}) {
    final now = DateTime.now();
    return _cachedExams.where((e) => e.date.isAfter(now)).take(limit).toList();
  }

  Duration getTimeRemaining(ExamEvent exam) {
    final now = DateTime.now();
    final diff = exam.date.difference(now);
    return diff.isNegative ? Duration.zero : diff;
  }

  Future<void> setManualSubject(String? subject) async {
    final prefs = await SharedPreferences.getInstance();
    if (subject == null || subject.isEmpty) {
      await prefs.remove(_manualSubjectKey);
    } else {
      await prefs.setString(_manualSubjectKey, subject);
    }
    await _syncToHomeWidget();
  }

  Future<String?> getManualSubject() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_manualSubjectKey);
  }

  Future<void> setColorMode(WidgetColorMode mode) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_colorModeKey, mode.name);
    await _syncToHomeWidget();
  }

  Future<WidgetColorMode> getColorMode() async {
    final prefs = await SharedPreferences.getInstance();
    final mode = prefs.getString(_colorModeKey);
    return mode == 'neonRed'
        ? WidgetColorMode.neonRed
        : WidgetColorMode.focusBlue;
  }

  Future<void> _syncToHomeWidget() async {
    try {
      final nearest = getNearestExam();
      final timeRemaining = getTimeRemaining(nearest);

      await HomeWidget.saveWidgetData('exam_name', nearest.subject);
      await HomeWidget.saveWidgetData('exam_code', nearest.component);
      await HomeWidget.saveWidgetData(
        'exam_date',
        nearest.date.toIso8601String(),
      );
      await HomeWidget.saveWidgetData('days_remaining', timeRemaining.inDays);
      await HomeWidget.saveWidgetData(
        'hours_remaining',
        timeRemaining.inHours.remainder(24),
      );
      await HomeWidget.saveWidgetData(
        'minutes_remaining',
        timeRemaining.inMinutes.remainder(60),
      );
      await HomeWidget.saveWidgetData(
        'seconds_remaining',
        timeRemaining.inSeconds.remainder(60),
      );
      await HomeWidget.saveWidgetData(
        'manual_subject',
        await getManualSubject(),
      );
      await HomeWidget.saveWidgetData(
        'color_mode',
        (await getColorMode()).name,
      );
      await HomeWidget.saveWidgetData(
        'last_sync',
        DateTime.now().toIso8601String(),
      );

      await HomeWidget.updateWidget(
        name: 'FlipClockEvolutionWidgetProvider',
        iOSName: 'FlipClockEvolutionWidget',
      );
    } catch (e) {
      print('Widget sync error: $e');
    }
  }

  Future<void> refreshWidget() async {
    await _syncToHomeWidget();
  }
}

class ExamEvent {
  final String board;
  final String subject;
  final String component;
  final DateTime date;
  final String startTime;
  final String endTime;

  const ExamEvent({
    required this.board,
    required this.subject,
    required this.component,
    required this.date,
    required this.startTime,
    required this.endTime,
  });

  String get displayName => '$subject ($component)';

  int get daysRemaining => date.difference(DateTime.now()).inDays;

  bool get isUrgent => daysRemaining < 1;
  bool get isUpcoming => date.isAfter(DateTime.now());

  Map<String, dynamic> toJson() => {
        'board': board,
        'subject': subject,
        'component': component,
        'date': date.toIso8601String(),
        'start_time': startTime,
        'end_time': endTime,
      };
}

enum WidgetColorMode { focusBlue, neonRed }
