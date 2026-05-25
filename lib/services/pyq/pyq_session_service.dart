import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class PyqSession {
  final String subjectCode;
  final String subjectName;
  final String year;
  final String series;
  final String paperVariant;
  final double progress;
  final DateTime lastAccessed;

  const PyqSession({
    required this.subjectCode,
    required this.subjectName,
    required this.year,
    required this.series,
    required this.paperVariant,
    required this.progress,
    required this.lastAccessed,
  });

  Map<String, dynamic> toJson() => {
        'subjectCode': subjectCode,
        'subjectName': subjectName,
        'year': year,
        'series': series,
        'paperVariant': paperVariant,
        'progress': progress,
        'lastAccessed': lastAccessed.toIso8601String(),
      };

  factory PyqSession.fromJson(Map<String, dynamic> json) => PyqSession(
        subjectCode: json['subjectCode'] ?? '',
        subjectName: json['subjectName'] ?? '',
        year: json['year'] ?? '',
        series: json['series'] ?? '',
        paperVariant: json['paperVariant'] ?? '',
        progress: (json['progress'] ?? 0.0).toDouble(),
        lastAccessed: json['lastAccessed'] != null
            ? DateTime.parse(json['lastAccessed'])
            : DateTime.now(),
      );
}

class PyqSessionService {
  static final PyqSessionService _instance = PyqSessionService._();
  static PyqSessionService get instance => _instance;
  PyqSessionService._();

  static const String _sessionPrefix = 'pyq_session_';
  static const String _progressPrefix = 'pyq_progress_';
  static const String _completedPrefix = 'pyq_completed_';
  static const String _totalPrefix = 'pyq_total_';

  Future<void> saveSession(PyqSession session) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(
        '$_sessionPrefix${session.subjectCode}',
        jsonEncode(session.toJson()),
      );
    } catch (e) {
      debugPrint('PyqSessionService: Failed to save session: $e');
    }
  }

  Future<PyqSession?> getSession(String subjectCode) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final jsonStr = prefs.getString('$_sessionPrefix$subjectCode');
      if (jsonStr == null) return null;
      return PyqSession.fromJson(jsonDecode(jsonStr));
    } catch (e) {
      debugPrint('PyqSessionService: Failed to load session: $e');
      return null;
    }
  }

  Future<void> saveProgress(String subjectCode, double progress) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setDouble('$_progressPrefix$subjectCode', progress);
    } catch (e) {
      debugPrint('PyqSessionService: Failed to save progress: $e');
    }
  }

  Future<double> getProgress(String subjectCode) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      return prefs.getDouble('$_progressPrefix$subjectCode') ?? 0.0;
    } catch (e) {
      debugPrint('PyqSessionService: Failed to load progress: $e');
      return 0.0;
    }
  }

  Future<void> savePaperCounts(String subjectCode, int total, int completed) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt('$_totalPrefix$subjectCode', total);
      await prefs.setInt('$_completedPrefix$subjectCode', completed);
      if (total > 0) {
        await prefs.setDouble('$_progressPrefix$subjectCode', completed / total);
      }
    } catch (e) {
      debugPrint('PyqSessionService: Failed to save paper counts: $e');
    }
  }

  Future<int> getTotalPapers(String subjectCode) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      return prefs.getInt('$_totalPrefix$subjectCode') ?? 0;
    } catch (e) {
      return 0;
    }
  }

  Future<int> getCompletedPapers(String subjectCode) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      return prefs.getInt('$_completedPrefix$subjectCode') ?? 0;
    } catch (e) {
      return 0;
    }
  }

  Future<int> getRemainingPapers(String subjectCode) async {
    final total = await getTotalPapers(subjectCode);
    final completed = await getCompletedPapers(subjectCode);
    return total - completed;
  }

  Future<void> markPaperCompleted(String subjectCode, String year, String series, String variant) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final key = 'pyq_completed_${subjectCode}_${year}_${series}_$variant';
      await prefs.setBool(key, true);
      
      final completed = await getCompletedPapers(subjectCode);
      await prefs.setInt('$_completedPrefix$subjectCode', completed + 1);
      
      final total = await getTotalPapers(subjectCode);
      if (total > 0) {
        await prefs.setDouble('$_progressPrefix$subjectCode', (completed + 1) / total);
      }
    } catch (e) {
      debugPrint('PyqSessionService: Failed to mark paper completed: $e');
    }
  }

  Future<bool> isPaperCompleted(String subjectCode, String year, String series, String variant) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final key = 'pyq_completed_${subjectCode}_${year}_${series}_$variant';
      return prefs.getBool(key) ?? false;
    } catch (e) {
      return false;
    }
  }

  Future<void> clearSession(String subjectCode) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove('$_sessionPrefix$subjectCode');
      await prefs.remove('$_progressPrefix$subjectCode');
      await prefs.remove('$_completedPrefix$subjectCode');
      await prefs.remove('$_totalPrefix$subjectCode');
    } catch (e) {
      debugPrint('PyqSessionService: Failed to clear session: $e');
    }
  }
}
