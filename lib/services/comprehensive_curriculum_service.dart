import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'supabase_service.dart';

class ComprehensiveCurriculumService {
  static final ComprehensiveCurriculumService _instance =
      ComprehensiveCurriculumService._();
  static ComprehensiveCurriculumService get instance => _instance;
  ComprehensiveCurriculumService._();

  final _supabase = SupabaseService.instance;
  bool _initialized = false;
  bool _supabaseAvailable = false;

  // In-memory cache
  List<Map<String, dynamic>> _subjects = [];
  List<Map<String, dynamic>> _chapters = [];
  List<Map<String, dynamic>> _subchapters = [];
  final Map<String, String> _syllabusUrls = {};

  // Error tracking
  String? _lastError;
  List<String> _missingData = [];

  Future<void> initialize() async {
    if (_initialized) return;

    _lastError = null;
    _missingData = [];

    // Try to load from Supabase first
    await _loadFromSupabase();

    // If Supabase fails, fall back to local CSV
    if (!_supabaseAvailable) {
      debugPrint('Supabase unavailable, falling back to local data');
      await _loadFromLocal();
    }

    _initialized = true;
    debugPrint(
        'CurriculumService: ${_supabaseAvailable ? "Supabase" : "Local"} mode - ${_subjects.length} subjects, ${_chapters.length} chapters, ${_subchapters.length} subchapters');

    if (_missingData.isNotEmpty) {
      debugPrint('Missing data: $_missingData');
    }
  }

  Future<void> _loadFromSupabase() async {
    try {
      // Fetch subjects
      final subjectsResult = await _supabase.query('curriculum_subjects', params: {
        'order': 'subject_name.asc',
      });

      if (subjectsResult.isNotEmpty) {
        _subjects = List<Map<String, dynamic>>.from(subjectsResult);
        _supabaseAvailable = true;
      }

      // Fetch chapters
      final chaptersResult = await _supabase.query('curriculum_chapters', params: {
        'order': 'chapter_number.asc',
      });

      if (chaptersResult.isNotEmpty) {
        _chapters = List<Map<String, dynamic>>.from(chaptersResult);
      }

      // Fetch subchapters
      final subchaptersResult = await _supabase.query('curriculum_subchapters', params: {
        'order': 'subchapter_number.asc',
      });

      if (subchaptersResult.isNotEmpty) {
        _subchapters = List<Map<String, dynamic>>.from(subchaptersResult);
      }

      // Fetch syllabus URLs
      final syllabiResult = await _supabase.query('curriculum_syllabi');

      if (syllabiResult.isNotEmpty) {
        for (final s in syllabiResult) {
          _syllabusUrls[s['subject_code']] = s['syllabus_url'] ?? '';
        }
      }

      // Cache locally for offline
      await _cacheToLocal();
    } catch (e) {
      _lastError = 'Supabase error: $e';
      debugPrint('Supabase load failed: $e');
      _supabaseAvailable = false;
    }
  }

  Future<void> _loadFromLocal() async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // Try cached data first
      final cachedSubjects = prefs.getString('cached_subjects');
      final cachedChapters = prefs.getString('cached_chapters');
      final cachedSubchapters = prefs.getString('cached_subchapters');

      if (cachedSubjects != null) {
        _subjects = List<Map<String, dynamic>>.from(jsonDecode(cachedSubjects));
      }
      if (cachedChapters != null) {
        _chapters = List<Map<String, dynamic>>.from(jsonDecode(cachedChapters));
      }
      if (cachedSubchapters != null) {
        _subchapters =
            List<Map<String, dynamic>>.from(jsonDecode(cachedSubchapters));
      }

      // If still empty, try parsing CSV
      if (_subjects.isEmpty) {
        await _parseCsvFiles();
      }
    } catch (e) {
      _lastError = 'Local load failed: $e';
      debugPrint('Local load failed: $e');
    }
  }

  Future<void> _parseCsvFiles() async {
    // This is handled by curriculum_data_parser.dart
    // Just ensure we have at least something
    if (_subjects.isEmpty) {
      _missingData.add('No curriculum data found - need to fix this');
    }
  }

  Future<void> _cacheToLocal() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('cached_subjects', jsonEncode(_subjects));
      await prefs.setString('cached_chapters', jsonEncode(_chapters));
      await prefs.setString('cached_subchapters', jsonEncode(_subchapters));
      await prefs.setBool('supabase_available', _supabaseAvailable);
    } catch (e) {
      debugPrint('Cache save failed: $e');
    }
  }

  // ==================== PUBLIC API ====================

  // Get all subjects
  List<Map<String, dynamic>> getSubjects() {
    if (_subjects.isEmpty) {
      _missingData.add('No subjects loaded - need to fix this');
    }
    return _subjects;
  }

  // Get subjects by board
  List<Map<String, dynamic>> getSubjectsByBoard(String board) {
    final result = _subjects.where((s) => s['board'] == board).toList();
    if (result.isEmpty && _subjects.isNotEmpty) {
      _missingData.add('No subjects for board $board - need to fix this');
    }
    return result;
  }

  // Get single subject
  Map<String, dynamic>? getSubject(String subjectCode) {
    final result =
        _subjects.where((s) => s['subject_code'] == subjectCode).toList();
    if (result.isEmpty) {
      _missingData.add('Subject $subjectCode not found - need to fix this');
    }
    return result.isNotEmpty ? result.first : null;
  }

  // Get chapters for a subject
  List<Map<String, dynamic>> getChapters(String subjectCode) {
    final result =
        _chapters.where((c) => c['subject_code'] == subjectCode).toList()
          ..sort((a, b) {
            final aNum = int.tryParse(a['chapter_number'] ?? '0') ?? 0;
            final bNum = int.tryParse(b['chapter_number'] ?? '0') ?? 0;
            return aNum.compareTo(bNum);
          });

    if (result.isEmpty) {
      _missingData.add('No chapters for $subjectCode - need to fix this');
    }
    return result;
  }

  // Get single chapter
  Map<String, dynamic>? getChapter(String subjectCode, String chapterNumber) {
    final result = _chapters
        .where((c) =>
            c['subject_code'] == subjectCode &&
            c['chapter_number'] == chapterNumber)
        .toList();
    return result.isNotEmpty ? result.first : null;
  }

  // Get subchapters for a chapter
  List<Map<String, dynamic>> getSubchapters(
      String subjectCode, String chapterNumber) {
    final result = _subchapters
        .where((s) =>
            s['subject_code'] == subjectCode &&
            s['chapter_number'] == chapterNumber)
        .toList()
      ..sort((a, b) {
        final aNum = double.tryParse(a['subchapter_number'] ?? '0') ?? 0;
        final bNum = double.tryParse(b['subchapter_number'] ?? '0') ?? 0;
        return aNum.compareTo(bNum);
      });

    if (result.isEmpty) {
      _missingData.add(
          'No subchapters for $subjectCode ch$chapterNumber - need to fix this');
    }
    return result;
  }

  // Get single subchapter
  Map<String, dynamic>? getSubchapter(
      String subjectCode, String chapterNumber, String subchapterNumber) {
    final result = _subchapters
        .where((s) =>
            s['subject_code'] == subjectCode &&
            s['chapter_number'] == chapterNumber &&
            s['subchapter_number'] == subchapterNumber)
        .toList();
    return result.isNotEmpty ? result.first : null;
  }

  // Get syllabus URL
  String? getSyllabusUrl(String subjectCode) {
    final url = _syllabusUrls[subjectCode];
    if (url == null || url.isEmpty) {
      _missingData.add('No syllabus URL for $subjectCode - need to fix this');
    }
    return url;
  }

  // Check if data loaded from Supabase
  bool get isSupabaseAvailable => _supabaseAvailable;

  // Get error status
  String? get lastError => _lastError;

  // Get missing data warnings
  List<String> get missingDataWarnings => _missingData;

  // Clear warnings (call after showing to user)
  void clearWarnings() {
    _missingData.clear();
  }

  // Total counts
  int get totalSubjects => _subjects.length;
  int get totalChapters => _chapters.length;
  int get totalSubchapters => _subchapters.length;

  // Refresh data from backend
  Future<void> refresh() async {
    _initialized = false;
    _supabaseAvailable = false;
    await initialize();
  }

  // Search subjects
  List<Map<String, dynamic>> searchSubjects(String query) {
    if (query.isEmpty) return _subjects;
    final q = query.toLowerCase();
    return _subjects
        .where((s) =>
            (s['subject_name']?.toString().toLowerCase().contains(q) ??
                false) ||
            (s['subject_code']?.toString().toLowerCase().contains(q) ?? false))
        .toList();
  }
}

// User Progress Service - synced to backend
class UserProgressService {
  static final UserProgressService _instance = UserProgressService._();
  static UserProgressService get instance => _instance;
  UserProgressService._();

  final _supabase = SupabaseService.instance;
  String _userId = '';

  void setUserId(String userId) {
    _userId = userId;
  }

  bool get hasUserId => _userId.isNotEmpty;

  // Save scroll progress
  Future<void> updateProgress({
    required String subjectCode,
    required String chapterNumber,
    required String subchapterNumber,
    required double scrollPercentage,
    required int timeSpentSeconds,
  }) async {
    if (!hasUserId) return;

    try {
      await _supabase.mutate('user_subchapter_progress', method: 'upsert', body: {
        'user_id': _userId,
        'subject_code': subjectCode,
        'chapter_number': chapterNumber,
        'subchapter_number': subchapterNumber,
        'scroll_percentage': scrollPercentage,
        'time_spent_seconds': timeSpentSeconds,
        'last_accessed': DateTime.now().toIso8601String(),
      });

      // Also save locally
      await _saveLocally(subjectCode, chapterNumber, subchapterNumber,
          scrollPercentage, timeSpentSeconds);
    } catch (e) {
      debugPrint('Progress update failed: $e');
      // Still save locally
      await _saveLocally(subjectCode, chapterNumber, subchapterNumber,
          scrollPercentage, timeSpentSeconds);
    }
  }

  Future<void> _saveLocally(String subjectCode, String chapterNumber,
      String subchapterNumber, double scroll, int time) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'progress_${subjectCode}_${chapterNumber}_$subchapterNumber';
    await prefs.setString(
        key,
        jsonEncode({
          'scroll': scroll,
          'time': time,
          'synced': DateTime.now().toIso8601String(),
        }));
  }

  // Get progress
  Future<Map<String, dynamic>?> getProgress({
    required String subjectCode,
    required String chapterNumber,
    required String subchapterNumber,
  }) async {
    if (!hasUserId) {
      return await _getLocalProgress(
          subjectCode, chapterNumber, subchapterNumber);
    }

    try {
      final result = await _supabase.query('user_subchapter_progress', params: {
        'subject_code': 'eq.$subjectCode',
        'chapter_number': 'eq.$chapterNumber',
        'subchapter_number': 'eq.$subchapterNumber',
      });
      return result.isNotEmpty ? result.first : null;
    } catch (e) {
      return await _getLocalProgress(
          subjectCode, chapterNumber, subchapterNumber);
    }
  }

  Future<Map<String, dynamic>?> _getLocalProgress(
      String subjectCode, String chapterNumber, String subchapterNumber) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'progress_${subjectCode}_${chapterNumber}_$subchapterNumber';
    final data = prefs.getString(key);
    return data != null ? jsonDecode(data) : null;
  }

  // Get all progress for a subject
  Future<List<Map<String, dynamic>>> getSubjectProgress(
      String subjectCode) async {
    if (!hasUserId) return [];

    try {
      return await _supabase.query('user_subchapter_progress', params: {
        'user_id': 'eq.$_userId',
        'subject_code': 'eq.$subjectCode',
      });
    } catch (e) {
      return [];
    }
  }

  // Calculate completion percentage
  Future<double> getSubjectCompletion(String subjectCode) async {
    final chapters =
        ComprehensiveCurriculumService.instance.getChapters(subjectCode);
    if (chapters.isEmpty) return 0.0;

    int totalSubchapters = 0;
    int completedSubchapters = 0;

    for (final chapter in chapters) {
      final subchapters = ComprehensiveCurriculumService.instance
          .getSubchapters(subjectCode, chapter['chapter_number']);
      for (final sub in subchapters) {
        totalSubchapters++;
        final progress = await getProgress(
            subjectCode: subjectCode,
            chapterNumber: chapter['chapter_number'],
            subchapterNumber: sub['subchapter_number']);
        if (progress != null && (progress['scroll'] ?? 0.0) >= 80.0) {
          completedSubchapters++;
        }
      }
    }

    return totalSubchapters > 0
        ? (completedSubchapters / totalSubchapters * 100)
        : 0.0;
  }
}

// Notes Service
class SubchapterNotesService {
  static final SubchapterNotesService _instance = SubchapterNotesService._();
  static SubchapterNotesService get instance => _instance;
  SubchapterNotesService._();

  final _supabase = SupabaseService.instance;

  Future<String?> getNotes({
    required String subjectCode,
    required String chapterNumber,
    required String subchapterNumber,
  }) async {
    try {
      final result = await _supabase.query('subchapter_notes', params: {
        'subject_code': 'eq.$subjectCode',
        'chapter_number': 'eq.$chapterNumber',
        'subchapter_number': 'eq.$subchapterNumber',
      });
      return result.isNotEmpty ? result.first['notes'] : null;
    } catch (e) {
      return null;
    }
  }

  Future<void> saveNotes({
    required String subjectCode,
    required String chapterNumber,
    required String subchapterNumber,
    required String notes,
  }) async {
    try {
      await _supabase.mutate('subchapter_notes', method: 'upsert', body: {
        'subject_code': subjectCode,
        'chapter_number': chapterNumber,
        'subchapter_number': subchapterNumber,
        'notes': notes,
        'updated_at': DateTime.now().toIso8601String(),
      });
    } catch (e) {
      debugPrint('Save notes failed: $e');
    }
  }
}
