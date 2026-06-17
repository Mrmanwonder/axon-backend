// lib/services/subject_cache.dart
import '../services/curriculum_catalog_service.dart';

class SubjectCache {
  static List<String>? _cachedSubjects;
  static Map<String, List<String>>? _cachedChapters;
  static DateTime? _lastLoadTime;
  static String? _cachedBoard;

  // LRU cache bounds - max board+subject combinations to cache
  static const int _maxChaptersEntries = 20;

  static Future<List<String>> getSubjects(String board) async {
    final now = DateTime.now();
    if (_cachedSubjects != null &&
        _cachedChapters != null &&
        _cachedBoard == board &&
        _lastLoadTime != null &&
        now.difference(_lastLoadTime!).inHours < 24) {
      return _cachedSubjects!;
    }
    final subjects =
        await CurriculumCatalogService.instance.subjectsForBoard(board);
    _cachedSubjects = subjects;
    _cachedBoard = board;
    _lastLoadTime = now;
    return subjects;
  }

  static Future<List<String>> getChapters(String board, String subject) async {
    final key = '${board}_$subject';
    if (_cachedChapters?.containsKey(key) ?? false) {
      // LRU: move to end by removing and re-adding
      final chapters = _cachedChapters![key]!;
      _cachedChapters!.remove(key);
      _cachedChapters![key] = chapters;
      return chapters;
    }
    final chapters = await CurriculumCatalogService.instance.chapterTitles(
      board: board,
      subject: subject,
    );
    _cachedChapters ??= {};

    // LRU eviction: remove oldest (first) if at capacity
    if (_cachedChapters!.length >= _maxChaptersEntries) {
      final oldestKey = _cachedChapters!.keys.first;
      _cachedChapters!.remove(oldestKey);
    }

    _cachedChapters![key] = chapters;
    return chapters;
  }

  static void clear() {
    _cachedSubjects = null;
    _cachedChapters = null;
    _lastLoadTime = null;
    _cachedBoard = null;
  }
}
