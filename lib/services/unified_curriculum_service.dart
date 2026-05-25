import 'package:flutter/foundation.dart';
import 'comprehensive_curriculum_service.dart';

class UnifiedCurriculumService {
  static final UnifiedCurriculumService _instance =
      UnifiedCurriculumService._();
  static UnifiedCurriculumService get instance => _instance;
  UnifiedCurriculumService._();

  final _comp = ComprehensiveCurriculumService.instance;

  bool _initialized = false;
  List<CurriculumBoardEntry> _boards = [];

  Future<void> initialize() async {
    if (_initialized) return;

    await _comp.initialize();
    await _loadBoardsFromSupabase();

    _initialized = true;
    debugPrint('UnifiedCurriculumService: Ready');
  }

  Future<void> _loadBoardsFromSupabase() async {
    try {
      // Get all subjects
      final subjects = _comp.getSubjects();
      final boardMap = <String, List<Map<String, dynamic>>>{};

      for (final s in subjects) {
        final board = s['board'] as String? ?? 'Unknown';
        boardMap.putIfAbsent(board, () => []).add(s);
      }

      // Create board objects
      _boards = [];
      for (final entry in boardMap.entries) {
        final boardSubjects = <CurriculumSubjectEntry>[];
        for (final s in entry.value) {
          final code = s['subject_code'] as String? ?? '';
          final name = s['subject_name'] as String? ?? '';

          final chapters = _comp.getChapters(code);
          final chapterList = <CurriculumChapterEntry>[];
          for (final c in chapters) {
            final chNum = c['chapter_number'] as String? ?? '';
            final subchaptersList = _comp.getSubchapters(code, chNum);
            final subList = subchaptersList
                .map<String>((sc) => sc['subchapter_title'] as String? ?? '')
                .toList();
            chapterList.add(CurriculumChapterEntry(
                title: c['chapter_title'] as String? ?? '',
                subchapters: subList));
          }

          boardSubjects.add(CurriculumSubjectEntry(
              code: code, name: name, chapters: chapterList));
        }

        _boards.add(CurriculumBoardEntry(
            id: _norm(entry.key), label: entry.key, subjects: boardSubjects));
      }
    } catch (e) {
      debugPrint('Board load failed: $e');
    }
  }

  String _norm(String s) =>
      s.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '');

  List<CurriculumBoardEntry> get boards => _boards;

  CurriculumBoardEntry? findBoard(String rawBoard) {
    final normalized = _norm(rawBoard);
    for (final board in _boards) {
      if (_norm(board.id) == normalized || _norm(board.label) == normalized) {
        return board;
      }
    }
    if (normalized.contains('a level') || normalized.contains('as')) {
      return _boards.where((b) => b.id.contains('a_level')).firstOrNull;
    }
    if (normalized.contains('igcse')) {
      return _boards.where((b) => b.id.contains('igcse')).firstOrNull;
    }
    return _boards.isNotEmpty ? _boards.first : null;
  }

  List<String> subjectsForBoard(String boardId) {
    final board = _boards
        .where((b) => b.id == boardId || _norm(b.label) == _norm(boardId))
        .firstOrNull;
    return board?.subjects.map((s) => s.name).toList() ?? [];
  }

  List<CurriculumSubjectEntry> get allSubjects {
    final result = <CurriculumSubjectEntry>[];
    for (final board in _boards) {
      result.addAll(board.subjects);
    }
    return result;
  }

  CurriculumSubjectEntry? getSubjectDetails(String subjectCode) {
    for (final board in _boards) {
      for (final subject in board.subjects) {
        if (subject.code == subjectCode) {
          return subject;
        }
      }
    }
    return null;
  }

  Future<int> chapterCount(
      {required String board, required String subject}) async {
    final subjectDetails = allSubjects
        .where((s) =>
            s.name.toLowerCase() == subject.toLowerCase() ||
            s.code.toLowerCase() == subject.toLowerCase())
        .firstOrNull;
    return subjectDetails?.chapters.length ?? 0;
  }

  Future<String> canonicalBoardLabel(String rawBoard) async {
    final board = findBoard(rawBoard);
    return board?.label ?? rawBoard;
  }

  String getMissingDataWarning() {
    final warnings = _comp.missingDataWarnings;
    if (warnings.isEmpty) return '';
    return warnings.join('\n');
  }
}

class CurriculumBoardEntry {
  final String id;
  final String label;
  final List<CurriculumSubjectEntry> subjects;

  CurriculumBoardEntry(
      {required this.id, required this.label, required this.subjects});
}

class CurriculumSubjectEntry {
  final String code;
  final String name;
  final List<CurriculumChapterEntry> chapters;

  CurriculumSubjectEntry(
      {required this.code, required this.name, required this.chapters});
}

class CurriculumChapterEntry {
  final String title;
  final List<String> subchapters;

  CurriculumChapterEntry({required this.title, required this.subchapters});
}

final curriculumService = UnifiedCurriculumService.instance;
