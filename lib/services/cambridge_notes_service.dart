import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:shared_preferences/shared_preferences.dart';

class CambridgeNotesService {
  static final CambridgeNotesService instance = CambridgeNotesService._();
  CambridgeNotesService._();

  Map<String, CurriculumNotes>? _notesMap;

  bool get isReady => _notesMap != null;

  Future<void> initialize() async {
    if (_notesMap != null) return;
    try {
      final jsonStr =
          await rootBundle.loadString('assets/data/study_notes.json');
      _notesMap = _decodeNotes(jsonStr);
    } catch (e) {
      try {
        final prefs = await SharedPreferences.getInstance();
        final cached = prefs.getString('cached_study_notes');
        if (cached != null) {
          _notesMap = _decodeNotes(cached);
        }
      } catch (_) {
        _notesMap = const {};
      }
    }
  }

  Future<void> initializeWithJson(String jsonStr) async {
    _notesMap = _decodeNotes(jsonStr);
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('cached_study_notes', jsonStr);
    } catch (_) {}
  }

  CurriculumNotes? forSubject(String subjectCode) {
    final key = _resolveSubjectKey(subjectCode);
    return key == null ? null : _notesMap?[key];
  }

  String? noteText(String subjectCode, String chapterId, String subchapter) {
    final subject = forSubject(subjectCode);
    if (subject == null) return null;
    final requestedChapter = _normalizeLookup(chapterId);
    CurriculumNoteChapter? chapter;
    for (final candidate in subject.chapters) {
      if (_normalizeLookup(candidate.id) == requestedChapter ||
          _normalizeLookup(candidate.title) == requestedChapter) {
        chapter = candidate;
        break;
      }
    }
    if (chapter == null) return null;
    return chapter.getNote(subchapter);
  }

  List<String> allSubjectCodes() => _notesMap?.keys.toList() ?? [];

  Map<String, CurriculumNotes> _decodeNotes(String jsonStr) {
    final decoded = jsonDecode(jsonStr) as Map<String, dynamic>;
    return decoded.map((code, data) {
      final map =
          data is Map ? Map<String, dynamic>.from(data) : <String, dynamic>{};
      return MapEntry(code, CurriculumNotes.fromJson(code, map));
    });
  }

  String? _resolveSubjectKey(String subjectCodeOrName) {
    final notes = _notesMap;
    if (notes == null || notes.isEmpty) return null;
    if (notes.containsKey(subjectCodeOrName)) return subjectCodeOrName;

    final requested = _normalizeLookup(subjectCodeOrName);
    for (final entry in notes.entries) {
      if (_normalizeLookup(entry.key) == requested ||
          _normalizeLookup(entry.value.subjectName) == requested ||
          _normalizeLookup(subjectName(entry.key)) == requested) {
        return entry.key;
      }
    }
    return null;
  }

  static String _normalizeLookup(String value) =>
      value.trim().toLowerCase().replaceAll(RegExp(r'\s+'), ' ');

  static String subjectName(String code) {
    const names = {
      '9618': 'Computer Science',
      '9471': 'Computer Science',
      '9608': 'Computer Science',
      '9709': 'Mathematics',
      '0580': 'Mathematics',
      '9231': 'Further Mathematics',
      '9702': 'Physics',
      '0625': 'Physics',
      '9701': 'Chemistry',
      '0620': 'Chemistry',
      '9700': 'Biology',
      '0610': 'Biology',
      '9708': 'Economics',
      '0455': 'Economics',
      '9609': 'Business',
      '9706': 'Accounting',
      '9389': 'History',
      '0470': 'History',
      '9696': 'Geography',
      '0460': 'Geography',
      '9093': 'English',
      '0500': 'English',
      '9699': 'Sociology',
      '9698': 'Psychology',
      '9694': 'Environmental Management',
    };
    return names[code] ?? 'Unknown';
  }
}

class CurriculumNotes {
  final String subjectCode;
  final String subjectName;
  final List<CurriculumNoteChapter> chapters;

  const CurriculumNotes({
    required this.subjectCode,
    required this.subjectName,
    required this.chapters,
  });

  factory CurriculumNotes.fromJson(String code, Map<String, dynamic> json) {
    final chaptersList = (json['chapters'] as List?)
            ?.whereType<Map>()
            .map((c) => Map<String, dynamic>.from(c))
            .toList() ??
        [];
    return CurriculumNotes(
      subjectCode: code,
      subjectName:
          (json['name'] as String?) ?? CambridgeNotesService.subjectName(code),
      chapters:
          chaptersList.map((c) => CurriculumNoteChapter.fromJson(c)).toList(),
    );
  }
}

class CurriculumNoteChapter {
  final String id;
  final String title;
  final List<CurriculumNoteSubchapter> subchapters;

  const CurriculumNoteChapter({
    required this.id,
    required this.title,
    required this.subchapters,
  });

  factory CurriculumNoteChapter.fromJson(Map<String, dynamic> json) {
    final subs = (json['subchapters'] as List?)
            ?.whereType<Map>()
            .map((s) => Map<String, dynamic>.from(s))
            .toList() ??
        [];
    return CurriculumNoteChapter(
      id: json['id'] as String? ?? '',
      title: json['title'] as String? ?? '',
      subchapters:
          subs.map((s) => CurriculumNoteSubchapter.fromJson(s)).toList(),
    );
  }

  String? getNote(String subchapterId) {
    final requested = CambridgeNotesService._normalizeLookup(subchapterId);
    for (final sub in subchapters) {
      if (CambridgeNotesService._normalizeLookup(sub.id) == requested ||
          CambridgeNotesService._normalizeLookup(sub.title) == requested) {
        return sub.text;
      }
    }
    return null;
  }
}

class CurriculumNoteSubchapter {
  final String id;
  final String title;
  final String text;

  const CurriculumNoteSubchapter({
    required this.id,
    required this.title,
    required this.text,
  });

  factory CurriculumNoteSubchapter.fromJson(Map<String, dynamic> json) {
    return CurriculumNoteSubchapter(
      id: json['id'] as String? ?? '',
      title: json['title'] as String? ?? '',
      text: json['text'] as String? ?? '',
    );
  }
}
