import 'package:flutter/foundation.dart';
import 'supabase_proxy_service.dart';

class SupabaseDataService {
  static final SupabaseDataService _instance = SupabaseDataService._();
  static SupabaseDataService get instance => _instance;

  final _proxy = SupabaseProxyService.instance;

  SupabaseDataService._();

  Future<List<Map<String, dynamic>>> getBoards() async {
    try {
      return await _proxy.query('boards', params: {'order': 'name.asc'});
    } catch (e) {
      debugPrint('Error fetching boards: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getAllGlobalSubjects() async {
    try {
      return await _proxy.query('subjects', params: {'order': 'name.asc'});
    } catch (e) {
      debugPrint('Error fetching all subjects: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getSubjectsByBoard(String boardId) async {
    try {
      return await _proxy.query('subjects', params: {
        'board_id': 'eq.$boardId',
        'order': 'name.asc',
      });
    } catch (e) {
      debugPrint('Error fetching subjects: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getGlobalChapters(
      String subjectCode) async {
    try {
      return await _proxy.query('chapters', params: {
        'subject_code': 'eq.$subjectCode',
        'order': 'order_index.asc',
      });
    } catch (e) {
      debugPrint('Error fetching chapters: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getAllGlobalChapters() async {
    try {
      return await _proxy.query('chapters', params: {
        'order': 'subject_code.asc,order_index.asc',
      });
    } catch (e) {
      debugPrint('Error fetching all chapters: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getNotesByChapter(String chapterId) async {
    try {
      return await _proxy.query('user_notes', params: {
        'chapter_id': 'eq.$chapterId',
        'deleted': 'eq.false',
        'order': 'updated_at.desc',
      });
    } catch (e) {
      debugPrint('Error fetching chapter notes: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getUserAllNotes() async {
    try {
      return await _proxy.query('user_notes', params: {
        'deleted': 'eq.false',
        'order': 'updated_at.desc',
      });
    } catch (e) {
      debugPrint('Error fetching all notes: $e');
      return [];
    }
  }

  Future<void> saveUserNote({
    required String id,
    required String userId,
    required String subjectCode,
    required String chapterId,
    String? title,
    String? content,
  }) async {
    final now = DateTime.now().toIso8601String();
    try {
      await _proxy.mutate('user_notes',
          method: 'upsert',
          body: {
            'id': id,
            'user_id': userId,
            'subject_code': subjectCode,
            'chapter_id': chapterId,
            'title': title,
            'content': content,
            'created_at': now,
            'updated_at': now,
            'deleted': false,
          });
    } catch (e) {
      debugPrint('Error saving note: $e');
    }
  }

  Future<void> deleteUserNote(String noteId) async {
    try {
      await _proxy.mutate('user_notes',
          method: 'update',
          body: {'deleted': true, 'updated_at': DateTime.now().toIso8601String()},
          params: {'id': 'eq.$noteId'});
    } catch (e) {
      debugPrint('Error deleting note: $e');
    }
  }

  Future<List<Map<String, dynamic>>> getGlobalNotes(
      String subjectCode, String chapterName) async {
    try {
      final data = await _proxy.query('global_notes', params: {
        'subject_code': 'eq.$subjectCode',
        'chapter_id': 'eq.${_normalizeChapterName(chapterName)}',
        'order': 'created_at.desc',
      });

      if (data.isNotEmpty) return data;

      final allNotes = await _proxy.query('global_notes', params: {
        'subject_code': 'eq.$subjectCode',
        'order': 'created_at.desc',
      });

      final filtered = allNotes.where((note) {
        final noteChapterId =
            (note['chapter_id'] ?? '').toString().toLowerCase();
        final searchChapter = chapterName.toLowerCase();
        return noteChapterId.contains(searchChapter) ||
            (note['title'] ?? '')
                .toString()
                .toLowerCase()
                .contains(searchChapter);
      }).toList();

      return filtered;
    } catch (e) {
      debugPrint('Error fetching global notes: $e');
      return [];
    }
  }

  String _normalizeChapterName(String chapterName) {
    return chapterName.trim().toLowerCase().replaceAll(' ', '_');
  }

  Future<List<Map<String, dynamic>>> getAllGlobalNotes() async {
    try {
      return await _proxy.query('global_notes', params: {
        'order': 'subject_code.asc,created_at.desc',
      });
    } catch (e) {
      debugPrint('Error fetching all global notes: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getChaptersBySubjectCode(
      String subjectCode) async {
    try {
      return await _proxy.query('chapters', params: {
        'subject_code': 'eq.$subjectCode',
        'order': 'order_index.asc',
      });
    } catch (e) {
      debugPrint('Error fetching chapters: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getUserSubjects(String userId) async {
    try {
      return await _proxy.query('user_subjects', params: {
        'user_id': 'eq.$userId',
        'order': 'created_at.asc',
      });
    } catch (e) {
      debugPrint('Error fetching user subjects: $e');
      return [];
    }
  }

  Future<void> saveUserSubjectSelection({
    required String id,
    required String userId,
    required String boardId,
    required String subjectCode,
    required String subjectName,
  }) async {
    try {
      await _proxy.mutate('user_subjects', method: 'upsert', body: {
        'id': id,
        'user_id': userId,
        'board_id': boardId,
        'subject_code': subjectCode,
        'subject_name': subjectName,
        'created_at': DateTime.now().toIso8601String(),
      });
    } catch (e) {
      debugPrint('Error saving user subject: $e');
    }
  }

  Future<void> removeUserSubject(String id, String userId) async {
    try {
      await _proxy.mutate('user_subjects',
          method: 'delete', params: {'id': 'eq.$id', 'user_id': 'eq.$userId'});
    } catch (e) {
      debugPrint('Error removing user subject: $e');
    }
  }
}
