import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'supabase_service.dart';

class OfflineSyncService {
  final _supabase = SupabaseService.instance;
  Timer? _syncTimer;
  bool _isSyncing = false;
  String? _currentUserId;

  OfflineSyncService._();

  void init(String userId) {
    _currentUserId = userId;
    _startPeriodicSync();
  }

  void _startPeriodicSync() {
    _syncTimer?.cancel();
    _syncTimer = Timer.periodic(const Duration(minutes: 5), (_) {
      syncAll();
    });
  }

  void dispose() {
    _syncTimer?.cancel();
  }

  Future<void> syncAll() async {
    if (_isSyncing || _currentUserId == null) return;
    _isSyncing = true;

    try {
      await Future.wait([
        _syncNotesFromCloud(),
        _syncPyqsFromCloud(),
        _syncMocksFromCloud(),
        _syncStudyProgressFromCloud(),
        _uploadLocalChanges(),
      ]);
    } catch (e) {
      debugPrint('Offline sync failed: $e');
    } finally {
      _isSyncing = false;
    }
  }

  Future<void> _uploadLocalChanges() async {
    if (_currentUserId == null) return;

    try {
      final unsyncedNotes =
          await LocalDatabase.instance.getUnsyncedNotes(_currentUserId!);
      for (final note in unsyncedNotes) {
        await _uploadNoteToCloud(note);
        await LocalDatabase.instance.markNoteSynced(note['id']);
      }

      final unsyncedPyqs =
          await LocalDatabase.instance.getUnsyncedPyqs(_currentUserId!);
      for (final pyq in unsyncedPyqs) {
        await _uploadPyqToCloud(pyq);
        await LocalDatabase.instance.markPyqSynced(pyq['id']);
      }

      final unsyncedMocks =
          await LocalDatabase.instance.getUnsyncedMocks(_currentUserId!);
      for (final mock in unsyncedMocks) {
        await _uploadMockToCloud(mock);
        await LocalDatabase.instance.markMockSynced(mock['id']);
      }

      final unsyncedProgress = await LocalDatabase.instance
          .getUnsyncedStudyProgress(_currentUserId!);
      for (final progress in unsyncedProgress) {
        await _uploadStudyProgressToCloud(progress);
        await LocalDatabase.instance.markStudyProgressSynced(progress['id']);
      }
    } catch (e) {
      debugPrint('Upload failed: $e');
    }
  }

  Future<void> _syncNotesFromCloud() async {
    if (_currentUserId == null) return;
    try {
      final cloudData = await _supabase.query('user_notes', params: {
        'user_id': 'eq.${_currentUserId!}',
      });

      for (final row in cloudData) {
        final localData =
            await LocalDatabase.instance.getAllNotes(_currentUserId!);
        final localNote =
            localData.where((n) => n['id'] == row['id']).firstOrNull;

        if (localNote == null) {
          await LocalDatabase.instance.upsertNote({
            'id': row['id'],
            'user_id': row['user_id'],
            'subject_code': row['subject_code'],
            'title': row['title'],
            'content': row['content'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
            'synced': 1,
            'deleted': row['deleted'] ?? 0,
          });
        } else {
          final localUpdated =
              DateTime.tryParse(localNote['updated_at'] ?? '') ??
                  DateTime(1970);
          final cloudUpdated =
              DateTime.tryParse(row['updated_at'] ?? '') ?? DateTime(1970);
          if (cloudUpdated.isAfter(localUpdated)) {
            await LocalDatabase.instance.upsertNote({
              'id': row['id'],
              'user_id': row['user_id'],
              'subject_code': row['subject_code'],
              'title': row['title'],
              'content': row['content'],
              'created_at': row['created_at'],
              'updated_at': row['updated_at'],
              'synced': 1,
              'deleted': row['deleted'] ?? 0,
            });
          }
        }
      }
    } catch (e) {
      debugPrint('Notes sync failed: $e');
    }
  }

  Future<void> _syncPyqsFromCloud() async {
    if (_currentUserId == null) return;
    try {
      final cloudData = await _supabase.query('user_pyqs', params: {
        'user_id': 'eq.${_currentUserId!}',
      });

      for (final row in cloudData) {
        final localData =
            await LocalDatabase.instance.getAllPyqs(_currentUserId!);
        final localPyq =
            localData.where((p) => p['id'] == row['id']).firstOrNull;

        if (localPyq == null) {
          await LocalDatabase.instance.upsertPyq({
            'id': row['id'],
            'user_id': row['user_id'],
            'subject_code': row['subject_code'],
            'year': row['year'],
            'variant': row['variant'],
            'paper_type': row['paper_type'],
            'score': row['score'],
            'time_spent_seconds': row['time_spent_seconds'],
            'completed_at': row['completed_at'],
            'updated_at': row['updated_at'],
            'synced': 1,
            'deleted': row['deleted'] ?? 0,
          });
        } else {
          final localUpdated =
              DateTime.tryParse(localPyq['updated_at'] ?? '') ?? DateTime(1970);
          final cloudUpdated =
              DateTime.tryParse(row['updated_at'] ?? '') ?? DateTime(1970);
          if (cloudUpdated.isAfter(localUpdated)) {
            await LocalDatabase.instance.upsertPyq({
              'id': row['id'],
              'user_id': row['user_id'],
              'subject_code': row['subject_code'],
              'year': row['year'],
              'variant': row['variant'],
              'paper_type': row['paper_type'],
              'score': row['score'],
              'time_spent_seconds': row['time_spent_seconds'],
              'completed_at': row['completed_at'],
              'updated_at': row['updated_at'],
              'synced': 1,
              'deleted': row['deleted'] ?? 0,
            });
          }
        }
      }
    } catch (e) {
      debugPrint('PYQs sync failed: $e');
    }
  }

  Future<void> _syncMocksFromCloud() async {
    if (_currentUserId == null) return;
    try {
      final cloudData = await _supabase.query('user_mocks', params: {
        'user_id': 'eq.${_currentUserId!}',
      });

      for (final row in cloudData) {
        final localData =
            await LocalDatabase.instance.getAllMocks(_currentUserId!);
        final localMock =
            localData.where((m) => m['id'] == row['id']).firstOrNull;

        if (localMock == null) {
          await LocalDatabase.instance.upsertMock({
            'id': row['id'],
            'user_id': row['user_id'],
            'subject_code': row['subject_code'],
            'paper_code': row['paper_code'],
            'title': row['title'],
            'total_marks': row['total_marks'],
            'obtained_marks': row['obtained_marks'],
            'time_taken_seconds': row['time_taken_seconds'],
            'attempt_date': row['attempt_date'],
            'updated_at': row['updated_at'],
            'synced': 1,
            'deleted': row['deleted'] ?? 0,
          });
        } else {
          final localUpdated =
              DateTime.tryParse(localMock['updated_at'] ?? '') ??
                  DateTime(1970);
          final cloudUpdated =
              DateTime.tryParse(row['updated_at'] ?? '') ?? DateTime(1970);
          if (cloudUpdated.isAfter(localUpdated)) {
            await LocalDatabase.instance.upsertMock({
              'id': row['id'],
              'user_id': row['user_id'],
              'subject_code': row['subject_code'],
              'paper_code': row['paper_code'],
              'title': row['title'],
              'total_marks': row['total_marks'],
              'obtained_marks': row['obtained_marks'],
              'time_taken_seconds': row['time_taken_seconds'],
              'attempt_date': row['attempt_date'],
              'updated_at': row['updated_at'],
              'synced': 1,
              'deleted': row['deleted'] ?? 0,
            });
          }
        }
      }
    } catch (e) {
      debugPrint('Mocks sync failed: $e');
    }
  }

  Future<void> _syncStudyProgressFromCloud() async {
    if (_currentUserId == null) return;
    try {
      final cloudData = await _supabase.query('study_progress', params: {
        'user_id': 'eq.${_currentUserId!}',
      });

      for (final row in cloudData) {
        final localData =
            await LocalDatabase.instance.getAllStudyProgress(_currentUserId!);
        final localProgress =
            localData.where((p) => p['id'] == row['id']).firstOrNull;

        if (localProgress == null) {
          await LocalDatabase.instance.upsertStudyProgress({
            'id': row['id'],
            'user_id': row['user_id'],
            'subject_code': row['subject_code'],
            'chapter_id': row['chapter_id'],
            'progress_percentage': row['progress_percentage'],
            'last_accessed': row['last_accessed'],
            'updated_at': row['updated_at'],
            'synced': 1,
          });
        } else {
          final localUpdated =
              DateTime.tryParse(localProgress['updated_at'] ?? '') ??
                  DateTime(1970);
          final cloudUpdated =
              DateTime.tryParse(row['updated_at'] ?? '') ?? DateTime(1970);
          if (cloudUpdated.isAfter(localUpdated)) {
            await LocalDatabase.instance.upsertStudyProgress({
              'id': row['id'],
              'user_id': row['user_id'],
              'subject_code': row['subject_code'],
              'chapter_id': row['chapter_id'],
              'progress_percentage': row['progress_percentage'],
              'last_accessed': row['last_accessed'],
              'updated_at': row['updated_at'],
              'synced': 1,
            });
          }
        }
      }
    } catch (e) {
      debugPrint('Study progress sync failed: $e');
    }
  }

  Future<void> _uploadNoteToCloud(Map<String, dynamic> note) async {
    final isDeleted = note['deleted'] == 1;
    if (isDeleted) {
      try {
        await _supabase.mutate('user_notes', method: 'delete', params: {'id': 'eq.${note['id']}'});
      } catch (e) {
        debugPrint('[OfflineSyncService] Failed to delete note from cloud: $e');
      }
    } else {
      try {
        await _supabase.mutate('user_notes', method: 'upsert', body: {
          'id': note['id'],
          'user_id': note['user_id'],
          'subject_code': note['subject_code'],
          'title': note['title'],
          'content': note['content'],
          'created_at': note['created_at'],
          'updated_at': note['updated_at'],
          'deleted': false,
        });
      } catch (e) {
        debugPrint('Note upload failed: $e');
      }
    }
  }

  Future<void> _uploadPyqToCloud(Map<String, dynamic> pyq) async {
    final isDeleted = pyq['deleted'] == 1;
    if (isDeleted) {
      try {
        await _supabase.mutate('user_pyqs', method: 'delete', params: {'id': 'eq.${pyq['id']}'});
      } catch (e) {
        debugPrint('[OfflineSyncService] Failed to delete PYQ from cloud: $e');
      }
    } else {
      try {
        await _supabase.mutate('user_pyqs', method: 'upsert', body: {
          'id': pyq['id'],
          'user_id': pyq['user_id'],
          'subject_code': pyq['subject_code'],
          'year': pyq['year'],
          'variant': pyq['variant'],
          'paper_type': pyq['paper_type'],
          'score': pyq['score'],
          'time_spent_seconds': pyq['time_spent_seconds'],
          'completed_at': pyq['completed_at'],
          'updated_at': pyq['updated_at'],
          'deleted': false,
        });
      } catch (e) {
        debugPrint('PYQ upload failed: $e');
      }
    }
  }

  Future<void> _uploadMockToCloud(Map<String, dynamic> mock) async {
    final isDeleted = mock['deleted'] == 1;
    if (isDeleted) {
      try {
        await _supabase.mutate('user_mocks', method: 'delete', params: {'id': 'eq.${mock['id']}'});
      } catch (e) {
        debugPrint('[OfflineSyncService] Failed to delete mock from cloud: $e');
      }
    } else {
      try {
        await _supabase.mutate('user_mocks', method: 'upsert', body: {
          'id': mock['id'],
          'user_id': mock['user_id'],
          'subject_code': mock['subject_code'],
          'paper_code': mock['paper_code'],
          'title': mock['title'],
          'total_marks': mock['total_marks'],
          'obtained_marks': mock['obtained_marks'],
          'time_taken_seconds': mock['time_taken_seconds'],
          'attempt_date': mock['attempt_date'],
          'updated_at': mock['updated_at'],
          'deleted': false,
        });
      } catch (e) {
        debugPrint('Mock upload failed: $e');
      }
    }
  }

  Future<void> _uploadStudyProgressToCloud(
      Map<String, dynamic> progress) async {
    try {
      await _supabase.mutate('study_progress', method: 'upsert', body: {
        'id': progress['id'],
        'user_id': progress['user_id'],
        'subject_code': progress['subject_code'],
        'chapter_id': progress['chapter_id'],
        'progress_percentage': progress['progress_percentage'],
        'last_accessed': progress['last_accessed'],
        'updated_at': progress['updated_at'],
      });
    } catch (e) {
      debugPrint('Study progress upload failed: $e');
    }
  }

  Future<void> saveNoteLocally({
    required String id,
    required String userId,
    String? subjectCode,
    String? title,
    String? content,
  }) async {
    final now = DateTime.now().toIso8601String();
    await LocalDatabase.instance.upsertNote({
      'id': id,
      'user_id': userId,
      'subject_code': subjectCode,
      'title': title,
      'content': content,
      'created_at': now,
      'updated_at': now,
    });
    _uploadNoteToCloud({
      'id': id,
      'user_id': userId,
      'subject_code': subjectCode,
      'title': title,
      'content': content,
      'created_at': now,
      'updated_at': now,
    });
  }

  Future<void> savePyqResultLocally({
    required String id,
    required String userId,
    required String subjectCode,
    required int year,
    String? variant,
    String? paperType,
    double? score,
    int? timeSpentSeconds,
  }) async {
    final now = DateTime.now().toIso8601String();
    await LocalDatabase.instance.upsertPyq({
      'id': id,
      'user_id': userId,
      'subject_code': subjectCode,
      'year': year,
      'variant': variant,
      'paper_type': paperType,
      'score': score,
      'time_spent_seconds': timeSpentSeconds,
      'completed_at': now,
      'updated_at': now,
    });
  }

  Future<void> saveMockResultLocally({
    required String id,
    required String userId,
    required String subjectCode,
    String? paperCode,
    String? title,
    double? totalMarks,
    double? obtainedMarks,
    int? timeTakenSeconds,
    String? attemptDate,
  }) async {
    final now = DateTime.now().toIso8601String();
    await LocalDatabase.instance.upsertMock({
      'id': id,
      'user_id': userId,
      'subject_code': subjectCode,
      'paper_code': paperCode,
      'title': title,
      'total_marks': totalMarks,
      'obtained_marks': obtainedMarks,
      'time_taken_seconds': timeTakenSeconds,
      'attempt_date': attemptDate ?? now,
      'updated_at': now,
    });
  }

  Future<void> saveStudyProgressLocally({
    required String id,
    required String userId,
    required String subjectCode,
    String? chapterId,
    double? progressPercentage,
  }) async {
    final now = DateTime.now().toIso8601String();
    await LocalDatabase.instance.upsertStudyProgress({
      'id': id,
      'user_id': userId,
      'subject_code': subjectCode,
      'chapter_id': chapterId,
      'progress_percentage': progressPercentage,
      'last_accessed': now,
      'updated_at': now,
    });
  }

  Future<void> deleteNote(String id) async {
    await LocalDatabase.instance.deleteNoteLocally(id);
  }

  Future<void> deletePyq(String id) async {
    await LocalDatabase.instance.deletePyqLocally(id);
  }

  Future<void> deleteMock(String id) async {
    await LocalDatabase.instance.deleteMockLocally(id);
  }
}
