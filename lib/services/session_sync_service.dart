// lib/services/session_sync_service.dart
// ─────────────────────────────────────────────────────────────────
// Secure Session Sync Service
// All operations verify user ownership before accessing data
// ─────────────────────────────────────────────────────────────────

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'firestore_service.dart';

import 'leaderboard_service.dart';
import 'security_service.dart' as sec;

export 'leaderboard_service.dart';

class SecurityException implements Exception {
  final String message;
  SecurityException(this.message);

  @override
  String toString() => message;
}

class SessionSyncService {
  static final FirebaseFirestore _db = AxonFirestore.instance;
  static final FirebaseAuth _auth = FirebaseAuth.instance;
  static final sec.SecurityService _security = sec.SecurityService();

  /// Persist a completed study session to Firestore and update the
  /// global leaderboard entry for this user.
  ///
  /// Security: Verifies current user owns the session before writing
  static Future<void> syncSession({
    required String subject,
    required String chapter,
    required double intensityIndex,
    required double studyHours,
    required int breakCount,
    required int streak,
    required double totalWeekHours,
    String board = '',
  }) async {
    final user = _auth.currentUser;
    if (user == null) {
      throw SecurityException('User must be authenticated to sync session');
    }

    final uid = user.uid;

    // Verify the user ID is valid
    if (uid.isEmpty || !RegExp(r'^[A-Za-z0-9_-]{1,128}$').hasMatch(uid)) {
      throw SecurityException('Invalid user ID');
    }

    // Sanitize inputs to prevent injection
    final sanitizedSubject = _sanitizeString(subject, 50);
    final sanitizedChapter = _sanitizeString(chapter, 100);
    final sanitizedBoard = _sanitizeString(board, 50);

    final sessionId = '${DateTime.now().millisecondsSinceEpoch}';

    // Write session document to user's private collection
    // Path: users/{uid}/sessions/{sessionId}
    await _db
        .collection(AxonCollections.usersPrivate)
        .doc(uid)
        .collection('sessions')
        .doc(sessionId)
        .set({
      'uid': uid,
      'board': sanitizedBoard,
      'subject': sanitizedSubject,
      'chapter': sanitizedChapter,
      'intensity_index': _clampDouble(intensityIndex, 0.0, 1.0),
      'study_hours': _clampDouble(studyHours, 0.0, 24.0),
      'break_count': _clampInt(breakCount, 0, 100),
      'streak': _clampInt(streak, 0, 365),
      'started_at': FieldValue.serverTimestamp(),
      'date': DateTime.now().toIso8601String(),
    });

    // Update global leaderboard (separate collection, allowed to be public)
    await LeaderboardService.publishScore(
      sessions: 1,
      minutes: (studyHours * 60).toInt(),
      streak: streak,
      performance: intensityIndex,
    );
  }

  /// Read user's sessions with ownership verification
  static Future<List<Map<String, dynamic>>> getUserSessions(
      String userId) async {
    // Security: Verify ownership
    if (!_security.isOwner(userId)) {
      throw SecurityException(
          'Access denied: Cannot read another user\'s sessions');
    }

    final snapshot = await _db
        .collection(AxonCollections.usersPrivate)
        .doc(userId)
        .collection('sessions')
        .orderBy('started_at', descending: true)
        .limit(100)
        .get();

    return snapshot.docs.map((doc) {
      final data = doc.data();
      return {
        'id': doc.id,
        ...data,
      };
    }).toList();
  }

  /// Delete a session with ownership verification
  static Future<void> deleteSession(String userId, String sessionId) async {
    // Security: Verify ownership
    if (!_security.isOwner(userId)) {
      throw SecurityException(
          'Access denied: Cannot delete another user\'s session');
    }

    await _db
        .collection(AxonCollections.usersPrivate)
        .doc(userId)
        .collection('sessions')
        .doc(sessionId)
        .delete();
  }

  // Helper: Sanitize string input
  static String _sanitizeString(String input, int maxLength) {
    // Remove any potential injection characters
    return input
        .replaceAll(RegExp(r'[<>{}"$]'), '')
        .trim()
        .substring(0, input.length.clamp(0, maxLength));
  }

  // Helper: Clamp double value
  static double _clampDouble(double value, double min, double max) {
    return value.clamp(min, max);
  }

  // Helper: Clamp int value
  static int _clampInt(int value, int min, int max) {
    return value.clamp(min, max);
  }
}

class AxonCrashlytics {
  static Future<void> recordError(
    dynamic exception,
    StackTrace? stack, {
    String? reason,
    bool fatal = false,
  }) async {
    debugPrint(
        '[AxonCrashlytics] ${fatal ? 'FATAL' : 'error'}: ${reason ?? ''}\n$exception\n$stack');
  }

  static Future<void> setUser(String uid) async {}

  static Future<void> setKey(String key, dynamic value) async {}

  static void initFlutterErrorHandler() {
    // FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterFatalError;
  }
}
