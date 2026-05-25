// lib/services/leaderboard_service.dart
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'firestore_service.dart';

class LeaderboardEntry {
  final String uid;
  final String displayName;
  final String? photoUrl;
  final int totalSessions;
  final int totalMinutes;
  final int currentStreak;
  final double predictedPerformance;

  const LeaderboardEntry({
    required this.uid,
    required this.displayName,
    this.photoUrl,
    required this.totalSessions,
    required this.totalMinutes,
    required this.currentStreak,
    required this.predictedPerformance,
  });
}

class LeaderboardService {
  static const String _backendUrl = 'https://bhavu.up.railway.app';
  final FirebaseFirestore _firestore = AxonFirestore.instance;

  Future<List<LeaderboardEntry>> getLeaderboard({int limit = 20}) async {
    try {
      final snapshot = await _firestore
          .collection(AxonCollections.usersPublic)
          .orderBy('total_sessions', descending: true)
          .limit(limit)
          .get();

      return snapshot.docs.map((doc) {
        final data = doc.data();
        return LeaderboardEntry(
          uid: doc.id,
          displayName: data['display_name']?.toString() ?? 'Student',
          photoUrl: data['photo_url']?.toString(),
          totalSessions: (data['total_sessions'] as num?)?.toInt() ?? 0,
          totalMinutes: (data['total_minutes'] as num?)?.toInt() ?? 0,
          currentStreak: (data['current_streak'] as num?)?.toInt() ?? 0,
          predictedPerformance:
              (data['predicted_performance'] as num?)?.toDouble() ?? 0.0,
        );
      }).toList();
    } catch (e) {
      return [];
    }
  }

  Future<LeaderboardEntry?> getUserRank(String uid) async {
    try {
      final allUsers = await getLeaderboard(limit: 100);
      for (int i = 0; i < allUsers.length; i++) {
        if (allUsers[i].uid == uid) {
          return LeaderboardEntry(
            uid: allUsers[i].uid,
            displayName: allUsers[i].displayName,
            photoUrl: allUsers[i].photoUrl,
            totalSessions: allUsers[i].totalSessions,
            totalMinutes: allUsers[i].totalMinutes,
            currentStreak: allUsers[i].currentStreak,
            predictedPerformance: allUsers[i].predictedPerformance,
          );
        }
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  static Future<void> publishScore({
    required int sessions,
    required int minutes,
    required int streak,
    required double performance,
  }) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    try {
      await AxonPaths.privateUserCollection(user.uid, 'leaderboard_stats')
          .doc('current')
          .set({
        'sessions_delta': sessions,
        'minutes_delta': minutes,
        'current_streak': streak,
        'predicted_performance': performance,
        'last_updated': FieldValue.serverTimestamp(),
      }, SetOptions(merge: true));

      // Write to users_public for public leaderboard read access
      await AxonFirestore.instance
          .collection(AxonCollections.usersPublic)
          .doc(user.uid)
          .set({
        'display_name': user.displayName ?? 'Student',
        'photo_url': user.photoURL,
        'total_sessions': sessions,
        'total_minutes': minutes,
        'current_streak': streak,
        'predicted_performance': performance,
        'last_updated': FieldValue.serverTimestamp(),
      }, SetOptions(merge: true));
      await syncPublicMirror();
    } catch (e) {
      debugPrint('Failed to publish score: $e');
    }
  }

  static Future<void> syncPublicMirror() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;
    try {
      final token = await user.getIdToken();
      await http
          .post(
            Uri.parse('$_backendUrl/syncLeaderboardProfile'),
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $token',
            },
            body: jsonEncode(const {}),
          )
          .timeout(const Duration(seconds: 20));
    } catch (_) {}
  }
}
