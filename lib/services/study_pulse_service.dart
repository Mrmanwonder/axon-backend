import 'dart:convert';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;

import '../models/study_pulse_analytics.dart';
import 'firestore_service.dart';

class StudyPulseService {
  StudyPulseService({http.Client? client}) : _client = client ?? http.Client();

  static const String _backendUrl = 'https://bhavu.up.railway.app';
  final http.Client _client;

  Stream<StudyPulseAnalytics?> watchCurrentAnalytics(String uid) {
    return AxonPaths.privateUserCollection(uid, 'analytics')
        .doc('current')
        .snapshots()
        .map((snapshot) {
      final data = snapshot.data();
      if (!snapshot.exists || data == null) {
        return null;
      }
      return StudyPulseAnalytics.fromJson(data);
    });
  }

  Future<void> analyzeStudyPulse({String? userId, String? sessionId}) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final token = await user.getIdToken();
    if (token == null || token.isEmpty) return;

    await _client.post(
      Uri.parse('$_backendUrl/analyze-study-pulse'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode({
        'user_id': userId ?? user.uid,
        if (sessionId != null && sessionId.isNotEmpty) 'session_id': sessionId,
      }),
    );
  }
}
