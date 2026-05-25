import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'exam_planner_service.dart';

class MockExamSession {
  final String paperId;
  final String paperTitle;
  final String subject;
  final int durationMinutes;
  final int maxMarks;
  final int remainingSeconds;
  final int currentQuestion;
  final List<int> flaggedQuestions;
  final int violationCount;
  final DateTime startedAt;
  final DateTime updatedAt;

  const MockExamSession({
    required this.paperId,
    required this.paperTitle,
    required this.subject,
    required this.durationMinutes,
    required this.maxMarks,
    required this.remainingSeconds,
    required this.currentQuestion,
    required this.flaggedQuestions,
    required this.violationCount,
    required this.startedAt,
    required this.updatedAt,
  });

  MockExamSession copyWith({
    int? remainingSeconds,
    int? currentQuestion,
    List<int>? flaggedQuestions,
    int? violationCount,
    DateTime? updatedAt,
  }) {
    return MockExamSession(
      paperId: paperId,
      paperTitle: paperTitle,
      subject: subject,
      durationMinutes: durationMinutes,
      maxMarks: maxMarks,
      remainingSeconds: remainingSeconds ?? this.remainingSeconds,
      currentQuestion: currentQuestion ?? this.currentQuestion,
      flaggedQuestions: flaggedQuestions ?? this.flaggedQuestions,
      violationCount: violationCount ?? this.violationCount,
      startedAt: startedAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  Map<String, dynamic> toJson() => {
        'paperId': paperId,
        'paperTitle': paperTitle,
        'subject': subject,
        'durationMinutes': durationMinutes,
        'maxMarks': maxMarks,
        'remainingSeconds': remainingSeconds,
        'currentQuestion': currentQuestion,
        'flaggedQuestions': flaggedQuestions,
        'violationCount': violationCount,
        'startedAt': startedAt.toIso8601String(),
        'updatedAt': updatedAt.toIso8601String(),
      };

  factory MockExamSession.fromJson(Map<String, dynamic> json) {
    return MockExamSession(
      paperId: (json['paperId'] ?? '').toString(),
      paperTitle: (json['paperTitle'] ?? '').toString(),
      subject: (json['subject'] ?? '').toString(),
      durationMinutes: (json['durationMinutes'] ?? 75) as int,
      maxMarks: (json['maxMarks'] ?? 100) as int,
      remainingSeconds: (json['remainingSeconds'] ?? 0) as int,
      currentQuestion: (json['currentQuestion'] ?? 1) as int,
      flaggedQuestions: (json['flaggedQuestions'] as List? ?? const [])
          .map((value) => (value as num).toInt())
          .toList(),
      violationCount: (json['violationCount'] ?? 0) as int,
      startedAt:
          DateTime.tryParse((json['startedAt'] ?? '').toString()) ?? DateTime.now(),
      updatedAt:
          DateTime.tryParse((json['updatedAt'] ?? '').toString()) ?? DateTime.now(),
    );
  }
}

class MockExamResult {
  final int score;
  final String reflection;
  final int violationCount;

  const MockExamResult({
    required this.score,
    required this.reflection,
    required this.violationCount,
  });
}

class MockExamService {
  MockExamService._();

  static final MockExamService instance = MockExamService._();

  static const String _activeSessionKey = 'mock_exam_active_session';
  static const String _lastReflectionKey = 'mock_exam_last_reflection';
  static const String _accountabilityKey = 'mock_exam_accountability_pending';

  Future<void> startSession(PastPaperPack paper) async {
    final prefs = await SharedPreferences.getInstance();
    final now = DateTime.now();
    final session = MockExamSession(
      paperId: paper.id,
      paperTitle: '${paper.subject} ${paper.year} ${paper.variant}',
      subject: paper.subject,
      durationMinutes: paper.duration,
      maxMarks: paper.maxMarks,
      remainingSeconds: paper.duration * 60,
      currentQuestion: 1,
      flaggedQuestions: const [],
      violationCount: 0,
      startedAt: now,
      updatedAt: now,
    );
    await prefs.setString(_activeSessionKey, jsonEncode(session.toJson()));
  }

  Future<MockExamSession?> getActiveSession() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_activeSessionKey);
    if (raw == null || raw.isEmpty) return null;
    try {
      return MockExamSession.fromJson(
        Map<String, dynamic>.from(jsonDecode(raw) as Map),
      );
    } catch (e) {
      debugPrint('[MockExamService] Failed to load active session: $e');
      return null;
    }
  }

  Future<void> updateCheckpoint({
    required int currentQuestion,
    required List<int> flaggedQuestions,
    required int remainingSeconds,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final current = await getActiveSession();
    if (current == null) return;
    final updated = current.copyWith(
      currentQuestion: currentQuestion,
      flaggedQuestions: List<int>.from(flaggedQuestions)..sort(),
      remainingSeconds: remainingSeconds,
      updatedAt: DateTime.now(),
    );
    await prefs.setString(_activeSessionKey, jsonEncode(updated.toJson()));
  }

  Future<int> recordViolation(String reason) async {
    final prefs = await SharedPreferences.getInstance();
    final current = await getActiveSession();
    if (current == null) return 0;

    final updated = current.copyWith(
      violationCount: current.violationCount + 1,
      updatedAt: DateTime.now(),
    );
    await prefs.setString(_activeSessionKey, jsonEncode(updated.toJson()));

    final pendingMessage =
        'Protocol breach recorded for ${current.paperTitle}: $reason. '
        'Resume under control and finish the paper without breaking focus again.';
    await prefs.setString(_accountabilityKey, pendingMessage);
    return updated.violationCount;
  }

  Future<void> abandonSession({String reason = 'mock ended early'}) async {
    await recordViolation(reason);
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_activeSessionKey);
  }

  Future<void> completeSession({
    required int score,
    required String reflection,
    required int remainingSeconds,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final current = await getActiveSession();
    final payload = {
      'score': score,
      'reflection': reflection,
      'remainingSeconds': remainingSeconds,
      'completedAt': DateTime.now().toIso8601String(),
      'paperId': current?.paperId ?? '',
      'paperTitle': current?.paperTitle ?? '',
      'subject': current?.subject ?? '',
      'violationCount': current?.violationCount ?? 0,
    };
    await prefs.setString(_lastReflectionKey, jsonEncode(payload));
    await prefs.remove(_activeSessionKey);
    await prefs.remove(_accountabilityKey);
  }

  Future<String?> consumePendingAccountabilityMessage() async {
    final message = await getPendingAccountabilityMessage();
    if (message == null || message.isEmpty) return null;
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_accountabilityKey);
    return message;
  }

  Future<String?> getPendingAccountabilityMessage() async {
    final prefs = await SharedPreferences.getInstance();
    final message = prefs.getString(_accountabilityKey);
    return message == null || message.isEmpty ? null : message;
  }
}
