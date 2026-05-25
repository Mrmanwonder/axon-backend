import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'gemma_service.dart';

enum AxonAiTaskKind {
  multimodalGrading,
  longTermPlanning,
  activeRecall,
  motivationalChat,
  conceptRetrieval,
}

extension AxonAiTaskKindX on AxonAiTaskKind {
  String get wireName {
    switch (this) {
      case AxonAiTaskKind.multimodalGrading:
        return 'multimodal_grading';
      case AxonAiTaskKind.longTermPlanning:
        return 'long_term_planning';
      case AxonAiTaskKind.activeRecall:
        return 'active_recall';
      case AxonAiTaskKind.motivationalChat:
        return 'motivational_chat';
      case AxonAiTaskKind.conceptRetrieval:
        return 'concept_retrieval';
    }
  }
}

class AiContextPacket {
  final int version;
  final DateTime createdAt;
  final String sourceBrain;
  final String taskKind;
  final String studentPrompt;
  final String responseSummary;
  final String subject;
  final String chapter;
  final String objectiveId;
  final List<String> tags;
  final Map<String, dynamic> metadata;

  const AiContextPacket({
    required this.version,
    required this.createdAt,
    required this.sourceBrain,
    required this.taskKind,
    required this.studentPrompt,
    required this.responseSummary,
    this.subject = '',
    this.chapter = '',
    this.objectiveId = '',
    this.tags = const [],
    this.metadata = const {},
  });

  factory AiContextPacket.fromJson(Map<String, dynamic> json) {
    return AiContextPacket(
      version: (json['version'] as num?)?.toInt() ?? 1,
      createdAt:
          DateTime.tryParse('${json['created_at'] ?? ''}') ?? DateTime.now(),
      sourceBrain: (json['source_brain'] ?? '').toString(),
      taskKind: (json['task_kind'] ?? '').toString(),
      studentPrompt: (json['student_prompt'] ?? '').toString(),
      responseSummary: (json['response_summary'] ?? '').toString(),
      subject: (json['subject'] ?? '').toString(),
      chapter: (json['chapter'] ?? '').toString(),
      objectiveId: (json['objective_id'] ?? '').toString(),
      tags:
          (json['tags'] as List? ?? const []).map((e) => e.toString()).toList(),
      metadata: Map<String, dynamic>.from(json['metadata'] as Map? ?? const {}),
    );
  }

  Map<String, dynamic> toJson() => {
        'version': version,
        'created_at': createdAt.toIso8601String(),
        'source_brain': sourceBrain,
        'task_kind': taskKind,
        'student_prompt': studentPrompt,
        'response_summary': responseSummary,
        'subject': subject,
        'chapter': chapter,
        'objective_id': objectiveId,
        'tags': tags,
        'metadata': metadata,
      };

  String toCompactJson() => jsonEncode(toJson());
}

class OfflineAiStatus {
  final bool ready;
  final String model;
  final String runtime;
  final String reason;

  const OfflineAiStatus({
    required this.ready,
    required this.model,
    required this.runtime,
    required this.reason,
  });

  factory OfflineAiStatus.fromMap(Map<Object?, Object?> map) {
    return OfflineAiStatus(
      ready: map['ready'] == true,
      model: (map['model'] ?? '').toString(),
      runtime: (map['runtime'] ?? '').toString(),
      reason: (map['reason'] ?? '').toString(),
    );
  }
}

class OfflineAiService {
  OfflineAiService._();

  static final OfflineAiService instance = OfflineAiService._();

  bool _isAvailable = false;
  bool get isAvailable => _isAvailable;

  static const MethodChannel _channel =
      MethodChannel('com.axon.app/offline_ai');

  Future<void> initialize() async {
    try {
      final result = await getStatus();
      _isAvailable = result.ready;
    } catch (e) {
      _isAvailable = false;
    }
  }

  Future<OfflineAiStatus> getStatus() async {
    // Try native channel first
    try {
      final dynamic raw = await _channel.invokeMethod('getOfflineModelStatus');
      if (raw is Map) {
        return OfflineAiStatus.fromMap(Map<Object?, Object?>.from(raw));
      }
    } catch (_) {}

    // Check if Gemma service is ready
    if (GemmaService.instance.isReady) {
      return const OfflineAiStatus(
        ready: true,
        model: 'gemma-2b-studyplan-q4',
        runtime: 'llamadart',
        reason: 'Gemma model loaded and ready for offline inference',
      );
    }

    // Try to initialize Gemma - it might work!
    try {
      final initialized = await GemmaService.instance.initialize();
      if (initialized) {
        return const OfflineAiStatus(
          ready: true,
          model: 'gemma-2b-studyplan-q4',
          runtime: 'llamadart',
          reason: 'Gemma loaded successfully',
        );
      }
    } catch (_) {}

    _isAvailable = false;
    return const OfflineAiStatus(
      ready: false,
      model: 'gemma-2b-studyplan-q4',
      runtime: 'fallback',
      reason: 'Model not loaded - using fallback responses',
    );
  }

  Future<String> runTextTask({
    required AxonAiTaskKind taskKind,
    required String prompt,
    AiContextPacket? contextPacket,
  }) async {
    // Try local Gemma first
    if (!GemmaService.instance.isReady) {
      try {
        debugPrint('Attempting to initialize Gemma...');
        await GemmaService.instance.initialize();
      } catch (e) {
        debugPrint('Gemma init failed: $e');
      }
    }

    if (GemmaService.instance.isReady) {
      try {
        String? context;
        if (contextPacket != null) {
          context =
              'Task: ${contextPacket.taskKind}, Subject: ${contextPacket.subject}, Chapter: ${contextPacket.chapter}';
        }

        final result = await GemmaService.instance.chat(
          message: prompt,
          context: context,
          maxNewTokens: 1024,
        );

        if (result.trim().isNotEmpty && !result.contains('[OFFLINE_ERROR')) {
          return result.trim();
        }
      } catch (e) {
        debugPrint('Gemma inference failed: $e');
      }
    }

    // Fallback to rule-based responses
    return _ruleBasedResponse(taskKind, prompt);
  }

  String _ruleBasedResponse(AxonAiTaskKind taskKind, String prompt) {
    final q = prompt.toLowerCase().trim();

    switch (taskKind) {
      case AxonAiTaskKind.motivationalChat:
        if (q.contains('hello') || q.contains('hi ') || q == 'hi') {
          return 'Hello! I\'m Axon\'s offline assistant. I can help with study questions when the AI service is unavailable.';
        }
        if (q.contains('thank')) {
          return 'You\'re welcome! Keep up the great work! 📚';
        }
        if (q.contains('motivat') ||
            q.contains('tired') ||
            q.contains('bored')) {
          return 'Every expert was once a beginner. Consistency beats intensity — show up every day, even for 15 minutes. You\'ve got this! 💪';
        }
        return 'Keep going! Every study session builds toward your goals. Small consistent efforts lead to big results. You\'ve got this!';
      case AxonAiTaskKind.activeRecall:
        if (q.contains('study') || q.contains('plan')) {
          return 'I recommend breaking your study into 45-minute focused blocks with 10-minute breaks. What subject are you working on?';
        }
        if (q.contains('math') ||
            q.contains('physics') ||
            q.contains('chemistry') ||
            q.contains('biology')) {
          return 'Great subject! Try explaining concepts out loud (Feynman technique) and practice with past papers.';
        }
        return 'Try active recall: read a concept, close your notes, and write what you remember. Then check and correct. This is the most effective study method!';
      case AxonAiTaskKind.longTermPlanning:
        if (q.contains('help') || q.contains('what can you')) {
          return 'I can help with study techniques, subject advice, and motivation. Try asking about a specific subject!';
        }
        return 'Create a study schedule based on your exam dates. Prioritize weaker subjects. Study in focused 25-min blocks with 5-min breaks. Review within 24 hours of learning.';
      case AxonAiTaskKind.conceptRetrieval:
        if (q.contains('math') ||
            q.contains('physics') ||
            q.contains('chemistry') ||
            q.contains('biology')) {
          return 'Great subject! Try explaining concepts out loud (Feynman technique) and practice with past papers.';
        }
        return 'Break concepts into smaller parts. Connect new ideas to what you already know. Use diagrams and analogies. Teach the concept to someone else or explain it out loud.';
      case AxonAiTaskKind.multimodalGrading:
        return 'I\'m in offline mode. For best feedback, please connect to the internet.';
    }
  }
}
