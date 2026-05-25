import 'package:flutter/foundation.dart';
import 'grok_service.dart';
import '../../models/models.dart';
import 'app_state.dart';

class SpacedRepetitionCard {
  final String id;
  final String subject;
  final String topic;
  final String question;
  final String answer;
  final String? hint;
  final String? explanation;

  SpacedRepetitionCard({
    required this.id,
    required this.subject,
    required this.topic,
    required this.question,
    required this.answer,
    this.hint,
    this.explanation,
  });
}

class _CachedAiResponse {
  final String response;
  final DateTime timestamp;
  _CachedAiResponse(this.response) : timestamp = DateTime.now();
}

class AxonAiService {
  AxonAiService._();
  static final AxonAiService instance = AxonAiService._();

  static const String _sourceAiGenerated = '[SOURCE:AI]';

  // LRU cache with size bounds
  static const int _maxCacheSize = 50; // Max cached responses
  final Map<String, _CachedAiResponse> _responseCache = {};
  final List<String> _cacheOrder = []; // Track insertion order for LRU

  String? _getCached(String key) {
    final cached = _responseCache[key];
    if (cached == null) return null;
    if (DateTime.now().difference(cached.timestamp).inHours >= 1) {
      _responseCache.remove(key);
      _cacheOrder.remove(key);
      return null;
    }
    // LRU: move to end
    _cacheOrder.remove(key);
    _cacheOrder.add(key);
    return cached.response;
  }

  void _setCached(String key, String response) {
    // Evict oldest if at capacity
    while (_responseCache.length >= _maxCacheSize && _cacheOrder.isNotEmpty) {
      final oldest = _cacheOrder.removeAt(0);
      _responseCache.remove(oldest);
    }
    _responseCache[key] = _CachedAiResponse(response);
    _cacheOrder.add(key);
  }

  Future<String> generateMotivation({
    required MetricsState metrics,
    required String studentName,
    String? userId,
    MotivationStyle? motivationStyle,
  }) async {
    if (userId != null) {
      final cached = _getCached('motivation_$userId');
      if (cached != null) return cached;
    }
    final prompt = '''
Student: $studentName
Performance: ${(metrics.predictedPerformance * 100).round()}%
Sleep: ${metrics.sleepHours.toStringAsFixed(1)}h
Screen: ${metrics.screenTimeHours.toStringAsFixed(1)}h
Study: ${metrics.activeStudyHours.toStringAsFixed(1)}h

Give a short motivational message tailored to these metrics.
Keep it personal, natural, and complete in 3 to 5 sentences.
''';
    try {
      final result = _sourceAiGenerated + await GrokService().chat(prompt);
      if (userId != null) _setCached('motivation_$userId', result);
      return result;
    } catch (e) {
      return 'Keep going! You can do this.';
    }
  }

  Future<String> generateSuggestions({
    required MetricsState metrics,
    required String primarySubject,
    String? userId,
  }) async {
    if (userId != null) {
      final cached = _getCached('suggestions_$userId');
      if (cached != null) return cached;
    }
    final prompt = 'Subject: ${primarySubject.isEmpty ? 'General' : primarySubject}\nPerformance: ${(metrics.predictedPerformance * 100).round()}%\nGive 3 actionable study suggestions.';
    try {
      final result = _sourceAiGenerated + await GrokService().chat(prompt);
      if (userId != null) _setCached('suggestions_$userId', result);
      return result;
    } catch (e) {
      return '1. Review past papers\n2. Practice daily\n3. Join study group';
    }
  }

  Future<String> answerQuestion({
    required String question,
    String? context,
  }) async {
    final prompt = '$question\nContext: ${context ?? ""}';
    try {
      await GrokService().initialize();
      return await GrokService().chat(prompt);
    } catch (e) {
      return 'Unable to answer at this time.';
    }
  }

  Future<String> explainAnswer({
    required String question,
    required String studentAnswer,
    required String markingScheme,
  }) async {
    final prompt = 'Question: $question\nAnswer: $studentAnswer\nMarking: $markingScheme';
    try {
      return await GrokService().chat(prompt);
    } catch (e) {
      return 'Check against marking scheme.';
    }
  }

  Future<String> generateChapterNotes({
    required String subject,
    required String chapter,
    required String bookContext,
  }) async {
    if (bookContext.isEmpty) return 'No source for $subject $chapter';
    final prompt = 'Notes for $subject $chapter: $bookContext';
    try {
      return await GrokService().chat(prompt);
    } catch (e) {
      return 'Notes unavailable.';
    }
  }

  Future<String> generateMockPaper({
    required String subject,
    required String chapter,
    required String bookContext,
  }) async {
    if (bookContext.isEmpty) return 'No source available.';
    final prompt = 'Mock questions for $subject $chapter';
    try {
      return await GrokService().chat(prompt);
    } catch (e) {
      return 'Questions unavailable.';
    }
  }

  Future<String> generateQuiz({
    required String subject,
    required String topic,
    int numQuestions = 5,
  }) async {
    final prompt = '$numQuestions MCQ for $topic in $subject';
    try {
      return await GrokService().chat(prompt);
    } catch (e) {
      return 'Quiz unavailable.';
    }
  }

  Future<List<SpacedRepetitionCard>> generateRevisionCards({
    required String subject,
    required String sourceTitle,
    required List questions,
    String markingSchemeText = '',
  }) async {
    if (questions.isEmpty) return [];
    final prompt = 'Create revision cards for $subject';
    try {
      final response = await GrokService().chat(prompt);
      return [SpacedRepetitionCard(id: sourceTitle, subject: subject, topic: 'AI Generated', question: sourceTitle, answer: response)];
    } catch (e) {
      return [];
    }
  }

  Future<void> initialize() async {
    await GrokService().initialize();
  }

  final ValueNotifier<bool> isWorking = ValueNotifier(true);
}