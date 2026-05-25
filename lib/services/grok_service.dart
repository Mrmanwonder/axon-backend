// lib/services/grok_service.dart
// ─────────────────────────────────────────────────────────────────
// Grok AI Service - Primary AI for Axon
// Uses OpenRouter (Gemini Flash) as primary for speed
// Falls back to Grok, Deepseek, Vercel
// Supports streaming for instant responses
// ─────────────────────────────────────────────────────────────────

import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'secure_credentials_service.dart';

enum GrokStatus { initializing, ready, error, exhausted }

enum ApiProvider { openrouter, grok, deepseek, vercel }

class GrokService {
  static final GrokService _instance = GrokService._internal();
  factory GrokService() => _instance;
  GrokService._internal();

  GrokStatus _status = GrokStatus.initializing;
  String? _lastError;
  String? _grokApiKey;
  String? _vercelApiKey;
  String? _deepseekApiKey;
  String? _openrouterApiKey;
  List<Map<String, String>> _conversationHistory = [];
  final List<Map<String, String>> _contextDocuments = [];
  static const int _maxHistoryLength = 20;
  static const int _maxContextChars = 50000;
  static const Duration _timeout = Duration(seconds: 30);
  static const Duration _streamTimeout = Duration(seconds: 25);

  bool get isReady => _status == GrokStatus.ready;
  bool get isExhausted => _status == GrokStatus.exhausted;
  String? get lastError => _lastError;
  List<Map<String, String>> get contextDocuments => _contextDocuments;

  Future<bool> initialize() async {
    try {
      _status = GrokStatus.initializing;

      final credentials = SecureCredentialsService();
      await credentials.initialize();

      final allCreds = await credentials.getAllCredentials();

      _grokApiKey = allCreds.effectiveGrokKey;
      _vercelApiKey = allCreds.effectiveVercelKey;
      _deepseekApiKey = allCreds.effectiveDeepseekKey;
      _openrouterApiKey = allCreds.effectiveOpenrouterKey;

      if (_grokApiKey == null &&
          _vercelApiKey == null &&
          _deepseekApiKey == null &&
          _openrouterApiKey == null) {
        _status = GrokStatus.error;
        _lastError = 'No API key configured. Add keys in Settings > API Keys';
        debugPrint('GrokService: $lastError');
        return false;
      }

      _status = GrokStatus.ready;
      debugPrint('GrokService: initialized successfully');
      return true;
    } catch (e) {
      _status = GrokStatus.error;
      _lastError = e.toString();
      debugPrint('GrokService init error: $e');
      return false;
    }
  }

  List<ApiProvider> get _availableProviders {
    final providers = <ApiProvider>[];
    // Fastest first: OpenRouter (Gemini Flash) -> Grok -> DeepSeek -> Vercel
    if (_openrouterApiKey != null && _openrouterApiKey!.isNotEmpty) {
      providers.add(ApiProvider.openrouter);
    }
    if (_grokApiKey != null && _grokApiKey!.isNotEmpty) {
      providers.add(ApiProvider.grok);
    }
    if (_deepseekApiKey != null && _deepseekApiKey!.isNotEmpty) {
      providers.add(ApiProvider.deepseek);
    }
    if (_vercelApiKey != null && _vercelApiKey!.isNotEmpty) {
      providers.add(ApiProvider.vercel);
    }
    return providers;
  }

  /// Stream response for instant display — uses OpenRouter with SSE
  Stream<String> chatStream(
    String message, {
    String? systemPrompt,
    List<Map<String, String>>? context,
    int? maxTokens,
    double? temperature,
  }) async* {
    if (_status == GrokStatus.initializing) {
      await initialize();
    }

    if (_status == GrokStatus.error && _availableProviders.isEmpty) {
      yield '[ERROR: Service not initialized - $_lastError]';
      return;
    }

    // Let pending frame render before heavy work
    await Future.delayed(Duration.zero);

    // Build messages
    final msgs =
        _buildMessages(message, systemPrompt: systemPrompt, context: context);

    // Try streaming providers in order
    final providers = _availableProviders;

    for (final provider in providers) {
      if (provider == ApiProvider.openrouter) {
        final stream = _streamOpenrouter(msgs,
            maxTokens: maxTokens ?? 512, temperature: temperature ?? 0.45);
        StringBuffer? full;
        await for (final chunk in stream) {
          if (chunk.isEmpty) continue;
          full ??= StringBuffer();
          full.write(chunk);
          yield chunk;
        }
        // If we got content, save to history and return
        if (full != null && full.toString().trim().isNotEmpty) {
          return;
        }
        // Otherwise fall through to next provider
        continue;
      }
      // Non-streaming fallback for other providers
      try {
        final response = await _callProviderApi(provider, msgs,
            maxTokens: maxTokens ?? 512, temperature: temperature ?? 0.45);
        if (response != null &&
            !response.contains('error') &&
            response.isNotEmpty) {
          final cleaned = _stripThinkTags(response);
          _conversationHistory.add({'role': 'user', 'content': message});
          _conversationHistory.add({'role': 'assistant', 'content': cleaned});
          yield cleaned;
          return;
        }
      } catch (_) {}
    }

    yield '[ERROR: All APIs unavailable]';
  }

  /// Build the messages list for API calls
  List<Map<String, String>> _buildMessages(
    String message, {
    String? systemPrompt,
    List<Map<String, String>>? context,
  }) {
    final messages = <Map<String, String>>[];

    messages.add({
      'role': 'system',
      'content': systemPrompt ?? _getDefaultSystemPrompt(),
    });

    if (_contextDocuments.isNotEmpty) {
      final contextText =
          _contextDocuments.map((d) => d['content'] ?? '').join('\n\n---\n\n');
      messages.add({
        'role': 'system',
        'content': 'Context from documents:\n$contextText',
      });
    }

    if (context != null && context.isNotEmpty) {
      final seen = <String>{};
      for (final item in context.take(_maxHistoryLength)) {
        final role = item['role'] == 'assistant' ? 'assistant' : 'user';
        final content = (item['content'] ?? '').trim();
        if (content.isEmpty || seen.contains(content)) continue;
        seen.add(content);
        messages.add({'role': role, 'content': content});
      }
    }

    messages.addAll(_conversationHistory);

    if (_conversationHistory.length >= _maxHistoryLength) {
      _conversationHistory = _conversationHistory.sublist(
        _conversationHistory.length - _maxHistoryLength,
      );
    }

    messages.add({'role': 'user', 'content': message});
    return messages;
  }

  /// Streaming via OpenRouter SSE for instant responses
  Stream<String> _streamOpenrouter(
    List<Map<String, String>> messages, {
    int maxTokens = 512,
    double temperature = 0.45,
  }) async* {
    if (_openrouterApiKey == null || _openrouterApiKey!.isEmpty) {
      yield '[ERROR: OpenRouter key not configured]';
      return;
    }

    final modelsToTry = [
      'owl/owl-alpha',
      'google/gemma-4-26b-it',
      'meta-llama/llama-3.3-70b-instruct:free',
      'google/gemma-4-31b-it:free',
      'meta-llama/llama-3.2-3b-instruct:free',
      'liquid/lfm-2.5-1.2b-instruct:free',
    ];

    for (final model in modelsToTry) {
      final client = http.Client();
      try {
        final uri = Uri.parse('https://openrouter.ai/api/v1/chat/completions');
        final body = {
          'model': model,
          'messages': messages,
          'max_tokens': maxTokens,
          'temperature': temperature,
          'stream': true,
        };

        final request = http.Request('POST', uri)
          ..headers['Content-Type'] = 'application/json'
          ..headers['Authorization'] = 'Bearer $_openrouterApiKey'
          ..body = jsonEncode(body);

        final streamedResponse =
            await client.send(request).timeout(_streamTimeout);

        if (streamedResponse.statusCode == 200) {
          final fullResponse = StringBuffer();
          await for (final chunk
              in streamedResponse.stream.transform(utf8.decoder)) {
            final lines = chunk.split('\n');
            for (final line in lines) {
              if (!line.startsWith('data: ')) continue;
              final data = line.substring(6).trim();
              if (data == '[DONE]') break;
              try {
                final json = jsonDecode(data);
                final delta =
                    json['choices']?[0]?['delta']?['content'] as String?;
                if (delta != null && delta.isNotEmpty) {
                  fullResponse.write(delta);
                  yield delta;
                }
              } catch (_) {}
            }
          }

          final cleaned = _stripThinkTags(fullResponse.toString()).trim();
          if (cleaned.isNotEmpty) {
            _conversationHistory.add(
                {'role': 'user', 'content': messages.last['content'] ?? ''});
            _conversationHistory.add({'role': 'assistant', 'content': cleaned});
          }
          client.close();
          return; // Success, exit function
        } else {
          debugPrint(
              'GrokService: OpenRouter stream model $model returned ${streamedResponse.statusCode}');
          client.close();
        }
      } catch (e) {
        debugPrint('GrokService: OpenRouter model $model failed: $e');
        client.close();
      }
    }

    yield '[ERROR: All OpenRouter models failed to respond]';
  }

  /// Non-streaming chat (backward compatible)
  Future<String> chat(
    String message, {
    String? systemPrompt,
    List<Map<String, String>>? context,
    int? maxTokens,
    double? temperature,
  }) async {
    if (_status == GrokStatus.initializing) {
      await initialize();
    }

    if (_status == GrokStatus.error &&
        _openrouterApiKey == null &&
        _grokApiKey == null &&
        _deepseekApiKey == null &&
        _vercelApiKey == null) {
      return '[ERROR: Service not initialized - $_lastError]';
    }

    final messages =
        _buildMessages(message, systemPrompt: systemPrompt, context: context);

    String? lastError;
    final providers = _availableProviders;

    for (final provider in providers) {
      String? response;
      try {
        response = await _callProviderApi(provider, messages,
            maxTokens: maxTokens, temperature: temperature);

        if (response != null &&
            !response.contains('error') &&
            response.isNotEmpty) {
          final cleaned = _stripThinkTags(response);
          _conversationHistory.add({
            'role': 'user',
            'content': message,
          });
          _conversationHistory.add({
            'role': 'assistant',
            'content': cleaned,
          });
          return cleaned;
        }

        if (response != null) {
          lastError = response;
          if (response.contains('insufficient') ||
              response.contains('exhausted') ||
              response.contains('rate_limit') ||
              response.contains('429') ||
              response.contains('quota')) {
            debugPrint(
                'GrokService: $provider exhausted, trying next provider...');
          }
        }
      } catch (e) {
        debugPrint('GrokService: $provider failed: $e');
        lastError = e.toString();
      }
    }

    debugPrint('GrokService: All API providers failed');
    if (lastError != null) {
      return '[ERROR: API call failed - $lastError]. Check your API keys in .env file.';
    }
    return '[ERROR: API call failed - No API available]. Check your API keys in .env file.';
  }

  Future<String?> _callProviderApi(
    ApiProvider provider,
    List<Map<String, String>> messages, {
    int? maxTokens,
    double? temperature,
  }) async {
    switch (provider) {
      case ApiProvider.grok:
        return await _callGrokApi(messages,
            maxTokens: maxTokens, temperature: temperature);
      case ApiProvider.vercel:
        return await _callVercelApi(messages,
            maxTokens: maxTokens, temperature: temperature);
      case ApiProvider.deepseek:
        return await _callDeepseekApi(messages,
            maxTokens: maxTokens, temperature: temperature);
      case ApiProvider.openrouter:
        return await _callOpenrouterApi(messages,
            maxTokens: maxTokens, temperature: temperature);
    }
  }

  Future<String?> _callGrokApi(
    List<Map<String, String>> messages, {
    int? maxTokens,
    double? temperature,
  }) async {
    try {
      final uri = Uri.parse('https://api.x.ai/v1/chat/completions');
      final body = {
        'model': 'grok-3-beta',
        'messages': messages,
        'max_tokens': maxTokens ?? 4096,
        'temperature': temperature ?? 0.7,
      };

      final response = await http
          .post(
            uri,
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $_grokApiKey',
            },
            body: jsonEncode(body),
          )
          .timeout(_timeout);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['choices']?[0]?['message']?['content'] ??
            '[EMPTY_RESPONSE]';
      } else if (response.statusCode == 429 ||
          response.statusCode == 503 ||
          response.body.contains('insufficient')) {
        return '${response.statusCode}: exhausted';
      } else {
        debugPrint(
            'GrokService: Grok API error ${response.statusCode}: ${response.body}');
        return 'error: ${response.statusCode}';
      }
    } catch (e) {
      debugPrint('GrokService: Grok API exception: $e');
      return null;
    }
  }

  Future<String?> _callVercelApi(
    List<Map<String, String>> messages, {
    int? maxTokens,
    double? temperature,
  }) async {
    try {
      final uri = Uri.parse('https://api.vercel.ai/v1/chat/completions');
      final body = {
        'model': 'gpt-4',
        'messages': messages,
        'max_tokens': maxTokens ?? 4096,
        'temperature': temperature ?? 0.7,
      };

      final response = await http
          .post(
            uri,
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $_vercelApiKey',
            },
            body: jsonEncode(body),
          )
          .timeout(_timeout);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['choices']?[0]?['message']?['content'] ??
            '[EMPTY_RESPONSE]';
      } else {
        debugPrint(
            'GrokService: Vercel API error ${response.statusCode}: ${response.body}');
        return 'error: ${response.statusCode}';
      }
    } catch (e) {
      debugPrint('GrokService: Vercel API exception: $e');
      return null;
    }
  }

  Future<String?> _callDeepseekApi(
    List<Map<String, String>> messages, {
    int? maxTokens,
    double? temperature,
  }) async {
    try {
      final uri = Uri.parse('https://api.deepseek.com/v1/chat/completions');
      final body = {
        'model': 'deepseek-chat',
        'messages': messages,
        'max_tokens': maxTokens ?? 4096,
        'temperature': temperature ?? 0.7,
      };

      final response = await http
          .post(
            uri,
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $_deepseekApiKey',
            },
            body: jsonEncode(body),
          )
          .timeout(_timeout);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['choices']?[0]?['message']?['content'] ??
            '[EMPTY_RESPONSE]';
      } else if (response.statusCode == 429 || response.statusCode == 503) {
        return '${response.statusCode}: exhausted';
      } else {
        debugPrint(
            'GrokService: Deepseek API error ${response.statusCode}: ${response.body}');
        return 'error: ${response.statusCode}';
      }
    } catch (e) {
      debugPrint('GrokService: Deepseek API exception: $e');
      return null;
    }
  }

  Future<String?> _callOpenrouterApi(
    List<Map<String, String>> messages, {
    int? maxTokens,
    double? temperature,
  }) async {
    if (_openrouterApiKey == null || _openrouterApiKey!.isEmpty) {
      return 'error: OpenRouter key not configured';
    }

    final modelsToTry = [
      'owl/owl-alpha',
      'google/gemma-4-26b-it',
      'meta-llama/llama-3.3-70b-instruct:free',
      'google/gemma-4-31b-it:free',
      'meta-llama/llama-3.2-3b-instruct:free',
      'liquid/lfm-2.5-1.2b-instruct:free',
    ];

    for (final model in modelsToTry) {
      try {
        final uri = Uri.parse('https://openrouter.ai/api/v1/chat/completions');
        final body = {
          'model': model,
          'messages': messages,
          'max_tokens': maxTokens ?? 4096,
          'temperature': temperature ?? 0.7,
        };

        final response = await http
            .post(
              uri,
              headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer $_openrouterApiKey',
              },
              body: jsonEncode(body),
            )
            .timeout(_timeout);

        if (response.statusCode == 200) {
          final data = jsonDecode(response.body);
          return data['choices']?[0]?['message']?['content'] ??
              '[EMPTY_RESPONSE]';
        } else {
          debugPrint(
              'GrokService: Openrouter API model $model failed with status ${response.statusCode}');
        }
      } catch (e) {
        debugPrint('GrokService: Openrouter API model $model failed: $e');
      }
    }

    return 'error: All OpenRouter models failed';
  }

  Future<String> multiTurnChat({
    required List<String> userMessages,
    List<String>? assistantMessages,
    String? systemHint,
    int? maxTokens,
    double? temperature,
  }) async {
    final messages = <Map<String, String>>[];

    messages.add({
      'role': 'system',
      'content': systemHint ?? _getDefaultSystemPrompt(),
    });

    for (var i = 0; i < userMessages.length; i++) {
      messages.add({
        'role': 'user',
        'content': userMessages[i],
      });
      if (assistantMessages != null && i < assistantMessages.length) {
        messages.add({
          'role': 'assistant',
          'content': assistantMessages[i],
        });
      }
    }

    return chat(
      userMessages.isNotEmpty ? userMessages.last : '',
      systemPrompt: systemHint ?? _getDefaultSystemPrompt(),
      maxTokens: maxTokens,
      temperature: temperature,
    );
  }

  void ingestContext(String text) {
    if (_contextDocuments.length >= 10) {
      _contextDocuments.removeAt(0);
    }

    _contextDocuments.add({
      'source': 'user_context',
      'content': text.length > _maxContextChars
          ? text.substring(0, _maxContextChars)
          : text,
    });

    debugPrint(
        'GrokService: ingested context, total: ${_contextDocuments.length}');
  }

  void ingestDocument(String source, String content) {
    if (_contextDocuments.length >= 10) {
      _contextDocuments.removeAt(0);
    }

    _contextDocuments.add({
      'source': source,
      'content': content.length > _maxContextChars
          ? content.substring(0, _maxContextChars)
          : content,
    });
  }

  void clearContext() {
    _contextDocuments.clear();
    debugPrint('GrokService: cleared context');
  }

  void clearHistory() {
    _conversationHistory.clear();
    debugPrint('GrokService: cleared history');
  }

  Future<String> generateStudyPlan({
    required String subject,
    required List<String> topics,
    int daysRemaining = 30,
    String? weakTopics,
  }) async {
    final prompt =
        '''You are Axon, an expert study planning AI. Create a focused study plan for $subject.

Days remaining until exam: $daysRemaining
Topics to cover: ${topics.join(', ')}

${weakTopics != null ? 'Areas needing extra practice: $weakTopics' : ''}

Create a day-by-day plan with:
1. Each day's topic focus
2. 2-3 subtopics per day
3. Practice recommendations
4. Time allocation per topic

Format as a structured plan.''';

    return chat(prompt, systemPrompt: _getStudyPlanPrompt());
  }

  Future<String> explainConcept({
    required String topic,
    String? level,
  }) async {
    final prompt =
        '''Explain the concept of "$topic" for a ${level ?? 'high school'} student.

Include:
1. Simple definition
2. Key formulas (if any)
3. Real-world examples
4. Common mistakes to avoid
5. Practice problem suggestion''';

    return chat(prompt, systemPrompt: _getConceptExplanationPrompt());
  }

  Future<String> generateQuiz({
    required String subject,
    required String topic,
    int numQuestions = 5,
  }) async {
    final prompt =
        '''Generate $numQuestions multiple choice questions for "$topic" in $subject.

Format as:
1. Question text
A) option a
B) option b
C) option c
D) option d
CORRECT: [letter]
EXPLANATION: [brief explanation]''';

    return chat(prompt, systemPrompt: _getQuizPrompt());
  }

  Future<String> evaluateAnswer({
    required String question,
    required String userAnswer,
    required String correctAnswer,
  }) async {
    final prompt = '''Question: $question
User's answer: $userAnswer
Correct answer: $correctAnswer

Evaluate and provide:
1. Correct/Incorrect
2. Brief explanation
3. Key points to remember''';

    return chat(prompt, systemPrompt: _getAnswerEvaluationPrompt());
  }

  String _getDefaultSystemPrompt() =>
      '''You are Axon, an expert AI study assistant for Cambridge CAIE/IGCSE students.

Tone & Style:
- Be direct: Start with the answer or a high-level summary. Never repeat the user's query back.
- Never use conversational filler like "It looks like you're asking about..." or "To help you effectively..."
- Prioritize clarity and conciseness — if a concept can be explained in two sentences instead of five, do it.
- Use \$\$...\$\$ for display LaTeX and \$...\$ for inline. All math must use LaTeX.
- Keep line breaks generous for readability.

Always be clear, concise, and encouraging. Use examples where helpful.''';

  String _getStudyPlanPrompt() =>
      '''You are an expert Cambridge CAIE study planner.

Create realistic, achievable daily plans. Consider:
- Student has ${'{daysRemaining}'} days
- 2-3 hours per day available
- Mix of new content and practice
- Regular review of weak areas

Be specific with topics and time allocation.''';

  String _getConceptExplanationPrompt() => '''You are a patient, expert tutor.

Explain concepts clearly at the appropriate level. Use:
- Simple language
- Visual descriptions (text-based)
- Memory aids
- Common pitfalls

Encourage understanding over memorization.''';

  String _getQuizPrompt() =>
      '''Generate clear, focused multiple choice questions.

Make sure:
- Only one correct answer
- Plausible distractors
- Questions test understanding
- Explanations are brief but clear''';

  String _getAnswerEvaluationPrompt() => '''You are a helpful feedback provider.

Be constructive:
- Start with positive feedback
- Explain mistakes clearly
- Provide the correct approach
- Suggest improvement areas''';

  String _stripThinkTags(String text) {
    return text
        .replaceAll(RegExp(r'<think>[\s\S]*?</think>', multiLine: true), '')
        .trim();
  }

  void reset() {
    clearContext();
    clearHistory();
    _status = GrokStatus.ready;
    debugPrint('GrokService: reset');
  }

  bool get hasMultipleProviders => _availableProviders.length > 1;
  int get providerCount => _availableProviders.length;
}
