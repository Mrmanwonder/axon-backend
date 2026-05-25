import 'dart:async';
import 'api_client.dart';

enum AiProvider { openrouter, grok, deepseek, vercel }

enum AiServiceStatus { initializing, ready, error }

class AiProxyService {
  final ApiClient _apiClient;

  AiServiceStatus _status = AiServiceStatus.initializing;
  String? _lastError;
  final List<Map<String, String>> _conversationHistory = [];
  final List<Map<String, String>> _contextDocuments = [];

  static const int _maxHistoryLength = 20;
  static const int _maxContextChars = 50000;

  AiProxyService({required ApiClient apiClient}) : _apiClient = apiClient;

  AiServiceStatus get status => _status;
  String? get lastError => _lastError;
  bool get isReady => _status == AiServiceStatus.ready;

  /// Streaming chat — calls backend proxy at /api/ai/chat
  Stream<String> chatStream(
    String message, {
    String? systemPrompt,
    List<Map<String, String>>? context,
    int? maxTokens,
    double? temperature,
  }) async* {
    _status = AiServiceStatus.ready;

    final body = _buildRequestBody(
      message,
      systemPrompt: systemPrompt,
      context: context,
      maxTokens: maxTokens,
      temperature: temperature,
      stream: true,
    );

    Stream<String> proxyStream;
    try {
      proxyStream = await _apiClient.postStream(
        '/api/ai/chat',
        body: body,
        timeout: const Duration(seconds: 60),
      );
    } catch (e) {
      yield '[ERROR: Unable to connect to AI service]';
      return;
    }

    final fullResponse = StringBuffer();
    await for (final chunk in proxyStream) {
      if (chunk.isEmpty) continue;
      fullResponse.write(chunk);
      yield chunk;
    }

    final response = fullResponse.toString().trim();
    if (response.isNotEmpty) {
      _conversationHistory.add({'role': 'user', 'content': message});
      _conversationHistory.add({'role': 'assistant', 'content': response});
      _trimHistory();
    }
  }

  /// Non-streaming chat
  Future<String> chat(
    String message, {
    String? systemPrompt,
    List<Map<String, String>>? context,
    int? maxTokens,
    double? temperature,
  }) async {
    _status = AiServiceStatus.ready;

    final body = _buildRequestBody(
      message,
      systemPrompt: systemPrompt,
      context: context,
      maxTokens: maxTokens,
      temperature: temperature,
      stream: false,
    );

    final result = await _apiClient.post(
      '/api/ai/chat',
      body: body,
      timeout: const Duration(seconds: 45),
    );

    if (result.isError) {
      _lastError = result.message;
      _status = AiServiceStatus.error;
      switch (result.errorType) {
        case ApiErrorType.rateLimited:
          return '[SYSTEM: Rate limited — please wait before sending another request]';
        case ApiErrorType.network:
          return '[SYSTEM: No internet connection]';
        case ApiErrorType.auth:
          return '[SYSTEM: Authentication required — please sign in]';
        default:
          return '[SYSTEM: Unable to process request — ${result.message}]';
      }
    }

    final data = result.data;
    if (data == null) {
      return '[SYSTEM: Empty response from server]';
    }

    final response = (data['response'] ?? data['choices']?[0]?['message']?['content'] ?? '').toString().trim();
    if (response.isEmpty) {
      return '[SYSTEM: Empty response from AI]';
    }

    _conversationHistory.add({'role': 'user', 'content': message});
    _conversationHistory.add({'role': 'assistant', 'content': response});
    _trimHistory();

    return response;
  }

  Map<String, dynamic> _buildRequestBody(
    String message, {
    String? systemPrompt,
    List<Map<String, String>>? context,
    int? maxTokens,
    double? temperature,
    bool stream = false,
  }) {
    final messages = <Map<String, String>>[];
    messages.add({
      'role': 'system',
      'content': systemPrompt ?? _defaultSystemPrompt(),
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
      for (final item in context.take(_maxHistoryLength)) {
        final role = item['role'] == 'assistant' ? 'assistant' : 'user';
        final content = (item['content'] ?? '').trim();
        if (content.isEmpty) continue;
        messages.add({'role': role, 'content': content});
      }
    }

    messages.addAll(_conversationHistory);
    messages.add({'role': 'user', 'content': message});

    return {
      'messages': messages,
      'stream': stream,
      if (maxTokens != null) 'max_tokens': maxTokens,
      if (temperature != null) 'temperature': temperature,
    };
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
  }

  void clearHistory() {
    _conversationHistory.clear();
  }

  void _trimHistory() {
    while (_conversationHistory.length > _maxHistoryLength) {
      _conversationHistory.removeAt(0);
    }
  }

  String _defaultSystemPrompt() =>
      'You are Axon, an expert AI study assistant for Cambridge CAIE/IGCSE students. '
      'Be direct, clear, and concise. Use \$\$...\$\$ for display LaTeX and \$...\$ for inline. '
      'Prioritize clarity - explain in two sentences instead of five. '
      'Always be encouraging. Use examples where helpful.';
}
