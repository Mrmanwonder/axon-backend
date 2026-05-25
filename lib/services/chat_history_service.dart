import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class ChatHistoryService {
  static final ChatHistoryService _instance = ChatHistoryService._();
  static ChatHistoryService get instance => _instance;
  ChatHistoryService._();

  static const String _chatHistoryKey = 'axon_chat_history';
  static const String _lastChatKey = 'axon_last_chat';

  List<Map<String, String>> _history = [];
  String? _lastChatId;

  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();
    final historyJson = prefs.getString(_chatHistoryKey);
    final lastChatId = prefs.getString(_lastChatKey);
    
    if (historyJson != null) {
      try {
        _history = (jsonDecode(historyJson) as List)
            .map((e) => Map<String, String>.from(e as Map))
            .toList();
      } catch (_) {
        _history = [];
      }
    }
    _lastChatId = lastChatId;
  }

  List<Map<String, String>> get history => List.unmodifiable(_history);
  String? get lastChatId => _lastChatId;

  bool get hasPreviousChat => _history.isNotEmpty;

  Future<void> addMessage(String role, String content) async {
    _history.add({role: content});
    await _save();
  }

  Future<void> clearHistory() async {
    _history.clear();
    _lastChatId = null;
    await _save();
  }

  Future<void> startNewChat() async {
    if (_history.isNotEmpty) {
      _lastChatId = DateTime.now().millisecondsSinceEpoch.toString();
    }
    _history.clear();
    await _save();
  }

  Future<void> _save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_chatHistoryKey, jsonEncode(_history));
    if (_lastChatId != null) {
      await prefs.setString(_lastChatKey, _lastChatId!);
    } else {
      await prefs.remove(_lastChatKey);
    }
  }
}