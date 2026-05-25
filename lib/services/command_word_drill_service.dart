import 'dart:convert';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;

import '../models/command_word_drill_models.dart';
import 'backend_config.dart';

class CommandWordDrillService {
  CommandWordDrillService({http.Client? client})
      : _client = client ?? http.Client();

  static const String _backendUrl = BackendConfig.baseUrl;
  final http.Client _client;

  Future<CommandWordDrillBundle> generateDrill({
    required String objectiveId,
    List<String> commandWords = const [],
  }) async {
    final token = await FirebaseAuth.instance.currentUser?.getIdToken();
    final response = await _client.post(
      Uri.parse('$_backendUrl/generate/command-word-drill'),
      headers: {
        'Content-Type': 'application/json',
        if (token != null) 'Authorization': 'Bearer $token',
      },
      body: jsonEncode({
        'objective_id': objectiveId,
        'command_words': commandWords,
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception('Failed to generate drill (${response.statusCode})');
    }

    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    return CommandWordDrillBundle.fromJson(
      Map<String, dynamic>.from(payload['drill'] as Map),
    );
  }

  Future<CommandWordDrillEvaluation> evaluateDrill({
    required String objectiveId,
    required List<CommandWordDrillCard> cards,
    required Map<String, String> responses,
  }) async {
    final token = await FirebaseAuth.instance.currentUser?.getIdToken();
    final response = await _client.post(
      Uri.parse('$_backendUrl/generate/evaluate-command-word-drill'),
      headers: {
        'Content-Type': 'application/json',
        if (token != null) 'Authorization': 'Bearer $token',
      },
      body: jsonEncode({
        'objective_id': objectiveId,
        'prompt_cards': cards.map((item) => item.toJson()).toList(),
        'responses': responses.entries
            .map((entry) => {
                  'command_word': entry.key,
                  'response': entry.value,
                })
            .toList(),
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception('Failed to evaluate drill (${response.statusCode})');
    }

    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    return CommandWordDrillEvaluation.fromJson(
      Map<String, dynamic>.from(payload['evaluation'] as Map),
    );
  }
}
