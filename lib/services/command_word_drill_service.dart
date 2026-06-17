import '../models/command_word_drill_models.dart';
import 'api_client.dart';
import 'backend_config.dart';

class CommandWordDrillService {
  CommandWordDrillService({ApiClient? apiClient})
      : _apiClient = apiClient ?? ApiClient(baseUrl: BackendConfig.baseUrl);

  final ApiClient _apiClient;

  Future<CommandWordDrillBundle> generateDrill({
    required String objectiveId,
    List<String> commandWords = const [],
  }) async {
    final result = await _apiClient.post(
      '/generate/command-word-drill',
      body: {
        'objective_id': objectiveId,
        'command_words': commandWords,
      },
    );

    if (result.isError) {
      throw Exception(
          'Failed to generate drill: ${result.message} (${result.statusCode})');
    }

    final data = result.data;
    if (data == null) {
      throw Exception('Failed to generate drill: empty response');
    }

    final drill = data['drill'] as Map<String, dynamic>?;
    if (drill == null) {
      throw Exception('Failed to generate drill: missing drill data');
    }

    return CommandWordDrillBundle.fromJson(
      Map<String, dynamic>.from(drill),
    );
  }

  Future<CommandWordDrillEvaluation> evaluateDrill({
    required String objectiveId,
    required List<CommandWordDrillCard> cards,
    required Map<String, String> responses,
  }) async {
    final result = await _apiClient.post(
      '/generate/evaluate-command-word-drill',
      body: {
        'objective_id': objectiveId,
        'prompt_cards': cards.map((item) => item.toJson()).toList(),
        'responses': responses.entries
            .map((entry) => {
                  'command_word': entry.key,
                  'response': entry.value,
                })
            .toList(),
      },
    );

    if (result.isError) {
      throw Exception(
          'Failed to evaluate drill: ${result.message} (${result.statusCode})');
    }

    final data = result.data;
    if (data == null) {
      throw Exception('Failed to evaluate drill: empty response');
    }

    final evaluation = data['evaluation'] as Map<String, dynamic>?;
    if (evaluation == null) {
      throw Exception('Failed to evaluate drill: missing evaluation data');
    }

    return CommandWordDrillEvaluation.fromJson(
      Map<String, dynamic>.from(evaluation),
    );
  }
}
