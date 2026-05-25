class CommandWordDrillCard {
  final String commandWord;
  final String prompt;
  final String depthExpectation;

  const CommandWordDrillCard({
    required this.commandWord,
    required this.prompt,
    required this.depthExpectation,
  });

  factory CommandWordDrillCard.fromJson(Map<String, dynamic> json) {
    return CommandWordDrillCard(
      commandWord: (json['command_word'] ?? '').toString(),
      prompt: (json['prompt'] ?? '').toString(),
      depthExpectation: (json['depth_expectation'] ?? '').toString(),
    );
  }

  Map<String, dynamic> toJson() => {
        'command_word': commandWord,
        'prompt': prompt,
        'depth_expectation': depthExpectation,
      };
}

class CommandWordDrillBundle {
  final String objectiveId;
  final String board;
  final String subject;
  final String paper;
  final String topic;
  final String subTopic;
  final String title;
  final String description;
  final List<CommandWordDrillCard> cards;

  const CommandWordDrillBundle({
    required this.objectiveId,
    required this.board,
    required this.subject,
    required this.paper,
    required this.topic,
    required this.subTopic,
    required this.title,
    required this.description,
    required this.cards,
  });

  factory CommandWordDrillBundle.fromJson(Map<String, dynamic> json) {
    return CommandWordDrillBundle(
      objectiveId: (json['objective_id'] ?? '').toString(),
      board: (json['board'] ?? '').toString(),
      subject: (json['subject'] ?? '').toString(),
      paper: (json['paper'] ?? '').toString(),
      topic: (json['topic'] ?? '').toString(),
      subTopic: (json['sub_topic'] ?? '').toString(),
      title: (json['title'] ?? '').toString(),
      description: (json['description'] ?? '').toString(),
      cards: (json['cards'] as List? ?? const [])
          .whereType<Map>()
          .map((item) => CommandWordDrillCard.fromJson(Map<String, dynamic>.from(item)))
          .toList(),
    );
  }
}

class CommandWordDrillEvaluationItem {
  final String commandWord;
  final double awardedScore;
  final double maxScore;
  final bool depthSatisfied;
  final String missingDepth;
  final String feedback;

  const CommandWordDrillEvaluationItem({
    required this.commandWord,
    required this.awardedScore,
    required this.maxScore,
    required this.depthSatisfied,
    required this.missingDepth,
    required this.feedback,
  });

  factory CommandWordDrillEvaluationItem.fromJson(Map<String, dynamic> json) {
    return CommandWordDrillEvaluationItem(
      commandWord: (json['command_word'] ?? '').toString(),
      awardedScore: (json['awarded_score'] as num?)?.toDouble() ?? 0,
      maxScore: (json['max_score'] as num?)?.toDouble() ?? 0,
      depthSatisfied: json['depth_satisfied'] == true,
      missingDepth: (json['missing_depth'] ?? '').toString(),
      feedback: (json['feedback'] ?? '').toString(),
    );
  }
}

class CommandWordDrillEvaluation {
  final double overallScore;
  final String feedback;
  final List<CommandWordDrillEvaluationItem> items;

  const CommandWordDrillEvaluation({
    required this.overallScore,
    required this.feedback,
    required this.items,
  });

  factory CommandWordDrillEvaluation.fromJson(Map<String, dynamic> json) {
    return CommandWordDrillEvaluation(
      overallScore: (json['overall_score'] as num?)?.toDouble() ?? 0,
      feedback: (json['feedback'] ?? '').toString(),
      items: (json['items'] as List? ?? const [])
          .whereType<Map>()
          .map((item) => CommandWordDrillEvaluationItem.fromJson(Map<String, dynamic>.from(item)))
          .toList(),
    );
  }
}
