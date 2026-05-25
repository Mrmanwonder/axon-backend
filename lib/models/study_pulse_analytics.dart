class TopicReadiness {
  final String objectiveId;
  final double accuracy;
  final double decay;
  final double readiness;
  final String risk;

  const TopicReadiness({
    required this.objectiveId,
    required this.accuracy,
    required this.decay,
    required this.readiness,
    required this.risk,
  });

  factory TopicReadiness.fromJson(Map<String, dynamic> json) => TopicReadiness(
        objectiveId:
            (json['objective_id'] ?? json['topic_id'] ?? '').toString(),
        accuracy: (json['accuracy'] as num?)?.toDouble() ?? 0,
        decay: (json['decay'] as num?)?.toDouble() ?? 0,
        readiness: (json['readiness'] as num?)?.toDouble() ?? 0,
        risk: (json['risk'] ?? 'red').toString(),
      );
}

class AdvisorInsight {
  final String headline;
  final String insight;
  final String action;

  const AdvisorInsight({
    required this.headline,
    required this.insight,
    required this.action,
  });

  factory AdvisorInsight.fromJson(Map<String, dynamic> json) => AdvisorInsight(
        headline: (json['headline'] ?? '').toString(),
        insight: (json['insight'] ?? '').toString(),
        action: (json['action'] ?? '').toString(),
      );
}

class StudyPulseAnalytics {
  final double coverage;
  final double accuracy;
  final double averageReadiness;
  final double examRisk;
  final double readinessScore;
  final String predictedGradeBand;
  final String projectedGrade;
  final bool crisisMode;
  final List<TopicReadiness> readinessHeatmap;
  final AdvisorInsight advisor;

  const StudyPulseAnalytics({
    required this.coverage,
    required this.accuracy,
    required this.averageReadiness,
    required this.examRisk,
    required this.readinessScore,
    required this.predictedGradeBand,
    required this.projectedGrade,
    required this.crisisMode,
    required this.readinessHeatmap,
    required this.advisor,
  });

  factory StudyPulseAnalytics.fromJson(Map<String, dynamic> json) =>
      StudyPulseAnalytics(
        coverage: (json['coverage'] as num?)?.toDouble() ?? 0,
        accuracy: (json['accuracy'] as num?)?.toDouble() ?? 0,
        averageReadiness:
            (json['average_readiness'] as num?)?.toDouble() ?? 0,
        examRisk: (json['exam_risk'] as num?)?.toDouble() ?? 0,
        readinessScore: (json['readiness_score'] as num?)?.toDouble() ?? 0,
        predictedGradeBand: (json['predicted_grade_band'] ?? '').toString(),
        projectedGrade:
            (json['projected_grade'] ?? json['predicted_grade_band'] ?? '')
                .toString(),
        crisisMode: json['crisis_mode'] == true,
        readinessHeatmap: (json['readiness_heatmap'] as List? ?? const [])
            .whereType<Map>()
            .map((item) => TopicReadiness.fromJson(Map<String, dynamic>.from(item)))
            .toList(),
        advisor: AdvisorInsight.fromJson(
          Map<String, dynamic>.from(json['advisor'] as Map? ?? const {}),
        ),
      );
}
