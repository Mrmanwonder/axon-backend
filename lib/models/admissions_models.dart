class AdmissionsTarget {
  final String id;
  final String universityName;
  final String country;
  final String courseName;
  final String classification;
  final String applicationSystem;
  final String status;
  final String rationale;
  final String entryRequirements;
  final String sourceUrl;
  final DateTime? deadlineAt;
  final String fitBand;
  final double readinessScore;

  const AdmissionsTarget({
    required this.id,
    required this.universityName,
    required this.country,
    required this.courseName,
    required this.classification,
    required this.applicationSystem,
    required this.status,
    required this.rationale,
    this.entryRequirements = '',
    this.sourceUrl = '',
    this.deadlineAt,
    this.fitBand = '',
    this.readinessScore = 0,
  });

  factory AdmissionsTarget.fromJson(String id, Map<String, dynamic> json) {
    return AdmissionsTarget(
      id: id,
      universityName: (json['university_name'] ?? '').toString(),
      country: (json['country'] ?? '').toString(),
      courseName: (json['course_name'] ?? '').toString(),
      classification: (json['classification'] ?? '').toString(),
      applicationSystem: (json['application_system'] ?? '').toString(),
      status: (json['status'] ?? 'planned').toString(),
      rationale: (json['rationale'] ?? '').toString(),
      entryRequirements: (json['entry_requirements'] ?? '').toString(),
      sourceUrl: (json['source_url'] ?? '').toString(),
      deadlineAt: DateTime.tryParse((json['deadline_at'] ?? '').toString()),
      fitBand: (json['fit_band'] ?? '').toString(),
      readinessScore: (json['readiness_score'] is num)
          ? (json['readiness_score'] as num).toDouble()
          : double.tryParse('${json['readiness_score'] ?? 0}') ?? 0,
    );
  }
}

class UniversitySubjectRequirement {
  final String subject;
  final String minimumGrade;
  final bool required;

  const UniversitySubjectRequirement({
    required this.subject,
    required this.minimumGrade,
    this.required = true,
  });

  factory UniversitySubjectRequirement.fromJson(Map<String, dynamic> json) {
    return UniversitySubjectRequirement(
      subject: (json['subject'] ?? '').toString(),
      minimumGrade: (json['minimum_grade'] ?? '').toString(),
      required: json['required'] != false,
    );
  }

  Map<String, dynamic> toJson() => {
        'subject': subject,
        'minimum_grade': minimumGrade,
        'required': required,
      };
}

class UniversityProgram {
  final String id;
  final String universityName;
  final String location;
  final String degree;
  final String courseName;
  final String duration;
  final List<String> coreModules;
  final Map<String, List<UniversitySubjectRequirement>> requirementsByBoard;
  final String minimumThreshold;
  final String sourceUrl;

  const UniversityProgram({
    required this.id,
    required this.universityName,
    required this.location,
    required this.degree,
    required this.courseName,
    required this.duration,
    required this.coreModules,
    required this.requirementsByBoard,
    required this.minimumThreshold,
    this.sourceUrl = '',
  });

  List<UniversitySubjectRequirement> requirementsFor({
    required String board,
    required List<String> subjects,
  }) {
    final normalizedBoard = board.toLowerCase();
    final boardKey = requirementsByBoard.keys.firstWhere(
      (key) => normalizedBoard.contains(key.toLowerCase()),
      orElse: () => requirementsByBoard.keys.isNotEmpty
          ? requirementsByBoard.keys.first
          : '',
    );
    final requirements = requirementsByBoard[boardKey] ?? const [];
    if (subjects.isEmpty) return requirements;
    final normalizedSubjects = subjects.map((item) => item.toLowerCase()).toList();
    return requirements
        .where(
          (requirement) => normalizedSubjects.any(
            (subject) => subject.contains(requirement.subject.toLowerCase()) ||
                requirement.subject.toLowerCase().contains(subject),
          ),
        )
        .toList();
  }

  String requirementSummaryFor({
    required String board,
    required List<String> subjects,
  }) {
    final mapped = requirementsFor(board: board, subjects: subjects);
    final source = mapped.isEmpty
        ? requirementsByBoard.values.expand((items) => items).toList()
        : mapped;
    if (source.isEmpty) return minimumThreshold;
    return source
        .map((item) => '${item.subject}: ${item.minimumGrade}')
        .join(' · ');
  }
}

class AdmissionsMarkEntry {
  final String id;
  final String subject;
  final double currentMark;
  final String currentGrade;
  final String targetGrade;
  final DateTime updatedAt;

  const AdmissionsMarkEntry({
    required this.id,
    required this.subject,
    required this.currentMark,
    required this.currentGrade,
    required this.targetGrade,
    required this.updatedAt,
  });

  factory AdmissionsMarkEntry.fromJson(String id, Map<String, dynamic> json) {
    return AdmissionsMarkEntry(
      id: id,
      subject: (json['subject'] ?? '').toString(),
      currentMark: (json['current_mark'] is num)
          ? (json['current_mark'] as num).toDouble()
          : double.tryParse('${json['current_mark'] ?? 0}') ?? 0,
      currentGrade: (json['current_grade'] ?? '').toString(),
      targetGrade: (json['target_grade'] ?? '').toString(),
      updatedAt: DateTime.tryParse((json['updated_at'] ?? '').toString()) ??
          DateTime.now(),
    );
  }
}

class AdmissionsMilestone {
  final String id;
  final String targetId;
  final String title;
  final DateTime dueAt;
  final bool completed;
  final String dependencyType;
  final String phase;
  final List<String> dependsOnIds;
  final List<String> evidenceAssetIds;
  final String status;

  const AdmissionsMilestone({
    required this.id,
    required this.targetId,
    required this.title,
    required this.dueAt,
    required this.completed,
    required this.dependencyType,
    this.phase = '',
    this.dependsOnIds = const [],
    this.evidenceAssetIds = const [],
    this.status = 'pending',
  });

  bool get isBlocked => !completed && dependsOnIds.isNotEmpty;

  factory AdmissionsMilestone.fromJson(String id, Map<String, dynamic> json) {
    return AdmissionsMilestone(
      id: id,
      targetId: (json['target_id'] ?? '').toString(),
      title: (json['title'] ?? '').toString(),
      dueAt:
          DateTime.tryParse((json['due_at'] ?? '').toString()) ?? DateTime.now(),
      completed: json['completed'] == true,
      dependencyType: (json['dependency_type'] ?? '').toString(),
      phase: (json['phase'] ?? '').toString(),
      dependsOnIds: (json['depends_on_ids'] as List? ?? const [])
          .map((item) => item.toString())
          .where((item) => item.isNotEmpty)
          .toList(),
      evidenceAssetIds: (json['evidence_asset_ids'] as List? ?? const [])
          .map((item) => item.toString())
          .where((item) => item.isNotEmpty)
          .toList(),
      status: (json['status'] ?? 'pending').toString(),
    );
  }
}

class AdmissionsVaultAsset {
  final String id;
  final String title;
  final String assetType;
  final String url;
  final String provider;
  final String projectName;
  final List<String> tags;
  final List<String> linkedTargetIds;
  final String extractedText;
  final String targetDegree;
  final int relevanceRating;

  const AdmissionsVaultAsset({
    required this.id,
    required this.title,
    required this.assetType,
    required this.url,
    this.provider = '',
    this.projectName = '',
    this.tags = const [],
    this.linkedTargetIds = const [],
    this.extractedText = '',
    this.targetDegree = '',
    this.relevanceRating = 0,
  });

  factory AdmissionsVaultAsset.fromJson(String id, Map<String, dynamic> json) {
    return AdmissionsVaultAsset(
      id: id,
      title: (json['title'] ?? '').toString(),
      assetType: (json['asset_type'] ?? '').toString(),
      url: (json['url'] ?? '').toString(),
      provider: (json['provider'] ?? '').toString(),
      projectName: (json['project_name'] ?? '').toString(),
      tags: (json['tags'] as List? ?? const [])
          .map((item) => item.toString())
          .where((item) => item.isNotEmpty)
          .toList(),
      linkedTargetIds: (json['linked_target_ids'] as List? ?? const [])
          .map((item) => item.toString())
          .where((item) => item.isNotEmpty)
          .toList(),
      extractedText: (json['extracted_text'] ?? '').toString(),
      targetDegree: (json['target_degree'] ?? '').toString(),
      relevanceRating: (json['relevance_rating'] is num)
          ? (json['relevance_rating'] as num).toInt()
          : int.tryParse('${json['relevance_rating'] ?? 0}') ?? 0,
    );
  }
}

class AdmissionsMilestoneMapNode {
  final AdmissionsMilestone milestone;
  final List<String> blockerTitles;

  const AdmissionsMilestoneMapNode({
    required this.milestone,
    required this.blockerTitles,
  });

  bool get isReady =>
      milestone.completed || blockerTitles.isEmpty || !milestone.isBlocked;
}

class AdmissionsMilestoneProgress {
  final int completedCount;
  final int totalCount;
  final int blockedCount;
  final int evidenceCount;

  const AdmissionsMilestoneProgress({
    required this.completedCount,
    required this.totalCount,
    this.blockedCount = 0,
    this.evidenceCount = 0,
  });

  double get progressRatio {
    if (totalCount <= 0) return 0;
    return completedCount / totalCount;
  }

  String get progressLabel => '$completedCount / $totalCount complete';
}
