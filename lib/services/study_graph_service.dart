// lib/services/study_graph_service.dart
// ─────────────────────────────────────────────────────────────────
// Study Graph Service
// Builds and manages the personalized study knowledge graph.
// Tracks chapter mastery, prerequisite relationships, and gaps.
// ─────────────────────────────────────────────────────────────────

import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class StudyGraphService {
  static final StudyGraphService _instance = StudyGraphService._internal();
  factory StudyGraphService() => _instance;
  StudyGraphService._internal();

  static const String _graphKey = 'study_graph_v2';

  // ─────────────────────────────────────────────────────────────────
  // PUBLIC API
  // ─────────────────────────────────────────────────────────────────

  Future<StudyGraph> loadGraph(String board, List<String> subjects) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('${_graphKey}_${board}_${subjects.length}');

    if (raw != null && raw.isNotEmpty) {
      try {
        return StudyGraph.fromJson(jsonDecode(raw));
      } catch (_) {}
    }

    return _buildDefaultGraph(board, subjects);
  }

  Future<void> saveGraph(StudyGraph graph) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      '${_graphKey}_${graph.board}_${graph.subjects.length}',
      jsonEncode(graph.toJson()),
    );
  }

  Future<void> updateChapterMastery(
    String board,
    String subject,
    String chapter, {
    required double mastery,
    double? predictedExamScore,
    int? studyCount,
  }) async {
    final graph = await loadGraph(board, [subject]);

    final updatedNodes = graph.nodes.map((n) {
      if (n.chapter == chapter && n.subject == subject) {
        return ChapterNode(
          chapter: chapter,
          subject: subject,
          mastery: mastery,
          predictedScore: predictedExamScore ?? n.predictedScore,
          studyCount: studyCount ?? n.studyCount,
          prerequisites: n.prerequisites,
          dependents: n.dependents,
          isWeak: mastery < 0.5,
          lastStudied: DateTime.now(),
        );
      }
      return n;
    }).toList();

    final updated = StudyGraph(
      board: graph.board,
      subjects: graph.subjects,
      nodes: updatedNodes,
      edges: graph.edges,
      createdAt: graph.createdAt,
      lastUpdated: DateTime.now(),
    );

    await saveGraph(updated);
  }

  Future<List<ChapterNode>> getWeakestChapters(
    String board,
    String subject, {
    int limit = 5,
  }) async {
    final graph = await loadGraph(board, [subject]);
    return graph.nodes.where((n) => n.subject == subject).toList()
      ..sort((a, b) => a.mastery.compareTo(b.mastery))
      ..take(limit);
  }

  Future<List<ChapterNode>> getRecentChapters(
    String board,
    String subject, {
    int limit = 5,
  }) async {
    final graph = await loadGraph(board, [subject]);
    final sorted = graph.nodes
        .where((n) => n.subject == subject && n.lastStudied != null)
        .toList()
      ..sort((a, b) => b.lastStudied!.compareTo(a.lastStudied!));
    return sorted.take(limit).toList();
  }

  Future<List<String>> getRecommendedNextChapters(
    String board,
    String subject,
  ) async {
    final graph = await loadGraph(board, [subject]);
    final weakChapters = graph.nodes
        .where((n) => n.subject == subject && n.mastery < 0.6)
        .toList();

    weakChapters.sort((a, b) {
      final aPrereqsMet = _arePrerequisitesMet(a, graph.nodes);
      final bPrereqsMet = _arePrerequisitesMet(b, graph.nodes);
      if (aPrereqsMet && !bPrereqsMet) return -1;
      if (!aPrereqsMet && bPrereqsMet) return 1;
      return a.mastery.compareTo(b.mastery);
    });

    return weakChapters.map((n) => n.chapter).toList();
  }

  Future<void> buildFromAssessment(
    String board,
    List<String> subjects,
    Map<String, Map<String, double>> chapterMasteries,
  ) async {
    final nodes = <ChapterNode>[];
    final edges = <GraphEdge>[];

    for (final subject in subjects) {
      final chapters = _getChapterOrderForSubject(board, subject);
      final subjectMasteries = chapterMasteries[subject] ?? {};

      for (int i = 0; i < chapters.length; i++) {
        final chapter = chapters[i];
        final mastery = subjectMasteries[chapter] ?? 0.3;

        final prereqs = <String>[];
        if (i > 0) prereqs.add(chapters[i - 1]);

        nodes.add(ChapterNode(
          chapter: chapter,
          subject: subject,
          mastery: mastery,
          studyCount: mastery > 0.5 ? 2 : 0,
          prerequisites: prereqs,
          dependents: i < chapters.length - 1 ? [chapters[i + 1]] : [],
          isWeak: mastery < 0.5,
          lastStudied: null,
        ));

        if (i > 0) {
          edges.add(GraphEdge(
            from: chapters[i - 1],
            to: chapters[i],
            weight: 0.8,
            type: EdgeType.prerequisite,
          ));
        }
      }
    }

    final graph = StudyGraph(
      board: board,
      subjects: subjects,
      nodes: nodes,
      edges: edges,
      createdAt: DateTime.now(),
      lastUpdated: DateTime.now(),
    );

    await saveGraph(graph);
  }

  Future<Map<String, double>> getOverallMasteryBySubject(
    String board,
    String subject,
  ) async {
    final graph = await loadGraph(board, [subject]);
    final subjectNodes = graph.nodes.where((n) => n.subject == subject);
    if (subjectNodes.isEmpty) return {};

    final totalMastery = subjectNodes.fold(0.0, (sum, n) => sum + n.mastery);
    final avgMastery = totalMastery / subjectNodes.length;

    return {
      'averageMastery': avgMastery,
      'weakChapterCount': subjectNodes.where((n) => n.isWeak).length.toDouble(),
      'strongChapterCount':
          subjectNodes.where((n) => n.mastery >= 0.7).length.toDouble(),
      'totalChapters': subjectNodes.length.toDouble(),
      'estimatedExamScore': avgMastery * 100,
    };
  }

  // ─────────────────────────────────────────────────────────────────
  // GRAPH BUILDERS
  // ─────────────────────────────────────────────────────────────────

  StudyGraph _buildDefaultGraph(String board, List<String> subjects) {
    final nodes = <ChapterNode>[];
    final edges = <GraphEdge>[];

    for (final subject in subjects) {
      final chapters = _getChapterOrderForSubject(board, subject);
      for (int i = 0; i < chapters.length; i++) {
        nodes.add(ChapterNode(
          chapter: chapters[i],
          subject: subject,
          mastery: 0.3,
          studyCount: 0,
          prerequisites: i > 0 ? [chapters[i - 1]] : [],
          dependents: i < chapters.length - 1 ? [chapters[i + 1]] : [],
          isWeak: true,
          lastStudied: null,
        ));

        if (i > 0) {
          edges.add(GraphEdge(
            from: chapters[i - 1],
            to: chapters[i],
            weight: 0.8,
            type: EdgeType.prerequisite,
          ));
        }
      }
    }

    return StudyGraph(
      board: board,
      subjects: subjects,
      nodes: nodes,
      edges: edges,
      createdAt: DateTime.now(),
      lastUpdated: DateTime.now(),
    );
  }

  List<String> _getChapterOrderForSubject(String board, String subject) {
    final lowerSubject = subject.toLowerCase();

    if (lowerSubject.contains('math')) {
      return _mathChapters;
    }
    if (lowerSubject.contains('physics')) {
      return _physicsChapters;
    }
    if (lowerSubject.contains('chemistry')) {
      return _chemistryChapters;
    }
    if (lowerSubject.contains('biology')) {
      return _biologyChapters;
    }
    if (lowerSubject.contains('economics')) {
      return _economicsChapters;
    }
    if (lowerSubject.contains('computer')) {
      return _computerScienceChapters;
    }
    if (lowerSubject.contains('english')) {
      return _englishChapters;
    }

    return _genericChapters;
  }

  bool _arePrerequisitesMet(ChapterNode node, List<ChapterNode> allNodes) {
    for (final prereq in node.prerequisites) {
      final prereqNode = allNodes.firstWhere(
        (n) => n.chapter == prereq && n.subject == node.subject,
        orElse: () => ChapterNode(
          chapter: prereq,
          subject: node.subject,
          mastery: 0,
        ),
      );
      if (prereqNode.mastery < 0.4) return false;
    }
    return true;
  }

  static const List<String> _mathChapters = [
    'Number & Algebra',
    'Functions',
    'Sequences & Series',
    'Trigonometry',
    'Calculus: Differentiation',
    'Calculus: Integration',
    'Vectors',
    'Probability & Statistics',
    'Coordinate Geometry',
    'Series & Sequences',
  ];

  static const List<String> _physicsChapters = [
    'Kinematics',
    'Forces & Newton\'s Laws',
    'Work, Energy & Power',
    'Waves & Superposition',
    'Electricity',
    'Electromagnetic Effects',
    'Circular Motion',
    'Gravitational Fields',
    'Atomic Physics',
    'Practical Physics',
  ];

  static const List<String> _chemistryChapters = [
    'Atomic Structure',
    'Chemical Bonding',
    'Stoichiometry',
    'Chemical Energetics',
    'Kinetics & Equilibria',
    'Redox Reactions',
    'Organic Chemistry',
    'Analytical Chemistry',
    'Electrochemistry',
    'Transition Metals',
  ];

  static const List<String> _biologyChapters = [
    'Cell Biology',
    'Biomolecules & Enzymes',
    'Cell Transport',
    'DNA & Protein Synthesis',
    'Genetic Inheritance',
    'Evolution & Natural Selection',
    'Ecology',
    'Human Physiology',
    'Plant Biology',
    'Practical Biology',
  ];

  static const List<String> _economicsChapters = [
    'Supply & Demand',
    'Elasticity',
    'Market Structures',
    'Market Failure',
    'National Income',
    'Inflation & Unemployment',
    'Fiscal Policy',
    'Monetary Policy',
    'International Trade',
    'Development Economics',
  ];

  static const List<String> _computerScienceChapters = [
    'Data Representation',
    'Computer Architecture',
    'Networks',
    'Operating Systems',
    'Algorithm Design',
    'Programming Fundamentals',
    'Databases',
    'Software Engineering',
    'AI & Machine Learning',
    'Ethics & Legislation',
  ];

  static const List<String> _englishChapters = [
    'Reading & Comprehension',
    'Writing Skills',
    'Summary & Synthesis',
    'Language & Tone',
    'Argument & Persuasion',
    'Literary Analysis',
    'Creative Writing',
    'Grammar & Vocabulary',
    'Text Types',
    'Speaking & Listening',
  ];

  static const List<String> _genericChapters = [
    'Chapter 1: Introduction',
    'Chapter 2: Core Concepts',
    'Chapter 3: Applications',
    'Chapter 4: Analysis',
    'Chapter 5: Synthesis',
  ];
}

// ─────────────────────────────────────────────────────────────────
// DATA MODELS
// ─────────────────────────────────────────────────────────────────

class StudyGraph {
  final String board;
  final List<String> subjects;
  final List<ChapterNode> nodes;
  final List<GraphEdge> edges;
  final DateTime createdAt;
  final DateTime lastUpdated;

  StudyGraph({
    required this.board,
    required this.subjects,
    required this.nodes,
    required this.edges,
    required this.createdAt,
    required this.lastUpdated,
  });

  Map<String, dynamic> toJson() => {
        'board': board,
        'subjects': subjects,
        'nodes': nodes.map((n) => n.toJson()).toList(),
        'edges': edges.map((e) => e.toJson()).toList(),
        'createdAt': createdAt.toIso8601String(),
        'lastUpdated': lastUpdated.toIso8601String(),
      };

  factory StudyGraph.fromJson(Map<String, dynamic> json) {
    DateTime createdAt;
    DateTime lastUpdated;
    try {
      createdAt = json['createdAt'] != null
          ? DateTime.parse(json['createdAt'])
          : DateTime.now();
    } catch (_) {
      createdAt = DateTime.now();
    }
    try {
      lastUpdated = json['lastUpdated'] != null
          ? DateTime.parse(json['lastUpdated'])
          : DateTime.now();
    } catch (_) {
      lastUpdated = DateTime.now();
    }
    return StudyGraph(
      board: json['board'] ?? '',
      subjects: List<String>.from(json['subjects'] ?? []),
      nodes: (json['nodes'] as List?)
              ?.map((n) => ChapterNode.fromJson(n))
              .toList() ??
          [],
      edges: (json['edges'] as List?)
              ?.map((e) => GraphEdge.fromJson(e))
              .toList() ??
          [],
      createdAt: createdAt,
      lastUpdated: lastUpdated,
    );
  }

  int get totalChapters => nodes.length;

  double get overallMastery {
    if (nodes.isEmpty) return 0;
    return nodes.fold(0.0, (sum, n) => sum + n.mastery) / nodes.length;
  }

  int get weakChapterCount => nodes.where((n) => n.isWeak).length;

  Map<String, double> getMasteryBySubject() {
    final map = <String, List<ChapterNode>>{};
    for (final node in nodes) {
      map.putIfAbsent(node.subject, () => []).add(node);
    }
    return map.map((k, v) {
      final avg = v.fold(0.0, (sum, n) => sum + n.mastery) / v.length;
      return MapEntry(k, avg);
    });
  }
}

class ChapterNode {
  final String chapter;
  final String subject;
  final double mastery;
  final double predictedScore;
  final int studyCount;
  final List<String> prerequisites;
  final List<String> dependents;
  final bool isWeak;
  final DateTime? lastStudied;

  ChapterNode({
    required this.chapter,
    required this.subject,
    this.mastery = 0.3,
    this.predictedScore = 0,
    this.studyCount = 0,
    this.prerequisites = const [],
    this.dependents = const [],
    this.isWeak = true,
    this.lastStudied,
  });

  Map<String, dynamic> toJson() => {
        'chapter': chapter,
        'subject': subject,
        'mastery': mastery,
        'predictedScore': predictedScore,
        'studyCount': studyCount,
        'prerequisites': prerequisites,
        'dependents': dependents,
        'isWeak': isWeak,
        'lastStudied': lastStudied?.toIso8601String(),
      };

  factory ChapterNode.fromJson(Map<String, dynamic> json) {
    DateTime? lastStudied;
    if (json['lastStudied'] != null) {
      try {
        lastStudied = DateTime.parse(json['lastStudied']);
      } catch (_) {}
    }
    return ChapterNode(
      chapter: json['chapter'] ?? '',
      subject: json['subject'] ?? '',
      mastery: (json['mastery'] ?? 0.3).toDouble(),
      predictedScore: (json['predictedScore'] ?? 0).toDouble(),
      studyCount: json['studyCount'] ?? 0,
      prerequisites: List<String>.from(json['prerequisites'] ?? []),
      dependents: List<String>.from(json['dependents'] ?? []),
      isWeak: json['isWeak'] ?? true,
      lastStudied: lastStudied,
    );
  }
}

enum EdgeType { prerequisite, related, skipped }

class GraphEdge {
  final String from;
  final String to;
  final double weight;
  final EdgeType type;

  GraphEdge({
    required this.from,
    required this.to,
    this.weight = 1.0,
    this.type = EdgeType.prerequisite,
  });

  Map<String, dynamic> toJson() => {
        'from': from,
        'to': to,
        'weight': weight,
        'type': type.name,
      };

  factory GraphEdge.fromJson(Map<String, dynamic> json) {
    return GraphEdge(
      from: json['from'] ?? '',
      to: json['to'] ?? '',
      weight: (json['weight'] ?? 1.0).toDouble(),
      type: EdgeType.values.firstWhere(
        (t) => t.name == json['type'],
        orElse: () => EdgeType.prerequisite,
      ),
    );
  }
}

final studyGraphServiceProvider = StudyGraphService();
