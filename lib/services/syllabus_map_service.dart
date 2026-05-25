import 'study_catalog.dart';

class SyllabusObjectiveRef {
  final String board;
  final String subject;
  final String paper;
  final String topic;
  final String objectiveId;
  final String learningObjective;

  const SyllabusObjectiveRef({
    required this.board,
    required this.subject,
    required this.paper,
    required this.topic,
    required this.objectiveId,
    required this.learningObjective,
  });
}

class SyllabusEvidence {
  final String sourceType;
  final String syllabusPath;
  final List<String> trustReasons;
  final List<String> objectiveIds;
  final String paper;

  const SyllabusEvidence({
    required this.sourceType,
    required this.syllabusPath,
    required this.trustReasons,
    required this.objectiveIds,
    required this.paper,
  });
}

class SyllabusMapService {
  SyllabusMapService._();

  static final SyllabusMapService instance = SyllabusMapService._();

  static const Map<String, Map<String, List<SyllabusObjectiveRef>>> _seed = {
    'IGCSE': {
      'Physics': [
        SyllabusObjectiveRef(
          board: 'IGCSE',
          subject: 'Physics',
          paper: 'Paper 4',
          topic: 'Motion',
          objectiveId: '1.1.1',
          learningObjective: 'Use speed, velocity, and acceleration correctly.',
        ),
        SyllabusObjectiveRef(
          board: 'IGCSE',
          subject: 'Physics',
          paper: 'Paper 4',
          topic: 'Forces',
          objectiveId: '1.6.1',
          learningObjective: 'Apply momentum and force relationships in written and calculation questions.',
        ),
        SyllabusObjectiveRef(
          board: 'IGCSE',
          subject: 'Physics',
          paper: 'Paper 6',
          topic: 'Electricity',
          objectiveId: '4.2.1',
          learningObjective: 'Interpret circuit evidence and explain electrical relationships.',
        ),
      ],
      'Mathematics': [
        SyllabusObjectiveRef(
          board: 'IGCSE',
          subject: 'Mathematics',
          paper: 'Paper 2',
          topic: 'Algebra',
          objectiveId: '2.1.1',
          learningObjective: 'Manipulate algebraic expressions accurately under exam conditions.',
        ),
        SyllabusObjectiveRef(
          board: 'IGCSE',
          subject: 'Mathematics',
          paper: 'Paper 4',
          topic: 'Trigonometry',
          objectiveId: '4.3.1',
          learningObjective: 'Solve right-angle and non-right-angle trigonometry problems.',
        ),
      ],
    },
    'A Level': {
      'Physics': [
        SyllabusObjectiveRef(
          board: 'A Level',
          subject: 'Physics',
          paper: 'Paper 2',
          topic: 'Momentum',
          objectiveId: 'P2.4',
          learningObjective: 'Model momentum and impulse with correct sign conventions and units.',
        ),
        SyllabusObjectiveRef(
          board: 'A Level',
          subject: 'Physics',
          paper: 'Paper 4',
          topic: 'Circular Motion',
          objectiveId: 'P4.2',
          learningObjective: 'Explain circular motion using resultant force and angular relationships.',
        ),
      ],
      'Mathematics': [
        SyllabusObjectiveRef(
          board: 'A Level',
          subject: 'Mathematics',
          paper: 'Pure 1',
          topic: 'Differentiation',
          objectiveId: 'M1.5',
          learningObjective: 'Differentiate composite and standard polynomial functions fluently.',
        ),
      ],
    },
    'IB': {
      'Physics': [
        SyllabusObjectiveRef(
          board: 'IB',
          subject: 'Physics',
          paper: 'Paper 2',
          topic: 'Fields',
          objectiveId: 'IB-PHY-4.1',
          learningObjective: 'Use field models and representations in structured exam responses.',
        ),
      ],
    },
  };

  SyllabusEvidence buildEvidence({
    required String board,
    required String subject,
    required String chapter,
    required int dueReviewCount,
    required int daysSinceLastStudied,
    required int daysToExam,
    required bool isPrimarySubject,
  }) {
    final normalizedBoard = _normalizeBoard(board);
    final normalizedSubject = StudyCatalog.normalizeSubject(subject);
    final objective = _findBestObjective(
      board: normalizedBoard,
      subject: normalizedSubject,
      chapter: chapter,
    );

    final trustReasons = <String>[
      if (dueReviewCount > 0)
        '$dueReviewCount spaced-repetition card${dueReviewCount == 1 ? '' : 's'} are due for this topic.',
      if (daysSinceLastStudied >= 0)
        'You last studied this ${daysSinceLastStudied == 0 ? 'today' : '$daysSinceLastStudied day${daysSinceLastStudied == 1 ? '' : 's'} ago'}.',
      if (daysToExam >= 0 && daysToExam < 999)
        'The next $normalizedSubject exam window is in $daysToExam day${daysToExam == 1 ? '' : 's'}.',
      if (isPrimarySubject) 'This subject is currently your main focus subject.',
    ];

    final syllabusPath =
        '${objective.board} -> ${objective.subject} -> ${objective.paper} -> ${objective.topic} -> ${objective.objectiveId}';

    return SyllabusEvidence(
      sourceType: 'OFFICIAL_SYLLABUS',
      syllabusPath: syllabusPath,
      trustReasons: trustReasons,
      objectiveIds: [objective.objectiveId],
      paper: objective.paper,
    );
  }

  SyllabusObjectiveRef _findBestObjective({
    required String board,
    required String subject,
    required String chapter,
  }) {
    final bySubject = _seed[board]?[subject];
    if (bySubject != null && bySubject.isNotEmpty) {
      final lowerChapter = chapter.toLowerCase();
      for (final objective in bySubject) {
        if (lowerChapter.contains(objective.topic.toLowerCase()) ||
            objective.topic.toLowerCase().contains(lowerChapter)) {
          return objective;
        }
      }
      return bySubject.first;
    }

    final generatedTopic = chapter.trim().isEmpty ? subject : chapter.trim();
    final generatedId =
        '${subject.replaceAll(RegExp(r'[^A-Za-z0-9]+'), '').toUpperCase()}-${generatedTopic.replaceAll(RegExp(r'[^A-Za-z0-9]+'), '').toUpperCase()}';
    return SyllabusObjectiveRef(
      board: board,
      subject: subject,
      paper: 'Core Paper',
      topic: generatedTopic,
      objectiveId: generatedId,
      learningObjective:
          'Strengthen recall and application for $generatedTopic in exam conditions.',
    );
  }

  String _normalizeBoard(String board) {
    final lower = board.trim().toLowerCase();
    if (lower.contains('o level') || lower.contains('olevel')) {
      return 'CAIE O Level';
    }
    if (lower.contains('a level') || lower.contains('alevel')) {
      return 'CAIE A Level';
    }
    if (lower.contains('igcse') || lower.contains('caie') || lower.contains('cambridge')) {
      return 'CAIE IGCSE';
    }
    return board.trim().isEmpty ? 'CAIE IGCSE' : board.trim();
  }
}
