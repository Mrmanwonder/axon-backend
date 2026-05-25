// lib/services/unified_data_service.dart
import 'curriculum_catalog_service.dart';
import 'subject_resource_linker.dart';
import 'local_database_service.dart';

class UnifiedSubjectData {
  final String code;
  final String name;
  final String board;
  final List<CurriculumChapter> chapters;
  final List<CurriculumPaper> papers;
  final SubjectResourceLink resources;
  final Map<String, dynamic>? progress;

  const UnifiedSubjectData({
    required this.code,
    required this.name,
    required this.board,
    this.chapters = const [],
    this.papers = const [],
    required this.resources,
    this.progress,
  });

  bool get hasPapers => papers.isNotEmpty;

  int get totalChapters => papers.isNotEmpty
      ? papers.fold(0, (sum, p) => sum + p.chapters.length)
      : chapters.length;

  List<CurriculumChapter> get allChapters {
    if (papers.isNotEmpty) {
      return papers.expand((p) => p.chapters).toList();
    }
    return chapters;
  }
}

class UnifiedDataService {
  static final UnifiedDataService _instance = UnifiedDataService._internal();
  factory UnifiedDataService() => _instance;
  UnifiedDataService._internal();

  bool _initialized = false;
  late SubjectResourceLinkerService _linker;
  late CurriculumCatalogService _curriculum;
  late LocalDatabaseService _database;

  Future<void> initialize() async {
    if (_initialized) return;

    _curriculum = CurriculumCatalogService.instance;
    _linker = SubjectResourceLinkerService();
    _database = LocalDatabaseService();

    await _linker.initialize();
    _initialized = true;
  }

  Future<List<UnifiedSubjectData>> getAllSubjects() async {
    if (!_initialized) await initialize();

    final boards = await _curriculum.loadBoards();
    final subjects = <UnifiedSubjectData>[];

    for (final board in boards) {
      for (final subject in board.subjects) {
        if (subject.code.isEmpty) continue;

        final resources = _linker.getLinksForCode(subject.code);
        if (resources == null) continue;

        subjects.add(UnifiedSubjectData(
          code: subject.code,
          name: subject.name,
          board: board.id,
          chapters: subject.chapters,
          papers: subject.papers,
          resources: resources,
        ));
      }
    }

    return subjects;
  }

  Future<UnifiedSubjectData?> getSubjectByCode(String code) async {
    if (!_initialized) await initialize();

    final boards = await _curriculum.loadBoards();

    for (final board in boards) {
      for (final subject in board.subjects) {
        if (subject.code == code) {
          final resources = _linker.getLinksForCode(code);
          if (resources == null) return null;

          return UnifiedSubjectData(
            code: subject.code,
            name: subject.name,
            board: board.id,
            chapters: subject.chapters,
            papers: subject.papers,
            resources: resources,
          );
        }
      }
    }

    return null;
  }

  Future<List<UnifiedSubjectData>> searchSubjects(String query) async {
    final all = await getAllSubjects();
    final q = query.toLowerCase();
    return all
        .where((s) => s.name.toLowerCase().contains(q) || s.code.contains(q))
        .toList();
  }

  Future<List<UnifiedSubjectData>> getSubjectsByBoard(String boardId) async {
    final all = await getAllSubjects();
    return all.where((s) => s.board == boardId).toList();
  }

  Future<Map<String, dynamic>> getDashboardData(String userId) async {
    if (!_initialized) await initialize();

    final allSubjects = await getAllSubjects();

    // Get user progress from database
    final progress = await _database.getUserChapterProgress(userId, '');
    final mocks = await _database.getUserMockScores(userId);
    final sessions = await _database.getUserStudySessions(userId);

    return {
      'totalSubjects': allSubjects.length,
      'subjects': allSubjects
          .map((s) => {
                'code': s.code,
                'name': s.name,
                'chapters': s.totalChapters,
                'resources': s.resources.allResources.length,
                'pastPapers': s.resources.pastPapers.length,
                'mocks': s.resources.mocks.length,
                'flashcards': s.resources.flashcards.length,
              })
          .toList(),
      'userProgress': {
        'chaptersCompleted': progress.length,
        'mockScores': mocks.length,
        'studySessions': sessions.length,
      },
      'resourceCounts': _linker.getResourceCounts(),
    };
  }

  Future<void> trackProgress({
    required String userId,
    required String subjectCode,
    required String chapterId,
    required double completionPercentage,
    int? timeSpent,
  }) async {
    await _database.insertOrUpdateChapterProgress({
      'user_id': userId,
      'subject_code': subjectCode,
      'chapter_id': chapterId,
      'completion_percentage': completionPercentage,
      'time_spent_seconds': timeSpent ?? 0,
      'last_accessed': DateTime.now().millisecondsSinceEpoch,
    });
  }

  Future<void> recordMockScore({
    required String userId,
    required String subjectCode,
    required double totalMarks,
    required double obtainedMarks,
    String? paperCode,
  }) async {
    await _database.insertMockScore({
      'user_id': userId,
      'subject_code': subjectCode,
      'paper_code': paperCode,
      'total_marks': totalMarks,
      'obtained_marks': obtainedMarks,
      'percentage': (obtainedMarks / totalMarks) * 100,
      'attempt_date': DateTime.now().millisecondsSinceEpoch,
    });
  }

  Future<void> recordStudySession({
    required String userId,
    required String subjectCode,
    required int durationSeconds,
    String? sessionType,
    int? topicsCovered,
  }) async {
    final now = DateTime.now().millisecondsSinceEpoch;
    await _database.insertStudySession({
      'user_id': userId,
      'subject_code': subjectCode,
      'start_time': now - (durationSeconds * 1000),
      'end_time': now,
      'duration_seconds': durationSeconds,
      'session_type': sessionType ?? 'general',
      'topics_covered': topicsCovered ?? 0,
    });
  }
}
