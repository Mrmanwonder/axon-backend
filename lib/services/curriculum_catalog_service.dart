import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/foundation.dart';
import 'supabase_proxy_service.dart';
import 'firestore_service.dart';

class ExamBoardOption {
  final String id;
  final String label;
  final String fullName;
  final String description;
  final String regions;
  final List<String> supportedLevels;

  const ExamBoardOption({
    required this.id,
    required this.label,
    required this.fullName,
    required this.description,
    required this.regions,
    required this.supportedLevels,
  });
}

class QualificationLevel {
  final String id;
  final String label;
  final String description;

  const QualificationLevel({
    required this.id,
    required this.label,
    required this.description,
  });

  static const igcse = QualificationLevel(
    id: 'igcse',
    label: 'IGCSE',
    description: 'International General Certificate of Secondary Education',
  );

  static const asLevel = QualificationLevel(
    id: 'as_level',
    label: 'AS Level',
    description: 'Advanced Subsidiary Level (first half of A Level)',
  );

  static const aLevel = QualificationLevel(
    id: 'a_level',
    label: 'A Level',
    description: 'Advanced Level (full qualification)',
  );

  static const List<QualificationLevel> all = [igcse, asLevel, aLevel];
}

class SupportedBoardOption {
  final String id;
  final String label;
  final String description;
  final String regions;

  const SupportedBoardOption({
    required this.id,
    required this.label,
    required this.description,
    required this.regions,
  });
}

class CurriculumChapter {
  final String id;
  final String title;
  final List<String> subchapters;

  const CurriculumChapter({
    required this.id,
    required this.title,
    required this.subchapters,
  });

  factory CurriculumChapter.fromJson(Map<String, dynamic> json) {
    // Support both old format (subchapters: List<String>) and new format (topics: List)
    final topicsList = json['topics'] as List?;
    final subchaptersList = json['subchapters'] as List?;

    List<String> subs;
    if (subchaptersList != null) {
      subs = subchaptersList.map((item) => item.toString()).toList();
    } else if (topicsList != null) {
      // Extract topic names from new format
      subs = topicsList.map((item) {
        final topicMap = item is Map ? item : {};
        return (topicMap['name'] ?? topicMap['title'] ?? 'Topic').toString();
      }).toList();
    } else {
      subs = [];
    }

    return CurriculumChapter(
      id: (json['id'] ?? json['chapter'] ?? '').toString(),
      title: (json['name'] ?? json['title'] ?? '').toString(),
      subchapters: subs,
    );
  }
}

class CurriculumPaper {
  final String code;
  final String name;
  final List<CurriculumChapter> chapters;

  const CurriculumPaper({
    required this.code,
    required this.name,
    required this.chapters,
  });
}

class CurriculumSubject {
  final String code;
  final String name;
  final List<CurriculumChapter> chapters;
  final List<String> aliases;
  final List<CurriculumPaper> papers;

  const CurriculumSubject({
    required this.code,
    required this.name,
    required this.chapters,
    this.aliases = const [],
    this.papers = const [],
  });

  factory CurriculumSubject.fromJson(Map<String, dynamic> json) {
    return CurriculumSubject(
      code: (json['code'] ?? json['subject_code'] ?? '').toString(),
      name: (json['name'] ?? '').toString(),
      chapters: (json['chapters'] as List? ?? const [])
          .map((item) =>
              CurriculumChapter.fromJson(Map<String, dynamic>.from(item)))
          .toList(),
      aliases: (json['aliases'] as List? ?? const [])
          .map((item) => item.toString())
          .toList(),
    );
  }
}

class CurriculumBoard {
  final String id;
  final String label;
  final List<CurriculumSubject> subjects;
  final List<String> aliases;

  const CurriculumBoard({
    required this.id,
    required this.label,
    required this.subjects,
    this.aliases = const [],
  });

  factory CurriculumBoard.fromJson(Map<String, dynamic> json) {
    return CurriculumBoard(
      id: (json['id'] ?? '').toString(),
      label: (json['label'] ?? '').toString(),
      subjects: (json['subjects'] as List? ?? const [])
          .map((item) =>
              CurriculumSubject.fromJson(Map<String, dynamic>.from(item)))
          .toList(),
      aliases: (json['aliases'] as List? ?? const [])
          .map((item) => item.toString())
          .toList(),
    );
  }
}

class CurriculumCatalogService {
  static final CurriculumCatalogService _instance =
      CurriculumCatalogService._();
  static CurriculumCatalogService get instance => _instance;

  CurriculumCatalogService._();

  List<CurriculumBoard>? _cache;
  final _proxy = SupabaseProxyService.instance;

  List<CurriculumSubject>? _subjectsCache;
  Map<String, List<CurriculumChapter>>? _chaptersCache;
  bool _localDataInitialized = false;

  // Step 1: Exam boards — only CAIE (Cambridge Assessment International Education)
  List<ExamBoardOption> get examBoards => const [
        ExamBoardOption(
          id: 'caie',
          label: 'CAIE',
          fullName: 'Cambridge Assessment International Education',
          description:
              'Cambridge International. Offers IGCSE, O Level, AS & A Level. Global zones (1–6 + UK). May/June & Oct/Nov series.',
          regions: 'Global',
          supportedLevels: ['igcse', 'as_level', 'a_level'],
        ),
      ];

  // Legacy: flat board+level combinations for backward compatibility
  List<SupportedBoardOption> get supportedBoards {
    final result = <SupportedBoardOption>[];
    for (final board in examBoards) {
      if (board.supportedLevels.contains('igcse')) {
        result.add(SupportedBoardOption(
          id: '${board.id}_igcse',
          label: '${board.label} IGCSE',
          description: '${board.fullName} — IGCSE qualification.',
          regions: board.regions,
        ));
      }
      if (board.supportedLevels.contains('as_level')) {
        result.add(SupportedBoardOption(
          id: '${board.id}_as_level',
          label: '${board.label} AS Level',
          description: '${board.fullName} — AS Level qualification.',
          regions: board.regions,
        ));
      }
      if (board.supportedLevels.contains('a_level')) {
        result.add(SupportedBoardOption(
          id: '${board.id}_a_level',
          label: '${board.label} A Level',
          description: '${board.fullName} — A Level qualification.',
          regions: board.regions,
        ));
      }
    }
    return result;
  }

  String _norm(String value) {
    return value.trim().toLowerCase().replaceAll(RegExp(r'\s+'), ' ');
  }

  Future<List<CurriculumBoard>> _loadBoards() async {
    if (_cache != null) return _cache!;

    final allBoards = <CurriculumBoard>[];

    try {
      final raw = await rootBundle.loadString('data/curriculum_catalog.json');
      final decoded = jsonDecode(raw) as Map<String, dynamic>;
      final boards = (decoded['boards'] as List? ?? const [])
          .whereType<Map>()
          .map((item) =>
              CurriculumBoard.fromJson(Map<String, dynamic>.from(item)))
          .where((item) => item.id.isNotEmpty)
          .toList();
      allBoards.addAll(boards);
    } catch (_) {}

    try {
      final raw =
          await rootBundle.loadString('data/caie_as_alvl_curriculum.json');
      final decoded = jsonDecode(raw) as Map<String, dynamic>;
      final boards = (decoded['boards'] as List? ?? const [])
          .whereType<Map>()
          .map((item) =>
              CurriculumBoard.fromJson(Map<String, dynamic>.from(item)))
          .toList();

      for (final board in boards) {
        final existingIndex = allBoards.indexWhere((b) =>
            b.id == 'caie_a_level' ||
            b.aliases.any((a) => _norm(a).contains('a level')));
        if (existingIndex >= 0) {
          final existingCodes =
              allBoards[existingIndex].subjects.map((s) => s.code).toSet();
          final newSubjects = board.subjects
              .where((s) => !existingCodes.contains(s.code))
              .toList();
          allBoards[existingIndex] = CurriculumBoard(
            id: 'caie_a_level',
            label: 'CAIE A Level',
            aliases: const [
              'CAIE A Level',
              'A Level',
              'Cambridge AS & A Level'
            ],
            subjects: [...allBoards[existingIndex].subjects, ...newSubjects],
          );
        } else {
          allBoards.add(board);
        }
      }
    } catch (_) {}

    _cache = allBoards;
    return _cache!;
  }

  Future<List<CurriculumBoard>> loadBoards() => _loadBoards();

  Future<CurriculumBoard?> findBoard(String rawBoard) async {
    final normalized = _norm(rawBoard);
    final boards = await _loadBoards();

    // Exact match first
    for (final board in boards) {
      if (_norm(board.id) == normalized || _norm(board.label) == normalized) {
        return board;
      }
      if (board.aliases.any((alias) => _norm(alias) == normalized)) {
        return board;
      }
    }

    // Two-step detection: extract board + level from combined string
    final boardId = _extractBoardId(normalized);
    final level = _extractLevel(normalized);

    if (boardId != null) {
      final exactMatch = boards.where((b) => b.id == boardId).firstOrNull;
      if (exactMatch != null) return exactMatch;

      // Map to legacy CAIE board IDs
      final legacyId = level == 'igcse'
          ? 'caie_igcse'
          : level == 'as_level'
              ? 'caie_as_level'
              : 'caie_a_level';
      return boards.where((b) => b.id == legacyId).firstOrNull;
    }

    // Fallback: all unrecognized input maps to CAIE
    if (normalized.contains('a level') || normalized.contains('as level')) {
      return boards.where((b) => b.id == 'caie_a_level').firstOrNull;
    }
    if (normalized.contains('igcse') ||
        normalized.contains('o level') ||
        normalized.contains('olevel')) {
      return boards.where((b) => b.id == 'caie_igcse').firstOrNull;
    }
    return boards.isNotEmpty ? boards.first : null;
  }

  // Extract board ID from a normalized string — only CAIE
  String? _extractBoardId(String normalized) {
    if (normalized.contains('caie') ||
        normalized.contains('cambridge') ||
        normalized.contains('cie')) {
      return 'caie';
    }
    return null;
  }

  // Extract qualification level from a normalized string
  String _extractLevel(String normalized) {
    if (normalized.contains('as level') && !normalized.contains('a level')) {
      return 'as_level';
    }
    if (normalized.contains('a level') ||
        normalized.contains('alevel') ||
        normalized.contains('ial') ||
        normalized.contains('advanced level')) {
      return 'a_level';
    }
    if (normalized.contains('igcse') ||
        normalized.contains('international gcse')) {
      return 'igcse';
    }
    if (normalized.contains('o level') || normalized.contains('olevel')) {
      return 'igcse'; // O Level maps to IGCSE subjects
    }
    if (normalized.contains('gcse')) {
      return 'igcse';
    }
    return 'igcse'; // Default
  }

  Future<String> canonicalBoardLabel(String rawBoard) async {
    return (await findBoard(rawBoard))?.label ?? rawBoard.trim();
  }

  Future<List<String>> subjectsForBoard(String rawBoard) async {
    final board = await findBoard(rawBoard);
    if (board == null || board.subjects.isEmpty) {
      // Fallback: return all available subjects from cache
      await initializeLocalData();
      if (_subjectsCache != null) {
        return _subjectsCache!.map((s) => s.name).toList()..sort();
      }
      return const [];
    }
    return board.subjects.map((s) => s.name).toList()..sort();
  }

  /// Pre-load all curriculum data from local JSON for instant access
  Future<void> initializeLocalData() async {
    if (_localDataInitialized) return;

    // Load boards from local JSON immediately
    final boards = await _loadBoards();

    // Build subjects cache from local data
    final subjects = <CurriculumSubject>[];
    for (final board in boards) {
      subjects.addAll(board.subjects);
    }
    _subjectsCache = subjects;

    // Build chapters cache from local data
    final chapters = <String, List<CurriculumChapter>>{};
    for (final subject in subjects) {
      if (subject.code.isNotEmpty) {
        chapters[subject.code] = subject.chapters;
      }
    }
    _chaptersCache = chapters;

    _localDataInitialized = true;
    debugPrint(
        'CurriculumCatalogService: Local data initialized with ${subjects.length} subjects');
  }

  Future<List<CurriculumSubject>> getAllSubjects() async {
    if (_subjectsCache != null) return _subjectsCache!;

    // Try local first
    await initializeLocalData();
    if (_subjectsCache != null) return _subjectsCache!;

    // Fallback to supabase only if local fails
    try {
      debugPrint(
          'CurriculumCatalogService: Fetching subjects from Supabase...');
      final data = await _proxy.query('subjects', params: {'order': 'name.asc'});
      _subjectsCache = data
          .map((item) =>
              CurriculumSubject.fromJson(Map<String, dynamic>.from(item)))
          .toList();
      debugPrint(
          'CurriculumCatalogService: Fetched ${_subjectsCache?.length} subjects from Supabase');
      return _subjectsCache!;
    } catch (e) {
      debugPrint(
          'CurriculumCatalogService: Supabase subjects fetch failed: $e');
      return _subjectsCache ?? [];
    }
  }

  Future<List<CurriculumChapter>> getChapters(String subjectCode) async {
    if (_chaptersCache != null && _chaptersCache!.containsKey(subjectCode)) {
      return _chaptersCache![subjectCode]!;
    }

    // Try local first
    await initializeLocalData();
    if (_chaptersCache != null && _chaptersCache!.containsKey(subjectCode)) {
      return _chaptersCache![subjectCode]!;
    }

    // Fallback
    return _chaptersCache?[subjectCode] ?? [];
  }

  Future<Map<String, List<String>>> getUserChapters(String uid) async {
    try {
      final doc = await AxonPaths.privateUserDoc(uid).get();
      if (!doc.exists || doc.data() == null) {
        return {};
      }
      final data = doc.data()!;
      final studyCatalog = data['study_catalog'];
      if (studyCatalog is Map) {
        final result = <String, List<String>>{};
        for (final entry in studyCatalog.entries) {
          final subject = entry.key.toString();
          final chapters = (entry.value as List?)
                  ?.map((e) => e.toString())
                  .where((c) => c.isNotEmpty)
                  .toList() ??
              [];
          result[subject] = chapters;
        }
        return result;
      }
      return {};
    } catch (_) {
      return {};
    }
  }

  Future<CurriculumSubject?> findSubject({
    required String board,
    required String subject,
  }) async {
    final normalizedBoard = _norm(board);
    final normalizedSubject = _norm(subject);

// Try to match subject code from Supabase
    try {
      // Extract numerical code from subject string
      final codeMatch = RegExp(r'\d+').firstMatch(normalizedSubject);
      String? searchCode;
      if (codeMatch != null) {
        searchCode = codeMatch.group(0);
      } else {
        // Search by name
        final data = await _proxy.query('subjects', params: {
          'name': 'ilike.*$normalizedSubject*',
          'limit': '1',
        });
        if (data.isNotEmpty) {
          final subjData = data.first;
          searchCode = subjData['code'] as String;
        }
      }

      if (searchCode != null) {
        // Get chapters from Supabase
        final chapterData = await _proxy.query('chapters', params: {
          'subject_code': 'eq.$searchCode',
          'order': 'order_index.asc',
        });
        final chapters = chapterData
            .map(
                (c) => CurriculumChapter.fromJson(Map<String, dynamic>.from(c)))
            .toList();

        final subjData = await _proxy.query('subjects', params: {
          'code': 'eq.$searchCode',
          'limit': '1',
        });
        final name =
            subjData.isNotEmpty ? subjData.first['name'] as String : subject;

        return CurriculumSubject(
          code: searchCode,
          name: name,
          chapters: chapters,
        );
      }
    } catch (_) {}

    // Fallback to local JSON
    final boards = await _loadBoards();

    for (final b in boards) {
      if (_norm(b.id).contains(normalizedBoard) ||
          b.aliases.any((a) => _norm(a).contains(normalizedBoard))) {
        for (final s in b.subjects) {
          if (_norm(s.code) == normalizedSubject ||
              _norm(s.name).contains(normalizedSubject) ||
              s.aliases.any((a) => _norm(a).contains(normalizedSubject))) {
            return s;
          }
        }
      }
    }

    return _fallbackSubject(normalizedSubject);
  }

  CurriculumSubject? _fallbackSubject(String normalized) {
    if (normalized.contains('0580') ||
        (normalized.contains('math') && !normalized.contains('further'))) {
      return const CurriculumSubject(
        code: '0580',
        name: 'Mathematics',
        aliases: ['Math', 'Maths'],
        chapters: [
          CurriculumChapter(id: '1', title: 'Number', subchapters: [
            'Number operations',
            'Powers and roots',
            'Factors',
            'Fractions',
            'Percentages',
            'Rounding'
          ]),
          CurriculumChapter(id: '2', title: 'Algebra', subchapters: [
            'Expressions',
            'Linear equations',
            'Inequalities',
            'Quadratics',
            'Sequences',
            'Graphs'
          ]),
          CurriculumChapter(id: '3', title: 'Geometry', subchapters: [
            'Lines and angles',
            'Triangles',
            'Circles',
            'Congruence',
            'Transformations',
            'Trigonometry'
          ]),
          CurriculumChapter(
              id: '4',
              title: 'Measure',
              subchapters: ['Units', 'Perimeter', 'Area', 'Volume']),
          CurriculumChapter(
              id: '5',
              title: 'Statistics',
              subchapters: ['Data collection', 'Diagrams', 'Averages']),
          CurriculumChapter(
              id: '6',
              title: 'Probability',
              subchapters: ['Basic probability', 'Calculations', 'Diagrams']),
        ],
      );
    }
    if (normalized.contains('0610') || normalized.contains('biolog')) {
      return const CurriculumSubject(
        code: '0610',
        name: 'Biology',
        aliases: ['Bio', 'Biology'],
        chapters: [
          CurriculumChapter(
              id: '1',
              title: 'Characteristics',
              subchapters: ['MRS GREN', 'Classification']),
          CurriculumChapter(
              id: '2',
              title: 'Organisation',
              subchapters: ['Cell structure', 'Cell division']),
          CurriculumChapter(
              id: '3',
              title: 'Movement',
              subchapters: ['Diffusion', 'Osmosis', 'Active transport']),
          CurriculumChapter(
              id: '4',
              title: 'Molecules',
              subchapters: ['Carbohydrates', 'Lipids', 'Proteins', 'Water']),
          CurriculumChapter(
              id: '5',
              title: 'Enzymes',
              subchapters: ['Enzyme action', 'Factors affecting']),
        ],
      );
    }
    if (normalized.contains('0620') || normalized.contains('chemis')) {
      return const CurriculumSubject(
        code: '0620',
        name: 'Chemistry',
        aliases: ['Chem', 'Chemistry'],
        chapters: [
          CurriculumChapter(
              id: '1',
              title: 'States of Matter',
              subchapters: ['Kinetic theory', 'Gas properties']),
          CurriculumChapter(
              id: '2',
              title: 'Atomic Structure',
              subchapters: ['Atomic model', 'Electronic structure']),
          CurriculumChapter(
              id: '3',
              title: 'Bonding',
              subchapters: ['Ionic', 'Covalent', 'Metallic']),
          CurriculumChapter(
              id: '4',
              title: 'Reactions',
              subchapters: ['Equations', 'Calculations']),
        ],
      );
    }
    if (normalized.contains('0652') || normalized.contains('physic')) {
      return const CurriculumSubject(
        code: '0652',
        name: 'Physics',
        aliases: ['Phys', 'Physics'],
        chapters: [
          CurriculumChapter(
              id: '1', title: 'Motion', subchapters: ['Kinematics', 'Graphs']),
          CurriculumChapter(
              id: '2',
              title: 'Forces',
              subchapters: ['Newton laws', 'Momentum']),
          CurriculumChapter(
              id: '3', title: 'Energy', subchapters: ['Work', 'Power']),
          CurriculumChapter(
              id: '4',
              title: 'Waves',
              subchapters: ['Properties', 'Light', 'Sound']),
        ],
      );
    }
    return CurriculumSubject(
      code: normalized.length > 4 ? normalized.substring(0, 4) : '0000',
      name: normalized.substring(0, 1).toUpperCase() + normalized.substring(1),
      chapters: const [],
    );
  }

  // Helper to check if subjects are loaded from local JSON
  Future<Map<String, List<String>>> chapterTree({
    required String board,
    required String subject,
  }) async {
    final subjectData = await findSubject(board: board, subject: subject);
    if (subjectData == null) return {};

    return {
      for (final chapter in subjectData.chapters)
        chapter.title: chapter.subchapters,
    };
  }

  Future<List<String>> chapterTitles({
    required String board,
    required String subject,
  }) async {
    final subjectData = await findSubject(board: board, subject: subject);
    if (subjectData == null) return [];
    return subjectData.chapters.map((c) => c.title).toList();
  }

  Future<int> chapterCount({
    required String board,
    required String subject,
  }) async {
    final chapters = await chapterTitles(board: board, subject: subject);
    return chapters.length;
  }

  Future<String?> getSubjectCode(String subjectName) async {
    final subjects = await getAllSubjects();
    final normalized = _norm(subjectName);

    // 1. Direct code match
    final isCode = RegExp(r'^\d{4}$').hasMatch(normalized);
    if (isCode) return normalized;

    // 2. Exact name match
    for (final s in subjects) {
      if (_norm(s.name) == normalized) return s.code;
    }

    // 3. Alias match
    for (final s in subjects) {
      if (s.aliases.any((a) => _norm(a) == normalized)) return s.code;
    }

    // 4. Partial match
    for (final s in subjects) {
      if (_norm(s.name).contains(normalized)) {
        debugPrint(
            'CurriculumCatalogService: Resolved "$subjectName" to code "${s.code}" (partial match)');
        return s.code;
      }
    }

    debugPrint(
        'CurriculumCatalogService: Failed to resolve subject code for "$subjectName"');
    return null;
  }
}

extension<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
