import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'resource_sources.dart';

class ResourceRecommendationService {
  static final ResourceRecommendationService _instance =
      ResourceRecommendationService._internal();
  factory ResourceRecommendationService() => _instance;
  ResourceRecommendationService._internal();

  Future<ChapterResources> getResourcesForChapter({
    required String board,
    required String subject,
    required String chapter,
    bool includeVideos = true,
    bool includeNotes = true,
    bool includePastPapers = true,
    bool includeTextbook = true,
    List<String>? weakChapterKeys,
  }) async {
    final boardNorm = normalizeBoard(board);
    final recs = <EnhancedResource>[];

    // 1. Fetch from online sources
    if (includePastPapers) {
      recs.addAll(await _fetchOnlinePastPapers(subject, chapter, boardNorm));
    }
    if (includeVideos) {
      recs.addAll(_buildOnlineVideos(subject, chapter));
    }
    if (includeNotes) {
      recs.addAll(_buildOnlineNotes(subject, chapter, boardNorm));
    }
    if (includeTextbook) {
      recs.addAll(_buildOnlineTextbooks(subject, chapter, boardNorm));
    }

    // 2. Also scan local files (as fallback)
    final localResources = await _scanLocalFiles(subject, chapter, boardNorm);
    for (final r in localResources) {
      if (!recs.any((existing) => existing.id == r.id)) {
        recs.add(r);
      }
    }

    recs.sort((a, b) => b.qualityScore.compareTo(a.qualityScore));

    return ChapterResources(
      resources: recs,
      totalCount: recs.length,
      hasMore: false,
    );
  }

  // ═══════════════════════════════════════════════════════════════════
  // ONLINE PAST PAPERS
  // ═══════════════════════════════════════════════════════════════════

  Future<List<EnhancedResource>> _fetchOnlinePastPapers(
      String subject, String chapter, String board) async {
    final papers = <EnhancedResource>[];
    final subjectCode = _getSubjectCode(subject);
    final normalizedBoard = board.toLowerCase();

    // Cambridge IGCSE past papers
    if (normalizedBoard.contains('cambridge') ||
        normalizedBoard.contains('igcse') ||
        normalizedBoard.contains('cie')) {
      for (final year in [2024, 2023, 2022, 2021, 2020]) {
        papers.add(EnhancedResource(
          id: 'cie_pp_${subjectCode}_$year',
          title: '$subject - May/June $year',
          subtitle: 'Cambridge IGCSE Paper',
          url: 'https://pastpapers.org/cie/$subjectCode/$year',
          type: ResourceType.pastPaper,
          source: 'Cambridge IGCSE',
          sourceType: ResourceSourceType.pastPaper,
          boardTags: ['cambridge', 'igcse'],
          resourceTypes: ['Question Paper'],
          qualityScore: 0.95,
          accessType: AccessType.free,
          isHealthy: true,
          isVerified: true,
          topics: [subject, chapter, year.toString()],
        ));

        papers.add(EnhancedResource(
          id: 'cie_ms_${subjectCode}_$year',
          title: '$subject - May/June $year (Mark Scheme)',
          subtitle: 'Cambridge IGCSE Mark Scheme',
          url: 'https://pastpapers.org/cie/$subjectCode/${year}_ms',
          type: ResourceType.pastPaper,
          source: 'Cambridge IGCSE',
          sourceType: ResourceSourceType.pastPaper,
          boardTags: ['cambridge', 'igcse'],
          resourceTypes: ['Mark Scheme'],
          qualityScore: 0.95,
          accessType: AccessType.free,
          isHealthy: true,
          isVerified: true,
          topics: [subject, chapter, year.toString()],
        ));
      }
    }

    // Cambridge A-Level
    if (normalizedBoard.contains('alevel')) {
      for (final year in [2024, 2023, 2022, 2021]) {
        papers.add(EnhancedResource(
          id: 'cie_al_${subjectCode}_$year',
          title: '$subject - $year A-Level',
          subtitle: 'Cambridge A-Level Paper',
          url: 'https://pastpapers.org/cie/alevel/$subjectCode/$year',
          type: ResourceType.pastPaper,
          source: 'Cambridge A-Level',
          sourceType: ResourceSourceType.pastPaper,
          boardTags: ['cambridge', 'alevel'],
          resourceTypes: ['Question Paper'],
          qualityScore: 0.95,
          accessType: AccessType.free,
          isHealthy: true,
          isVerified: true,
          topics: [subject, chapter, year.toString()],
        ));
      }
    }

    // IB Diploma
    if (normalizedBoard.contains('ib') || normalizedBoard.contains('dp')) {
      papers.add(EnhancedResource(
        id: 'ib_${subject}_hl',
        title: '$subject HL - IB Past Papers',
        subtitle: 'IB Diploma HL',
        url: 'https://pastpapers.org/ib/${subject.toLowerCase()}/hl',
        type: ResourceType.pastPaper,
        source: 'IB Archive',
        sourceType: ResourceSourceType.pastPaper,
        boardTags: ['ib', 'dp'],
        resourceTypes: ['Question Paper', 'Mark Scheme'],
        qualityScore: 0.95,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: true,
        topics: [subject, chapter, 'HL'],
      ));

      papers.add(EnhancedResource(
        id: 'ib_${subject}_sl',
        title: '$subject SL - IB Past Papers',
        subtitle: 'IB Diploma SL',
        url: 'https://pastpapers.org/ib/${subject.toLowerCase()}/sl',
        type: ResourceType.pastPaper,
        source: 'IB Archive',
        sourceType: ResourceSourceType.pastPaper,
        boardTags: ['ib', 'dp'],
        resourceTypes: ['Question Paper', 'Mark Scheme'],
        qualityScore: 0.95,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: true,
        topics: [subject, chapter, 'SL'],
      ));
    }

    return papers;
  }

  String _getSubjectCode(String subject) {
    final codes = {
      'mathematics': '0580',
      'additional mathematics': '0606',
      'physics': '0620',
      'chemistry': '0620',
      'biology': '0610',
      'computer science': '0478',
      'economics': '0455',
      'english': '0500',
      'history': '0470',
      'geography': '0460',
    };
    return codes[subject.toLowerCase()] ??
        subject.toLowerCase().replaceAll(' ', '-');
  }

  // ═══════════════════════════════════════════════════════════════════
  // ONLINE VIDEOS
  // ═══════════════════════════════════════════════════════════════════

  List<EnhancedResource> _buildOnlineVideos(String subject, String chapter) {
    final search = Uri.encodeComponent('$subject $chapter');
    final subSlug = subject.toLowerCase().replaceAll(' ', '_');
    final chapSlug = chapter.toLowerCase().replaceAll(' ', '_');
    return [
      EnhancedResource(
        id: 'khan_${subSlug}_$chapSlug',
        title: 'Khan Academy — $subject: $chapter',
        subtitle: 'Free video lessons & practice',
        url: 'https://www.khanacademy.org/search?page_search_query=$search',
        type: ResourceType.video,
        source: 'Khan Academy',
        sourceType: ResourceSourceType.khanAcademy,
        boardTags: ['khan'],
        resourceTypes: ['Video', 'Practice'],
        qualityScore: 0.95,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: true,
        topics: [subject, chapter],
      ),
      EnhancedResource(
        id: 'youtube_${subject}_$chapSlug',
        title: 'YouTube — $subject: $chapter',
        subtitle: 'Search results',
        url: 'https://www.youtube.com/results?search_query=$search',
        type: ResourceType.video,
        source: 'YouTube',
        sourceType: ResourceSourceType.youtube,
        boardTags: ['youtube'],
        resourceTypes: ['Video'],
        qualityScore: 0.7,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: false,
        topics: [subject, chapter],
      ),
      EnhancedResource(
        id: 'freeschool_$subject',
        title: 'FreeSchool — $subject',
        subtitle: 'Educational videos',
        url: 'https://www.youtube.com/c/FreeSchool/videos',
        type: ResourceType.video,
        source: 'YouTube',
        sourceType: ResourceSourceType.youtube,
        boardTags: ['youtube'],
        resourceTypes: ['Video'],
        qualityScore: 0.75,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: false,
        topics: [subject],
      ),
      EnhancedResource(
        id: 'physicsclassroom',
        title: 'Physics Classroom',
        subtitle: 'Tutorials & simulations',
        url: 'https://www.physicsclassroom.com/',
        type: ResourceType.video,
        source: 'Physics Classroom',
        sourceType: ResourceSourceType.youtube,
        boardTags: ['physics'],
        resourceTypes: ['Video', 'Simulation'],
        qualityScore: 0.9,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: true,
        topics: ['Physics'],
      ),
    ];
  }

  // ═══════════════════════════════════════════════════════════════════
  // ONLINE NOTES
  // ═══════════════════════════════════════════════════════════════════

  List<EnhancedResource> _buildOnlineNotes(
      String subject, String chapter, String board) {
    final subSlug = subject.toLowerCase().replaceAll(' ', '_');
    final chapSlug = chapter.toLowerCase().replaceAll(' ', '_');
    return [
      EnhancedResource(
        id: 'notes_${subSlug}_$chapSlug',
        title: 'Revision Notes — $subject: $chapter',
        subtitle: 'Quick revision guide',
        url:
            'https://www.savemyexams.co.uk/revision-notes/$board/$subject/$chapter',
        type: ResourceType.notes,
        source: 'Save My Exams',
        sourceType: ResourceSourceType.custom,
        boardTags: [board],
        resourceTypes: ['Revision Notes'],
        qualityScore: 0.85,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: true,
        topics: [subject, chapter],
      ),
      EnhancedResource(
        id: 'quizlet_${subSlug}_$chapSlug',
        title: 'Quizlet — $subject: $chapter',
        subtitle: 'Flashcards & quizzes',
        url: 'https://quizlet.com/subject/$subject',
        type: ResourceType.notes,
        source: 'Quizlet',
        sourceType: ResourceSourceType.custom,
        boardTags: [],
        resourceTypes: ['Flashcards'],
        qualityScore: 0.75,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: false,
        topics: [subject, chapter],
      ),
      EnhancedResource(
        id: 'britannica_$subject',
        title: 'Encyclopedia — $subject',
        subtitle: 'Reference articles',
        url: 'https://www.britannica.com/search?query=$subject',
        type: ResourceType.textbook,
        source: 'Britannica',
        sourceType: ResourceSourceType.textbook,
        boardTags: [],
        resourceTypes: ['Reference', 'Article'],
        qualityScore: 0.9,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: true,
        topics: [subject],
      ),
    ];
  }

  // ═══════════════════════════════════════════════════════════════════
  // ONLINE TEXTBOOKS
  // ═══════════════════════════════════════════════════════════════════

  List<EnhancedResource> _buildOnlineTextbooks(
      String subject, String chapter, String board) {
    final subSlug = subject.toLowerCase().replaceAll(' ', '-');
    final search = Uri.encodeComponent('$subject $chapter textbook pdf');
    return [
      EnhancedResource(
        id: 'scribd_$subSlug',
        title: 'Scribd — $subject',
        subtitle: 'Ebooks & textbooks',
        url: 'https://www.scribd.com/search?query=$search',
        type: ResourceType.textbook,
        source: 'Scribd',
        sourceType: ResourceSourceType.textbook,
        boardTags: [],
        resourceTypes: ['Ebook', 'Textbook'],
        qualityScore: 0.9,
        accessType: AccessType.freemium,
        isHealthy: true,
        isVerified: true,
        topics: [subject],
      ),
      EnhancedResource(
        id: 'docpub_$subSlug',
        title: 'Documen.pub — $subject',
        subtitle: 'Free PDF textbooks',
        url: 'https://documen.pub/category/$subSlug/',
        type: ResourceType.textbook,
        source: 'Documen.pub',
        sourceType: ResourceSourceType.textbook,
        boardTags: [],
        resourceTypes: ['PDF', 'Textbook'],
        qualityScore: 0.85,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: false,
        topics: [subject],
      ),
      EnhancedResource(
        id: 'libgen_$subSlug',
        title: 'Library Genesis — $subject',
        subtitle: 'Academic textbooks',
        url: 'https://libgen.is/search.php?req=$subject&res=100',
        type: ResourceType.textbook,
        source: 'Library Genesis',
        sourceType: ResourceSourceType.textbook,
        boardTags: [],
        resourceTypes: ['PDF', 'Ebook'],
        qualityScore: 0.95,
        accessType: AccessType.free,
        isHealthy: true,
        isVerified: true,
        topics: [subject],
      ),
    ];
  }

  // ═══════════════════════════════════════════════════════════════════
  // LOCAL FILES (Fallback)
  // ═══════════════════════════════════════════════════════════════════

  Future<List<EnhancedResource>> _scanLocalFiles(
      String subject, String chapter, String board) async {
    final resources = <EnhancedResource>[];

    try {
      final dirs = await Future.wait([
        getApplicationDocumentsDirectory(),
        getExternalStorageDirectory(),
      ]);

      final searchDirs = [
        'PastPapers',
        'Papers',
        'Downloads',
        'Documents',
        'assets/papers',
        'assets/videos',
        'assets/notes'
      ];

      for (final baseDir in dirs.whereType<Directory>()) {
        for (final subDir in searchDirs) {
          try {
            final path = '${baseDir.path}/$subDir';
            final dir = Directory(path);
            if (await dir.exists()) {
              await for (final entity
                  in dir.list(recursive: true, followLinks: false)) {
                if (entity is File) {
                  final resource =
                      _matchLocalFile(entity.path, subject, chapter, board);
                  if (resource != null) {
                    resources.add(resource);
                  }
                }
              }
            }
          } catch (_) {}
        }
      }
    } catch (_) {}

    return resources;
  }

  EnhancedResource? _matchLocalFile(
      String path, String subject, String chapter, String board) {
    final fileName = path.split(Platform.pathSeparator).last.toLowerCase();
    if (!fileName.endsWith('.pdf') &&
        !fileName.endsWith('.doc') &&
        !fileName.endsWith('.mp4') &&
        !fileName.endsWith('.mkv')) {
      return null;
    }

    final subjectKeywords = _getSubjectKeywords(subject);
    if (!subjectKeywords.any((k) => fileName.contains(k))) {
      return null;
    }

    final type = fileName.contains('paper') || fileName.contains('question')
        ? ResourceType.pastPaper
        : (fileName.endsWith('.mp4') || fileName.endsWith('.mkv'))
            ? ResourceType.video
            : ResourceType.notes;

    return EnhancedResource(
      id: 'local_${fileName.hashCode}',
      title: fileName.replaceAll(RegExp(r'[-_]'), ' ').replaceAll(
          RegExp(r'\.(pdf|doc|mp4|mkv)$', caseSensitive: false), ''),
      subtitle: 'Local file',
      url: path,
      type: type,
      source: 'Local Files',
      sourceType: ResourceSourceType.custom,
      boardTags: [board],
      resourceTypes: type == ResourceType.video ? ['Video'] : ['Document'],
      qualityScore: 0.85,
      accessType: AccessType.free,
      isHealthy: true,
      isVerified: true,
      topics: [subject],
    );
  }

  List<String> _getSubjectKeywords(String subject) {
    final map = {
      'mathematics': ['math', '0580', '0606'],
      'physics': ['physics', '0620', '5054'],
      'chemistry': ['chemistry', '0620', '5070'],
      'biology': ['biology', '0610', '5090'],
      'computer science': ['computer', '0478', '0450', 'cs', 'ict'],
      'economics': ['economics', '0455'],
    };
    return map[subject.toLowerCase()] ?? [subject.toLowerCase()];
  }
}

class EnhancedResource {
  final String id;
  final String title;
  final String? subtitle;
  final String url;
  final ResourceType type;
  final String source;
  final ResourceSourceType sourceType;
  final List<String> boardTags;
  final List<String> resourceTypes;
  final double qualityScore;
  final AccessType accessType;
  final bool isHealthy;
  final bool isVerified;
  final List<String> topics;

  const EnhancedResource({
    required this.id,
    required this.title,
    this.subtitle,
    required this.url,
    required this.type,
    required this.source,
    required this.sourceType,
    required this.boardTags,
    required this.resourceTypes,
    required this.qualityScore,
    required this.accessType,
    required this.isHealthy,
    required this.isVerified,
    required this.topics,
  });
}

enum ResourceType { video, notes, pastPaper, textbook, examPrep }

class ChapterResources {
  final List<EnhancedResource> resources;
  final int totalCount;
  final bool hasMore;

  const ChapterResources({
    required this.resources,
    required this.totalCount,
    required this.hasMore,
  });
}
