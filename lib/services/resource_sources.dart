enum ResourceSourceType {
  khanAcademy,
  youtube,
  ncert,
  byjus,
  vedantu,
  unpkg,
  pastPaper,
  textbook,
  coursera,
  udemy,
  custom,
}

enum AccessType {
  free,
  freemium,
  paid,
  subscription,
}

enum SourceQuality {
  low,
  medium,
  high,
  top,
}

enum AxonBoard {
  cbse,
  icse,
  cambridge,
  ibDP,
  state,
  jee,
  neet,
}

String normalizeBoard(String board) {
  final lower = board.toLowerCase();
  if (lower.contains('ocr')) return 'ocr_a_level';
  if (lower.contains('edexcel') && lower.contains('igcse')) {
    return 'edexcel_igcse';
  }
  if (lower.contains('edexcel')) return 'edexcel_a_level';
  if (lower.contains('ib') || lower.contains('dp')) return 'ibdp';
  if (lower.contains('o level') || lower.contains('olevel')) {
    return 'caie_o_level';
  }
  if (lower.contains('a level') || lower.contains('alevel') || lower.contains('a-level')) {
    return 'caie_a_level';
  }
  if (lower.contains('cambridge') || lower.contains('cie') || lower.contains('caie') || lower.contains('igcse')) {
    return 'caie_igcse';
  }
  if (lower.contains('cbse')) return 'cbse';
  if (lower.contains('icse')) return 'icse';
  if (lower.contains('jee')) return 'jee';
  if (lower.contains('neet')) return 'neet';
  return 'caie_igcse';
}

class BoardConfig {
  final String board;
  final List<String> subjects;
  final Map<String, List<String>> chapterMap;

  const BoardConfig({
    required this.board,
    required this.subjects,
    required this.chapterMap,
  });
}

class ResourceSourcesServiceProvider {
  static final ResourceSourcesServiceProvider _instance =
      ResourceSourcesServiceProvider._internal();
  factory ResourceSourcesServiceProvider() => _instance;
  ResourceSourcesServiceProvider._internal();

  List<ResourceSource> getSourcesForSubject(String board, String subject) {
    return ResourceSource.getDefaults();
  }

  List<ResourceSource> getSourcesForBoard(String board) {
    return ResourceSource.getDefaults();
  }

  BoardConfig? getConfig(String board) {
    switch (normalizeBoard(board)) {
      case 'caie_igcse':
      case 'caie_o_level':
        return BoardConfig(
          board: normalizeBoard(board),
          subjects: [
            'Mathematics',
            'Physics',
            'Chemistry',
            'Biology',
            'Computer Science',
            'Economics',
            'History',
            'Geography',
            'English',
            'French',
            'Spanish',
          ],
          chapterMap: {},
        );
      case 'caie_a_level':
      case 'ocr_a_level':
      case 'edexcel_a_level':
        return BoardConfig(
          board: normalizeBoard(board),
          subjects: [
            'Mathematics',
            'Further Mathematics',
            'Physics',
            'Chemistry',
            'Biology',
            'Computer Science',
            'Economics',
            'History',
            'Geography',
            'English',
          ],
          chapterMap: {},
        );
      case 'edexcel_igcse':
        return BoardConfig(
          board: 'edexcel_igcse',
          subjects: [
            'Mathematics',
            'Physics',
            'Chemistry',
            'Biology',
            'Computer Science',
            'Economics',
            'History',
            'Geography',
            'English',
          ],
          chapterMap: {},
        );
      case 'ibdp':
        return BoardConfig(
          board: 'ibdp',
          subjects: [
            'Mathematics',
            'Physics',
            'Chemistry',
            'Biology',
            'Computer Science',
            'Economics',
            'History',
            'Geography',
            'English',
          ],
          chapterMap: {},
        );
      default:
        return BoardConfig(
          board: board,
          subjects: [
            'Math',
            'Science',
            'English',
            'Physics',
            'Chemistry',
            'Biology'
          ],
          chapterMap: {},
        );
    }
  }
}

final resourceSourcesServiceProvider = ResourceSourcesServiceProvider();

class ResourceSource {
  final String id;
  final String name;
  final String url;
  final ResourceSourceType sourceType;
  final AccessType accessType;
  final double qualityScore;
  final List<String> boardTags;
  final List<String> supportedSubjects;
  final bool isHealthy;
  final List<String> resourceTypes;
  final SourceQuality quality;

  const ResourceSource({
    required this.id,
    required this.name,
    required this.url,
    required this.sourceType,
    required this.accessType,
    this.qualityScore = 0.5,
    this.boardTags = const [],
    this.supportedSubjects = const [],
    this.isHealthy = true,
    this.resourceTypes = const ['video', 'notes'],
    this.quality = SourceQuality.medium,
  });

  ResourceSourceType get type => sourceType;

  String buildChapterUrl(String subject, String chapter) {
    print('ResourceSource.buildChapterUrl($subject, $chapter) called');
    final searchTerm = '$subject $chapter'.replaceAll(' ', '+');
    return '$url/search?query=$searchTerm';
  }

  static List<ResourceSource> getDefaults() {
    print('ResourceSource.getDefaults() called');
    return [
      const ResourceSource(
        id: 'khanacademy',
        name: 'Khan Academy',
        url: 'https://www.khanacademy.org',
        sourceType: ResourceSourceType.khanAcademy,
        accessType: AccessType.free,
        qualityScore: 0.9,
        quality: SourceQuality.high,
        boardTags: ['CBSE', 'ICSE', 'State'],
        supportedSubjects: ['Math', 'Science', 'English'],
        resourceTypes: ['video', 'notes', 'practice'],
      ),
      const ResourceSource(
        id: 'youtube',
        name: 'YouTube',
        url: 'https://www.youtube.com',
        sourceType: ResourceSourceType.youtube,
        accessType: AccessType.free,
        qualityScore: 0.6,
        quality: SourceQuality.medium,
        boardTags: ['CBSE', 'ICSE', 'State', 'JEE', 'NEET'],
        supportedSubjects: [
          'Math',
          'Science',
          'English',
          'Physics',
          'Chemistry',
          'Biology'
        ],
        resourceTypes: ['video'],
      ),
      const ResourceSource(
        id: 'ncert',
        name: 'NCERT',
        url: 'https://ncert.nic.in',
        sourceType: ResourceSourceType.ncert,
        accessType: AccessType.free,
        qualityScore: 0.95,
        quality: SourceQuality.top,
        boardTags: ['CBSE'],
        supportedSubjects: [
          'Math',
          'Science',
          'English',
          'Physics',
          'Chemistry',
          'Biology'
        ],
        resourceTypes: ['textbook', 'notes'],
      ),
    ];
  }
}
