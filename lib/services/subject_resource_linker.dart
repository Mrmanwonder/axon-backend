// lib/services/subject_resource_linker.dart
import 'curriculum_catalog_service.dart';
import 'study_resources_service.dart';

class ResourceLink {
  final String type;
  final String name;
  final String url;
  final String? description;

  const ResourceLink({
    required this.type,
    required this.name,
    required this.url,
    this.description,
  });
}

class SubjectResourceLink {
  final String code;
  final String name;
  final String level;
  final List<ResourceLink> pastPapers;
  final List<ResourceLink> mocks;
  final List<ResourceLink> flashcards;
  final List<ResourceLink> notes;
  final List<ResourceLink> videos;
  final List<ResourceLink> otherResources;

  const SubjectResourceLink({
    required this.code,
    required this.name,
    required this.level,
    this.pastPapers = const [],
    this.mocks = const [],
    this.flashcards = const [],
    this.notes = const [],
    this.videos = const [],
    this.otherResources = const [],
  });

  List<ResourceLink> get allResources => [
        ...pastPapers,
        ...mocks,
        ...flashcards,
        ...notes,
        ...videos,
        ...otherResources,
      ];
}

class SubjectResourceLinkerService {
  static final SubjectResourceLinkerService _instance =
      SubjectResourceLinkerService._internal();
  factory SubjectResourceLinkerService() => _instance;
  SubjectResourceLinkerService._internal();

  final Map<String, SubjectResourceLink> _links = {};
  bool _initialized = false;

  Future<void> initialize() async {
    if (_initialized) return;
    await _buildLinks();
    _initialized = true;
  }

  Future<void> _buildLinks() async {
    // Get all subjects from curriculum
    final boards = await CurriculumCatalogService.instance.loadBoards();

    for (final board in boards) {
      for (final subject in board.subjects) {
        if (subject.code.isEmpty) continue;

        // Get resource data from StudyResourcesService
        final resource = StudyResourcesService().getResources(subject.code);

        final links = SubjectResourceLink(
          code: subject.code,
          name: subject.name,
          level: board.label,
          pastPapers: _convertToResourceLinks(
              resource?.pastPaperUrls ?? [], 'Past Papers'),
          mocks: _convertToResourceLinks(
              resource?.examPrepUrls ?? [], 'Mock Exams'),
          flashcards: _convertToResourceLinks(
              resource?.flashcardsUrls ?? [], 'Flashcards'),
          notes: _convertToResourceLinks(resource?.notesUrls ?? [], 'Notes'),
          videos: _convertToResourceLinks(resource?.videoUrls ?? [], 'Videos'),
          otherResources:
              _convertToResourceLinks(resource?.otherUrls ?? [], 'Other'),
        );

        _links[subject.code] = links;
      }
    }
  }

  List<ResourceLink> _convertToResourceLinks(List<String> urls, String type) {
    return urls.map((url) {
      final name = _extractNameFromUrl(url);
      return ResourceLink(type: type, name: name, url: url);
    }).toList();
  }

  String _extractNameFromUrl(String url) {
    final uri = Uri.tryParse(url);
    if (uri == null) return url;

    final host = uri.host.toLowerCase();
    if (host.contains('papacambridge')) return 'PapaCambridge';
    if (host.contains('savemyexams')) return 'SaveMyExams';
    if (host.contains('physicsandmathstutor')) return 'PMT';
    if (host.contains('znotes')) return 'ZNotes';
    if (host.contains('quizlet')) return 'Quizlet';
    if (host.contains('cognito')) return 'Cognito';

    return host.isNotEmpty ? host.split('.').last : url;
  }

  SubjectResourceLink? getLinksForCode(String code) {
    return _links[code];
  }

  SubjectResourceLink? getLinksForSubject(String subjectName, String boardId) {
    for (final entry in _links.entries) {
      final board = _links[entry.key];
      if (board != null &&
          board.name.toLowerCase() == subjectName.toLowerCase()) {
        return entry.value;
      }
    }
    return null;
  }

  List<SubjectResourceLink> getAllLinks() {
    return _links.values.toList();
  }

  List<SubjectResourceLink> getLinksForLevel(String level) {
    return _links.values
        .where((l) => l.level.toLowerCase().contains(level.toLowerCase()))
        .toList();
  }

  List<SubjectResourceLink> searchLinks(String query) {
    final q = query.toLowerCase();
    return _links.values
        .where((l) => l.name.toLowerCase().contains(q) || l.code.contains(q))
        .toList();
  }

  Future<List<String>> getAllSubjectCodes() async {
    if (!_initialized) await initialize();
    return _links.keys.toList()..sort();
  }

  Map<String, int> getResourceCounts() {
    return {
      'totalSubjects': _links.length,
      'pastPapers':
          _links.values.fold(0, (sum, l) => sum + l.pastPapers.length),
      'mocks': _links.values.fold(0, (sum, l) => sum + l.mocks.length),
      'flashcards':
          _links.values.fold(0, (sum, l) => sum + l.flashcards.length),
      'notes': _links.values.fold(0, (sum, l) => sum + l.notes.length),
      'videos': _links.values.fold(0, (sum, l) => sum + l.videos.length),
    };
  }
}
