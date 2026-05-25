import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

class PastPaperService {
  static const String _offlineModeKey = 'offline_mode_enabled';
  static const String _papersManifestKey = 'past_papers_manifest';

  static final PastPaperService _instance = PastPaperService._internal();
  factory PastPaperService() => _instance;
  PastPaperService._internal();

  // Subject to exam code mapping for Cambridge/O Level
  static const Map<String, List<String>> subjectExamCodes = {
    'Accounting': ['9706', '0452'],
    'Mathematics': ['9709', '0580', '4037'],
    'Further Mathematics': ['9231'],
    'Physics': ['9702', '0625'],
    'Chemistry': ['9701', '0620'],
    'Biology': ['9700', '0610'],
    'Computer Science': ['9618', '9608', '0478'],
    'Economics': ['9708', '0455'],
    'Geography': ['0460', '9696'],
    'History': ['9389', '0470'],
    'English': ['0500', '0510'],
    'French': ['0520', '9645'],
    'Spanish': ['0580', '9665'],
    'Chinese': ['0523', '9715'],
    'Art and Design': ['0400', '9479'],
    'Business Studies': ['9609', '9707', '0450'],
  };

  static const Map<String, String> _subjectAliases = {
    'math': 'Mathematics',
    'maths': 'Mathematics',
    'pure mathematics': 'Mathematics',
    'further maths': 'Further Mathematics',
    'fm': 'Further Mathematics',
    'cs': 'Computer Science',
    'computer studies': 'Computer Science',
    'business': 'Business Studies',
    'business studies': 'Business Studies',
    'english language': 'English',
  };

  static String normalizeSubjectName(String subject) =>
      subject.trim().replaceAll(RegExp(r'\s+'), ' ');

  static List<String> examCodesForSubject(String subject) {
    final normalized = normalizeSubjectName(subject).toLowerCase();
    final directCode = subject.trim().toUpperCase();
    if (RegExp(r'\d').hasMatch(directCode) &&
        RegExp(r'^[A-Z0-9]{4,}$').hasMatch(directCode)) {
      return [directCode];
    }

    final alias = _subjectAliases[normalized];
    if (alias != null) {
      return List<String>.from(subjectExamCodes[alias] ?? const []);
    }

    for (final entry in subjectExamCodes.entries) {
      if (normalizeSubjectName(entry.key).toLowerCase() == normalized) {
        return List<String>.from(entry.value);
      }
    }
    return const [];
  }

  // ─────────────────────────────────────────────────────────────────
  // Offline Mode
  // ─────────────────────────────────────────────────────────────────

  Future<bool> isOfflineModeEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_offlineModeKey) ?? false;
  }

  Future<void> setOfflineMode(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_offlineModeKey, enabled);
  }

  // ─────────────────────────────────────────────────────────────────
  // Paper Management
  // ─────────────────────────────────────────────────────────────────

  Future<String> get _papersDirectory async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}/past_papers');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir.path;
  }

  Future<List<String>> getUserSubjects() async {
    final prefs = await SharedPreferences.getInstance();
    final subjectsList = prefs.getStringList('userSubjects');
    if (subjectsList != null) {
      return subjectsList;
    }
    return [];
  }

  Future<void> saveUserSubjects(List<String> subjects) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('userSubjects', subjects);
  }

  Future<List<PastPaperInfo>> getPapersForSubject(String subject) async {
    final papers = <PastPaperInfo>[];
    final path = await _papersDirectory;
    final normalizedSubject = normalizeSubjectName(subject);
    final subjectDir = Directory('$path/$normalizedSubject');
    final legacyRootDir = Directory(path);
    final examCodes = examCodesForSubject(normalizedSubject);
    final scannedPaths = <String>{};

    Future<void> scanDirectory(Directory dir,
        {bool filterBySubject = false}) async {
      if (!await dir.exists()) return;

      await for (final entity in dir.list(recursive: true)) {
        if (entity is! File || !entity.path.toLowerCase().endsWith('.pdf')) {
          continue;
        }
        final canonicalPath = entity.path;
        if (!scannedPaths.add(canonicalPath)) continue;

        final fileName = entity.uri.pathSegments.isEmpty
            ? entity.path.split(Platform.pathSeparator).last
            : entity.uri.pathSegments.last;
        final info = _parseFileName(fileName, canonicalPath);
        if (info == null) continue;
        if (filterBySubject &&
            !examCodes.contains(info.examCode) &&
            normalizeSubjectName(info.subject).toLowerCase() !=
                normalizedSubject.toLowerCase()) {
          continue;
        }
        papers.add(info);
      }
    }

    await scanDirectory(subjectDir);
    if (papers.isEmpty) {
      await scanDirectory(legacyRootDir, filterBySubject: true);
    }

    papers.sort((a, b) => b.year.compareTo(a.year));
    return papers;
  }

  PastPaperInfo? _parseFileName(String fileName, String path) {
    // Format: 9700_s23_qp_11.pdf
    // 9700 = exam code
    // s23 = season (s23 = Summer 2023, w23 = Winter 2023)
    // qp = question paper (ms = mark scheme, gt = grade thresholds, er = examiner report)
    // 11 = variant (11, 12, 13, 21, 22, 23, etc.)

    try {
      final match = RegExp(r'^([a-z0-9]+)_([smw])(\d{2})_([a-z]+)_(\d+)\.pdf$')
          .firstMatch(fileName.toLowerCase());
      if (match == null) return null;

      final examCode = match.group(1)!.toUpperCase();
      final season = switch (match.group(2)!) {
        's' => 'Summer',
        'm' => 'March',
        _ => 'Winter',
      };
      final year = 2000 + int.parse(match.group(3)!);
      final paperType = _getPaperType(match.group(4)!);
      final variant = match.group(5)!;

      return PastPaperInfo(
        fileName: fileName,
        filePath: path,
        examCode: examCode,
        year: year,
        season: season,
        paperType: paperType,
        variant: variant,
        subject: _getSubjectFromCode(examCode),
      );
    } catch (e) {
      return null;
    }
  }

  String _getPaperType(String code) {
    switch (code) {
      case 'qp':
        return 'Question Paper';
      case 'ms':
        return 'Mark Scheme';
      case 'gt':
        return 'Grade Thresholds';
      case 'er':
        return 'Examiner Report';
      default:
        return code;
    }
  }

  String _getSubjectFromCode(String examCode) {
    for (final entry in subjectExamCodes.entries) {
      if (entry.value.contains(examCode)) {
        return entry.key;
      }
    }
    return 'Unknown';
  }

  Future<int> getDownloadedPapersCount() async {
    final path = await _papersDirectory;
    final dir = Directory(path);
    if (!await dir.exists()) return 0;

    int count = 0;
    await for (final entity in dir.list(recursive: true)) {
      if (entity is File && entity.path.toLowerCase().endsWith('.pdf')) {
        count++;
      }
    }
    return count;
  }

  Future<void> deleteAllPapers() async {
    final path = await _papersDirectory;
    final dir = Directory(path);
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }

  // Get papers that are question papers only
  Future<List<PastPaperInfo>> getQuestionPapersOnly(String subject) async {
    final allPapers = await getPapersForSubject(subject);
    return allPapers.where((p) => p.paperType == 'Question Paper').toList();
  }

  // Save download manifest
  Future<void> saveDownloadManifest(Map<String, List<String>> manifest) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_papersManifestKey, jsonEncode(manifest));
  }

  Future<Map<String, List<String>>> getDownloadManifest() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_papersManifestKey);
    if (raw != null) {
      try {
        return Map<String, List<String>>.from(
          (jsonDecode(raw) as Map)
              .map((k, v) => MapEntry(k, List<String>.from(v))),
        );
      } catch (_) {
        return {};
      }
    }
    return {};
  }
}

class PastPaperInfo {
  final String fileName;
  final String filePath;
  final String examCode;
  final int year;
  final String season;
  final String paperType;
  final String variant;
  final String subject;

  PastPaperInfo({
    required this.fileName,
    required this.filePath,
    required this.examCode,
    required this.year,
    required this.season,
    required this.paperType,
    required this.variant,
    required this.subject,
  });

  bool get isQuestionPaper => paperType == 'Question Paper';
  bool get isMarkScheme => paperType == 'Mark Scheme';

  String get displayName => '$examCode $year $season $paperType $variant';
}
