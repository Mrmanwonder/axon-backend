import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:developer' as dev;
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum CrawlerResourceType {
  syllabus,
  dateSheet,
  markScheme,
  topical,
  pastPaper,
  video,
  notes,
  formulaSheet,
  examinerReport,
}

class AxonCrawlTarget {
  final String name;
  final String baseUrl;
  final List<CrawlerResourceType> types;
  final List<String> subjects;
  final List<String> boardTags;
  final String? subjectCodePattern;

  const AxonCrawlTarget({
    required this.name,
    required this.baseUrl,
    required this.types,
    this.subjects = const [],
    this.boardTags = const [],
    this.subjectCodePattern,
  });
}

class AxonCrawlResult {
  final String title;
  final String url;
  final String fileName;
  final CrawlerResourceType type;
  final String? board;
  final String? subject;
  final int? year;
  final String? season;
  final String? paperType;
  final String? variant;
  final String hash;
  final DateTime crawledAt;

  AxonCrawlResult({
    required this.title,
    required this.url,
    required this.fileName,
    required this.type,
    this.board,
    this.subject,
    this.year,
    this.season,
    this.paperType,
    this.variant,
    required this.hash,
    required this.crawledAt,
  });

  Map<String, dynamic> toJson() => {
        'title': title,
        'url': url,
        'fileName': fileName,
        'type': type.name,
        'board': board,
        'subject': subject,
        'year': year,
        'season': season,
        'paperType': paperType,
        'variant': variant,
        'hash': hash,
        'crawledAt': crawledAt.toIso8601String(),
      };
}

class AxonCrawlerService {
  static final AxonCrawlerService _instance = AxonCrawlerService._internal();
  factory AxonCrawlerService() => _instance;
  AxonCrawlerService._internal();

  final http.Client _client = http.Client();
  final Set<String> _processedHashes = {};
  bool _isInitialized = false;
  String? _saveDirectory;

  static const List<AxonCrawlTarget> _defaultTargets = [
    // Cambridge IGCSE & O Level - All subjects and series
    AxonCrawlTarget(
      name: 'PapaCambridge IGCSE',
      baseUrl: 'https://pastpapers.papacambridge.com',
      types: [
        CrawlerResourceType.pastPaper,
        CrawlerResourceType.markScheme,
        CrawlerResourceType.examinerReport,
        CrawlerResourceType.syllabus,
      ],
      subjects: [],
      boardTags: ['cambridge', 'igcse', 'olevel'],
    ),
    AxonCrawlTarget(
      name: 'PapaCambridge AS/A Level',
      baseUrl: 'https://pastpapers.papacambridge.com',
      types: [
        CrawlerResourceType.pastPaper,
        CrawlerResourceType.markScheme,
        CrawlerResourceType.examinerReport,
        CrawlerResourceType.syllabus,
      ],
      subjects: [],
      boardTags: ['cambridge', 'alevel', 'as level'],
    ),
    AxonCrawlTarget(
      name: 'GCE Guide IGCSE',
      baseUrl: 'https://papers.gceguide.com',
      types: [
        CrawlerResourceType.pastPaper,
        CrawlerResourceType.markScheme,
      ],
      subjects: [],
      boardTags: ['cambridge', 'igcse'],
    ),
    AxonCrawlTarget(
      name: 'GCE Guide AS/A Level',
      baseUrl: 'https://papers.gceguide.com',
      types: [
        CrawlerResourceType.pastPaper,
        CrawlerResourceType.markScheme,
      ],
      subjects: [],
      boardTags: ['cambridge', 'alevel'],
    ),
    AxonCrawlTarget(
      name: 'Syllabus Cambridge',
      baseUrl: 'https://syllabus.papacambridge.com/syllabus/caie',
      types: [CrawlerResourceType.syllabus],
      subjects: [],
      boardTags: ['cambridge', 'igcse', 'alevel'],
    ),
    // ICT specific - 9618
    AxonCrawlTarget(
      name: 'ICT Papers 9618',
      baseUrl: 'https://pastpapers.papacambridge.com/IGCSE/ICT-9618',
      types: [CrawlerResourceType.pastPaper, CrawlerResourceType.markScheme],
      subjects: ['9618'],
      boardTags: ['cambridge', 'igcse'],
    ),
    // Computer Science - 9618 and 9608
    AxonCrawlTarget(
      name: 'Computer Science 9618',
      baseUrl:
          'https://pastpapers.papacambridge.com/AS%20Level/Computer%20Science-9618',
      types: [CrawlerResourceType.pastPaper, CrawlerResourceType.markScheme],
      subjects: ['9618'],
      boardTags: ['cambridge', 'as', 'alevel'],
    ),
    // Feb/March Series specific (s25, s24, s23)
    AxonCrawlTarget(
      name: 'March Series IGCSE',
      baseUrl: 'https://pastpapers.papacambridge.com/IGCSE',
      types: [CrawlerResourceType.pastPaper],
      subjects: [],
      boardTags: ['cambridge', 'igcse'],
      subjectCodePattern: 's', // March series
    ),
    // June Series (m25, m24)
    AxonCrawlTarget(
      name: 'June Series IGCSE',
      baseUrl: 'https://pastpapers.papacambridge.com/IGCSE',
      types: [CrawlerResourceType.pastPaper],
      subjects: [],
      boardTags: ['cambridge', 'igcse'],
      subjectCodePattern: 'm', // June series
    ),
    // November Series (w24, w23)
    AxonCrawlTarget(
      name: 'November Series IGCSE',
      baseUrl: 'https://pastpapers.papacambridge.com/IGCSE',
      types: [CrawlerResourceType.pastPaper],
      subjects: [],
      boardTags: ['cambridge', 'igcse'],
      subjectCodePattern: 'w', // November series
    ),
    // AS/A Level March (Feb/March) papers
    AxonCrawlTarget(
      name: 'March Series ASALevel',
      baseUrl: 'https://pastpapers.papacambridge.com/AS%20Level',
      types: [CrawlerResourceType.pastPaper],
      subjects: [],
      boardTags: ['cambridge', 'as', 'alevel'],
    ),
    // All subject codes
    AxonCrawlTarget(
      name: 'Subject Codes Collection',
      baseUrl: 'https://pastpapers.papacambridge.com',
      types: [CrawlerResourceType.pastPaper],
      subjects: [
        '9709', // Math
        '9702', // Physics
        '9701', // Chemistry
        '9700', // Biology
        '9618', // ICT
        '9608', // Computer Science
        '9708', // Economics
        '9609', // Business
        '9706', // Accounting
        '9389', // History
        '9696', // Geography
      ],
      boardTags: ['cambridge'],
    ),
  ];

  Future<void> initialize() async {
    if (_isInitialized) return;

    final dir = await getApplicationDocumentsDirectory();
    _saveDirectory = '${dir.path}/axon_crawl_cache';
    Directory(_saveDirectory!).createSync(recursive: true);

    await _loadProcessedHashes();
    _isInitialized = true;
    dev.log('AxonCrawler initialized at: $_saveDirectory');
  }

  Future<void> _loadProcessedHashes() async {
    final prefs = await SharedPreferences.getInstance();
    final hashes = prefs.getStringList('crawled_hashes') ?? [];
    _processedHashes.addAll(hashes);
  }

  Future<void> _saveProcessedHashes() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('crawled_hashes', _processedHashes.toList());
  }

  String get saveDirectory => _saveDirectory ?? '';

  /// Parse URL to extract resource metadata
  AxonCrawlResult? _parseResourceUrl(String url, CrawlerResourceType type) {
    try {
      final uri = Uri.parse(url);
      final pathSegments = uri.pathSegments;
      final fileName = pathSegments.last;

      // Extract year and season - supports all Cambridge series:
      // s = March (Feb/March), w = November (Oct/Nov), m = June (May/June)
      // Also handles: feb, march, jun, nov, oct, season codes
      final yearMatch = RegExp(r'([swm])(\d{2})').firstMatch(fileName);
      String? season;
      int? year;
      if (yearMatch != null) {
        season = yearMatch.group(1);
        year = 2000 + int.parse(yearMatch.group(2)!);
      }

      // Also try to extract from full filename patterns like "s24", "m25", "w24"
      final seasonYearMatch =
          RegExp(r'(feb|mar|jun|nov|oct)[_\s-]?(\d{2})', caseSensitive: false)
              .firstMatch(fileName);
      if (seasonYearMatch != null) {
        final monthStr = seasonYearMatch.group(1)!.toLowerCase();
        if (monthStr == 'feb' || monthStr == 'mar') {
          season = 's'; // March series
        } else if (monthStr == 'jun') {
          season = 'm'; // June series
        } else if (monthStr == 'oct' || monthStr == 'nov') {
          season = 'w'; // November series
        }
        year ??= 2000 + int.parse(seasonYearMatch.group(2)!);
      }

      // Extract subject code (4 digits) - supports all Cambridge codes including:
      // 9618 (ICT), 9608 (Computer Science - legacy), 9702 (Physics), etc.
      final codeMatch = RegExp(r'^(\d{4})').firstMatch(fileName);
      String? subjectCode;
      if (codeMatch != null) {
        subjectCode = codeMatch.group(1);
      }

      // Extract paper type (qp, ms, gt, er, in)
      String? paperType;
      if (fileName.contains('_qp_')) {
        paperType = 'qp';
      } else if (fileName.contains('_ms_')) {
        paperType = 'ms';
      } else if (fileName.contains('_gt_')) {
        paperType = 'gt';
      } else if (fileName.contains('_er_')) {
        paperType = 'er';
      } else if (fileName.contains('_in_')) {
        paperType = 'in';
      }

      // Calculate hash
      final hash = md5.convert(utf8.encode(url)).toString();

      return AxonCrawlResult(
        title: fileName,
        url: url,
        fileName: fileName,
        type: type,
        year: year,
        season: season,
        paperType: paperType,
        subject: subjectCode,
        hash: hash,
        crawledAt: DateTime.now(),
      );
    } catch (e) {
      return null;
    }
  }

  /// Scrape links from a target URL
  Future<List<String>> scrapeLinks(String targetUrl,
      {CrawlerResourceType? filterType}) async {
    try {
      final response = await _client.get(
        Uri.parse(targetUrl),
        headers: {
          'User-Agent': 'AxonBot/1.0 (Educational Indexer)',
          'Accept': 'text/html,application/xhtml+xml',
        },
      );

      if (response.statusCode != 200) {
        dev.log('Failed to fetch $targetUrl: ${response.statusCode}');
        return [];
      }

      final links = <String>[];
      final hrefPattern = RegExp(r'href="([^"]+)"', caseSensitive: false);
      final matches = hrefPattern.allMatches(response.body);

      for (final match in matches) {
        String href = match.group(1) ?? '';

        if (href.isEmpty) continue;

        // Make absolute URL
        if (href.startsWith('/')) {
          href = '${Uri.parse(targetUrl).origin}$href';
        } else if (!href.startsWith('http')) {
          href = '$targetUrl/$href';
        }

        // Filter by type
        bool isValid = false;
        final lowerHref = href.toLowerCase();

        if (filterType != null) {
          switch (filterType) {
            case CrawlerResourceType.syllabus:
              isValid = lowerHref.contains('syllabus') ||
                  lowerHref.contains('/syllabus/');
              break;
            case CrawlerResourceType.dateSheet:
              isValid = lowerHref.contains('timetable') ||
                  lowerHref.contains('date sheet');
              break;
            case CrawlerResourceType.markScheme:
              isValid = lowerHref.contains('_ms_') ||
                  lowerHref.contains('mark-scheme');
              break;
            case CrawlerResourceType.topical:
              isValid = lowerHref.contains('topical');
              break;
            case CrawlerResourceType.pastPaper:
              isValid = lowerHref.endsWith('.pdf') &&
                  (lowerHref.contains('_qp_') || lowerHref.contains('paper'));
              break;
            case CrawlerResourceType.examinerReport:
              isValid = lowerHref.contains('_er_') ||
                  lowerHref.contains('examiners report');
              break;
            default:
              isValid = lowerHref.endsWith('.pdf');
          }
        } else {
          isValid = lowerHref.endsWith('.pdf');
        }

        if (isValid) {
          links.add(href);
        }
      }

      return links.toSet().toList();
    } catch (e) {
      dev.log('Error scraping $targetUrl: $e');
      return [];
    }
  }

  /// Download a single resource
  Future<AxonCrawlResult?> downloadResource(String url,
      {CrawlerResourceType type = CrawlerResourceType.pastPaper}) async {
    if (_saveDirectory == null) await initialize();

    try {
      final response = await _client.get(
        Uri.parse(url),
        headers: {'User-Agent': 'AxonBot/1.0'},
      );

      if (response.statusCode != 200) {
        dev.log('Failed to download $url: ${response.statusCode}');
        return null;
      }

      // Calculate hash to check duplicates
      final hash = md5.convert(response.bodyBytes).toString();
      if (_processedHashes.contains(hash)) {
        dev.log('Skipped (already processed): $url');
        return null;
      }

      _processedHashes.add(hash);

      // Parse metadata
      final result = _parseResourceUrl(url, type);
      if (result == null) return null;

      // Save file
      final filePath = '$_saveDirectory/${result.fileName}';
      final file = File(filePath);
      await file.writeAsBytes(response.bodyBytes);

      dev.log('Downloaded: ${result.fileName}');
      return result;
    } catch (e) {
      dev.log('Error downloading $url: $e');
      return null;
    }
  }

  /// Crawl a specific target with batching
  Future<List<AxonCrawlResult>> crawlTarget(
    AxonCrawlTarget target, {
    int batchSize = 5,
    Function(String)? onProgress,
  }) async {
    final results = <AxonCrawlResult>[];

    for (final type in target.types) {
      onProgress?.call('Crawling ${target.name} for ${type.name}...');

      // Scrape links
      final links = await scrapeLinks(target.baseUrl, filterType: type);
      dev.log('Found ${links.length} ${type.name} resources');

      // Download in batches
      for (var i = 0; i < links.length; i += batchSize) {
        final batch = links.skip(i).take(batchSize);
        final batchResults = await Future.wait(
          batch.map((url) => downloadResource(url, type: type)),
        );

        results.addAll(batchResults.whereType<AxonCrawlResult>());

        // Small delay to avoid rate limiting
        await Future.delayed(const Duration(milliseconds: 500));
      }
    }

    await _saveProcessedHashes();
    return results;
  }

  /// Run full crawl on all targets
  Future<List<AxonCrawlResult>> runFullCrawl({
    List<AxonCrawlTarget>? targets,
    int batchSize = 5,
    Function(String)? onProgress,
    Function(double)? onProgressPercent,
  }) async {
    if (_saveDirectory == null) await initialize();

    final targetsToUse = targets ?? _defaultTargets;
    final allResults = <AxonCrawlResult>[];

    for (var i = 0; i < targetsToUse.length; i++) {
      onProgressPercent?.call(i / targetsToUse.length);

      final results = await crawlTarget(
        targetsToUse[i],
        batchSize: batchSize,
        onProgress: onProgress,
      );

      allResults.addAll(results);
    }

    onProgressPercent?.call(1.0);
    return allResults;
  }

  /// Get crawled files for a specific type
  Future<List<File>> getCrawledFiles({CrawlerResourceType? type}) async {
    if (_saveDirectory == null) return [];

    final dir = Directory(_saveDirectory!);
    if (!dir.existsSync()) return [];

    final files = <File>[];
    await for (final entity in dir.list()) {
      if (entity is File && entity.path.endsWith('.pdf')) {
        if (type != null) {
          // Filter by type based on filename
          final name = entity.path.toLowerCase();
          bool matches = false;
          switch (type) {
            case CrawlerResourceType.markScheme:
              matches = name.contains('_ms_');
              break;
            case CrawlerResourceType.syllabus:
              matches = name.contains('syllabus');
              break;
            case CrawlerResourceType.dateSheet:
              matches = name.contains('timetable');
              break;
            case CrawlerResourceType.examinerReport:
              matches = name.contains('_er_');
              break;
            default:
              matches = true;
          }
          if (matches) files.add(entity);
        } else {
          files.add(entity);
        }
      }
    }

    return files;
  }

  /// Clear all crawled files
  Future<void> clearCache() async {
    if (_saveDirectory == null) return;

    final dir = Directory(_saveDirectory!);
    if (dir.existsSync()) {
      await dir.delete(recursive: true);
      dir.createSync(recursive: true);
    }

    _processedHashes.clear();
    await _saveProcessedHashes();
  }

  /// Get cache statistics
  Future<Map<String, dynamic>> getCacheStats() async {
    final files = await getCrawledFiles();
    final totalSize = await Future.wait(
      files.map((f) => f.length()),
    ).then((lengths) => lengths.fold<int>(0, (a, b) => a + b));

    return {
      'totalFiles': files.length,
      'totalSizeMB': (totalSize / 1024 / 1024).toStringAsFixed(2),
      'processedHashes': _processedHashes.length,
    };
  }

  void dispose() {
    _client.close();
  }
}
