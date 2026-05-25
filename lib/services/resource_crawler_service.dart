import 'dart:async';
import 'dart:convert';
import 'dart:developer' as dev;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:crypto/crypto.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'backend_health_service.dart';
import 'firestore_service.dart';
import 'resource_quality_service.dart';
import 'resource_recommendation_service.dart';
import 'security_service.dart';

class ChapterResource {
  final String title;
  final String url;
  final String type;
  final bool verified;
  final bool healthy;
  final String source;
  final String sourceType;
  final List<String> boardTags;
  final double qualityScore;

  const ChapterResource({
    required this.title,
    required this.url,
    required this.type,
    this.verified = false,
    this.healthy = true,
    this.source = '',
    this.sourceType = '',
    this.boardTags = const [],
    this.qualityScore = 0.0,
  });

  Map<String, dynamic> toJson() => {
        'title': title,
        'url': url,
        'type': type,
        'verified': verified,
        'healthy': healthy,
        'source': source,
        'sourceType': sourceType,
        'boardTags': boardTags,
        'qualityScore': qualityScore,
      };

  factory ChapterResource.fromJson(Map<String, dynamic> json) {
    return ChapterResource(
      title: json['title']?.toString() ?? '',
      url: json['url']?.toString() ?? '',
      type: json['type']?.toString() ?? '',
      verified: json['verified'] == true,
      healthy: json['healthy'] != false,
      source: json['source']?.toString() ?? '',
      sourceType: json['sourceType']?.toString() ?? '',
      boardTags:
          (json['boardTags'] as List?)?.map((e) => e.toString()).toList() ??
              const [],
      qualityScore: (json['qualityScore'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class ResourceCrawlerService {
  static final ResourceCrawlerService _instance =
      ResourceCrawlerService._internal();
  factory ResourceCrawlerService() => _instance;
  ResourceCrawlerService._internal();

  final FirebaseFirestore _db = AxonFirestore.instance;
  final SecurityService _security = SecurityService();

  String _resourceCacheKey(String board, String subject, String chapter) =>
      'res_${_sanitizeString(board, 20)}_${_sanitizeString(subject, 20)}_${_sanitizeString(chapter, 20)}'
          .replaceAll(' ', '_')
          .toLowerCase();

  Future<void> initialize() async {
    // Initialize crawler service
  }

  bool _isValidUid(String uid) {
    if (uid.isEmpty || uid.length > 128) return false;
    return RegExp(r'^[A-Za-z0-9_-]+$').hasMatch(uid);
  }

  Future<List<ChapterResource>> crawlSubject({
    required String uid,
    required String board,
    required String subject,
  }) async {
    final available =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return [];
    if (uid.isEmpty || !_isValidUid(uid)) return [];
    if (subject.isEmpty || subject.length > 50) return [];

    final safeBoard = _sanitizeString(board, 50);
    final safeSubject = _sanitizeString(subject, 50);

    final catalog = await _loadStudyCatalog();
    final chapters = catalog[safeSubject] ?? <String>[];

    if (chapters.isEmpty) {
      chapters.addAll(_inferChaptersForSubject(safeBoard, safeSubject));
    }

    final allResources = <ChapterResource>[];
    for (int i = 0; i < chapters.length; i += 3) {
      final batchEnd = (i + 3).clamp(0, chapters.length);
      final batch = chapters.sublist(i, batchEnd);
      await Future.wait(batch.map((chapter) async {
        final resources = await crawlChapter(
            board: safeBoard, subject: safeSubject, chapter: chapter);
        await saveResources(
            uid: uid,
            board: safeBoard,
            subject: safeSubject,
            chapter: chapter,
            resources: resources);
        allResources.addAll(resources);
      }));
    }
    return allResources;
  }

  Future<List<ChapterResource>> crawlChapter({
    required String board,
    required String subject,
    required String chapter,
  }) async {
    try {
      dev.log('Crawler: fetching resources for $subject - $chapter');

      final recService = ResourceRecommendationService();
      final qualityService = ResourceQualityService();

      final chapterRecs = await recService.getResourcesForChapter(
        board: board,
        subject: subject,
        chapter: chapter,
      );

      final qualityScores = await qualityService.getQualityScores();

      return chapterRecs.resources.map((r) {
        return ChapterResource(
          title: r.title,
          url: r.url,
          type: r.type.name,
          verified: r.isVerified,
          healthy: r.isHealthy,
          source: r.source,
          sourceType: r.sourceType.name,
          boardTags: r.boardTags,
          qualityScore: qualityScores[r.id] ?? r.qualityScore,
        );
      }).toList();
    } catch (_) {
      return const [];
    }
  }

  String _sanitizeString(String input, int maxLength) {
    var cleaned = input
        .replaceAll(RegExp(r'[<>{}\"$]'), '')
        .replaceAll(RegExp(r'[\(\)]'), '')
        .replaceAll(RegExp(r'[\x00-\x1F\x7F]'), '')
        .trim();
    cleaned = cleaned.length > 100 ? cleaned.substring(0, 100) : cleaned;
    return cleaned.substring(0, cleaned.length.clamp(0, maxLength));
  }

  Future<void> _crawlWithRetry({
    required String uid,
    required String board,
    required String subject,
    required String chapter,
    int maxRetries = 5,
  }) async {
    var retryCount = 0;
    while (retryCount < maxRetries) {
      try {
        final resources = await crawlChapter(
            board: board, subject: subject, chapter: chapter);
        await saveResources(
            uid: uid,
            board: board,
            subject: subject,
            chapter: chapter,
            resources: resources);
        return;
      } catch (e) {
        retryCount++;
        if (retryCount >= maxRetries) rethrow;
        await Future.delayed(Duration(seconds: 1 << retryCount));
      }
    }
  }

  Future<Map<String, List<String>>> _loadStudyCatalog() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('study_catalog');
    if (raw == null || raw.isEmpty) return {};
    try {
      return Map<String, List<String>>.from(
        (jsonDecode(raw) as Map<String, dynamic>).map(
          (k, v) => MapEntry(k, List<String>.from(v)),
        ),
      );
    } catch (_) {
      return {};
    }
  }

  List<String> _inferChaptersForSubject(String board, String subject) {
    final defaultChapters = <String, List<String>>{
      'Physics': [
        'Kinematics',
        'Forces',
        'Energy',
        'Waves',
        'Electricity',
        'Magnetism',
        'Atomic Physics'
      ],
      'Chemistry': [
        'Atomic Structure',
        'Bonding',
        'Stoichiometry',
        'Reactions',
        'Organic',
        'Electrochemistry'
      ],
      'Biology': [
        'Cell Biology',
        'Transport',
        'Biomolecules',
        'Ecology',
        'Genetics',
        'Evolution'
      ],
      'Mathematics': [
        'Algebra',
        'Functions',
        'Sequences',
        'Trigonometry',
        'Calculus',
        'Probability',
        'Statistics'
      ],
    };

    for (final entry in defaultChapters.entries) {
      if (subject.toLowerCase().contains(entry.key.toLowerCase())) {
        return entry.value;
      }
    }
    return ['Chapter 1', 'Chapter 2', 'Chapter 3', 'Chapter 4', 'Chapter 5'];
  }

  Future<void> saveResources({
    required String uid,
    required String board,
    required String subject,
    required String chapter,
    required List<ChapterResource> resources,
  }) async {
    if (!_security.isOwner(uid)) {
      throw SecurityException(
          'Access denied: Cannot save resources for another user');
    }

    if (resources.isEmpty) return;
    if (uid.isEmpty || !_isValidUid(uid)) return;

    final contentHash = sha256
        .convert(utf8.encode(
            '$board$subject$chapter${resources.map((r) => r.url).join()}'))
        .toString()
        .substring(0, 20);
    final docId = contentHash;

    await _db
        .collection(AxonCollections.usersPrivate)
        .doc(uid)
        .collection('resources')
        .doc(docId)
        .set({
      'ownerId': uid,
      'board': _sanitizeString(board, 50),
      'subject': _sanitizeString(subject, 50),
      'chapter': _sanitizeString(chapter, 50),
      'content_hash': contentHash,
      'resources': resources.map((r) => r.toJson()).toList(),
      'crawled_at': FieldValue.serverTimestamp(),
      'updated_at': FieldValue.serverTimestamp(),
    }, SetOptions(merge: true));

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _resourceCacheKey(board, subject, chapter),
      jsonEncode(resources.map((r) => r.toJson()).toList()),
    );
  }

  Future<List<ChapterResource>> loadResources({
    required String board,
    required String subject,
    required String chapter,
    required String uid,
  }) async {
    if (!_security.isOwner(uid)) {
      throw SecurityException(
          'Access denied: Cannot load another user\'s resources');
    }

    if (uid.isEmpty || !_isValidUid(uid)) return [];

    final docId = _resourceCacheKey(board, subject, chapter);

    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(docId);
    if (raw != null && raw.isNotEmpty) {
      try {
        return (jsonDecode(raw) as List)
            .whereType<Map>()
            .map((e) => ChapterResource.fromJson(Map<String, dynamic>.from(e)))
            .toList();
      } catch (_) {}
    }

    try {
      final snapshot = await _db
          .collection(AxonCollections.usersPrivate)
          .doc(uid)
          .collection('resources')
          .where('board', isEqualTo: board)
          .where('subject', isEqualTo: subject)
          .where('chapter', isEqualTo: chapter)
          .limit(1)
          .get();

      if (snapshot.docs.isEmpty) {
        final crawled = await crawlChapter(
          board: board,
          subject: subject,
          chapter: chapter,
        );
        if (crawled.isNotEmpty) {
          await saveResources(
            uid: uid,
            board: board,
            subject: subject,
            chapter: chapter,
            resources: crawled,
          );
          return crawled;
        }
        return [];
      }

      final data = snapshot.docs.first.data();
      final resources = (data['resources'] as List?)
              ?.map(
                  (e) => ChapterResource.fromJson(Map<String, dynamic>.from(e)))
              .toList() ??
          [];

      await prefs.setString(
        docId,
        jsonEncode(resources.map((r) => r.toJson()).toList()),
      );

      return resources;
    } catch (_) {
      return [];
    }
  }

  Stream<List<ChapterResource>> watchResources({
    required String board,
    required String subject,
    required String chapter,
    required String uid,
  }) async* {
    if (!_security.isOwner(uid)) {
      yield [];
      return;
    }

    yield await loadResources(
        board: board, subject: subject, chapter: chapter, uid: uid);
  }

  Future<void> triggerCrawlForChapter({
    required String uid,
    required String board,
    required String subject,
    required String chapter,
  }) async {
    if (!_security.isOwner(uid)) {
      throw SecurityException(
          'Access denied: Cannot trigger crawl for another user');
    }

    await _crawlWithRetry(
        uid: uid, board: board, subject: subject, chapter: chapter);
  }

  Future<void> scheduleNightlyCrawl({
    required String uid,
    required String board,
  }) async {
    if (!_security.isOwner(uid)) {
      throw SecurityException(
          'Access denied: Cannot schedule crawl for another user');
    }

    if (uid.isEmpty || !_isValidUid(uid)) return;

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('nightly_crawl_uid', uid);
    await prefs.setString('nightly_crawl_board', board);
    await prefs.setInt(
        'nightly_crawl_scheduled', DateTime.now().millisecondsSinceEpoch);
  }

  Future<void> scheduleInitialCrawl({
    required String uid,
    required String board,
    required List<String> subjects,
  }) async {
    if (!_security.isOwner(uid)) {
      throw SecurityException(
          'Access denied: Cannot schedule crawl for another user');
    }

    for (final subject in subjects) {
      await _crawlWithRetry(
        uid: uid,
        board: board,
        subject: subject,
        chapter: '',
      );
    }
  }
}
