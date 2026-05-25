import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:workmanager/workmanager.dart';

import 'axon_crawler_service.dart';
import 'datesheet_parser.dart';
import 'exam_data_service.dart';
import 'grok_service.dart';

// ═══════════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE SUBJECT DATABASE - CAIE ONLY
// ═══════════════════════════════════════════════════════════════════════════════

Map<String, Map<String, List<Map<String, dynamic>>>> get allSubjects => {
      'CAIE': {
        'IGCSE': [
          {'code': '0610', 'name': 'Biology'},
          {'code': '0620', 'name': 'Chemistry'},
          {'code': '0625', 'name': 'Physics'},
          {'code': '0580', 'name': 'Mathematics'},
          {'code': '0500', 'name': 'English'},
          {'code': '0455', 'name': 'Economics'},
          {'code': '0450', 'name': 'Business Studies'},
          {'code': '0478', 'name': 'Computer Science'},
          {'code': '0460', 'name': 'Geography'},
          {'code': '0416', 'name': 'History'},
          {'code': '0501', 'name': 'French'},
          {'code': '0505', 'name': 'German'},
          {'code': '0502', 'name': 'Spanish'},
          {'code': '0509', 'name': 'Chinese'},
          {'code': '0508', 'name': 'Arabic'},
          {'code': '0549', 'name': 'Hindi'},
          {'code': '0400', 'name': 'Art and Design'},
          {'code': '0445', 'name': 'Design and Technology'},
          {'code': '0410', 'name': 'Music'},
          {'code': '0411', 'name': 'Drama'},
          {'code': '0452', 'name': 'Accounting'},
          {'code': '0476', 'name': 'English as a Second Language'},
          {'code': '0438', 'name': 'Biology (0438)'},
          {'code': '0439', 'name': 'Chemistry (0439)'},
          {'code': '0443', 'name': 'Physics (0443)'},
          {'code': '0970', 'name': 'Biology (0970)'},
          {'code': '0971', 'name': 'Chemistry (0971)'},
          {'code': '0972', 'name': 'Physics (0972)'},
          {'code': '0581', 'name': 'Mathematics (0581)'},
          {'code': '0606', 'name': 'Mathematics (0606)'},
          {'code': '0984', 'name': 'Computer Science (0984)'},
          {'code': '0987', 'name': 'Economics (0987)'},
          {'code': '0980', 'name': 'Mathematics (0980)'},
          {'code': '0444', 'name': 'Mathematics (UK)'},
          {'code': '0465', 'name': 'English ESL'},
          {'code': '0510', 'name': 'English (0510)'},
          {'code': '0520', 'name': 'French (0520)'},
          {'code': '0525', 'name': 'German (0525)'},
          {'code': '0474', 'name': 'Spanish (0474)'},
          {'code': '0523', 'name': 'Chinese (0523)'},
          {'code': '0547', 'name': 'Chinese (0547)'},
          {'code': '0544', 'name': 'Arabic (0544)'},
          {'code': '0539', 'name': 'Urdu'},
          {'code': '0490', 'name': 'Religious Studies'},
          {'code': '0457', 'name': 'Global Perspectives'},
          {'code': '0426', 'name': 'Global Perspectives (0426)'},
          {'code': '0449', 'name': 'Bangladesh Studies'},
          {'code': '0448', 'name': 'Pakistan Studies'},
          {'code': '0447', 'name': 'India Studies'},
          {'code': '0986', 'name': 'Business Studies (0986)'},
          {'code': '0985', 'name': 'Accounting (0985)'},
          {'code': '0989', 'name': 'Art and Design (0989)'},
          {'code': '0415', 'name': 'Art and Design (0415)'},
          {'code': '0429', 'name': 'Music (0429)'},
          {'code': '0978', 'name': 'Music (0978)'},
          {'code': '0428', 'name': 'Drama (0428)'},
          {'code': '0994', 'name': 'Drama (0994)'},
          {'code': '0979', 'name': 'Design and Technology (0979)'},
          {'code': '0648', 'name': 'Food and Nutrition'},
          {'code': '0680', 'name': 'Environmental Management'},
          {'code': '0453', 'name': 'Development Studies'},
          {'code': '0600', 'name': 'Agriculture'},
          {'code': '0637', 'name': 'Child Development'},
          {'code': '0471', 'name': 'Travel and Tourism'},
          {'code': '0413', 'name': 'Physical Education'},
          {'code': '0995', 'name': 'Physical Education (0995)'},
          {'code': '0493', 'name': 'Islamiyat'},
          {'code': '0695', 'name': 'Vietnamese First Language'},
          {'code': '0480', 'name': 'Latin'},
          {'code': '0408', 'name': 'World Literature'},
          {'code': '0514', 'name': 'Czech First Language'},
          {'code': '0503', 'name': 'Dutch'},
          {'code': '0543', 'name': 'Greek'},
          {'code': '0513', 'name': 'Turkish'},
          {'code': '0696', 'name': 'Malay First Language'},
          {'code': '0697', 'name': 'Marine Science'},
          {'code': '0698', 'name': 'Setswana First Language'},
          {'code': '0262', 'name': 'Swahili'},
          {'code': '0538', 'name': 'Bahasa Indonesia'},
          {'code': '0454', 'name': 'Enterprise'},
          {'code': '0652', 'name': 'Physical Science'},
          {'code': '0608', 'name': 'Twenty-First Century Science'},
          {'code': '0653', 'name': 'Science'},
          {'code': '0654', 'name': 'Sciences'},
          {'code': '0442', 'name': 'Sciences (0442)'},
          {'code': '0973', 'name': 'Sciences (0973)'},
          {'code': '0518', 'name': 'Thai'},
          {'code': '0516', 'name': 'Russian'},
          {'code': '0535', 'name': 'Italian'},
          {'code': '0546', 'name': 'Malay'},
          {'code': '0545', 'name': 'Indonesian'},
          {'code': '0531', 'name': 'IsiZulu'},
          {'code': '0532', 'name': 'Kazakh'},
          {'code': '0521', 'name': 'Korean'},
          {'code': '0548', 'name': 'Afrikaans'},
          {'code': '0512', 'name': 'Afrikaans (0512)'},
          {'code': '0477', 'name': 'English (0477)'},
          {'code': '0486', 'name': 'English (0486)'},
          {'code': '0427', 'name': 'English (0427)'},
          {'code': '0522', 'name': 'English (0522)'},
          {'code': '0524', 'name': 'English (0524)'},
          {'code': '0627', 'name': 'English (0627)'},
          {'code': '0772', 'name': 'English (0772)'},
          {'code': '0990', 'name': 'English (0990)'},
          {'code': '0991', 'name': 'English (0991)'},
          {'code': '0993', 'name': 'English (0993)'},
          {'code': '0475', 'name': 'English (0475)'},
          {'code': '0472', 'name': 'English (0472)'},
          {'code': '0992', 'name': 'English (0992)'},
          {'code': '0528', 'name': 'French (0528)'},
          {'code': '0685', 'name': 'French (0685)'},
          {'code': '0529', 'name': 'German (0529)'},
          {'code': '0677', 'name': 'German (0677)'},
          {'code': '0519', 'name': 'Japanese (0519)'},
          {'code': '0534', 'name': 'Chinese (0534)'},
          {'code': '0527', 'name': 'Arabic (0527)'},
          {'code': '0530', 'name': 'Spanish (0530)'},
          {'code': '0533', 'name': 'Spanish (0533)'},
          {'code': '0537', 'name': 'Spanish (0537)'},
          {'code': '0417', 'name': 'ICT'},
          {'code': '0983', 'name': 'ICT (0983)'},
          {'code': '0420', 'name': 'Computer Studies'},
          {'code': '0441', 'name': 'Computer Studies (0441)'},
          {'code': '0976', 'name': 'Geography (0976)'},
          {'code': '0470', 'name': 'History (0470)'},
          {'code': '0977', 'name': 'History (0977)'},
          {'code': '0409', 'name': 'History American'},
        ],
        'AS and A Level': [
          {'code': '9700', 'name': 'Biology'},
          {'code': '9184', 'name': 'Biology (9184)'},
          {'code': '9701', 'name': 'Chemistry'},
          {'code': '9185', 'name': 'Chemistry (9185)'},
          {'code': '9702', 'name': 'Physics'},
          {'code': '8780', 'name': 'Physical Science'},
          {'code': '9709', 'name': 'Mathematics'},
          {'code': '9231', 'name': 'Mathematics (9231)'},
          {'code': '9280', 'name': 'Mathematics (9280)'},
          {'code': '9608', 'name': 'Computer Science'},
          {'code': '9618', 'name': 'Computer Science (9618)'},
          {'code': '9691', 'name': 'Computing'},
          {'code': '9713', 'name': 'Applied ICT'},
          {'code': '9093', 'name': 'English'},
          {'code': '8693', 'name': 'English (8693)'},
          {'code': '8695', 'name': 'English (8695)'},
          {'code': '8287', 'name': 'English (8287)'},
          {'code': '8274', 'name': 'English (8274)'},
          {'code': '9695', 'name': 'English Literature'},
          {'code': '9276', 'name': 'English Literature (9276)'},
          {'code': '8021', 'name': 'English General Paper'},
          {'code': '8001', 'name': 'General Paper'},
          {'code': '8004', 'name': 'General Paper (8004)'},
          {'code': '9716', 'name': 'French'},
          {'code': '9281', 'name': 'French (9281)'},
          {'code': '8682', 'name': 'French (8682)'},
          {'code': '8277', 'name': 'French (8277)'},
          {'code': '9898', 'name': 'French Language and Literature'},
          {'code': '9717', 'name': 'German'},
          {'code': '8683', 'name': 'German (8683)'},
          {'code': '8027', 'name': 'German (8027)'},
          {'code': '9897', 'name': 'German Language and Literature'},
          {'code': '9718', 'name': 'Portuguese'},
          {'code': '8684', 'name': 'Portuguese (8684)'},
          {'code': '8672', 'name': 'Portuguese (8672)'},
          {'code': '9719', 'name': 'Spanish'},
          {'code': '8685', 'name': 'Spanish (8685)'},
          {'code': '8673', 'name': 'Spanish (8673)'},
          {'code': '8278', 'name': 'Spanish (8278)'},
          {'code': '8279', 'name': 'Spanish (8279)'},
          {'code': '9282', 'name': 'Spanish (9282)'},
          {'code': '9844', 'name': 'Spanish (9844)'},
          {'code': '9715', 'name': 'Chinese'},
          {'code': '8681', 'name': 'Chinese (8681)'},
          {'code': '8669', 'name': 'Chinese (8669)'},
          {'code': '8238', 'name': 'Chinese (8238)'},
          {'code': '9868', 'name': 'Chinese (9868)'},
          {'code': '9680', 'name': 'Arabic'},
          {'code': '8680', 'name': 'Arabic (8680)'},
          {'code': '9687', 'name': 'Hindi'},
          {'code': '8687', 'name': 'Hindi (8687)'},
          {'code': '9686', 'name': 'Urdu'},
          {'code': '8686', 'name': 'Urdu (8686)'},
          {'code': '9676', 'name': 'Urdu (9676)'},
          {'code': '9689', 'name': 'Tamil'},
          {'code': '8689', 'name': 'Tamil (8689)'},
          {'code': '9690', 'name': 'Telugu'},
          {'code': '8690', 'name': 'Telugu (8690)'},
          {'code': '9688', 'name': 'Marathi'},
          {'code': '8688', 'name': 'Marathi (8688)'},
          {'code': '8281', 'name': 'Japanese'},
          {'code': '8679', 'name': 'Afrikaans'},
          {'code': '8779', 'name': 'Afrikaans (8779)'},
          {'code': '9679', 'name': 'Afrikaans (9679)'},
          {'code': '9696', 'name': 'Geography'},
          {'code': '9278', 'name': 'Geography (9278)'},
          {'code': '9697', 'name': 'History'},
          {'code': '9389', 'name': 'History (9389)'},
          {'code': '9489', 'name': 'History (9489)'},
          {'code': '9279', 'name': 'History (9279)'},
          {'code': '9698', 'name': 'Psychology'},
          {'code': '9990', 'name': 'Psychology (9990)'},
          {'code': '9699', 'name': 'Sociology'},
          {'code': '9694', 'name': 'Thinking Skills'},
          {'code': '9239', 'name': 'Global Perspectives and Research'},
          {'code': '8987', 'name': 'Global Perspectives'},
          {'code': '8275', 'name': 'Global Perspectives (8275)'},
          {'code': '9274', 'name': 'Classical Studies'},
          {'code': '9084', 'name': 'Law'},
          {'code': '9609', 'name': 'Business'},
          {'code': '9707', 'name': 'Business Studies'},
          {'code': '9708', 'name': 'Economics'},
          {'code': '9706', 'name': 'Accounting'},
          {'code': '9704', 'name': 'Art and Design'},
          {'code': '9479', 'name': 'Art and Design (9479)'},
          {'code': '9481', 'name': 'Design and Technology'},
          {'code': '9705', 'name': 'Design and Technology (9705)'},
          {'code': '9631', 'name': 'Design and Textiles'},
          {'code': '9703', 'name': 'Music'},
          {'code': '9483', 'name': 'Music (9483)'},
          {'code': '9385', 'name': 'Music (9385)'},
          {'code': '9482', 'name': 'Drama'},
          {'code': '9607', 'name': 'Media Studies'},
          {'code': '9626', 'name': 'Information Technology'},
          {'code': '9014', 'name': 'Hinduism'},
          {'code': '9487', 'name': 'Hinduism (9487)'},
          {'code': '8058', 'name': 'Hinduism (8058)'},
          {'code': '9013', 'name': 'Islamic Studies'},
          {'code': '9488', 'name': 'Islamic Studies (9488)'},
          {'code': '9011', 'name': 'Divinity'},
          {'code': '8041', 'name': 'Divinity (8041)'},
          {'code': '9484', 'name': 'Biblical Studies'},
          {'code': '9693', 'name': 'Marine Science'},
          {'code': '9395', 'name': 'Travel and Tourism'},
          {'code': '9396', 'name': 'Physical Education'},
          {'code': '8386', 'name': 'Sport and Physical Education'},
          {'code': '8291', 'name': 'Environmental Management'},
          {'code': '9336', 'name': 'Food Studies'},
          {'code': '8024', 'name': 'Nepal Studies'},
          {
            'code': '9980',
            'name': 'Cambridge International Project Qualification'
          },
        ],
        'O Level': [
          {'code': '5090', 'name': 'Biology'},
          {'code': '5070', 'name': 'Chemistry'},
          {'code': '5054', 'name': 'Physics'},
          {'code': '4024', 'name': 'Mathematics'},
          {'code': '4037', 'name': 'Mathematics Additional'},
          {'code': '1123', 'name': 'English Language'},
          {'code': '2010', 'name': 'Literature in English'},
          {'code': '2217', 'name': 'Geography'},
          {'code': '2147', 'name': 'History'},
          {'code': '2281', 'name': 'Economics'},
          {'code': '7115', 'name': 'Business Studies'},
          {'code': '7110', 'name': 'Principles of Accounts'},
          {'code': '7707', 'name': 'Accounting'},
          {'code': '2210', 'name': 'Computer Science'},
          {'code': '7010', 'name': 'Computer Studies'},
          {'code': '3015', 'name': 'French'},
          {'code': '3025', 'name': 'German'},
          {'code': '3035', 'name': 'Spanish'},
          {'code': '3180', 'name': 'Arabic'},
          {'code': '3204', 'name': 'Bengali'},
          {'code': '3202', 'name': 'Nepali'},
          {'code': '3195', 'name': 'Hindi'},
          {'code': '3247', 'name': 'Urdu'},
          {'code': '6010', 'name': 'Art'},
          {'code': '6090', 'name': 'Art and Design'},
          {'code': '6043', 'name': 'Design and Technology'},
          {'code': '6065', 'name': 'Food and Nutrition'},
          {'code': '2048', 'name': 'Religious Studies'},
          {'code': '2058', 'name': 'Islamiyat'},
          {'code': '2069', 'name': 'Global Perspectives'},
          {'code': '2251', 'name': 'Sociology'},
          {'code': '4040', 'name': 'Statistics'},
          {'code': '5129', 'name': 'Science Combined'},
          {'code': '5096', 'name': 'Human and Social Biology'},
          {'code': '7100', 'name': 'Commerce'},
          {'code': '7101', 'name': 'Commercial Studies'},
          {'code': '3205', 'name': 'Sinhala'},
          {'code': '3206', 'name': 'Tamil'},
          {'code': '3226', 'name': 'Tamil (3226)'},
          {'code': '3248', 'name': 'Urdu (3248)'},
          {'code': '3162', 'name': 'Swahili'},
          {'code': '3158', 'name': 'Setswana'},
          {'code': '7048', 'name': 'CDT Design and Communication'},
          {'code': '6050', 'name': 'Fashion and Fabrics'},
          {'code': '6130', 'name': 'Fashion and Textiles'},
          {'code': '5014', 'name': 'Environmental Management'},
          {'code': '5038', 'name': 'Agriculture'},
          {'code': '5180', 'name': 'Marine Science'},
          {'code': '7094', 'name': 'Bangladesh Studies'},
          {'code': '2059', 'name': 'Pakistan Studies'},
          {'code': '7096', 'name': 'Travel and Tourism'},
          {'code': '2134', 'name': 'History Modern World Affairs'},
          {'code': '2158', 'name': 'History World Affairs'},
          {'code': '2068', 'name': 'Islamic Studies'},
          {'code': '2055', 'name': 'Hinduism'},
          {'code': '2056', 'name': 'Islamic Religion and Culture'},
          {'code': '2035', 'name': 'Biblical Studies'},
        ],
      },
    };

int getTotalSubjectCount() {
  int count = 0;
  for (final board in allSubjects.values) {
    for (final level in board.values) {
      count += level.length;
    }
  }
  return count;
}

const String _dailyDateSheetWork = 'axon_daily_date_sheet_check';
const String _resourceSyncWork = 'axon_resource_sync';

@pragma('vm:entry-point')
void callbackDispatcher() {
  Workmanager().executeTask((task, inputData) async {
    switch (task) {
      case _dailyDateSheetWork:
        await _runDailyDateSheetCheck(inputData);
        break;
      case _resourceSyncWork:
        await _runResourceSync(inputData);
        break;
    }
    return true;
  });
}

Future<void> _runDailyDateSheetCheck(Map<String, dynamic>? inputData) async {
  try {
    final prefs = await SharedPreferences.getInstance();
    final board = prefs.getString('userBoard');
    final subjectsJson = prefs.getString('userSubjects');

    if (board == null || board.isEmpty) return;

    List<String> subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology'];
    if (subjectsJson != null) {
      try {
        subjects = List<String>.from(jsonDecode(subjectsJson));
      } catch (_) {}
    }

    final autoCrawl = AxonAutoCrawlService();
    await autoCrawl.initialize(AxonAutoCrawlConfig(
      board: board,
      subjects: subjects,
      dailyDateSheetCheck: true,
    ));
    await autoCrawl.forceDateSheetCheck();
  } catch (e) {
    debugPrint('Daily date sheet check failed: $e');
  }
}

Future<void> _runResourceSync(Map<String, dynamic>? inputData) async {
  try {
    final prefs = await SharedPreferences.getInstance();
    final board = prefs.getString('userBoard');

    if (board == null || board.isEmpty) return;

    final autoCrawl = AxonAutoCrawlService();
    await autoCrawl.initialize(AxonAutoCrawlConfig(board: board, subjects: []));
    await autoCrawl.syncPendingResources();
  } catch (e) {
    debugPrint('Resource sync failed: $e');
  }
}

enum CrawlPriority {
  dateSheet,
  syllabus,
  pastPapers,
  markSchemes,
  videos,
  topicals,
}

class AxonAutoCrawlConfig {
  final String board;
  final List<String> subjects;
  final List<String> examSeasons;
  final bool dailyDateSheetCheck;
  final int maxResourcesPerDay;
  final int gemmaVerificationBatchSize;
  final bool persistTasks;

  const AxonAutoCrawlConfig({
    required this.board,
    required this.subjects,
    this.examSeasons = const ['May/June', 'Oct/Nov'],
    this.dailyDateSheetCheck = true,
    this.maxResourcesPerDay = 50,
    this.gemmaVerificationBatchSize = 5,
    this.persistTasks = true,
  });
}

class CrawlTask {
  final String id;
  final String url;
  final CrawlPriority priority;
  final DateTime createdAt;
  final int retryCount;
  final String? metadata;
  final bool isVerified;

  CrawlTask({
    required this.id,
    required this.url,
    required this.priority,
    required this.createdAt,
    this.retryCount = 0,
    this.metadata,
    this.isVerified = false,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'url': url,
        'priority': priority.index,
        'createdAt': createdAt.toIso8601String(),
        'retryCount': retryCount,
        'metadata': metadata,
        'isVerified': isVerified,
      };

  factory CrawlTask.fromJson(Map<String, dynamic> json) => CrawlTask(
        id: json['id'],
        url: json['url'],
        priority: CrawlPriority.values[json['priority']],
        createdAt: DateTime.parse(json['createdAt']),
        retryCount: json['retryCount'] ?? 0,
        metadata: json['metadata'],
        isVerified: json['isVerified'] ?? false,
      );

  CrawlTask copyWith({int? retryCount, String? metadata, bool? isVerified}) {
    return CrawlTask(
      id: id,
      url: url,
      priority: priority,
      createdAt: createdAt,
      retryCount: retryCount ?? this.retryCount,
      metadata: metadata ?? this.metadata,
      isVerified: isVerified ?? this.isVerified,
    );
  }
}

class VerifiedResource {
  final String url;
  final String title;
  final CrawlPriority type;
  final double relevanceScore;
  final bool isVerified;
  final String? board;
  final String? subject;
  final String? gemmaAnalysis;
  final DateTime verifiedAt;
  final DateTime? expiresAt;

  VerifiedResource({
    required this.url,
    required this.title,
    required this.type,
    required this.relevanceScore,
    required this.isVerified,
    this.board,
    this.subject,
    this.gemmaAnalysis,
    required this.verifiedAt,
    this.expiresAt,
  });

  Map<String, dynamic> toJson() => {
        'url': url,
        'title': title,
        'type': type.index,
        'relevanceScore': relevanceScore,
        'isVerified': isVerified,
        'board': board,
        'subject': subject,
        'gemmaAnalysis': gemmaAnalysis,
        'verifiedAt': verifiedAt.toIso8601String(),
        'expiresAt': expiresAt?.toIso8601String(),
      };

  factory VerifiedResource.fromJson(Map<String, dynamic> json) =>
      VerifiedResource(
        url: json['url'],
        title: json['title'],
        type: CrawlPriority.values[json['type']],
        relevanceScore: (json['relevanceScore'] as num).toDouble(),
        isVerified: json['isVerified'],
        board: json['board'],
        subject: json['subject'],
        gemmaAnalysis: json['gemmaAnalysis'],
        verifiedAt: DateTime.parse(json['verifiedAt']),
        expiresAt: json['expiresAt'] != null
            ? DateTime.parse(json['expiresAt'])
            : null,
      );

  bool get isExpired => expiresAt != null && DateTime.now().isAfter(expiresAt!);
}

class GemmaBatchProcessor {
  final int batchSize;
  final Duration minInterval;
  final Map<String, VerifiedResource> _pendingVerification = {};
  Timer? _batchTimer;
  Function(List<VerifiedResource>)? _onBatchComplete;
  bool _isProcessing = false;

  GemmaBatchProcessor({
    this.batchSize = 5,
    this.minInterval = const Duration(seconds: 30),
  });

  void setBatchCompleteCallback(Function(List<VerifiedResource>) callback) {
    _onBatchComplete = callback;
  }

  void addForVerification(VerifiedResource resource) {
    _pendingVerification[resource.url] = resource;

    // Start timer if not running
    _batchTimer ??= Timer(minInterval, _processBatch);
  }

  Future<void> _processBatch() async {
    if (_isProcessing || _pendingVerification.isEmpty) return;

    _isProcessing = true;

    // Take batch
    final batch = _pendingVerification.entries.take(batchSize).toList();
    for (final entry in batch) {
      _pendingVerification.remove(entry.key);
    }

    // Process with Gemma (external call)
    final verified = await _verifyBatchWithGemma(
      batch.map((e) => e.value).toList(),
    );

    _onBatchComplete?.call(verified);

    _isProcessing = false;
    _batchTimer = null;

    // Schedule next if more pending
    if (_pendingVerification.isNotEmpty) {
      _batchTimer = Timer(minInterval, _processBatch);
    }
  }

  Future<List<VerifiedResource>> _verifyBatchWithGemma(
      List<VerifiedResource> resources) async {
    // Try to verify resource quality, fallback to as-is on failure
    try {
      final grokService = GrokService();
      if (grokService.isReady) {
        final verified = <VerifiedResource>[];
        for (final resource in resources) {
          final result = await grokService.chat(
            'Rate this resource from 1-10 for educational quality:\nTitle: ${resource.title}\nURL: ${resource.url}',
            systemPrompt: 'Return only a number between 1 and 10.',
          );
          final rating = double.tryParse(result.trim());
          verified.add(VerifiedResource(
            url: resource.url,
            title: resource.title,
            type: resource.type,
            relevanceScore: resource.relevanceScore,
            isVerified: rating != null && rating >= 6,
            board: resource.board,
            subject: resource.subject,
            gemmaAnalysis: rating != null
                ? 'Quality rating: $rating/10'
                : resource.gemmaAnalysis,
            verifiedAt: DateTime.now(),
            expiresAt: resource.expiresAt,
          ));
        }
        return verified;
      }
    } catch (e) {
      debugPrint('Resource quality verification failed: $e');
    }
    return resources;
  }

  int get pendingCount => _pendingVerification.length;

  void dispose() {
    _batchTimer?.cancel();
  }
}

class GemmaVerificationCache {
  final Map<String, _CachedGemmaResult> _cache = {};
  final int maxCacheSize;
  final Duration cacheDuration;

  GemmaVerificationCache({
    this.maxCacheSize = 500,
    this.cacheDuration = const Duration(hours: 24),
  });

  String _hashUrl(String url) => md5.convert(utf8.encode(url)).toString();

  bool isCached(String url) {
    final hash = _hashUrl(url);
    if (!_cache.containsKey(hash)) return false;
    final cached = _cache[hash]!;
    return DateTime.now().difference(cached.timestamp) < cacheDuration;
  }

  VerifiedResource? getCached(String url) {
    final hash = _hashUrl(url);
    return _cache[hash]?.result;
  }

  Future<void> add(String url, VerifiedResource result) async {
    final hash = _hashUrl(url);
    if (_cache.length >= maxCacheSize) {
      _evictOldest();
    }
    _cache[hash] = _CachedGemmaResult(result, DateTime.now());
  }

  void _evictOldest() {
    String? oldestKey;
    DateTime? oldestTime;
    for (final entry in _cache.entries) {
      if (oldestTime == null || entry.value.timestamp.isBefore(oldestTime)) {
        oldestKey = entry.key;
        oldestTime = entry.value.timestamp;
      }
    }
    if (oldestKey != null) _cache.remove(oldestKey);
  }
}

class _CachedGemmaResult {
  final VerifiedResource result;
  final DateTime timestamp;
  _CachedGemmaResult(this.result, this.timestamp);
}

class AxonAutoCrawlService {
  static final AxonAutoCrawlService _instance =
      AxonAutoCrawlService._internal();
  factory AxonAutoCrawlService() => _instance;
  AxonAutoCrawlService._internal();

  bool _isInitialized = false;
  bool _isRunning = false;
  Timer? _dailyCheckTimer;
  Timer? _resourceScanTimer;
  StreamController<List<VerifiedResource>>? _resourceController;

  List<CrawlTask> _taskQueue = [];
  List<VerifiedResource> _verifiedResources = [];
  final GemmaVerificationCache _gemmaCache = GemmaVerificationCache();
  late GemmaBatchProcessor _gemmaBatch;

  String? _saveDirectory;
  AxonAutoCrawlConfig? _config;
  Function(String, Map<String, dynamic>?)? _gemmaVerifier;

  Future<void> initialize(AxonAutoCrawlConfig config) async {
    if (_isInitialized) return;

    _config = config;
    _gemmaBatch = GemmaBatchProcessor(
      batchSize: config.gemmaVerificationBatchSize,
    );

    final dir = await getApplicationDocumentsDirectory();
    _saveDirectory = '${dir.path}/auto_crawl_cache';
    Directory(_saveDirectory!).createSync(recursive: true);

    await _loadPersistedState();
    _registerBackgroundTasks();

    _isInitialized = true;
    debugPrint(
        'AutoCrawl initialized with board: ${_config?.board}, subjects: ${_config?.subjects}');
  }

  void _registerBackgroundTasks() {
    // WorkManager integration for background tasks
    // In production, you'd call:
    // Workmanager().registerPeriodicTask(
    //   'daily_date_sheet',
    //   _dailyDateSheetWork,
    //   frequency: const Duration(hours: 24),
    //   constraints: Constraints(networkType: NetworkType.CONNECTED),
    // );
  }

  Future<void> _loadPersistedState() async {
    final prefs = await SharedPreferences.getInstance();

    // Load verified resources
    final resourcesJson = prefs.getString('axon_verified_resources');
    if (resourcesJson != null) {
      try {
        final list = jsonDecode(resourcesJson) as List;
        _verifiedResources = list
            .map((e) => VerifiedResource.fromJson(e))
            .where((r) => !r.isExpired)
            .toList();
      } catch (_) {}
    }

    // Load pending tasks
    final tasksJson = prefs.getString('axon_pending_tasks');
    if (tasksJson != null) {
      try {
        final list = jsonDecode(tasksJson) as List;
        _taskQueue = list.map((e) => CrawlTask.fromJson(e)).toList();
      } catch (_) {}
    }

    // Load last run times
    final lastDateSheet = prefs.getInt('last_date_sheet_check');
    if (lastDateSheet != null) {
      final lastCheck = DateTime.fromMillisecondsSinceEpoch(lastDateSheet);
      // Check if we need to run immediately
      final now = DateTime.now();
      if (now.difference(lastCheck).inHours >= 24) {
        // Will trigger in start()
      }
    }
  }

  Future<void> _persistState() async {
    final prefs = await SharedPreferences.getInstance();

    // Save verified resources (limit to recent 100)
    final recentResources = _verifiedResources.take(100).toList();
    await prefs.setString('axon_verified_resources',
        jsonEncode(recentResources.map((r) => r.toJson()).toList()));

    // Save pending tasks
    await prefs.setString('axon_pending_tasks',
        jsonEncode(_taskQueue.map((t) => t.toJson()).toList()));
  }

  void registerGemmaVerifier(Function(String, Map<String, dynamic>?) verifier) {
    _gemmaVerifier = verifier;

    // Set up batch processor callback
    _gemmaBatch.setBatchCompleteCallback((verified) {
      for (final resource in verified) {
        _addVerifiedResource(resource);
      }
    });
  }

  Future<void> start() async {
    if (!_isInitialized || _isRunning) return;

    _isRunning = true;
    _resourceController = StreamController<List<VerifiedResource>>.broadcast();

    // Check if date sheet check is due
    await _checkAndRunDateSheetIfDue();

    // Start other crawling
    _crawlSyllabus();
    _crawlResources();
    _scheduleDailyDateSheetCheck();
    _scheduleResourceScanning();

    // Process any persisted tasks
    await _processPersistedTasks();
  }

  Future<void> _checkAndRunDateSheetIfDue() async {
    final prefs = await SharedPreferences.getInstance();
    final lastCheck = prefs.getInt('last_date_sheet_check') ?? 0;
    final now = DateTime.now().millisecondsSinceEpoch;

    if (now - lastCheck >= 24 * 60 * 60 * 1000) {
      // 24 hours
      await _crawlDateSheets();
      await prefs.setInt('last_date_sheet_check', now);
    }
  }

  void stop() {
    _isRunning = false;
    _dailyCheckTimer?.cancel();
    _resourceScanTimer?.cancel();
    _gemmaBatch.dispose();
    _resourceController?.close();
    _persistState();
  }

  Stream<List<VerifiedResource>> get resourceStream =>
      _resourceController?.stream ?? const Stream.empty();

  List<VerifiedResource> get verifiedResources =>
      List.unmodifiable(_verifiedResources);

  // ═══════════════════════════════════════════════════════════════════
  // DATE SHEET CRAWLING (Highest Priority)
  // ═══════════════════════════════════════════════════════════════════

  Future<void> _crawlDateSheets() async {
    if (_config == null) return;

    final dateSheetUrls = await _buildDateSheetUrls();

    for (final url in dateSheetUrls) {
      _addTask(CrawlTask(
        id: md5.convert(utf8.encode(url)).toString(),
        url: url,
        priority: CrawlPriority.dateSheet,
        createdAt: DateTime.now(),
      ));
    }

    await _processQueue();
  }

  Future<List<String>> _buildDateSheetUrls() async {
    final urls = <String>[];

    // Initialize crawler service for link scraping
    final crawler = AxonCrawlerService();
    await crawler.initialize();

    // Strategy 1: Add zone timetables as fallback
    urls.addAll([
      'https://timetable.papacambridge.com/timetable/caie/uk',
      'https://timetable.papacambridge.com/timetable/caie/zone-1',
      'https://timetable.papacambridge.com/timetable/caie/zone-2',
      'https://timetable.papacambridge.com/timetable/caie/zone-3',
      'https://timetable.papacambridge.com/timetable/caie/zone-4',
      'https://timetable.papacambridge.com/timetable/caie/zone-5',
      'https://timetable.papacambridge.com/timetable/caie/zone-6',
    ]);

    // Strategy 2: Scrape PapaCambridge for detailed subject timetables
    final baseUrls = [
      'https://pastpapers.papacambridge.com/papers/caie/igcse',
      'https://pastpapers.papacambridge.com/papers/caie/as-and-a-level',
      'https://pastpapers.papacambridge.com/papers/caie/o-level',
    ];

    for (final baseUrl in baseUrls) {
      try {
        await crawler.scrapeLinks(
          baseUrl,
          filterType: CrawlerResourceType.dateSheet,
        );

        final pdfLinks = await crawler.scrapeLinks(baseUrl);

        for (final link in pdfLinks) {
          final lower = link.toLowerCase();
          if (lower.contains('timetable') ||
              lower.contains('date sheet') ||
              lower.contains('datesheet') ||
              lower.contains('schedule') ||
              (lower.contains('.pdf') && lower.contains('2026'))) {
            urls.add(link);
          }
        }
      } catch (e) {
        debugPrint('Failed to scrape $baseUrl: $e');
      }
    }

    // Strategy 3: Build direct subject-specific timetable URLs
    final board = _config?.board.toLowerCase() ?? '';
    final level = board.contains('alevel') || board.contains('a level')
        ? 'AS and A Level'
        : (board.contains('o level') || board.contains('olevel')
            ? 'O Level'
            : 'IGCSE');

    final boardKey = 'CAIE';
    final subjects = allSubjects[boardKey]?[level] ?? [];

    for (final subject in subjects) {
      final code = subject['code'] as String;
      final name = subject['name'] as String;

      final slug = name
          .toLowerCase()
          .replaceAll(' ', '-')
          .replaceAll('(', '')
          .replaceAll(')', '');

      urls.add(
          'https://pastpapers.papacambridge.com/timetable/caie/$slug-$code.pdf');
      urls.add(
          'https://pastpapers.papacambridge.com/timetable/caie/${code}_timetable.pdf');
    }

    // Deduplicate URLs
    return urls.toSet().toList();
  }

  void _scheduleDailyDateSheetCheck() {
    final now = DateTime.now();
    var nextRun = DateTime(now.year, now.month, now.day, 6, 0);
    if (nextRun.isBefore(now)) {
      nextRun = nextRun.add(const Duration(days: 1));
    }

    final delay = nextRun.difference(now);

    _dailyCheckTimer = Timer(delay, () {
      if (_isRunning) {
        _crawlDateSheets();
        _scheduleDailyDateSheetCheck();
      }
    });
  }

  Future<void> forceDateSheetCheck() async {
    if (_isRunning) {
      await _crawlDateSheets();

      // Update last check time
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt(
          'last_date_sheet_check', DateTime.now().millisecondsSinceEpoch);
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // SYLLABUS CRAWLING
  // ═══════════════════════════════════════════════════════════════════

  void _crawlSyllabus() {
    if (_config == null) return;

    final syllabusUrls = _buildSyllabusUrls();

    for (final url in syllabusUrls) {
      _addTask(CrawlTask(
        id: md5.convert(utf8.encode(url)).toString(),
        url: url,
        priority: CrawlPriority.syllabus,
        createdAt: DateTime.now(),
      ));
    }

    _processQueue();
  }

  List<String> _buildSyllabusUrls() {
    final urls = <String>[];
    final board = _config?.board.toLowerCase() ?? '';

    final base = 'https://syllabus.papacambridge.com/syllabus/caie';
    urls.add('$base/cambridge-igcse');
    urls.add('$base/cambridge-asa-level');
    urls.add('$base/cambridge-o-level');

    final level = board.contains('alevel') || board.contains('a level')
        ? 'AS and A Level'
        : (board.contains('o level') || board.contains('olevel')
            ? 'O Level'
            : 'IGCSE');

    final boardKey = 'CAIE';
    final subjects = allSubjects[boardKey]?[level] ?? [];
    for (final subject in subjects) {
      final code = subject['code'];
      final name = subject['name'] as String;
      final slug = name
          .toLowerCase()
          .replaceAll(' ', '-')
          .replaceAll('(', '')
          .replaceAll(')', '');
      urls.add('$base/$slug-$code');
    }

    return urls;
  }

  String _getBoardKey(String board) {
    return 'CAIE';
  }

  // ═══════════════════════════════════════════════════════════════════
  // RESOURCE SCRAPING
  // ═══════════════════════════════════════════════════════════════════

  void _crawlResources() {
    if (_config == null) return;

    final resourceUrls = _buildResourceUrls();

    for (final type in resourceUrls.keys) {
      for (final url in resourceUrls[type]!) {
        _addTask(CrawlTask(
          id: md5.convert(utf8.encode(url)).toString(),
          url: url,
          priority: type,
          createdAt: DateTime.now(),
        ));
      }
    }

    _processQueue();
  }

  Map<CrawlPriority, List<String>> _buildResourceUrls() {
    final urls = <CrawlPriority, List<String>>{};
    final board = _config?.board.toLowerCase() ?? '';

    // Safe handling of subjects - ensure it's always List<String>
    final userSubjects = _config?.subjects ?? <String>[];

    final boardKey = _getBoardKey(board);
    final boardSubjects = allSubjects[boardKey] ?? {};

    // Main past paper directories - CAIE
    if (boardKey == 'CAIE') {
        urls[CrawlPriority.pastPapers] = [
          'https://pastpapers.papacambridge.com/papers/caie/igcse',
          'https://pastpapers.papacambridge.com/papers/caie/as-and-a-level',
          'https://pastpapers.papacambridge.com/papers/caie/o-level',
        ];
      }

      // Add individual subject pages from database
      final subjectList = <Map<String, dynamic>>[];

      // Get subjects for each level
      for (final level in boardSubjects.keys) {
        subjectList.addAll(boardSubjects[level] ?? []);
      }

      // Add user's selected subjects matching
      for (final subjectName in userSubjects) {
        for (final level in boardSubjects.keys) {
          final matches = (boardSubjects[level] ?? []).where((s) =>
              s['name'].toString().toLowerCase() == subjectName.toLowerCase());
          subjectList.addAll(matches);
        }
      }

      // Generate URLs for each subject
      for (final subject in subjectList) {
        final code = subject['code'];
        final name = subject['name'] as String;
        final slug = name
            .toLowerCase()
            .replaceAll(' ', '-')
            .replaceAll('(', '')
            .replaceAll(')', '')
            .replaceAll('/', '-');

        if (boardKey == 'CAIE') {
          urls[CrawlPriority.pastPapers]!.add(
              'https://pastpapers.papacambridge.com/papers/caie/$slug-$code');
        }
      }

    return urls;
  }

  void _scheduleResourceScanning() {
    _resourceScanTimer = Timer.periodic(const Duration(hours: 4), (_) {
      if (_isRunning) {
        _crawlResources();
      }
    });
  }

  // ═══════════════════════════════════════════════════════════════════
  // TASK QUEUE MANAGEMENT
  // ═══════════════════════════════════════════════════════════════════

  void _addTask(CrawlTask task) {
    if (_taskQueue.any((t) => t.id == task.id)) return;

    final priorityIndex = CrawlPriority.values.indexOf(task.priority);
    var insertIndex = _taskQueue.length;

    for (var i = 0; i < _taskQueue.length; i++) {
      final taskPriorityIndex =
          CrawlPriority.values.indexOf(_taskQueue[i].priority);
      if (priorityIndex < taskPriorityIndex) {
        insertIndex = i;
        break;
      }
    }

    _taskQueue.insert(insertIndex, task);
  }

  Future<void> _processQueue() async {
    if (_taskQueue.isEmpty) return;

    final maxTasks = _config?.maxResourcesPerDay ?? 50;
    var processed = 0;

    while (_taskQueue.isNotEmpty && processed < maxTasks) {
      final task = _taskQueue.removeAt(0);

      try {
        final result = await _processTask(task);
        if (result != null) {
          _addVerifiedResource(result);
        }
      } catch (e) {
        if (task.retryCount < 3) {
          _taskQueue.add(task.copyWith(retryCount: task.retryCount + 1));
        }
      }

      processed++;
      await Future.delayed(const Duration(milliseconds: 200));
    }

    await _persistState();
  }

  Future<void> _processPersistedTasks() async {
    // Process any tasks that were persisted from previous session
    await _processQueue();
  }

  Future<void> syncPendingResources() async {
    // Called by background task to sync pending resources
    await _processQueue();
  }

  void _addVerifiedResource(VerifiedResource resource) {
    // Remove old version if exists
    _verifiedResources.removeWhere((r) => r.url == resource.url);
    _verifiedResources.insert(0, resource);

    // Emit update
    _resourceController?.add(_verifiedResources);
  }

  Future<VerifiedResource?> _processTask(CrawlTask task) async {
    // Check cache first
    if (_gemmaCache.isCached(task.url)) {
      return _gemmaCache.getCached(task.url);
    }

    final file = await _downloadResource(task.url);
    if (file == null) return null;

    // If this is a datesheet PDF, parse it into structured exam events
    if (task.priority == CrawlPriority.dateSheet &&
        file.path.toLowerCase().endsWith('.pdf')) {
      await _parseAndCacheDatesheet(file, task.url);
    }

    VerifiedResource? result;

    if (_gemmaVerifier != null && _shouldVerifyWithGemma(task)) {
      result = await _verifyWithGemma(task, file);
    } else {
      result = VerifiedResource(
        url: task.url,
        title: file.path.split('/').last,
        type: task.priority,
        relevanceScore: 0.5,
        isVerified: false,
        verifiedAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
      );

      // Queue for batch verification
      _gemmaBatch.addForVerification(result);
    }

    if (result != null) {
      await _gemmaCache.add(task.url, result);
    }

    return result;
  }

  Future<void> _parseAndCacheDatesheet(File pdfFile, String sourceUrl) async {
    try {
      final parsed = await DatesheetParser.instance.processDatesheetPdf(
        pdfFile,
        sourceUrl,
      );

      if (parsed) {
        // Reload exam cache so widgets see new data immediately
        final service = ExamDataService();
        await service.reloadCache();
        await service.refreshWidget();

        debugPrint(
          'Datesheet parsed and cache reloaded: ${pdfFile.path.split('/').last}',
        );
      }
    } catch (e) {
      debugPrint('Failed to parse datesheet: $e');
    }
  }

  bool _shouldVerifyWithGemma(CrawlTask task) {
    return task.priority == CrawlPriority.dateSheet ||
        task.priority == CrawlPriority.syllabus;
  }

  Future<VerifiedResource?> _verifyWithGemma(CrawlTask task, File file) async {
    try {
      final gemmaResult = await _gemmaVerifier?.call(task.url, {
        'board': _config?.board,
        'subjects': _config?.subjects,
        'task_type': task.priority.name,
      });

      if (gemmaResult != null) {
        return VerifiedResource(
          url: task.url,
          title: gemmaResult['title'] ?? file.path.split('/').last,
          type: task.priority,
          relevanceScore: (gemmaResult['relevance'] ?? 0.5).toDouble(),
          isVerified: gemmaResult['verified'] ?? false,
          board: gemmaResult['board'],
          subject: gemmaResult['subject'],
          gemmaAnalysis: gemmaResult['analysis'],
          verifiedAt: DateTime.now(),
          expiresAt: DateTime.now().add(const Duration(days: 30)),
        );
      }
    } catch (_) {}

    return null;
  }

  Future<File?> _downloadResource(String url) async {
    try {
      final response = await http.get(
        Uri.parse(url),
        headers: {'User-Agent': 'AxonBot/1.0'},
      );

      if (response.statusCode != 200) return null;

      final fileName = url.split('/').last;
      if (!fileName.toLowerCase().endsWith('.pdf')) return null;

      final file = File('$_saveDirectory/$fileName');
      await file.writeAsBytes(response.bodyBytes);

      return file;
    } catch (_) {
      return null;
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // PUBLIC API
  // ═══════════════════════════════════════════════════════════════════

  Future<Map<String, dynamic>> getCrawlStats() async {
    return {
      'isRunning': _isRunning,
      'pendingTasks': _taskQueue.length,
      'verifiedResources': _verifiedResources.length,
      'verifiedDateSheets': _verifiedResources
          .where((r) => r.type == CrawlPriority.dateSheet)
          .length,
      'verifiedSyllabus': _verifiedResources
          .where((r) => r.type == CrawlPriority.syllabus)
          .length,
      'gemmaPendingVerification': _gemmaBatch.pendingCount,
      'cacheSize': _gemmaCache.maxCacheSize,
    };
  }

  List<VerifiedResource> getResourcesByType(CrawlPriority type) {
    return _verifiedResources.where((r) => r.type == type).toList();
  }

  List<VerifiedResource> getResourcesByBoard(String board) {
    return _verifiedResources.where((r) => r.board == board).toList();
  }
}
