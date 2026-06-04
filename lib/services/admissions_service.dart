import 'dart:convert';
import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';

import '../models/admissions_models.dart';
import 'backend_config.dart';
import 'firestore_service.dart';
import 'grok_service.dart';

class AdmissionsService {
  AdmissionsService({http.Client? client}) : _client = client ?? http.Client();

  static const String _backendUrl = BackendConfig.baseUrl;
  final http.Client _client;

  CollectionReference<Map<String, dynamic>> _targets(String uid) =>
      AxonPaths.privateUserCollection(uid, 'admissions_targets');

  CollectionReference<Map<String, dynamic>> _milestones(String uid) =>
      AxonPaths.privateUserCollection(uid, 'admissions_milestones');

  CollectionReference<Map<String, dynamic>> _vault(String uid) =>
      AxonPaths.privateUserCollection(uid, 'admissions_vault');

  CollectionReference<Map<String, dynamic>> _marks(String uid) =>
      AxonPaths.privateUserCollection(uid, 'admissions_marks');

  static const List<UniversityProgram> _programCatalog = [
    UniversityProgram(
      id: 'imperial_cs_beng',
      universityName: 'Imperial College London',
      location: 'UK',
      degree: 'Computer Science',
      courseName: 'Computing BEng',
      duration: '3 years',
      coreModules: [
        'Algorithms',
        'Computer systems',
        'Databases',
        'Software engineering',
        'Artificial intelligence',
      ],
      minimumThreshold: 'A*A*A including Mathematics',
      requirementsByBoard: {
        'A Level': [
          UniversitySubjectRequirement(subject: 'Mathematics', minimumGrade: 'A*'),
          UniversitySubjectRequirement(subject: 'Further Mathematics', minimumGrade: 'A* preferred'),
          UniversitySubjectRequirement(subject: 'Computer Science', minimumGrade: 'A'),
          UniversitySubjectRequirement(subject: 'Physics', minimumGrade: 'A'),
        ],
        'IGCSE': [
          UniversitySubjectRequirement(subject: 'Mathematics', minimumGrade: 'A/7'),
          UniversitySubjectRequirement(subject: 'English', minimumGrade: 'B/6'),
        ],
      },
    ),
    UniversityProgram(
      id: 'cambridge_cs_ba',
      universityName: 'University of Cambridge',
      location: 'UK',
      degree: 'Computer Science',
      courseName: 'Computer Science BA',
      duration: '3 years',
      coreModules: [
        'Foundations of computer science',
        'Object-oriented programming',
        'Algorithms',
        'Machine learning',
        'Security',
      ],
      minimumThreshold: 'A*A*A with Mathematics required',
      requirementsByBoard: {
        'A Level': [
          UniversitySubjectRequirement(subject: 'Mathematics', minimumGrade: 'A*'),
          UniversitySubjectRequirement(subject: 'Further Mathematics', minimumGrade: 'A* strongly recommended'),
          UniversitySubjectRequirement(subject: 'Physics', minimumGrade: 'A'),
          UniversitySubjectRequirement(subject: 'Computer Science', minimumGrade: 'A'),
        ],
      },
    ),
    UniversityProgram(
      id: 'ucl_math_bsc',
      universityName: 'University College London',
      location: 'UK',
      degree: 'Mathematics',
      courseName: 'Mathematics BSc',
      duration: '3 years',
      coreModules: [
        'Analysis',
        'Algebra',
        'Mathematical methods',
        'Probability',
        'Differential equations',
      ],
      minimumThreshold: 'A*A*A with Mathematics A*',
      requirementsByBoard: {
        'A Level': [
          UniversitySubjectRequirement(subject: 'Mathematics', minimumGrade: 'A*'),
          UniversitySubjectRequirement(subject: 'Further Mathematics', minimumGrade: 'A recommended'),
          UniversitySubjectRequirement(subject: 'Physics', minimumGrade: 'A'),
        ],
      },
    ),
    UniversityProgram(
      id: 'nus_cs_bcomp',
      universityName: 'National University of Singapore',
      location: 'Singapore',
      degree: 'Computer Science',
      courseName: 'Computer Science BComp',
      duration: '4 years',
      coreModules: [
        'Programming methodology',
        'Data structures',
        'Computer organisation',
        'Software engineering',
        'AI and data science electives',
      ],
      minimumThreshold: 'Strong A-Level profile with Mathematics',
      requirementsByBoard: {
        'A Level': [
          UniversitySubjectRequirement(subject: 'Mathematics', minimumGrade: 'A'),
          UniversitySubjectRequirement(subject: 'Computer Science', minimumGrade: 'A preferred'),
          UniversitySubjectRequirement(subject: 'Physics', minimumGrade: 'A useful'),
        ],
      },
    ),
    UniversityProgram(
      id: 'ntu_math_bsc',
      universityName: 'Nanyang Technological University',
      location: 'Singapore',
      degree: 'Mathematics',
      courseName: 'Mathematical Sciences BSc',
      duration: '4 years',
      coreModules: [
        'Calculus',
        'Linear algebra',
        'Discrete mathematics',
        'Statistics',
        'Computational mathematics',
      ],
      minimumThreshold: 'Strong Mathematics grade profile',
      requirementsByBoard: {
        'A Level': [
          UniversitySubjectRequirement(subject: 'Mathematics', minimumGrade: 'A'),
          UniversitySubjectRequirement(subject: 'Further Mathematics', minimumGrade: 'A preferred'),
          UniversitySubjectRequirement(subject: 'Physics', minimumGrade: 'B+ useful'),
        ],
      },
    ),
  ];

  List<String> get availableLocations =>
      _programCatalog.map((item) => item.location).toSet().toList()..sort();

  List<String> get availableDegrees =>
      _programCatalog.map((item) => item.degree).toSet().toList()..sort();

  List<String> get availableUniversities =>
      _programCatalog.map((item) => item.universityName).toSet().toList()
        ..sort();

  List<UniversityProgram> searchPrograms({
    String location = '',
    String degree = '',
    String universityQuery = '',
  }) {
    final locationQuery = location.toLowerCase().trim();
    final degreeQuery = degree.toLowerCase().trim();
    final university = universityQuery.toLowerCase().trim();
    return _programCatalog.where((program) {
      final matchesLocation = locationQuery.isEmpty ||
          program.location.toLowerCase().contains(locationQuery);
      final matchesDegree =
          degreeQuery.isEmpty || program.degree.toLowerCase().contains(degreeQuery);
      final matchesUniversity = university.isEmpty ||
          program.universityName.toLowerCase().contains(university);
      return matchesLocation && matchesDegree && matchesUniversity;
    }).toList();
  }

  List<UniversityProgram> programsForUniversity(String universityName) {
    final query = universityName.toLowerCase().trim();
    return _programCatalog
        .where((program) => program.universityName.toLowerCase().contains(query))
        .toList();
  }

  Stream<List<AdmissionsTarget>> watchTargets(String uid) {
    return _targets(uid).snapshots().map(
          (snapshot) {
            final firestore = snapshot.docs
                .map((doc) => AdmissionsTarget.fromJson(doc.id, doc.data()))
                .toList()
              ..sort((a, b) {
                final aDeadline = a.deadlineAt ?? DateTime(2100);
                final bDeadline = b.deadlineAt ?? DateTime(2100);
                return aDeadline.compareTo(bDeadline);
              });
            if (firestore.isNotEmpty) {
              // Cache locally for offline fallback
              _cacheTargetsLocally(uid, firestore);
              return firestore;
            }
            // Fallback to cache if Firestore returns empty
            return _localCachedTargets(uid);
          },
        );
  }

  List<AdmissionsTarget> _localCachedTargets(String uid) {
    // Return from memory cache - populated by _cacheTargetsLocally
    return _targetCache[uid] ?? [];
  }

  final Map<String, List<AdmissionsTarget>> _targetCache = {};

  void _cacheTargetsLocally(String uid, List<AdmissionsTarget> targets) {
    _targetCache[uid] = targets;
    // Also persist to SharedPreferences
    try {
      SharedPreferences.getInstance().then((prefs) {
        prefs.setString('local_targets_$uid', jsonEncode(targets.map((t) => {
          'id': t.id,
          'university_name': t.universityName,
          'country': t.country,
          'course_name': t.courseName,
          'classification': t.classification,
          'application_system': t.applicationSystem,
          'status': t.status,
          'rationale': t.rationale,
          'entry_requirements': t.entryRequirements,
          'source_url': t.sourceUrl,
          'fit_band': t.fitBand,
          'readiness_score': t.readinessScore,
        }).toList()));
      });
    } catch (_) {}
  }

  Stream<List<AdmissionsMilestone>> watchMilestones(String uid) {
    return _milestones(uid).orderBy('due_at').snapshots().map(
          (snapshot) => snapshot.docs
              .map((doc) => AdmissionsMilestone.fromJson(doc.id, doc.data()))
              .toList(),
        );
  }

  Stream<List<AdmissionsVaultAsset>> watchVaultAssets(String uid) {
    return _vault(uid).orderBy('created_at', descending: true).snapshots().map(
          (snapshot) => snapshot.docs
              .map((doc) => AdmissionsVaultAsset.fromJson(doc.id, doc.data()))
              .toList(),
        );
  }

  Stream<List<AdmissionsMarkEntry>> watchMarks(String uid) {
    return _marks(uid).orderBy('subject').snapshots().map(
          (snapshot) => snapshot.docs
              .map((doc) => AdmissionsMarkEntry.fromJson(doc.id, doc.data()))
              .toList(),
        );
  }

  AdmissionsMilestoneProgress buildProgress(
    List<AdmissionsMilestone> milestones, {
    List<AdmissionsVaultAsset> assets = const [],
  }) {
    final completed = milestones.where((item) => item.completed).length;
    final blocked = milestones
        .where((item) => !item.completed && item.dependsOnIds.isNotEmpty)
        .length;
    return AdmissionsMilestoneProgress(
      completedCount: completed,
      totalCount: milestones.length,
      blockedCount: blocked,
      evidenceCount: assets.length,
    );
  }

  List<AdmissionsMilestoneMapNode> buildMilestoneMap(
    List<AdmissionsMilestone> milestones,
  ) {
    final byId = {
      for (final milestone in milestones) milestone.id: milestone,
    };
    return milestones.map((milestone) {
      final blockerTitles = milestone.dependsOnIds
          .map((id) => byId[id])
          .whereType<AdmissionsMilestone>()
          .where((item) => !item.completed)
          .map((item) => item.title)
          .toList();
      return AdmissionsMilestoneMapNode(
        milestone: milestone,
        blockerTitles: blockerTitles,
      );
    }).toList()
      ..sort((a, b) => a.milestone.dueAt.compareTo(b.milestone.dueAt));
  }

  Future<Map<String, dynamic>> generateUniversityFit({
    required Map<String, String> predictedGrades,
    required String targetCourse,
    List<String> countries = const ['UK', 'Singapore'],
    List<String> portfolioLinks = const [],
    double readinessScore = 0,
    List<Map<String, String>> portfolioEvidence = const [],
  }) async {
    final token = await FirebaseAuth.instance.currentUser?.getIdToken();
    final response = await _client.post(
      Uri.parse('$_backendUrl/generate/university-fit'),
      headers: {
        'Content-Type': 'application/json',
        if (token != null) 'Authorization': 'Bearer $token',
      },
      body: jsonEncode({
        'predicted_grades': predictedGrades,
        'target_course': targetCourse,
        'countries': countries,
        'portfolio_links': portfolioLinks,
        'portfolio_evidence': portfolioEvidence,
        'readiness_score': readinessScore,
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(
        'Failed to generate university fit (${response.statusCode})',
      );
    }

    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    return Map<String, dynamic>.from(payload['fit'] as Map);
  }

  Future<void> saveTarget({
    required String uid,
    required AdmissionsTarget target,
  }) async {
    try {
      await _targets(uid).doc(target.id).set({
        'university_name': target.universityName,
        'country': target.country,
        'course_name': target.courseName,
        'classification': target.classification,
        'application_system': target.applicationSystem,
        'status': target.status,
        'rationale': target.rationale,
        'entry_requirements': target.entryRequirements,
        'source_url': target.sourceUrl,
        'deadline_at': target.deadlineAt?.toIso8601String(),
        'fit_band': target.fitBand,
        'readiness_score': target.readinessScore,
      }, SetOptions(merge: true));
    } catch (e) {
      // Handle permission errors gracefully
      if (e is FirebaseException && e.code == 'permission-denied') {
        debugPrint('Permission denied when saving target: ${e.message}');
        // Don't throw the error to prevent app crash
        return;
      }
      // Re-throw other errors
      rethrow;
    }
  }

  Future<void> saveTargetWithDefaultMilestones({
    required String uid,
    required AdmissionsTarget target,
    DateTime? now,
  }) async {
    final anchor = now ?? DateTime.now();

    // Check if target already exists for this university + course to avoid duplicates
    final exists = await _targetExists(uid, target.universityName, target.courseName);
    if (exists) {
      debugPrint('AdmissionsService: Target already exists for ${target.universityName} - ${target.courseName}, skipping');
      return;
    }

    await saveTarget(uid: uid, target: target);

    // Smart milestones based on university and course
    final deadlineStr = target.deadlineAt != null
        ? 'by ${target.deadlineAt!.day}/${target.deadlineAt!.month}/${target.deadlineAt!.year}'
        : '';
    final uni = target.universityName;

    await saveMilestone(
      uid: uid,
      milestone: AdmissionsMilestone(
        id: '${target.id}_verify',
        targetId: target.id,
        title: 'Verify entry requirements for $uni',
        dueAt: anchor.add(const Duration(days: 3)),
        completed: false,
        dependencyType: 'verification',
        phase: 'research',
        status: 'pending',
      ),
    );
    await saveMilestone(
      uid: uid,
      milestone: AdmissionsMilestone(
        id: '${target.id}_evidence',
        targetId: target.id,
        title: 'Prepare personal statement for $uni $deadlineStr',
        dueAt: anchor.add(const Duration(days: 10)),
        completed: false,
        dependencyType: 'evidence',
        phase: 'portfolio',
        dependsOnIds: ['${target.id}_verify'],
        status: 'pending',
      ),
    );
    await saveMilestone(
      uid: uid,
      milestone: AdmissionsMilestone(
        id: '${target.id}_apply',
        targetId: target.id,
        title: 'Submit application to $uni',
        dueAt: target.deadlineAt ?? anchor.add(const Duration(days: 21)),
        completed: false,
        dependencyType: 'submission',
        phase: 'application',
        dependsOnIds: ['${target.id}_verify', '${target.id}_evidence'],
        status: 'pending',
      ),
    );

    // Cache locally for offline fallback
    await _cacheTarget(uid, target);
  }

  Future<bool> _targetExists(String uid, String universityName, String courseName) async {
    try {
      // Use single-field query + in-memory filter to avoid composite index dependency.
      // Firestore requires composite index for dual where() queries; fall back to a
      // single where() + in-memory filter to prevent silent failures that cause
      // duplicate targets and their milestone cascades.
      final snapshot = await _targets(uid)
          .where('university_name', isEqualTo: universityName)
          .get();
      final normalizedCourse = courseName.toLowerCase().trim();
      return snapshot.docs.any((doc) {
        final data = doc.data();
        final existingCourse = (data['course_name'] as String? ?? '').toLowerCase().trim();
        return existingCourse == normalizedCourse;
      });
    } catch (_) {
      // If query fails entirely, do NOT silently proceed — treat as "exists" to prevent
      // accidental duplicate creation. The user can manually remove any real duplicate.
      return true;
    }
  }

  Future<void> _cacheTarget(String uid, AdmissionsTarget target) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final key = 'cached_targets_$uid';
      final existing = prefs.getStringList(key) ?? [];
      existing.add(jsonEncode({
        'id': target.id,
        'university_name': target.universityName,
        'country': target.country,
        'course_name': target.courseName,
        'classification': target.classification,
        'application_system': target.applicationSystem,
        'status': target.status,
        'rationale': target.rationale,
        'entry_requirements': target.entryRequirements,
        'source_url': target.sourceUrl,
        'fit_band': target.fitBand,
        'readiness_score': target.readinessScore,
      }));
      await prefs.setStringList(key, existing);
    } catch (_) {}
  }

  Future<List<AdmissionsTarget>> getCachedTargets(String uid) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final key = 'cached_targets_$uid';
      final data = prefs.getStringList(key) ?? [];
      return data.map((json) {
        final map = jsonDecode(json) as Map<String, dynamic>;
        return AdmissionsTarget(
          id: map['id'] ?? '',
          universityName: map['university_name'] ?? '',
          country: map['country'] ?? '',
          courseName: map['course_name'] ?? '',
          classification: map['classification'] ?? '',
          applicationSystem: map['application_system'] ?? '',
          status: map['status'] ?? '',
          rationale: map['rationale'] ?? '',
          entryRequirements: map['entry_requirements'] ?? '',
          sourceUrl: map['source_url'] ?? '',
          fitBand: map['fit_band'] ?? '',
          readinessScore: (map['readiness_score'] as num?)?.toDouble() ?? 0,
        );
      }).toList();
    } catch (_) {
      return [];
    }
  }

  Future<AdmissionsTarget> createManualTarget({
    required String uid,
    required String universityName,
    required String courseName,
    String country = '',
    String classification = 'match',
    String applicationSystem = '',
    String entryRequirements = '',
    String sourceUrl = '',
    DateTime? deadlineAt,
    double readinessScore = 0,
  }) async {
    final targetId = 'target_${DateTime.now().millisecondsSinceEpoch}';
    final target = AdmissionsTarget(
      id: targetId,
      universityName: universityName.trim(),
      country: country.trim(),
      courseName: courseName.trim(),
      classification: classification.trim().isEmpty ? 'match' : classification,
      applicationSystem: applicationSystem.trim().isNotEmpty
          ? applicationSystem.trim()
          : _applicationSystemForCountry(country),
      status: 'planned',
      rationale: 'Manually added target from AXON analytics.',
      entryRequirements: entryRequirements.trim(),
      sourceUrl: sourceUrl.trim(),
      deadlineAt: deadlineAt,
      fitBand: classification.trim().isEmpty ? 'match' : classification.trim(),
      readinessScore: readinessScore,
    );
    await saveTargetWithDefaultMilestones(uid: uid, target: target);
    return target;
  }

  Future<AdmissionsTarget> saveProgramTarget({
    required String uid,
    required UniversityProgram program,
    required String bucket,
    required String board,
    required List<String> subjects,
    double readinessScore = 0,
  }) async {
    final normalizedBucket = _normalizeBucket(bucket);
    final target = AdmissionsTarget(
      id: 'target_${program.id}_${DateTime.now().millisecondsSinceEpoch}',
      universityName: program.universityName,
      country: program.location,
      courseName: program.courseName,
      classification: normalizedBucket,
      applicationSystem: _applicationSystemForCountry(program.location),
      status: 'planned',
      rationale:
          'Saved from AXON discovery. Bucket: ${_bucketLabel(normalizedBucket)}.',
      entryRequirements: program.requirementSummaryFor(
        board: board,
        subjects: subjects,
      ),
      sourceUrl: program.sourceUrl,
      fitBand: _bucketLabel(normalizedBucket),
      readinessScore: readinessScore,
    );
    await saveTargetWithDefaultMilestones(uid: uid, target: target);
    return target;
  }

  Future<void> saveMilestone({
    required String uid,
    required AdmissionsMilestone milestone,
  }) async {
    try {
      await _milestones(uid).doc(milestone.id).set({
        'target_id': milestone.targetId,
        'title': milestone.title,
        'due_at': milestone.dueAt.toIso8601String(),
        'completed': milestone.completed,
        'dependency_type': milestone.dependencyType,
        'phase': milestone.phase,
        'depends_on_ids': milestone.dependsOnIds,
        'evidence_asset_ids': milestone.evidenceAssetIds,
        'status': milestone.status,
      }, SetOptions(merge: true));
    } catch (e) {
      if (e is FirebaseException && e.code == 'permission-denied') {
        debugPrint('Permission denied when saving milestone: ${e.message}');
        return;
      }
      rethrow;
    }
  }

  Future<void> saveVaultAsset({
    required String uid,
    required String title,
    required String assetType,
    required String url,
    String provider = '',
    String projectName = '',
    List<String> tags = const [],
    List<String> linkedTargetIds = const [],
    String extractedText = '',
    String targetDegree = '',
    int relevanceRating = 0,
  }) async {
    await _vault(uid).add({
      'title': title,
      'asset_type': assetType,
      'url': url,
      'provider': provider,
      'project_name': projectName,
      'tags': tags,
      'linked_target_ids': linkedTargetIds,
      'extracted_text': extractedText,
      'target_degree': targetDegree,
      'relevance_rating': relevanceRating,
      'created_at': DateTime.now().toIso8601String(),
    });
  }

  Future<void> saveMarkEntry({
    required String uid,
    required String subject,
    required double currentMark,
    required String currentGrade,
    required String targetGrade,
  }) async {
    final id = subject.trim().toLowerCase().replaceAll(' ', '_');
    await _marks(uid).doc(id).set({
      'subject': subject.trim(),
      'current_mark': currentMark.clamp(0, 100),
      'current_grade': currentGrade.trim(),
      'target_grade': targetGrade.trim(),
      'updated_at': DateTime.now().toIso8601String(),
    }, SetOptions(merge: true));
  }

  Future<AdmissionsVaultAsset> uploadAchievementAsset({
    required String uid,
    required File file,
    required String title,
    required String targetDegree,
    String extractedText = '',
    List<String> linkedTargetIds = const [],
  }) async {
    final fileName = p.basename(file.path);
    final path =
        'users/$uid/admissions_vault/${DateTime.now().millisecondsSinceEpoch}_$fileName';
    final ref = FirebaseStorage.instance.ref(path);
    await ref.putFile(file);
    final url = await ref.getDownloadURL();
    final rating = extractedText.trim().isEmpty
        ? 0
        : await rateAchievementForDegree(
            achievementText: extractedText,
            targetDegree: targetDegree,
          );
    await saveVaultAsset(
      uid: uid,
      title: title,
      assetType: _assetTypeFromPath(file.path),
      url: url,
      provider: 'AXON Vault',
      projectName: targetDegree,
      tags: ['achievement', targetDegree],
      linkedTargetIds: linkedTargetIds,
      extractedText: extractedText,
      targetDegree: targetDegree,
      relevanceRating: rating,
    );
    return AdmissionsVaultAsset(
      id: path,
      title: title,
      assetType: _assetTypeFromPath(file.path),
      url: url,
      provider: 'AXON Vault',
      projectName: targetDegree,
      tags: ['achievement', targetDegree],
      linkedTargetIds: linkedTargetIds,
      extractedText: extractedText,
      targetDegree: targetDegree,
      relevanceRating: rating,
    );
  }

  Future<int> rateAchievementForDegree({
    required String achievementText,
    required String targetDegree,
  }) async {
    final prompt =
        'Rate this user achievement on a scale of 1-10 based on relevance, selectivity, and impact for a $targetDegree application. Return only the integer.\n\nAchievement:\n$achievementText';
    final response = await GrokService().chat(
      prompt,
      systemPrompt:
          'You are an admissions evidence evaluator. Return only one integer from 1 to 10. Do not explain.',
      maxTokens: 4,
      temperature: 0,
    );
    final rating = int.tryParse(response.trim());
    return (rating ?? 0).clamp(0, 10);
  }

  Future<void> persistGeneratedFit({
    required String uid,
    required String targetCourse,
    required Map<String, dynamic> fit,
    double readinessScore = 0,
  }) async {
    final targets = <Map<String, dynamic>>[];
    for (final bucket in ['safety', 'match', 'reach', 'dream']) {
      final entries = (fit[bucket] as List?) ?? const [];
      for (final entry in entries) {
        if (entry is Map<String, dynamic>) {
          targets.add({...entry, 'bucket': bucket});
        }
      }
    }

    for (var i = 0; i < targets.length; i++) {
      final item = targets[i];
      final targetId = 'target_${DateTime.now().millisecondsSinceEpoch}_$i';
      final target = AdmissionsTarget(
        id: targetId,
        universityName: (item['name'] ?? '').toString(),
        country: (item['country'] ?? '').toString(),
        courseName: targetCourse,
        classification: _normalizeBucket((item['bucket'] ?? 'reach').toString()),
        applicationSystem: _applicationSystemForCountry(
          (item['country'] ?? '').toString(),
        ),
        status: 'planned',
        rationale: (item['rationale'] ?? '').toString(),
        entryRequirements: (item['entry_requirements'] ?? '').toString(),
        fitBand: _bucketLabel(_normalizeBucket((item['bucket'] ?? 'reach').toString())),
        readinessScore: readinessScore,
      );
      await saveTargetWithDefaultMilestones(uid: uid, target: target);
    }

    final strategyNotes = (fit['strategy_notes'] ?? '').toString().trim();
    if (strategyNotes.isNotEmpty) {
      await saveVaultAsset(
        uid: uid,
        title: 'Admissions Strategy Notes',
        assetType: 'strategy_note',
        url: 'local://admissions-strategy',
        tags: ['admissions', 'strategy'],
        projectName: targetCourse,
      );
      debugPrint('Admissions strategy: $strategyNotes');
    }
  }

  String _applicationSystemForCountry(String country) {
    final normalized = country.toLowerCase();
    if (normalized.contains('uk')) return 'UCAS';
    if (normalized.contains('singapore')) return 'Direct';
    return 'Direct';
  }

  String _normalizeBucket(String bucket) {
    final lower = bucket.toLowerCase().trim();
    if (lower.contains('dream')) return 'dream';
    if (lower.contains('reach') || lower.contains('match') || lower.contains('target')) {
      return 'reach';
    }
    if (lower.contains('safe')) return 'safety';
    return 'reach';
  }

  String _bucketLabel(String bucket) {
    return switch (_normalizeBucket(bucket)) {
      'dream' => 'Dream',
      'safety' => 'Safety',
      _ => 'Reach',
    };
  }

  String _assetTypeFromPath(String path) {
    final extension = p.extension(path).toLowerCase();
    if (extension == '.pdf') return 'certificate';
    if (['.png', '.jpg', '.jpeg', '.webp'].contains(extension)) {
      return 'portfolio_image';
    }
    if (['.mp4', '.mov'].contains(extension)) return 'video';
    return 'portfolio_file';
  }

  String composeFitBand(Map<String, dynamic> item) {
    final gradeFit = (item['grade_fit'] ?? '').toString();
    final readinessFit = (item['readiness_fit'] ?? '').toString();
    if (gradeFit.isEmpty && readinessFit.isEmpty) {
      return '';
    }
    return [gradeFit, readinessFit]
        .where((part) => part.isNotEmpty)
        .join(' · ');
  }
}
