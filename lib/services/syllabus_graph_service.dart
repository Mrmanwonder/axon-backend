import 'package:cloud_firestore/cloud_firestore.dart';

import '../models/academic_engine_models.dart';
import 'firestore_service.dart';

class SyllabusGraphService {
  SyllabusGraphService({FirebaseFirestore? firestore})
      : _firestore = firestore ?? AxonFirestore.instance;

  final FirebaseFirestore _firestore;

  CollectionReference<Map<String, dynamic>> get _syllabusMaps =>
      _firestore.collection('syllabus_maps');

  DocumentReference<Map<String, dynamic>> _masteryDoc(
    String uid,
    String objectiveId,
  ) =>
      AxonPaths.privateUserCollection(uid, 'mastery').doc(objectiveId);

  CollectionReference<Map<String, dynamic>> _events(String uid) =>
      AxonPaths.privateUserCollection(uid, 'events');

  CollectionReference<Map<String, dynamic>> _mockResults(String uid) =>
      AxonPaths.privateUserCollection(uid, 'mock_results');

  Future<List<SyllabusLearningObjective>> fetchObjectives({
    required String board,
    required String subject,
  }) async {
    final snapshot = await _syllabusMaps
        .where('board', isEqualTo: board)
        .where('subject', isEqualTo: subject)
        .get();
    return snapshot.docs
        .map((doc) => SyllabusLearningObjective.fromJson(doc.data()))
        .toList();
  }

  Future<void> upsertMastery({
    required String uid,
    required MasteryRecord record,
  }) async {
    final currentMastery = record.decayedMastery();
    await _masteryDoc(uid, record.learningObjectiveId).set(
      {
        ...record.toJson(),
        'mastery_score': currentMastery,
        'updated_at': FieldValue.serverTimestamp(),
      },
      SetOptions(merge: true),
    );
  }

  Stream<List<MasteryRecord>> watchMastery(String uid) {
    return AxonPaths.privateUserCollection(uid, 'mastery')
        .snapshots()
        .map((snapshot) => snapshot.docs
            .map((doc) => MasteryRecord.fromJson(doc.data()))
            .toList());
  }

  Future<void> appendStudyEvent({
    required String uid,
    required StudyEventLog event,
  }) async {
    await _events(uid).doc(event.id).set({
      ...event.toJson(),
      'created_at': FieldValue.serverTimestamp(),
    });
  }

  Future<void> appendMockResult({
    required String uid,
    required MockResultRecord record,
  }) async {
    await _mockResults(uid).doc(record.id).set({
      ...record.toJson(),
      'created_at': FieldValue.serverTimestamp(),
    });
  }

  Stream<List<StudyEventLog>> watchRecentEvents(String uid, {int limit = 100}) {
    return _events(uid)
        .orderBy('occurred_at', descending: true)
        .limit(limit)
        .snapshots()
        .map((snapshot) => snapshot.docs
            .map((doc) => StudyEventLog(
                  id: (doc.data()['id'] ?? doc.id).toString(),
                  type: (doc.data()['type'] ?? '').toString(),
                  durationMinutes:
                      (doc.data()['duration_minutes'] as num?)?.toInt() ?? 0,
                  learningObjectiveId:
                      (doc.data()['learning_objective_id'] ??
                              doc.data()['objective_id'] ??
                              doc.data()['topic_id'] ??
                              '')
                          .toString(),
                  intensity: (doc.data()['intensity'] as num?)?.toDouble() ?? 0,
                  occurredAt: DateTime.tryParse(
                          (doc.data()['occurred_at'] ?? '').toString()) ??
                      DateTime.now(),
                  subject: (doc.data()['subject'] ?? '').toString(),
                ))
            .toList());
  }
}
