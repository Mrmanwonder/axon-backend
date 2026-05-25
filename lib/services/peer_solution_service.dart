import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';

import 'firestore_service.dart';

class PeerSolutionEntry {
  final String id;
  final String questionId;
  final String questionText;
  final String subject;
  final String chapter;
  final String topic;
  final String sourceTitle;
  final String authorUid;
  final String authorName;
  final String logicSummary;
  final String explanation;
  final String imageUrl;
  final String imagePublicId;
  final int voteScore;
  final int flagCount;
  final bool isPinned;
  final DateTime createdAt;

  const PeerSolutionEntry({
    required this.id,
    required this.questionId,
    required this.questionText,
    required this.subject,
    required this.chapter,
    required this.topic,
    required this.sourceTitle,
    required this.authorUid,
    required this.authorName,
    required this.logicSummary,
    required this.explanation,
    required this.imageUrl,
    required this.imagePublicId,
    required this.voteScore,
    required this.flagCount,
    required this.isPinned,
    required this.createdAt,
  });

  factory PeerSolutionEntry.fromJson(String id, Map<String, dynamic> json) {
    return PeerSolutionEntry(
      id: id,
      questionId: (json['question_id'] ?? '').toString(),
      questionText: (json['question_text'] ?? '').toString(),
      subject: (json['subject'] ?? '').toString(),
      chapter: (json['chapter'] ?? '').toString(),
      topic: (json['topic'] ?? '').toString(),
      sourceTitle: (json['source_title'] ?? '').toString(),
      authorUid: (json['author_uid'] ?? '').toString(),
      authorName: (json['author_name'] ?? 'Anonymous').toString(),
      logicSummary: (json['logic_summary'] ?? '').toString(),
      explanation: (json['explanation'] ?? '').toString(),
      imageUrl: (json['image_url'] ?? '').toString(),
      imagePublicId: (json['image_public_id'] ?? '').toString(),
      voteScore: (json['vote_score'] as num?)?.toInt() ?? 0,
      flagCount: (json['flag_count'] as num?)?.toInt() ?? 0,
      isPinned: json['is_pinned'] == true,
      createdAt: DateTime.tryParse((json['created_at'] ?? '').toString()) ??
          DateTime.now(),
    );
  }
}

class PeerSolutionService {
  PeerSolutionService._();

  static final PeerSolutionService instance = PeerSolutionService._();

  final FirebaseFirestore _db = AxonFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  CollectionReference<Map<String, dynamic>> get _solutions =>
      _db.collection('solution_wall_entries');

  DocumentReference<Map<String, dynamic>> _voteDoc(String solutionId, String uid) {
    return _solutions.doc(solutionId).collection('votes').doc(uid);
  }

  Stream<List<PeerSolutionEntry>> watchSolutions(String questionId) {
    return _solutions
        .where('question_id', isEqualTo: questionId)
        .where('flag_count', isLessThan: 5)
        .snapshots()
        .map((snapshot) {
      final entries = snapshot.docs
          .map((doc) => PeerSolutionEntry.fromJson(doc.id, doc.data()))
          .toList();
      entries.sort((a, b) {
        if (a.isPinned != b.isPinned) return a.isPinned ? -1 : 1;
        final vote = b.voteScore.compareTo(a.voteScore);
        if (vote != 0) return vote;
        return b.createdAt.compareTo(a.createdAt);
      });
      return entries;
    });
  }

  Future<void> submitSolution({
    required String questionId,
    required String questionText,
    required String subject,
    required String chapter,
    required String topic,
    required String sourceTitle,
    required String logicSummary,
    required String explanation,
    String imageUrl = '',
    String imagePublicId = '',
  }) async {
    final user = _auth.currentUser;
    if (user == null) {
      throw StateError('No authenticated user');
    }

    final displayName = (user.displayName ?? '').trim();
    await _solutions.add({
      'question_id': questionId,
      'question_text': questionText,
      'subject': subject,
      'chapter': chapter,
      'topic': topic,
      'source_title': sourceTitle,
      'author_uid': user.uid,
      'author_name': displayName.isEmpty ? 'Axon Student' : displayName,
      'logic_summary': logicSummary.trim(),
      'explanation': explanation.trim(),
      'image_url': imageUrl,
      'image_public_id': imagePublicId,
      'vote_score': 0,
      'flag_count': 0,
      'is_pinned': false,
      'created_at': DateTime.now().toIso8601String(),
    });
  }

  Future<int> getMyVote(String solutionId) async {
    final uid = _auth.currentUser?.uid;
    if (uid == null) return 0;
    final snapshot = await _voteDoc(solutionId, uid).get();
    return (snapshot.data()?['value'] as num?)?.toInt() ?? 0;
  }

  Future<void> vote(String solutionId, int nextValue) async {
    final uid = _auth.currentUser?.uid;
    if (uid == null) {
      throw StateError('No authenticated user');
    }
    await _db.runTransaction((txn) async {
      final solutionRef = _solutions.doc(solutionId);
      final voteRef = _voteDoc(solutionId, uid);
      final solutionSnap = await txn.get(solutionRef);
      final voteSnap = await txn.get(voteRef);
      if (!solutionSnap.exists) {
        throw StateError('Solution not found');
      }

      final previous = (voteSnap.data()?['value'] as num?)?.toInt() ?? 0;
      final applied = previous == nextValue ? 0 : nextValue.clamp(-1, 1);
      final delta = applied - previous;
      final currentScore =
          (solutionSnap.data()?['vote_score'] as num?)?.toInt() ?? 0;

      txn.set(voteRef, {
        'uid': uid,
        'value': applied,
        'updated_at': DateTime.now().toIso8601String(),
      });
      txn.update(solutionRef, {'vote_score': currentScore + delta});
    });
  }

  Future<void> flagSolution(String solutionId) async {
    await _solutions.doc(solutionId).update({
      'flag_count': FieldValue.increment(1),
      'updated_at': DateTime.now().toIso8601String(),
    });
  }

  Future<void> deleteSolution(String solutionId) async {
    final uid = _auth.currentUser?.uid;
    if (uid == null) return;
    final ref = _solutions.doc(solutionId);
    final snapshot = await ref.get();
    final ownerUid = snapshot.data()?['author_uid']?.toString() ?? '';
    if (ownerUid != uid) {
      debugPrint('PeerSolutionService: refusing to delete non-owned solution');
      return;
    }
    await ref.delete();
  }
}
