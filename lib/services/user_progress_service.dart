import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import '../services/firestore_service.dart';

class UserProgressService {
  UserProgressService._();
  static final UserProgressService instance = UserProgressService._();

  Future<void> updateSubchapterProgress({
    required String uid,
    required String subjectCode,
    required int chapterNumber,
    required int subchapterNumber,
    required double scrollPercentage,
    required int timeSpentSeconds,
  }) async {
    try {
      final docRef = AxonPaths.privateUserDoc(uid);
      final progressRef = docRef.collection('subchapter_progress').doc(
            '${subjectCode}_${chapterNumber}_$subchapterNumber',
          );

      await progressRef.set({
        'subjectCode': subjectCode,
        'chapterNumber': chapterNumber,
        'subchapterNumber': subchapterNumber,
        'scrollPercentage': scrollPercentage,
        'timeSpentSeconds': timeSpentSeconds,
        'lastUpdated': FieldValue.serverTimestamp(),
      }, SetOptions(merge: true));
    } catch (e) {
      debugPrint('UserProgressService.updateSubchapterProgress error: $e');
    }
  }

  Future<SubchapterProgress?> getSubchapterProgress({
    required String uid,
    required String subjectCode,
    required int chapterNumber,
    required int subchapterNumber,
  }) async {
    try {
      final docRef = AxonPaths.privateUserDoc(uid);
      final doc = await docRef
          .collection('subchapter_progress')
          .doc('${subjectCode}_${chapterNumber}_$subchapterNumber')
          .get();

      if (doc.exists && doc.data() != null) {
        return SubchapterProgress.fromMap(doc.data()!);
      }
    } catch (e) {
      debugPrint('UserProgressService.getSubchapterProgress error: $e');
    }
    return null;
  }

  Future<List<SubchapterProgress>> getAllProgressForSubject({
    required String uid,
    required String subjectCode,
  }) async {
    try {
      final docRef = AxonPaths.privateUserDoc(uid);
      final snapshot = await docRef
          .collection('subchapter_progress')
          .where('subjectCode', isEqualTo: subjectCode)
          .get();

      return snapshot.docs
          .map((doc) => SubchapterProgress.fromMap(doc.data()))
          .toList();
    } catch (e) {
      debugPrint('UserProgressService.getAllProgressForSubject error: $e');
      return [];
    }
  }

  Future<void> deleteSubchapterProgress({
    required String uid,
    required String subjectCode,
    required int chapterNumber,
    required int subchapterNumber,
  }) async {
    try {
      final docRef = AxonPaths.privateUserDoc(uid);
      await docRef
          .collection('subchapter_progress')
          .doc('${subjectCode}_${chapterNumber}_$subchapterNumber')
          .delete();
    } catch (e) {
      debugPrint('UserProgressService.deleteSubchapterProgress error: $e');
    }
  }
}

class SubchapterProgress {
  final String subjectCode;
  final int chapterNumber;
  final int subchapterNumber;
  final double scrollPercentage;
  final int timeSpentSeconds;
  final DateTime? lastUpdated;

  SubchapterProgress({
    required this.subjectCode,
    required this.chapterNumber,
    required this.subchapterNumber,
    required this.scrollPercentage,
    required this.timeSpentSeconds,
    this.lastUpdated,
  });

  factory SubchapterProgress.fromMap(Map<String, dynamic> map) {
    return SubchapterProgress(
      subjectCode: map['subjectCode'] ?? '',
      chapterNumber: map['chapterNumber'] ?? 0,
      subchapterNumber: map['subchapterNumber'] ?? 0,
      scrollPercentage: (map['scrollPercentage'] ?? 0.0).toDouble(),
      timeSpentSeconds: map['timeSpentSeconds'] ?? 0,
      lastUpdated: map['lastUpdated'] is Timestamp
          ? (map['lastUpdated'] as Timestamp).toDate()
          : null,
    );
  }

  Map<String, dynamic> toMap() => {
        'subjectCode': subjectCode,
        'chapterNumber': chapterNumber,
        'subchapterNumber': subchapterNumber,
        'scrollPercentage': scrollPercentage,
        'timeSpentSeconds': timeSpentSeconds,
        'lastUpdated': lastUpdated,
      };
}
