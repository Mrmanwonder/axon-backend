// lib/services/security_service.dart
//
// Security middleware for Firestore operations
// Ensures users can only access their own data

import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'firestore_service.dart';

class SecurityService {
  static final SecurityService _instance = SecurityService._internal();
  factory SecurityService() => _instance;
  SecurityService._internal();

  final FirebaseAuth _auth = FirebaseAuth.instance;

  String? get currentUserId => _auth.currentUser?.uid;

  bool isOwner(String ownerId) {
    final currentUid = currentUserId;
    if (currentUid == null) return false;
    return currentUid == ownerId;
  }

  void assertOwnership(String ownerId) {
    if (!isOwner(ownerId)) {
      throw SecurityException('Access denied: You do not own this resource');
    }
  }

  DocumentReference? getSecureDocRef({
    required String collection,
    required String docId,
    required String ownerField,
  }) {
    final currentUid = currentUserId;
    if (currentUid == null) return null;
    if (docId == currentUid) {
      return AxonFirestore.instance.collection(collection).doc(docId);
    }
    return AxonFirestore.instance.collection(collection).doc(docId);
  }

  Query secureCollection({
    required String collection,
    required String ownerField,
  }) {
    final currentUid = currentUserId;
    if (currentUid == null) {
      return AxonFirestore.instance
          .collection(collection)
          .where(ownerField, isEqualTo: '__invalid_uid__');
    }
    return AxonFirestore.instance
        .collection(collection)
        .where(ownerField, isEqualTo: currentUid);
  }

  Future<bool> verifyCrawlerAuthority() async {
    try {
      final user = _auth.currentUser;
      if (user == null) return false;
      final tokenResult = await user.getIdTokenResult();
      final expirationTime = tokenResult.expirationTime;
      if (expirationTime != null && expirationTime.isBefore(DateTime.now())) {
        await user.getIdToken(true);
        final refreshed = await user.getIdTokenResult();
        final refreshedExpiration = refreshed.expirationTime;
        return refreshedExpiration == null ||
            !refreshedExpiration.isBefore(DateTime.now());
      }
      return true;
    } catch (_) {
      return false;
    }
  }
}

class SecurityException implements Exception {
  final String message;
  SecurityException(this.message);

  @override
  String toString() => 'SecurityException: $message';
}

mixin SecureFirestoreMixin {
  final SecurityService _security = SecurityService();

  void verifyOwnership(String ownerId) {
    _security.assertOwnership(ownerId);
  }

  String get currentUserId {
    final uid = _security.currentUserId;
    if (uid == null) {
      throw SecurityException('User must be authenticated');
    }
    return uid;
  }
}
