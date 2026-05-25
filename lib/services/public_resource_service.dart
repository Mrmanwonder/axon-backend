import 'dart:developer' as dev;
import 'package:cloud_firestore/cloud_firestore.dart';

const String kPublicResourcesCollection = 'public_resources';

class PublicResourceEntry {
  final String uid;
  final String board;
  final String qualification;
  final String title;

  PublicResourceEntry({
    required this.uid,
    required this.board,
    required this.qualification,
    this.title = '',
  });

  factory PublicResourceEntry.fromFirestore(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>;
    return PublicResourceEntry(
      uid: data['uid'] as String? ?? '',
      board: data['board'] as String? ?? '',
      qualification: data['qualification'] as String? ?? '',
      title: data['title'] as String? ?? '',
    );
  }
}

class PublicResourceService {
  static final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  static Future<List<PublicResourceEntry>> fetchPublicResources() async {
    try {
      final snapshot = await _firestore
          .collection(kPublicResourcesCollection)
          .limit(50)
          .get();
      return snapshot.docs
          .map((doc) => PublicResourceEntry.fromFirestore(doc))
          .toList();
    } catch (e) {
      dev.log('Failed to fetch public resources: $e');
      return [];
    }
  }
}
