import 'dart:developer' as dev;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'firestore_service.dart';

class ManifestSyncService {
  static final FirebaseFirestore _firestore = AxonFirestore.instance;

  static Future<void> syncManifest() async {
    try {
      final manifestRef = _firestore.collection('manifests').doc('current');
      final doc = await manifestRef.get();
      if (!doc.exists) return;

      final data = doc.data() as Map<String, dynamic>;
      dev.log('Manifest synced: $data');
    } catch (e) {
      dev.log('Manifest sync error: $e');
    }
  }
}
