import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';

class AxonFirestore {
  AxonFirestore._();

  static const String databaseId = 'axon';
  static FirebaseFirestore? _instance;
  static bool _initialized = false;

  static FirebaseFirestore get instance {
    _instance ??= FirebaseFirestore.instanceFor(
      app: Firebase.app(),
      databaseId: databaseId,
    );
    return _instance!;
  }

  static Future<void> initialize() async {
    if (_initialized) return;
    final firestore = instance;
    firestore.settings = const Settings(
      persistenceEnabled: true,
      cacheSizeBytes: Settings.CACHE_SIZE_UNLIMITED,
    );
    _initialized = true;
  }
}

class AxonCollections {
  AxonCollections._();

  static const String usersLegacy = 'users';
  static const String usersPrivate = 'users_private';
  static const String usersPublic = 'users_public';
  static const String jobs = 'jobs';
  static const String syllabusMaps = 'syllabus_maps';
}

class AxonPaths {
  AxonPaths._();

  static CollectionReference<Map<String, dynamic>> privateUsers() =>
      AxonFirestore.instance.collection(AxonCollections.usersPrivate);

  static CollectionReference<Map<String, dynamic>> publicUsers() =>
      AxonFirestore.instance.collection(AxonCollections.usersPublic);

  static CollectionReference<Map<String, dynamic>> legacyUsers() =>
      AxonFirestore.instance.collection(AxonCollections.usersLegacy);

  static DocumentReference<Map<String, dynamic>> privateUserDoc(String uid) =>
      privateUsers().doc(uid);

  static DocumentReference<Map<String, dynamic>> publicUserDoc(String uid) =>
      publicUsers().doc(uid);

  static CollectionReference<Map<String, dynamic>> privateUserCollection(
    String uid,
    String collection,
  ) =>
      privateUserDoc(uid).collection(collection);
}
