import 'package:cloud_firestore/cloud_firestore.dart';
import 'encryption_service.dart';

extension SecureDocumentReference on DocumentReference<Map<String, dynamic>> {
  /// Encrypts the provided data, keeping the specified fields unencrypted for querying.
  Future<void> setSecure(Map<String, dynamic> data, {SetOptions? options, List<String> unencryptedFields = const []}) async {
    final Map<String, dynamic> unencryptedPart = {};
    final Map<String, dynamic> encryptedPart = {};

    for (final entry in data.entries) {
      if (unencryptedFields.contains(entry.key)) {
        unencryptedPart[entry.key] = entry.value;
      } else {
        encryptedPart[entry.key] = entry.value;
      }
    }

    final payloadToUpload = Map<String, dynamic>.from(unencryptedPart);
    if (encryptedPart.isNotEmpty) {
      final encryptedString = await EncryptionService.encryptMap(encryptedPart);
      payloadToUpload['_encrypted'] = encryptedString;
    }

    await set(payloadToUpload, options);
  }

  /// Updates a document securely. Since the payload is encrypted as a single string,
  /// this performs a read, decrypt, merge, encrypt, and write operation.
  Future<void> updateSecure(Map<String, dynamic> data, {List<String> unencryptedFields = const []}) async {
     final snapshot = await getSecure();
     if (snapshot == null) {
       throw Exception('Cannot update a non-existent document securely. Use setSecure instead.');
     }
     final mergedData = Map<String, dynamic>.from(snapshot)..addAll(data);
     await setSecure(mergedData, unencryptedFields: unencryptedFields);
  }

  /// Retrieves and decrypts the document.
  Future<Map<String, dynamic>?> getSecure() async {
    final snapshot = await get();
    if (!snapshot.exists || snapshot.data() == null) return null;
    
    final data = snapshot.data()!;
    if (data.containsKey('_encrypted')) {
      final decryptedMap = await EncryptionService.decryptMap(data['_encrypted'] as String);
      final result = Map<String, dynamic>.from(data)..remove('_encrypted');
      result.addAll(decryptedMap);
      return result;
    }
    
    return data;
  }

  /// Listens to document changes and decrypts incoming snapshots.
  Stream<Map<String, dynamic>?> snapshotsSecure() {
    return snapshots().asyncMap((snapshot) async {
      if (!snapshot.exists || snapshot.data() == null) return null;
      
      final data = snapshot.data()!;
      if (data.containsKey('_encrypted')) {
        final decryptedMap = await EncryptionService.decryptMap(data['_encrypted'] as String);
        final result = Map<String, dynamic>.from(data)..remove('_encrypted');
        result.addAll(decryptedMap);
        return result;
      }
      return data;
    });
  }
}

extension SecureQuery on Query<Map<String, dynamic>> {
  /// Retrieves and decrypts the query results.
  Future<List<Map<String, dynamic>>> getSecure() async {
    final snapshot = await get();
    final results = <Map<String, dynamic>>[];
    
    for (final doc in snapshot.docs) {
      final data = doc.data();
      if (data.containsKey('_encrypted')) {
        final decryptedMap = await EncryptionService.decryptMap(data['_encrypted'] as String);
        final result = Map<String, dynamic>.from(data)..remove('_encrypted');
        result.addAll(decryptedMap);
        result['id'] = doc.id;
        results.add(result);
      } else {
        final result = Map<String, dynamic>.from(data);
        result['id'] = doc.id;
        results.add(result);
      }
    }
    
    return results;
  }
  
  /// Listens to query results and decrypts incoming snapshots.
  Stream<List<Map<String, dynamic>>> snapshotsSecure() {
    return snapshots().asyncMap((snapshot) async {
      final results = <Map<String, dynamic>>[];
      for (final doc in snapshot.docs) {
        final data = doc.data();
        if (data.containsKey('_encrypted')) {
          final decryptedMap = await EncryptionService.decryptMap(data['_encrypted'] as String);
          final result = Map<String, dynamic>.from(data)..remove('_encrypted');
          result.addAll(decryptedMap);
          result['id'] = doc.id;
          results.add(result);
        } else {
          final result = Map<String, dynamic>.from(data);
          result['id'] = doc.id;
          results.add(result);
        }
      }
      return results;
    });
  }
}

extension SecureWriteBatch on WriteBatch {
  Future<void> setSecure(
    DocumentReference<Map<String, dynamic>> document,
    Map<String, dynamic> data, {
    SetOptions? options,
    List<String> unencryptedFields = const [],
  }) async {
    final Map<String, dynamic> unencryptedPart = {};
    final Map<String, dynamic> encryptedPart = {};

    for (final entry in data.entries) {
      if (unencryptedFields.contains(entry.key)) {
        unencryptedPart[entry.key] = entry.value;
      } else {
        encryptedPart[entry.key] = entry.value;
      }
    }

    final payloadToUpload = Map<String, dynamic>.from(unencryptedPart);
    if (encryptedPart.isNotEmpty) {
      final encryptedString = await EncryptionService.encryptMap(encryptedPart);
      payloadToUpload['_encrypted'] = encryptedString;
    }

    set(document, payloadToUpload, options);
  }
}
