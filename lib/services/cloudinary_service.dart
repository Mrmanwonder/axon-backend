import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:http_parser/http_parser.dart';
import 'package:path_provider/path_provider.dart';

import 'backend_health_service.dart';
import 'firestore_service.dart';
import 'leaderboard_service.dart';
import 'image_compression_service.dart';
import 'enhanced_security_service.dart';

class CloudinaryService {
  static const String _cloudName = 'dyoim3wmt';
  static const String _uploadPreset = 'axon_presets';

  final ImageCompressionService _imageCompression = ImageCompressionService();
  final EnhancedSecurityService _security = EnhancedSecurityService();

  static String? _extractPublicId(Map<String, dynamic> payload) {
    final direct = payload['public_id']?.toString();
    if (direct != null && direct.isNotEmpty) return direct;
    return null;
  }

  DocumentReference<Map<String, dynamic>> _doc(String uid) =>
      AxonPaths.privateUserDoc(uid);

  Future<void> _patchDoc(String uid, Map<String, dynamic> fields) async {
    final payload = {
      ...fields,
      'updated_at': FieldValue.serverTimestamp(),
    };
    try {
      await _doc(uid).update(payload);
    } on FirebaseException catch (e) {
      if (e.code != 'not-found') rethrow;
      await _doc(uid).set(payload, SetOptions(merge: true));
    }
  }

  Future<String?> uploadImage(File file) async {
    int attempts = 0;
    while (attempts < 2) {
      try {
        final compressed = await _imageCompression.compressImage(file.path);
        if (compressed == null) {
          debugPrint('Cloudinary: Image compression failed, uploading original');
        }

        final url =
            Uri.parse('https://api.cloudinary.com/v1_1/$_cloudName/upload');

        http.MultipartFile uploadFile;
        if (compressed != null) {
          uploadFile = http.MultipartFile.fromBytes(
            'file',
            compressed,
            filename: _security.sanitizeFilename(file.path.split('/').last),
            contentType: MediaType('image', 'jpeg'),
          );
        } else {
          uploadFile = await http.MultipartFile.fromPath(
            'file',
            file.path,
            contentType: MediaType('image', 'jpeg'),
          );
        }

        final request = http.MultipartRequest('POST', url)
          ..fields['upload_preset'] = _uploadPreset
          ..files.add(uploadFile);

        final response = await request.send();

        if (response.statusCode == 200) {
          final responseData = await response.stream.toBytes();
          final responseString = String.fromCharCodes(responseData);
          final jsonMap = jsonDecode(responseString);
          return jsonMap['secure_url'];
        } else {
          debugPrint('Cloudinary Upload Failed: ${response.statusCode}');
          return null;
        }
      } catch (e) {
        attempts++;
        if (attempts >= 2) {
          debugPrint('Cloudinary Upload Error after $attempts attempts: $e');
          return null;
        }
        debugPrint('Cloudinary Upload Error (attempt $attempts): $e - retrying...');
      }
    }
    return null;
  }

  Future<Map<String, String>?> uploadStudyArtifact(File file) async {
    int attempts = 0;
    while (attempts < 2) {
      try {
        final compressed = await _imageCompression.compressImage(file.path);

        final url =
            Uri.parse('https://api.cloudinary.com/v1_1/$_cloudName/upload');

        http.MultipartFile uploadFile;
        if (compressed != null) {
          uploadFile = http.MultipartFile.fromBytes(
            'file',
            compressed,
            filename: _security.sanitizeFilename(file.path.split('/').last),
            contentType: MediaType('image', 'jpeg'),
          );
        } else {
          uploadFile = await http.MultipartFile.fromPath(
            'file',
            file.path,
            contentType: MediaType('image', 'jpeg'),
          );
        }

        final request = http.MultipartRequest('POST', url)
          ..fields['upload_preset'] = _uploadPreset
          ..files.add(uploadFile);

        final response = await request.send();
        if (response.statusCode != 200) {
          debugPrint('Cloudinary Upload Failed: ${response.statusCode}');
          return null;
        }

        final body = await response.stream.bytesToString();
        final payload = jsonDecode(body) as Map<String, dynamic>;
        final secureUrl = (payload['secure_url'] ?? '').toString();
        final publicId = _extractPublicId(payload) ?? '';
        if (secureUrl.isEmpty || publicId.isEmpty) {
          return null;
        }
        return {
          'secure_url': secureUrl,
          'public_id': publicId,
        };
      } catch (e) {
        attempts++;
        if (attempts >= 2) {
          debugPrint('Cloudinary Artifact Upload Error after $attempts attempts: $e');
          return null;
        }
        debugPrint('Cloudinary Artifact Upload Error (attempt $attempts): $e - retrying...');
      }
    }
    return null;
  }

  Future<bool> updateProfilePhoto(String uid, File imageFile) async {
    // Compress image before upload (device-adaptive quality)
    final compressed = await _imageCompression.compressImage(imageFile.path);

    File fileToUpload = imageFile;
    if (compressed != null) {
      // Save compressed to temp file for upload
      final tempDir = await getTemporaryDirectory();
      final tempPath = '${tempDir.path}/compressed_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(tempPath).writeAsBytes(compressed);
      fileToUpload = File(tempPath);
    }

    final imageUrl = await uploadImage(fileToUpload);
    if (imageUrl == null) return false;

    final available =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return false;

    await _patchDoc(uid, {'photo_url': imageUrl});
    await LeaderboardService.syncPublicMirror();
    return true;
  }

  static String getThumbnailUrl(String photoUrl,
      {int width = 100, int height = 100}) {
    if (photoUrl.isEmpty) return '';

    final uploadIndex = photoUrl.indexOf('/upload/');
    if (uploadIndex == -1) return photoUrl;

    final baseUrl = photoUrl.substring(0, uploadIndex + 8);
    final rest = photoUrl.substring(uploadIndex + 8);

    return '$baseUrl/w_$width,h_$height,c_fill/$rest';
  }

  static String getProfileUrl(String photoUrl,
      {int width = 500, int height = 500}) {
    if (photoUrl.isEmpty) return '';

    final uploadIndex = photoUrl.indexOf('/upload/');
    if (uploadIndex == -1) return photoUrl;

    final baseUrl = photoUrl.substring(0, uploadIndex + 8);
    final rest = photoUrl.substring(uploadIndex + 8);

    return '$baseUrl/w_$width,h_$height,c_fill/$rest';
  }
}
