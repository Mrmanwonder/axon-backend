import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/models.dart';
import 'backend_health_service.dart';
import 'firestore_service.dart';
import 'study_catalog.dart';
import 'cloudinary_service.dart';
import 'leaderboard_service.dart';
import 'exam_service.dart';

class ProfileService {
  ProfileService({
    FirebaseFirestore? firestore,
    StudyCatalog? catalog,
  })  : _firestore = firestore,
        _catalog = catalog ?? StudyCatalog();

  final FirebaseFirestore? _firestore;
  final StudyCatalog _catalog;
  final ImagePicker _imagePicker = ImagePicker();
  final ExamService _examService = ExamService();

  DocumentReference<Map<String, dynamic>> _doc(String uid) =>
      (_firestore ?? AxonFirestore.instance)
          .collection(AxonCollections.usersPrivate)
          .doc(uid);

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
    await LeaderboardService.syncPublicMirror();
  }

  Future<void> _syncOfficialDeadlines({
    required String board,
    required List<String> subjects,
    String? administrativeZone,
  }) async {
    if (board.trim().isEmpty || subjects.isEmpty) return;
    final normalizedBoard = board.toLowerCase();
    final requiresZone = normalizedBoard.contains('caie') ||
        normalizedBoard.contains('cambridge') ||
        normalizedBoard.contains('igcse') ||
        normalizedBoard.contains('o level') ||
        normalizedBoard.contains('olevel') ||
        normalizedBoard.contains('a level');
    if (requiresZone &&
        (administrativeZone == null || administrativeZone.trim().isEmpty)) {
      return;
    }
    try {
      await _examService.syncOfficialDeadlines(
        board: board,
        subjects: subjects,
        administrativeZone: administrativeZone,
      );
    } catch (_) {
      // Deadline sync is best-effort; profile writes must still succeed.
    }
  }

  Stream<UserProfile?> streamProfile(User user) {
    if (BackendHealthService.instance.isFirestoreKnownUnavailable) {
      return Stream.value(null);
    }
    return _doc(user.uid).snapshots().map((snapshot) {
      if (!snapshot.exists) return null;
      final data = snapshot.data();
      if (data == null) return null;
      return UserProfile.fromFirestore(
        data,
        uid: user.uid,
        fallbackEmail: user.email,
      );
    });
  }

  Future<UserProfile?> getProfile(User user) async {
    final available =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return null;
    final snapshot = await _doc(user.uid).get();
    if (!snapshot.exists) return null;
    final data = snapshot.data();
    if (data == null) return null;
    return UserProfile.fromFirestore(
      data,
      uid: user.uid,
      fallbackEmail: user.email,
    );
  }

  Future<UserProfile> saveOnboarding({
    required User user,
    required String displayName,
    required String board,
    required List<String> subjects,
    required double targetHours,
    required MotivationStyle motivationStyle,
    String country = '',
    String timezone = '',
    List<String> examSeries = const [],
  }) async {
    final normalizedSubjects = StudyCatalog.normalizeSubjects(subjects);
    final profile = UserProfile(
      uid: user.uid,
      displayName: displayName,
      email: user.email ?? '',
      photoUrl: user.photoURL,
      motivationStyle: motivationStyle,
      board: board,
      subjects: normalizedSubjects,
      targetStudyHours: targetHours,
      onboardingComplete: true,
      createdAt: DateTime.now(),
      preferences: {
        'country': country,
        'timezone': timezone,
        'examSeries': examSeries,
      },
    );

    final available =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (available) {
      await _doc(user.uid).set({
        ...profile.toFirestore(),
        'created_at': FieldValue.serverTimestamp(),
        'updated_at': FieldValue.serverTimestamp(),
      }, SetOptions(merge: true));
      await LeaderboardService.syncPublicMirror();
      await _syncOfficialDeadlines(
        board: board,
        subjects: normalizedSubjects,
      );
    }

    await _catalog.replaceSubjects(normalizedSubjects);
    await _catalog.saveToFirestore(user.uid);
    return profile;
  }

  Future<void> updateMotivationStyle(String uid, MotivationStyle style) async {
    final available =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return;
    await _patchDoc(uid, {'motivation_style': style.index});
  }

  Future<void> updateSubjects(String uid, List<String> subjects) async {
    final normalizedSubjects = StudyCatalog.normalizeSubjects(subjects);
    final available =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (available) {
      await _patchDoc(uid, {'subjects': normalizedSubjects});
    }
    final snapshot = await _doc(uid).get();
    final board = (snapshot.data()?['board'] ?? '').toString();
    final preferences = Map<String, dynamic>.from(
      snapshot.data()?['preferences'] as Map? ?? const {},
    );
    final administrativeZone =
        (preferences['administrative_zone'] ?? '').toString();
    if (available) {
      await _syncOfficialDeadlines(
        board: board,
        subjects: normalizedSubjects,
        administrativeZone: administrativeZone,
      );
    }
    await _catalog.replaceSubjects(normalizedSubjects);
    if (available) {
      await _catalog.saveToFirestore(uid);
    }
  }

  Future<File?> pickProfilePhoto() async {
    try {
      PermissionStatus status = await Permission.photos.status;
      if (!status.isGranted && !status.isLimited) {
        status = await Permission.photos.request();
      }
      if (!status.isGranted && !status.isLimited) {
        status = await Permission.storage.status;
        if (!status.isGranted) {
          status = await Permission.storage.request();
        }
      }
      if (!status.isGranted && !status.isLimited) {
        debugPrint('Photo permission denied: $status');
        return null;
      }

      final XFile? picked = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 800,
        maxHeight: 800,
        imageQuality: 85,
      );

      if (picked == null) {
        debugPrint('No image selected');
        return null;
      }

      final file = File(picked.path);
      if (file.existsSync()) {
        return file;
      }

      debugPrint('Picked file does not exist');
      return null;
    } catch (e) {
      debugPrint('Error picking profile photo: $e');
      return null;
    }
  }

  Future<File?> takeProfilePhoto() async {
    try {
      PermissionStatus status = await Permission.camera.status;
      if (!status.isGranted) {
        status = await Permission.camera.request();
      }
      if (!status.isGranted) {
        debugPrint('Camera permission denied: $status');
        return null;
      }

      final XFile? picked = await _imagePicker.pickImage(
        source: ImageSource.camera,
        maxWidth: 800,
        maxHeight: 800,
        imageQuality: 85,
      );

      if (picked == null) {
        debugPrint('No photo taken');
        return null;
      }

      final file = File(picked.path);
      if (file.existsSync()) {
        return file;
      }

      debugPrint('Camera file does not exist');
      return null;
    } catch (e) {
      debugPrint('Error taking profile photo: $e');
      return null;
    }
  }

  Future<File?> selectAndCropProfilePhoto(ImageSource source) async {
    try {
      final picked = source == ImageSource.camera
          ? await takeProfilePhoto()
          : await pickProfilePhoto();
      if (picked == null) return null;

      try {
        final cropped = await cropProfilePhoto(picked);
        if (cropped != null) return cropped;
      } catch (e) {
        debugPrint('Cropping failed: $e');
      }
      return picked;
    } catch (e) {
      debugPrint('Error selecting profile photo: $e');
      return null;
    }
  }

  Future<File?> cropProfilePhoto(File imageFile) async {
    if (!imageFile.existsSync()) {
      debugPrint('Image file does not exist');
      return null;
    }

    try {
      final croppedFile = await ImageCropper().cropImage(
        sourcePath: imageFile.path,
        uiSettings: [
          AndroidUiSettings(
            toolbarTitle: '',
            toolbarColor: Colors.black,
            toolbarWidgetColor: Colors.white,
            initAspectRatio: CropAspectRatioPreset.square,
            lockAspectRatio: true,
            hideBottomControls: false,
            backgroundColor: Colors.black,
            activeControlsWidgetColor: const Color(0xFF3A86FF),
            dimmedLayerColor: Colors.black.withValues(alpha: 0.6),
            cropGridColor: const Color(0xFF3A86FF),
            cropFrameColor: const Color(0xFF3A86FF),
            cropFrameStrokeWidth: 2,
            cropGridRowCount: 0,
            cropGridColumnCount: 0,
          ),
          IOSUiSettings(
            title: 'Crop Photo',
            aspectRatioLockEnabled: true,
            resetAspectRatioEnabled: false,
            cancelButtonTitle: 'Cancel',
            doneButtonTitle: 'Done',
          ),
        ],
      );
      if (croppedFile != null) {
        final result = File(croppedFile.path);
        if (result.existsSync()) {
          return result;
        }
      }
      return null;
    } catch (e) {
      debugPrint('Crop error: $e');
      return null;
    }
  }

  Future<String?> uploadProfilePhoto(String uid, File imageFile) async {
    final cloudinary = CloudinaryService();
    final success = await cloudinary.updateProfilePhoto(uid, imageFile);
    if (success) {
      final url = await _getPhotoUrl(uid);
      return url;
    }
    return null;
  }

  Future<String?> _getPhotoUrl(String uid) async {
    try {
      final snapshot = await _doc(uid).get();
      if (snapshot.exists) {
        return snapshot.data()?['photo_url'];
      }
    } catch (_) {}
    return null;
  }

  Future<void> updateProfile(String uid, Map<String, dynamic> fields) async {
    final available =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return;
    await _patchDoc(uid, fields);
  }

  Future<void> updateDisplayName(String uid, String displayName) async {
    final available =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return;
    await _patchDoc(uid, {'display_name': displayName});
  }
}
