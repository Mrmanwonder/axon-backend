import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'firestore_service.dart';
import 'backend_config.dart';

enum FirestoreHealth {
  unknown,
  available,
  unavailable,
}

class BackendHealthService {
  BackendHealthService._();

  static final BackendHealthService instance = BackendHealthService._();

  FirestoreHealth _firestoreHealth = FirestoreHealth.unknown;
  DateTime? _lastCheckAt;
  static const Duration _cacheTtl = Duration(minutes: 5);
  static const String _backendUrl = BackendConfig.baseUrl;

  FirestoreHealth get firestoreHealth => _firestoreHealth;
  bool get isFirestoreKnownUnavailable =>
      _firestoreHealth == FirestoreHealth.unavailable;

  Future<void> initialize() async {
    await isFirestoreAvailable(forceRefresh: true);
    _startKeepAlive();
  }

  void _startKeepAlive() {
    Future.doWhile(() async {
      try {
        await http.get(Uri.parse('$_backendUrl/health')).timeout(
              const Duration(seconds: 10),
            );
      } catch (e) {
        debugPrint('[BackendHealthService] Health check failed: $e');
      }
      await Future.delayed(const Duration(minutes: 4));
      return true;
    });
  }

  Future<bool> isFirestoreAvailable({bool forceRefresh = false}) async {
    final now = DateTime.now();
    if (!forceRefresh &&
        _lastCheckAt != null &&
        now.difference(_lastCheckAt!) < _cacheTtl) {
      return _firestoreHealth == FirestoreHealth.available;
    }

    _lastCheckAt = now;
    try {
      await AxonFirestore.instance
          .collection('_backend_health')
          .limit(1)
          .get(const GetOptions(source: Source.server));
      _firestoreHealth = FirestoreHealth.available;
      return true;
    } on FirebaseException catch (e) {
      if (_looksLikeMissingDatabase(e)) {
        _firestoreHealth = FirestoreHealth.unavailable;
        return false;
      }
      if (_looksLikeExistingButProtectedDatabase(e)) {
        _firestoreHealth = FirestoreHealth.available;
        return true;
      }
      _firestoreHealth = FirestoreHealth.unknown;
      return false;
    } catch (e) {
      debugPrint('[BackendHealthService] isFirestoreAvailable failed: $e');
      _firestoreHealth = FirestoreHealth.unknown;
      return false;
    }
  }

  Future<T?> runIfFirestoreAvailable<T>(Future<T> Function() action) async {
    final available = await isFirestoreAvailable();
    if (!available) return null;
    try {
      return await action();
    } on FirebaseException catch (e) {
      if (_looksLikeMissingDatabase(e)) {
        _firestoreHealth = FirestoreHealth.unavailable;
      }
      rethrow;
    }
  }

  Future<bool> isHealthy({bool forceRefresh = false}) async {
    final fsAvailable = await isFirestoreAvailable(forceRefresh: forceRefresh);
    if (!fsAvailable) return false;
    final stopwatch = Stopwatch()..start();
    await runIfFirestoreAvailable(() => Future.value(true));
    stopwatch.stop();
    final latencyOk = stopwatch.elapsedMilliseconds < 500;
    return latencyOk;
  }

  bool _looksLikeMissingDatabase(FirebaseException e) {
    final message = (e.message ?? '').toLowerCase();
    return e.code == 'not-found' &&
        message.contains('database (default) does not exist');
  }

  bool _looksLikeExistingButProtectedDatabase(FirebaseException e) {
    return e.code == 'permission-denied' ||
        e.code == 'failed-precondition' ||
        e.code == 'unauthenticated';
  }
}
