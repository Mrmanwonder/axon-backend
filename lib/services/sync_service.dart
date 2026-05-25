// lib/services/sync_service.dart
// ─────────────────────────────────────────────────────────────────
// Offline-First Sync Service with Conflict Resolution
// Handles sessions, resources, profile, and cross-device continuity
// ─────────────────────────────────────────────────────────────────

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../models/models.dart';
import 'backend_health_service.dart';
import 'firestore_service.dart';

enum SyncStatus { idle, syncing, synced, offline, error }

enum SyncPriority { low, normal, high, critical }

class SyncOperation {
  final String id;
  final String type;
  final String collection;
  final String? documentId;
  final Map<String, dynamic> data;
  final DateTime createdAt;
  final SyncPriority priority;
  final int retryCount;
  final String? error;

  SyncOperation({
    required this.id,
    required this.type,
    required this.collection,
    this.documentId,
    required this.data,
    required this.createdAt,
    this.priority = SyncPriority.normal,
    this.retryCount = 0,
    this.error,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'type': type,
        'collection': collection,
        'documentId': documentId,
        'data': data,
        'createdAt': createdAt.toIso8601String(),
        'priority': priority.index,
        'retryCount': retryCount,
        'error': error,
      };

  factory SyncOperation.fromJson(Map<String, dynamic> json) {
    DateTime createdAt;
    try {
      createdAt = DateTime.parse(json['createdAt']);
    } catch (e) {
      debugPrint('[SyncService] Failed to parse SyncOperation createdAt: $e');
      createdAt = DateTime.now();
    }
    return SyncOperation(
      id: json['id'],
      type: json['type'],
      collection: json['collection'],
      documentId: json['documentId'],
      data: Map<String, dynamic>.from(json['data']),
      createdAt: createdAt,
      priority: SyncPriority.values[json['priority'] ?? 1],
      retryCount: json['retryCount'] ?? 0,
      error: json['error'],
    );
  }

  SyncOperation copyWith({
    int? retryCount,
    String? error,
  }) =>
      SyncOperation(
        id: id,
        type: type,
        collection: collection,
        documentId: documentId,
        data: data,
        createdAt: createdAt,
        priority: priority,
        retryCount: retryCount ?? this.retryCount,
        error: error,
      );
}

class SyncState {
  final SyncStatus status;
  final DateTime? lastSyncTime;
  final int pendingOperations;
  final List<String> activeDeviceIds;
  final String? currentDeviceId;
  final String? error;

  const SyncState({
    this.status = SyncStatus.idle,
    this.lastSyncTime,
    this.pendingOperations = 0,
    this.activeDeviceIds = const [],
    this.currentDeviceId,
    this.error,
  });

  bool get isOnline =>
      status == SyncStatus.synced || status == SyncStatus.syncing;
  bool get hasPendingSync => pendingOperations > 0;

  SyncState copyWith({
    SyncStatus? status,
    DateTime? lastSyncTime,
    int? pendingOperations,
    List<String>? activeDeviceIds,
    String? currentDeviceId,
    String? error,
  }) =>
      SyncState(
        status: status ?? this.status,
        lastSyncTime: lastSyncTime ?? this.lastSyncTime,
        pendingOperations: pendingOperations ?? this.pendingOperations,
        activeDeviceIds: activeDeviceIds ?? this.activeDeviceIds,
        currentDeviceId: currentDeviceId ?? this.currentDeviceId,
        error: error,
      );
}

class ActiveSession {
  final String sessionId;
  final String subject;
  final String chapter;
  final DateTime startedAt;
  final Duration elapsed;
  final int breakCount;
  final List<SessionPing> pings;
  final String deviceId;
  final bool isSyncing;
  final Map<String, dynamic> metadata;

  ActiveSession({
    required this.sessionId,
    required this.subject,
    required this.chapter,
    required this.startedAt,
    this.elapsed = Duration.zero,
    this.breakCount = 0,
    this.pings = const [],
    required this.deviceId,
    this.isSyncing = false,
    this.metadata = const {},
  });

  Map<String, dynamic> toJson() => {
        'sessionId': sessionId,
        'subject': subject,
        'chapter': chapter,
        'startedAt': startedAt.toIso8601String(),
        'elapsedSeconds': elapsed.inSeconds,
        'breakCount': breakCount,
        'pings': pings.map((p) => p.toJson()).toList(),
        'deviceId': deviceId,
        'isSyncing': isSyncing,
        'metadata': metadata,
      };

  factory ActiveSession.fromJson(Map<String, dynamic> json) {
    DateTime startedAt;
    try {
      startedAt = DateTime.parse(json['startedAt']);
    } catch (e) {
      debugPrint('[SyncService] Failed to parse ActiveSession startedAt: $e');
      startedAt = DateTime.now();
    }
    return ActiveSession(
      sessionId: json['sessionId'],
      subject: json['subject'],
      chapter: json['chapter'],
      startedAt: startedAt,
      elapsed: Duration(seconds: json['elapsedSeconds'] ?? 0),
      breakCount: json['breakCount'] ?? 0,
      pings: (json['pings'] as List?)
              ?.map((p) => SessionPing.fromJson(p))
              .toList() ??
          const [],
      deviceId: json['deviceId'] ?? '',
      isSyncing: json['isSyncing'] ?? false,
      metadata: Map<String, dynamic>.from(json['metadata'] ?? {}),
    );
  }

  ActiveSession copyWith({
    Duration? elapsed,
    int? breakCount,
    List<SessionPing>? pings,
    bool? isSyncing,
    Map<String, dynamic>? metadata,
  }) =>
      ActiveSession(
        sessionId: sessionId,
        subject: subject,
        chapter: chapter,
        startedAt: startedAt,
        elapsed: elapsed ?? this.elapsed,
        breakCount: breakCount ?? this.breakCount,
        pings: pings ?? this.pings,
        deviceId: deviceId,
        isSyncing: isSyncing ?? this.isSyncing,
        metadata: metadata ?? this.metadata,
      );
}

class ConflictRecord {
  final String localId;
  final String remoteId;
  final Map<String, dynamic> localData;
  final Map<String, dynamic> remoteData;
  final DateTime localTimestamp;
  final DateTime remoteTimestamp;
  final String resolution;

  ConflictRecord({
    required this.localId,
    required this.remoteId,
    required this.localData,
    required this.remoteData,
    required this.localTimestamp,
    required this.remoteTimestamp,
    this.resolution = 'pending',
  });

  Map<String, dynamic> toJson() => {
        'localId': localId,
        'remoteId': remoteId,
        'localData': localData,
        'remoteData': remoteData,
        'localTimestamp': localTimestamp.toIso8601String(),
        'remoteTimestamp': remoteTimestamp.toIso8601String(),
        'resolution': resolution,
      };

  factory ConflictRecord.fromJson(Map<String, dynamic> json) {
    DateTime localTimestamp;
    DateTime remoteTimestamp;
    try {
      localTimestamp = DateTime.parse(json['localTimestamp']);
    } catch (e) {
      debugPrint('[SyncService] Failed to parse ConflictRecord localTimestamp: $e');
      localTimestamp = DateTime.now();
    }
    try {
      remoteTimestamp = DateTime.parse(json['remoteTimestamp']);
    } catch (e) {
      debugPrint('[SyncService] Failed to parse ConflictRecord remoteTimestamp: $e');
      remoteTimestamp = DateTime.now();
    }
    return ConflictRecord(
      localId: json['localId'],
      remoteId: json['remoteId'],
      localData: Map<String, dynamic>.from(json['localData']),
      remoteData: Map<String, dynamic>.from(json['remoteData']),
      localTimestamp: localTimestamp,
      remoteTimestamp: remoteTimestamp,
      resolution: json['resolution'] ?? 'pending',
    );
  }
}

class SyncService {
  static final SyncService _instance = SyncService._internal();
  factory SyncService() => _instance;
  SyncService._internal();

  static final FirebaseFirestore _db = AxonFirestore.instance;
  static final FirebaseAuth _auth = FirebaseAuth.instance;

  final _syncStateController = StreamController<SyncState>.broadcast();
  Stream<SyncState> get syncStateStream => _syncStateController.stream;

  SyncState _state = const SyncState();
  SyncState get currentState => _state;

  static const String _deviceIdKey = 'axon_device_id';
  static const String _syncQueueKey = 'axon_sync_queue';
  static const String _activeSessionKey = 'axon_active_session';
  static const String _conflictLogKey = 'axon_conflicts';
  static const String _lastSyncKey = 'axon_last_sync';

  String? _deviceId;
  Timer? _syncTimer;
  Timer? _heartbeatTimer;
  StreamSubscription? _connectivitySubscription;
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;
    _isInitialized = true;

    await _loadDeviceId();
    _setupConnectivityListener();
    _startPeriodicSync();
    _startHeartbeat();
    await _processSyncQueue();
  }

  Future<String> _loadDeviceId() async {
    final prefs = await SharedPreferences.getInstance();
    _deviceId = prefs.getString(_deviceIdKey);
    if (_deviceId == null) {
      _deviceId =
          '${DateTime.now().millisecondsSinceEpoch}_${_generateRandomId(8)}';
      await prefs.setString(_deviceIdKey, _deviceId!);
    }
    _state = _state.copyWith(currentDeviceId: _deviceId);
    return _deviceId!;
  }

  String _generateRandomId(int length) {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    return List.generate(length, (i) => chars[i % chars.length]).join();
  }

  void _setupConnectivityListener() {
    Timer.periodic(const Duration(seconds: 30), (timer) async {
      final isOnline = await _checkConnectivity();
      if (isOnline && _state.status == SyncStatus.offline) {
        _updateState(_state.copyWith(status: SyncStatus.synced));
        await _processSyncQueue();
      } else if (!isOnline) {
        _updateState(_state.copyWith(status: SyncStatus.offline));
      }
    });
  }

  void _startPeriodicSync() {
    _syncTimer?.cancel();
    _syncTimer = Timer.periodic(const Duration(minutes: 5), (_) {
      _processSyncQueue();
    });
  }

  void _startHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = Timer.periodic(const Duration(minutes: 2), (_) async {
      if (_auth.currentUser != null && _deviceId != null) {
        await _updateDevicePresence();
      }
    });
  }

  Future<void> _updateDevicePresence() async {
    if (_auth.currentUser == null) return;
    final available = await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return;
    try {
      await _db
          .collection(AxonCollections.usersPrivate)
          .doc(_auth.currentUser!.uid)
          .collection('devices')
          .doc(_deviceId)
          .set({
        'lastSeen': FieldValue.serverTimestamp(),
        'deviceId': _deviceId,
        'isActive': true,
      }, SetOptions(merge: true));
    } catch (e) {
      debugPrint('[SyncService] Failed to update device presence: $e');
    }
  }

  void _updateState(SyncState newState) {
    _state = newState;
    _syncStateController.add(_state);
  }

  // ─────────────────────────────────────────────────────────────────
  // CRASH-SAFE SESSION RECOVERY
  // ─────────────────────────────────────────────────────────────────

  Future<void> saveActiveSession(ActiveSession session) async {
    final prefs = await SharedPreferences.getInstance();
    final sessionToSave = ActiveSession(
      sessionId: session.sessionId,
      subject: session.subject,
      chapter: session.chapter,
      startedAt: session.startedAt,
      elapsed: session.elapsed,
      breakCount: session.breakCount,
      pings: session.pings,
      deviceId: _deviceId ?? session.deviceId,
      isSyncing: session.isSyncing,
      metadata: session.metadata,
    );
    await prefs.setString(
        _activeSessionKey, jsonEncode(sessionToSave.toJson()));
    await prefs.setString(
        '${_activeSessionKey}_timestamp', DateTime.now().toIso8601String());
    await _persistActiveSessionToCloud(sessionToSave);
  }

  Future<ActiveSession?> getActiveSession() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_activeSessionKey);
    if (raw == null || raw.isEmpty) return null;

    try {
      final session = ActiveSession.fromJson(jsonDecode(raw));
      final timestamp = prefs.getString('${_activeSessionKey}_timestamp');
      if (timestamp != null) {
        final savedAt = DateTime.tryParse(timestamp);
        if (savedAt == null ||
            DateTime.now().difference(savedAt).inHours > 24) {
          await prefs.remove(_activeSessionKey);
          await prefs.remove('${_activeSessionKey}_timestamp');
          return null;
        }
      }
      return session;
    } catch (e) {
      debugPrint('[SyncService] Failed to load active session: $e');
      await prefs.remove(_activeSessionKey);
      await prefs.remove('${_activeSessionKey}_timestamp');
      return null;
    }
  }

  Future<void> clearActiveSession() async {
    final prefs = await SharedPreferences.getInstance();
    final localSession = await getActiveSession();
    await prefs.remove(_activeSessionKey);
    await prefs.remove('${_activeSessionKey}_timestamp');
    await _clearActiveSessionFromCloud(localSession);
  }

  Future<void> _persistActiveSessionToCloud(ActiveSession session) async {
    if (_auth.currentUser == null) return;
    final available = await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return;
    try {
      await _db
          .collection(AxonCollections.usersPrivate)
          .doc(_auth.currentUser!.uid)
          .collection('active_sessions')
          .doc(session.sessionId)
          .set({
        ...session.toJson(),
        'uid': _auth.currentUser!.uid,
        'isActive': true,
        'lastHeartbeatAt': FieldValue.serverTimestamp(),
      }, SetOptions(merge: true));
    } catch (e) {
      debugPrint('[SyncService] Failed to persist active session to cloud: $e');
    }
  }

  Future<void> _clearActiveSessionFromCloud(ActiveSession? session) async {
    if (_auth.currentUser == null || session == null) return;
    final available = await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return;
    try {
      await _db
          .collection(AxonCollections.usersPrivate)
          .doc(_auth.currentUser!.uid)
          .collection('active_sessions')
          .doc(session.sessionId)
          .set({
        'isActive': false,
        'endedAt': FieldValue.serverTimestamp(),
        'lastHeartbeatAt': FieldValue.serverTimestamp(),
      }, SetOptions(merge: true));
    } catch (e) {
      debugPrint('[SyncService] Failed to clear active session from cloud: $e');
    }
  }

  Future<int> getLiveStudyCount({
    Duration freshness = const Duration(minutes: 20),
  }) async {
    if (_auth.currentUser == null) return 0;
    final available = await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return 0;
    final cutoff = Timestamp.fromDate(DateTime.now().subtract(freshness));
    try {
      final aggregate = await _db
          .collectionGroup('active_sessions')
          .where('isActive', isEqualTo: true)
          .where('lastHeartbeatAt', isGreaterThan: cutoff)
          .count()
          .get();
      return aggregate.count ?? 0;
    } catch (e) {
      debugPrint('[SyncService] Failed to get live study count (aggregate): $e');
      try {
        final snapshot = await _db
            .collectionGroup('active_sessions')
            .where('isActive', isEqualTo: true)
            .where('lastHeartbeatAt', isGreaterThan: cutoff)
            .get();
        return snapshot.docs.length;
      } catch (e2) {
        debugPrint('[SyncService] Failed to get live study count (fallback): $e2');
        return 0;
      }
    }
  }

  Future<ActiveSession?> recoverSessionFromCloud() async {
    if (_auth.currentUser == null) return null;
    final available = await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return null;

    try {
      final snapshot = await _db
          .collection(AxonCollections.usersPrivate)
          .doc(_auth.currentUser!.uid)
          .collection('active_sessions')
          .where('isActive', isEqualTo: true)
          .limit(1)
          .get();

      if (snapshot.docs.isEmpty) return null;

      final doc = snapshot.docs.first;
      final remoteSession = ActiveSession.fromJson(doc.data());

      if (remoteSession.deviceId != _deviceId) {
        final localSession = await getActiveSession();
        if (localSession != null &&
            remoteSession.startedAt.isAfter(localSession.startedAt)) {
          return remoteSession;
        }
      }
    } catch (e) {
      debugPrint('[SyncService] Failed to recover session from cloud: $e');
    }
    return null;
  }

  Future<ActiveSession?> getRecoverableSession() async {
    final local = await getActiveSession();
    final remote = await recoverSessionFromCloud();

    if (local == null && remote == null) return null;
    if (local == null) return remote;
    if (remote == null) return local;

    if (remote.startedAt.isAfter(local.startedAt)) {
      await clearActiveSession();
      return remote;
    }
    return local;
  }

  // ─────────────────────────────────────────────────────────────────
  // SYNC QUEUE MANAGEMENT
  // ─────────────────────────────────────────────────────────────────

  Future<void> addToSyncQueue(SyncOperation operation) async {
    final prefs = await SharedPreferences.getInstance();
    final queue = await _getSyncQueue();
    queue.add(operation);
    await prefs.setString(
        _syncQueueKey, jsonEncode(queue.map((o) => o.toJson()).toList()));
    _updateState(_state.copyWith(pendingOperations: queue.length));

    final isOnline = await _checkConnectivity();
    if (isOnline) {
      _processSyncQueue();
    }
  }

  Future<List<SyncOperation>> _getSyncQueue() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_syncQueueKey);
    if (raw == null || raw.isEmpty) return [];

    try {
      final list = jsonDecode(raw) as List;
      return list.map((e) => SyncOperation.fromJson(e)).toList();
    } catch (e) {
      debugPrint('[SyncService] Failed to load sync queue: $e');
      return [];
    }
  }

  Future<void> _saveSyncQueue(List<SyncOperation> queue) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        _syncQueueKey, jsonEncode(queue.map((o) => o.toJson()).toList()));
    _updateState(_state.copyWith(pendingOperations: queue.length));
  }

  Future<void> _processSyncQueue() async {
    if (_auth.currentUser == null) return;

    final isOnline = await _checkConnectivity();
    if (!isOnline) {
      _updateState(_state.copyWith(status: SyncStatus.offline));
      return;
    }

    final queue = await _getSyncQueue();
    if (queue.isEmpty) {
      _updateState(_state.copyWith(status: SyncStatus.synced));
      return;
    }

    _updateState(_state.copyWith(status: SyncStatus.syncing));

    queue.sort((a, b) => b.priority.index.compareTo(a.priority.index));

    final failed = <SyncOperation>[];

    for (final operation in queue) {
      try {
        await _executeSyncOperation(operation);
      } catch (e) {
        if (operation.retryCount < 3) {
          failed.add(operation.copyWith(
            retryCount: operation.retryCount + 1,
            error: e.toString(),
          ));
        } else {
          await _logConflict(operation, e.toString());
        }
      }
    }

    await _saveSyncQueue(failed);

    if (failed.isEmpty) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_lastSyncKey, DateTime.now().toIso8601String());
    }

    _updateState(_state.copyWith(
      status: failed.isEmpty ? SyncStatus.synced : SyncStatus.error,
      lastSyncTime: failed.isEmpty ? DateTime.now() : _state.lastSyncTime,
      error: failed.isEmpty ? null : '${failed.length} operations failed',
    ));
  }

  Future<void> _executeSyncOperation(SyncOperation op) async {
    switch (op.type) {
      case 'create':
        final docRef = op.documentId != null
            ? _db.collection(op.collection).doc(op.documentId)
            : _db.collection(op.collection).doc();
        await docRef.set({
          ...op.data,
          'localId': op.id,
          'deviceId': _deviceId,
          'createdAt': FieldValue.serverTimestamp(),
          'updatedAt': FieldValue.serverTimestamp(),
        });
        break;

      case 'update':
        if (op.documentId != null) {
          await _db.collection(op.collection).doc(op.documentId).set({
            ...op.data,
            'deviceId': _deviceId,
            'updatedAt': FieldValue.serverTimestamp(),
          }, SetOptions(merge: true));
        }
        break;

      case 'delete':
        if (op.documentId != null) {
          await _db.collection(op.collection).doc(op.documentId).delete();
        }
        break;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // DATA SYNC
  // ─────────────────────────────────────────────────────────────────

  Future<void> syncSession(StudySession session) async {
    final available = await BackendHealthService.instance.isFirestoreAvailable();
    if (!available) return;
    final operation = SyncOperation(
      id: session.id,
      type: 'create',
      collection: 'users/${_auth.currentUser?.uid}/sessions',
      documentId: session.id,
      data: {
        'id': session.id,
        'subject': session.subject,
        'durationMinutes': session.durationMinutes,
        'breakCount': session.breakCount,
        'intensityIndex': session.intensityIndex,
        'pings': session.pings.map((p) => p.toJson()).toList(),
        'date': session.date.toIso8601String(),
        'deviceId': _deviceId,
      },
      createdAt: DateTime.now(),
      priority: SyncPriority.high,
    );
    await addToSyncQueue(operation);
  }

  Future<void> syncProfile(UserProfile profile) async {
    final operation = SyncOperation(
      id: 'profile_${DateTime.now().millisecondsSinceEpoch}',
      type: 'update',
      collection: 'users',
      documentId: profile.uid,
      data: profile.toFirestore(),
      createdAt: DateTime.now(),
      priority: SyncPriority.critical,
    );
    await addToSyncQueue(operation);
  }

  Future<void> syncResource(Map<String, dynamic> resource) async {
    final operation = SyncOperation(
      id: resource['id'] ?? 'resource_${DateTime.now().millisecondsSinceEpoch}',
      type: 'create',
      collection: 'users/${_auth.currentUser?.uid}/resources',
      data: {
        ...resource,
        'deviceId': _deviceId,
        'syncedAt': DateTime.now().toIso8601String(),
      },
      createdAt: DateTime.now(),
      priority: SyncPriority.normal,
    );
    await addToSyncQueue(operation);
  }

  Future<void> syncDailyMetrics(DailyMetrics metrics) async {
    final dateKey =
        '${metrics.date.year}_${metrics.date.month}_${metrics.date.day}';
    final operation = SyncOperation(
      id: 'metrics_$dateKey',
      type: 'update',
      collection: 'users/${_auth.currentUser?.uid}/metrics',
      documentId: dateKey,
      data: metrics.toJson(),
      createdAt: DateTime.now(),
      priority: SyncPriority.high,
    );
    await addToSyncQueue(operation);
  }

  // ─────────────────────────────────────────────────────────────────
  // CONFLICT RESOLUTION
  // ─────────────────────────────────────────────────────────────────

  Future<void> _logConflict(SyncOperation operation, String error) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_conflictLogKey);
    List<ConflictRecord> conflicts = [];

    if (raw != null && raw.isNotEmpty) {
      try {
        conflicts = (jsonDecode(raw) as List)
            .map((e) => ConflictRecord.fromJson(e))
            .toList();
      } catch (e) {
        debugPrint('[SyncService] Failed to decode conflict log: $e');
      }
    }

    conflicts.add(ConflictRecord(
      localId: operation.id,
      remoteId: operation.documentId ?? '',
      localData: operation.data,
      remoteData: {},
      localTimestamp: operation.createdAt,
      remoteTimestamp: DateTime.now(),
      resolution: 'failed',
    ));

    if (conflicts.length > 100) {
      conflicts = conflicts.sublist(conflicts.length - 100);
    }

    await prefs.setString(
        _conflictLogKey, jsonEncode(conflicts.map((c) => c.toJson()).toList()));
  }

  Future<Map<String, dynamic>> resolveConflict(
    ConflictRecord conflict,
    String strategy,
  ) async {
    switch (strategy) {
      case 'local_wins':
        return conflict.localData;
      case 'remote_wins':
        return conflict.remoteData;
      case 'merge':
        return {...conflict.remoteData, ...conflict.localData};
      default:
        return conflict.localData;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // CROSS-DEVICE CONTINUITY
  // ─────────────────────────────────────────────────────────────────

  Future<List<String>> getActiveDevices() async {
    if (_auth.currentUser == null) return [];

    try {
      final snapshot = await _db
          .collection(AxonCollections.usersPrivate)
          .doc(_auth.currentUser!.uid)
          .collection('devices')
          .where('isActive', isEqualTo: true)
          .where('lastSeen',
              isGreaterThan:
                  DateTime.now().subtract(const Duration(minutes: 10)))
          .get();

      final devices = snapshot.docs.map((d) => d.id).toList();
      _updateState(_state.copyWith(activeDeviceIds: devices));
      return devices;
    } catch (e) {
      debugPrint('[SyncService] Failed to get active devices: $e');
      return [];
    }
  }

  Future<void> transferSessionToDevice(
      String targetDeviceId, ActiveSession session) async {
    if (_auth.currentUser == null) return;

    final transferredSession = ActiveSession(
      sessionId: session.sessionId,
      subject: session.subject,
      chapter: session.chapter,
      startedAt: session.startedAt,
      elapsed: session.elapsed,
      breakCount: session.breakCount,
      pings: session.pings,
      deviceId: targetDeviceId,
      isSyncing: session.isSyncing,
      metadata: session.metadata,
    );

    await _db
        .collection(AxonCollections.usersPrivate)
        .doc(_auth.currentUser!.uid)
        .collection('pending_transfers')
        .doc(session.sessionId)
        .set({
      ...transferredSession.toJson(),
      'expiresAt':
          DateTime.now().add(const Duration(minutes: 30)).toIso8601String(),
    });
  }

  Future<ActiveSession?> receiveSessionTransfer() async {
    if (_auth.currentUser == null) return null;

    try {
      final snapshot = await _db
          .collection(AxonCollections.usersPrivate)
          .doc(_auth.currentUser!.uid)
          .collection('pending_transfers')
          .where('deviceId', isEqualTo: _deviceId)
          .where('expiresAt', isGreaterThan: DateTime.now().toIso8601String())
          .limit(1)
          .get();

      if (snapshot.docs.isNotEmpty) {
        final session = ActiveSession.fromJson(snapshot.docs.first.data());
        await snapshot.docs.first.reference.delete();
        await saveActiveSession(session);
        return session;
      }
    } catch (e) {
      debugPrint('[SyncService] Failed to receive session transfer: $e');
    }
    return null;
  }

  // ─────────────────────────────────────────────────────────────────
  // EXPORT / IMPORT
  // ─────────────────────────────────────────────────────────────────

  Future<Map<String, dynamic>> exportAllData() async {
    final prefs = await SharedPreferences.getInstance();
    final data = <String, dynamic>{
      'exportedAt': DateTime.now().toIso8601String(),
      'deviceId': _deviceId,
      'version': '1.0',
    };

    final keys = [
      'timer_history',
      'metrics_state',
      'userSubjects',
      'userBoard',
      'userTargetHours',
      'motivationStyle',
    ];

    for (final key in keys) {
      final raw = prefs.getString(key);
      if (raw != null) {
        data[key] = jsonDecode(raw);
      }
    }

    final boolVal = [
      'userOnboardingComplete',
      'goal_unlock_chime_day',
      'perfect_day_badge_day',
      'streak_milestone_day',
    ];

    for (final key in boolVal) {
      final val = prefs.getBool(key);
      if (val != null) {
        data[key] = val;
      }
    }

    final intVal = [
      'userTargetHours',
      'motivationStyle',
      'currentStreak',
      'longestStreak'
    ];
    for (final key in intVal) {
      final val = prefs.getInt(key);
      if (val != null) {
        data[key] = val;
      }
    }

    if (_auth.currentUser != null) {
      data['profile'] = (await _getCloudProfile())?.toFirestore();
      data['sessions'] = await _getCloudSessions();
    }

    return data;
  }

  Future<UserProfile?> _getCloudProfile() async {
    if (_auth.currentUser == null) return null;
    try {
      final doc = await _db
          .collection(AxonCollections.usersPrivate)
          .doc(_auth.currentUser!.uid)
          .get();
      if (doc.exists) {
        return UserProfile.fromFirestore(doc.data()!, uid: doc.id);
      }
    } catch (e) {
      debugPrint('[SyncService] Failed to get cloud profile: $e');
    }
    return null;
  }

  Future<List<Map<String, dynamic>>> _getCloudSessions() async {
    if (_auth.currentUser == null) return [];
    try {
      final snapshot = await _db
          .collection(AxonCollections.usersPrivate)
          .doc(_auth.currentUser!.uid)
          .collection('sessions')
          .orderBy('date', descending: true)
          .limit(500)
          .get();
      return snapshot.docs.map((d) => d.data()).toList();
    } catch (e) {
      debugPrint('[SyncService] Failed to get cloud sessions: $e');
      return [];
    }
  }

  Future<bool> importData(Map<String, dynamic> data) async {
    try {
      final prefs = await SharedPreferences.getInstance();

      if (data['profile'] != null) {
        await syncProfile(UserProfile.fromFirestore(
          data['profile'],
          uid: _auth.currentUser?.uid ?? '',
        ));
      }

      for (final key in ['timer_history', 'metrics_state']) {
        if (data[key] != null) {
          await prefs.setString(key, jsonEncode(data[key]));
        }
      }

      for (final key in ['userSubjects', 'userBoard']) {
        if (data[key] != null) {
          await prefs.setString(key, data[key].toString());
        }
      }

      if (data['sessions'] != null) {
        for (final session in data['sessions']) {
          await syncSession(StudySession.fromJson(session));
        }
      }

      return true;
    } catch (e) {
      debugPrint('[SyncService] Failed to import data: $e');
      return false;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // UTILITIES
  // ─────────────────────────────────────────────────────────────────

  Future<bool> _checkConnectivity() async {
    try {
      final result = await InternetAddress.lookup('google.com');
      return result.isNotEmpty && result[0].rawAddress.isNotEmpty;
    } catch (e) {
      debugPrint('[SyncService] Connectivity check failed: $e');
      return false;
    }
  }

  Future<void> forceSyncNow() async {
    _updateState(_state.copyWith(status: SyncStatus.syncing));
    await _processSyncQueue();
  }

  Future<DateTime?> getLastSyncTime() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_lastSyncKey);
    if (raw == null) return null;
    return DateTime.tryParse(raw);
  }

  Future<int> getPendingSyncCount() async {
    final queue = await _getSyncQueue();
    return queue.length;
  }

  void dispose() {
    _syncTimer?.cancel();
    _heartbeatTimer?.cancel();
    _connectivitySubscription?.cancel();
    _syncStateController.close();
  }
}

// Provider
final syncServiceProvider = Provider<SyncService>((ref) {
  final service = SyncService();
  ref.onDispose(() => service.dispose());
  return service;
});

final syncStateProvider = StreamProvider<SyncState>((ref) {
  return ref.watch(syncServiceProvider).syncStateStream;
});
