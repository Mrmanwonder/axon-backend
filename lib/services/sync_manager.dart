// lib/services/sync_manager.dart
import 'dart:async';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/daily_plan_task.dart';
import 'daily_plan_service.dart';
import 'notion_service.dart';
import 'google_calendar_service.dart';
import 'obsidian_service.dart';

class SyncManager {
  static final SyncManager _instance = SyncManager._internal();
  factory SyncManager() => _instance;
  SyncManager._internal();

  final DailyPlanService _planService = DailyPlanService();
  StreamSubscription? _planSubscription;
  String? _currentUid;
  bool _isWatching = false;

  // Services are now singletons
  GoogleCalendarService? _googleService;
  NotionService? _notionService;
  ObsidianService? _obsidianService;

  // Debouncer for auto-sync
  _Debouncer? _autoSyncDebouncer;

  // Track sync status
  bool _isSyncing = false;
  DateTime? _lastFullSync;

  /// Get or create the Google Calendar service instance
  GoogleCalendarService get googleCalendarService {
    _googleService ??= GoogleCalendarService();
    return _googleService!;
  }

  /// Get or create the Notion service instance
  NotionService get notionService {
    _notionService ??= NotionService();
    return _notionService!;
  }

  /// Get or create the Obsidian service instance
  ObsidianService get obsidianService {
    _obsidianService ??= ObsidianService();
    return _obsidianService!;
  }

  /// Start watching for plan changes and auto-sync
  Future<void> startWatching(String uid) async {
    if (_isWatching && _currentUid == uid) {
      debugPrint('Already watching for uid: $uid');
      return;
    }

    // Stop any existing watch
    await stopWatching();

    _currentUid = uid;
    _isWatching = true;
    _autoSyncDebouncer = _Debouncer(const Duration(seconds: 3));

    debugPrint('Starting to watch plan changes for uid: $uid');

    _planSubscription = _planService.watchTodayPlan(uid).listen(
      (tasks) {
        _autoSyncDebouncer?.call(() async {
          if (!_isSyncing) {
            await _syncAll(tasks);
          }
        });
      },
      onError: (error) {
        debugPrint('Error watching plan: $error');
      },
    );
  }

  /// Stop watching for changes
  Future<void> stopWatching() async {
    await _planSubscription?.cancel();
    _planSubscription = null;
    _isWatching = false;
    _currentUid = null;
    _autoSyncDebouncer?.dispose();
    _autoSyncDebouncer = null;
  }

  /// Manually trigger a full sync now
  Future<SyncResult> syncNow() async {
    return await _performSync();
  }

  /// Sync only to Google Calendar
  Future<SyncResult> syncGoogleCalendar() async {
    return await _syncService(
      serviceName: 'Google Calendar',
      syncFn: () async {
        final tasks = await _getCurrentTasks();
        if (tasks.isEmpty) {
          debugPrint('No tasks to sync to Google Calendar');
          return;
        }
        await googleCalendarService.syncStudyPlan(_convertTasks(tasks));
      },
    );
  }

  /// Sync only to Notion
  Future<SyncResult> syncNotion() async {
    return await _syncService(
      serviceName: 'Notion',
      syncFn: () async {
        final tasks = await _getCurrentTasks();
        if (tasks.isEmpty) {
          debugPrint('No tasks to sync to Notion');
          return;
        }
        await notionService.syncStudyPlan(_convertTasks(tasks));
      },
    );
  }

  /// Sync only to Obsidian
  Future<SyncResult> syncObsidian() async {
    return await _syncService(
      serviceName: 'Obsidian',
      syncFn: () async {
        final tasks = await _getCurrentTasks();
        if (tasks.isEmpty) {
          debugPrint('No tasks to sync to Obsidian');
          return;
        }
        await obsidianService.syncStudyPlan(_convertTasks(tasks));
      },
    );
  }

  /// Check if Google Calendar is connected and ready
  bool get isGoogleCalendarConnected => googleCalendarService.isConnected;

  /// Check if Notion is connected and ready
  bool get isNotionConnected => notionService.isConnected;

  /// Check if Obsidian is connected and ready
  bool get isObsidianConnected => obsidianService.isConnected;

  /// Get connection status for all services
  Map<String, bool> get connectionStatus => {
        'googleCalendar': googleCalendarService.isConnected,
        'notion': notionService.isConnected,
        'obsidian': obsidianService.isConnected,
      };

  Future<List<DailyPlanTask>> _getCurrentTasks() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null || _currentUid == null) {
      debugPrint('No user logged in for sync');
      return [];
    }

    try {
      final today = DateTime.now().toIso8601String().split('T').first;
      final tasks = await _planService.getTasksForDate(_currentUid!, today);
      return tasks;
    } catch (e) {
      debugPrint('Error fetching current tasks for sync: $e');
      return [];
    }
  }

  Future<SyncResult> _performSync() async {
    if (_isSyncing) {
      debugPrint('Sync already in progress');
      return SyncResult(
        success: false,
        message: 'Sync already in progress',
        syncedServices: {},
      );
    }

    _isSyncing = true;
    final results = <String, bool>{};
    String? errorMessage;

    try {
      final tasks = await _getCurrentTasks();

      if (tasks.isEmpty) {
        debugPrint('No tasks to sync');
        _isSyncing = false;
        return SyncResult(
          success: true,
          message: 'No tasks to sync',
          syncedServices: {},
        );
      }

      debugPrint('Starting full sync with ${tasks.length} tasks');

      // Sync to each service
      if (googleCalendarService.isConnected) {
        try {
          await googleCalendarService.syncStudyPlan(_convertTasks(tasks));
          results['googleCalendar'] = true;
          await _recordLastSync('google_calendar_last_sync');
        } catch (e, st) {
          debugPrint('Google Calendar sync error: $e\n$st');
          results['googleCalendar'] = false;
          errorMessage = 'Google Calendar: $e';
        }
      } else {
        debugPrint('Google Calendar not connected, skipping');
        results['googleCalendar'] = false;
      }

      if (notionService.isConnected) {
        try {
          await notionService.syncStudyPlan(_convertTasks(tasks));
          results['notion'] = true;
          await _recordLastSync('notion_last_sync');
        } catch (e, st) {
          debugPrint('Notion sync error: $e\n$st');
          results['notion'] = false;
          errorMessage = errorMessage ?? 'Notion: $e';
        }
      } else {
        debugPrint('Notion not connected, skipping');
        results['notion'] = false;
      }

      if (obsidianService.isConnected) {
        try {
          await obsidianService.syncStudyPlan(_convertTasks(tasks));
          results['obsidian'] = true;
          await _recordLastSync('obsidian_last_sync');
        } catch (e, st) {
          debugPrint('Obsidian sync error: $e\n$st');
          results['obsidian'] = false;
          errorMessage = errorMessage ?? 'Obsidian: $e';
        }
      } else {
        debugPrint('Obsidian not connected, skipping');
        results['obsidian'] = false;
      }

      final success = results.values.any((v) => v);
      _lastFullSync = DateTime.now();

      return SyncResult(
        success: success,
        message: errorMessage ??
            (success ? 'Sync completed' : 'No services connected'),
        syncedServices: results,
        syncedAt: _lastFullSync,
      );
    } catch (e, st) {
      debugPrint('Sync error: $e\n$st');
      return SyncResult(
        success: false,
        message: 'Sync failed: $e',
        syncedServices: results,
      );
    } finally {
      _isSyncing = false;
    }
  }

  Future<SyncResult> _syncService({
    required String serviceName,
    required Future<void> Function() syncFn,
  }) async {
    try {
      await syncFn();
      await _recordLastSync('${serviceName.toLowerCase()}_last_sync');
      return SyncResult(
        success: true,
        message: '$serviceName sync completed',
        syncedServices: {serviceName: true},
      );
    } catch (e, st) {
      debugPrint('$serviceName sync error: $e\n$st');
      return SyncResult(
        success: false,
        message: '$serviceName: $e',
        syncedServices: {serviceName: false},
      );
    }
  }

  Future<void> _syncAll(List<DailyPlanTask> tasks) async {
    if (tasks.isEmpty) return;

    debugPrint('Auto-syncing ${tasks.length} tasks');

    // Sync to connected services
    if (googleCalendarService.isConnected) {
      try {
        await googleCalendarService.syncStudyPlan(_convertTasks(tasks));
        await _recordLastSync('google_calendar_last_sync');
      } catch (e, st) {
        debugPrint('Auto-sync Google Calendar error: $e\n$st');
      }
    }
  }

  List<Map<String, dynamic>> _convertTasks(List<DailyPlanTask> tasks) {
    return tasks.map((task) {
      final description = StringBuffer();
      if (task.objectiveId.isNotEmpty) {
        description.writeln('Objective: ${task.objectiveId}');
      }
      if (task.reason.isNotEmpty) {
        description.writeln('Reason: ${task.reason}');
      }
      description.writeln('Type: ${task.taskType.value.replaceAll('_', ' ')}');
      if (task.subject.isNotEmpty) {
        description.writeln('Subject: ${task.subject}');
      }

      return {
        'id': task.id,
        'title': task.title,
        'description': description.toString().trim(),
        'start': task.startTime,
        'end': task.endTime,
        'subject': task.subject,
        'topic': task.paper,
        'completed': task.status == 'completed',
      };
    }).toList();
  }

  Future<void> _recordLastSync(String prefKey) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(prefKey, DateTime.now().millisecondsSinceEpoch);
  }

  /// Get the last sync time for a service
  Future<DateTime?> getLastSyncTime(String service) async {
    final prefs = await SharedPreferences.getInstance();
    final timestamp = prefs.getInt('${service.toLowerCase()}_last_sync');
    return timestamp != null
        ? DateTime.fromMillisecondsSinceEpoch(timestamp)
        : null;
  }
}

class _Debouncer {
  final Duration delay;
  Timer? _timer;
  _Debouncer(this.delay);

  void call(void Function() action) {
    _timer?.cancel();
    _timer = Timer(delay, action);
  }

  void dispose() {
    _timer?.cancel();
    _timer = null;
  }
}

/// Result of a sync operation
class SyncResult {
  final bool success;
  final String message;
  final Map<String, bool> syncedServices;
  final DateTime? syncedAt;

  SyncResult({
    required this.success,
    required this.message,
    required this.syncedServices,
    this.syncedAt,
  });

  @override
  String toString() {
    return 'SyncResult(success: $success, message: $message, services: $syncedServices)';
  }
}
