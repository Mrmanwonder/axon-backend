// lib/services/study_lock_service.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

class StudyLockService {
  static final StudyLockService _instance = StudyLockService._();
  static StudyLockService get instance => _instance;
  static const MethodChannel _channel = MethodChannel('com.axon.studylock');
  static const EventChannel _eventChannel = EventChannel('com.axon.studylock/events');

  static const String _blockedAppsKey = 'study_lock_blocked_apps';
  static const String _requiredMinutesKey = 'study_lock_required_minutes';
  static const String _loggedMinutesKey = 'study_lock_logged_minutes';
  static const String _isActiveKey = 'study_lock_active';
  static const String _accountabilityKey = 'study_lock_accountability_pending';

  Timer? _studyCheckTimer;
  bool _isInitialized = false;

  StreamSubscription<dynamic>? _blockedAppSub;

  void Function(String appName, String packageName)? onBlockedAppDetected;

  StudyLockService._();

  Future<void> initialize() async {
    if (_isInitialized) return;
    _isInitialized = true;

    _eventChannel.receiveBroadcastStream().listen((event) {
      if (event is Map) {
        final appName = event['appName'] as String? ?? 'Blocked App';
        final packageName = event['packageName'] as String? ?? '';
        debugLog('Blocked app detected: $appName ($packageName)');
        onBlockedAppDetected?.call(appName, packageName);
      }
    }, onError: (e) {
      debugLog('EventChannel error: $e');
    });

    await restoreTimerIfNeeded();
  }

  void dispose() {
    _blockedAppSub?.cancel();
    _studyCheckTimer?.cancel();
  }

  Future<void> setDistractionApps(
      List<String> packageNames, int requiredMinutes) async {
    final normalizedPackages = packageNames
        .map((packageName) => packageName.trim())
        .where((packageName) => packageName.isNotEmpty)
        .toSet()
        .toList(growable: false);

    if (normalizedPackages.isEmpty) {
      throw ArgumentError('At least one distraction app must be selected.');
    }

    final safeRequiredMinutes = requiredMinutes.clamp(1, 24 * 60);
    final prefs = await SharedPreferences.getInstance();
    final currentStudyMinutes = await _getCurrentStudyMinutes();

    await prefs.setStringList(_blockedAppsKey, normalizedPackages);
    await prefs.setInt(_requiredMinutesKey, safeRequiredMinutes);
    await prefs.setInt('study_lock_baseline_minutes', currentStudyMinutes);
    await prefs.setInt(_loggedMinutesKey, 0);
    await prefs.setBool(_isActiveKey, true);

    _startStudyTimeChecker();

    if (Platform.isAndroid) {
      _notifyNativeService();
    }

    debugLog('StudyLock: Activated for $safeRequiredMinutes study minutes, blocking ${normalizedPackages.length} apps');
  }

  void _startStudyTimeChecker() {
    _studyCheckTimer?.cancel();
    _studyCheckTimer = Timer.periodic(const Duration(seconds: 10), (_) {
      _checkStudyProgress();
    });
  }

  Future<void> _checkStudyProgress() async {
    final prefs = await SharedPreferences.getInstance();
    final isActive = prefs.getBool(_isActiveKey) ?? false;
    if (!isActive) {
      _studyCheckTimer?.cancel();
      _studyCheckTimer = null;
      return;
    }

    final required = prefs.getInt(_requiredMinutesKey) ?? 60;
    final baseline = prefs.getInt('study_lock_baseline_minutes') ?? 0;
    final current = await _getCurrentStudyMinutes();
    final logged = (current - baseline).clamp(0, required);

    await prefs.setInt(_loggedMinutesKey, logged);

    if (logged >= required && required > 0) {
      await _completeSession();
    }
  }

  Future<int> _getCurrentStudyMinutes() async {
    final prefs = await SharedPreferences.getInstance();
    final metricsState = prefs.getString('metrics_state');
    if (metricsState == null) return 0;

    try {
      final map = jsonDecode(metricsState) as Map<String, dynamic>;
      final hours = (map['activeStudyHours'] as num?)?.toDouble() ?? 0.0;
      return (hours * 60).round();
    } catch (e) {
      return 0;
    }
  }

  Future<void> _completeSession() async {
    _studyCheckTimer?.cancel();
    _studyCheckTimer = null;

    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_isActiveKey, false);
    await prefs.remove('study_lock_baseline_minutes');

    if (Platform.isAndroid) {
      _notifyNativeService();
    }

    debugLog('StudyLock: Session completed');
  }

  Future<void> deactivateStudyLock() async {
    _studyCheckTimer?.cancel();
    _studyCheckTimer = null;

    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_isActiveKey, false);
    await prefs.setInt(_loggedMinutesKey, 0);
    await prefs.remove('study_lock_baseline_minutes');
    await prefs.remove(_accountabilityKey);

    if (Platform.isAndroid) {
      _notifyNativeService();
    }
  }

  Future<void> syncOnResume() async {
    await _checkStudyProgress();
  }

  Future<void> restoreTimerIfNeeded() async {
    final prefs = await SharedPreferences.getInstance();
    final isActive = prefs.getBool(_isActiveKey) ?? false;

    if (isActive) {
      await syncOnResume();

      final stillActive = prefs.getBool(_isActiveKey) ?? false;
      if (stillActive) {
        _startStudyTimeChecker();
      }

      final pending = prefs.getString(_accountabilityKey);
      if (pending == null || pending.isEmpty) {
        await prefs.setString(
          _accountabilityKey,
          'Focus lock was interrupted before completion. Resume under discipline and finish the remaining minutes before taking distractions back.',
        );
      }

      debugLog('StudyLock: Timer restored');
    }
  }

  Future<bool> isActive() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_isActiveKey) ?? false;
  }

  Future<int> getLoggedMinutes() async {
    final prefs = await SharedPreferences.getInstance();
    final isActive = prefs.getBool(_isActiveKey);
    if (isActive != true) return prefs.getInt(_loggedMinutesKey) ?? 0;

    final required = prefs.getInt(_requiredMinutesKey) ?? 60;
    final baseline = prefs.getInt('study_lock_baseline_minutes') ?? 0;
    final current = await _getCurrentStudyMinutes();
    return (current - baseline).clamp(0, required);
  }

  Future<int> getRequiredMinutes() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(_requiredMinutesKey) ?? 60;
  }

  Future<List<String>> getBlockedApps() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getStringList(_blockedAppsKey) ?? [];
  }

  Future<Map<String, dynamic>> getStatus() async {
    final prefs = await SharedPreferences.getInstance();
    final isActive = prefs.getBool(_isActiveKey) ?? false;
    final required = prefs.getInt(_requiredMinutesKey) ?? 60;
    final logged = await getLoggedMinutes();

    return {
      'isActive': isActive,
      'loggedMinutes': logged,
      'requiredMinutes': required,
      'blockedApps': prefs.getStringList(_blockedAppsKey) ?? [],
    };
  }

  Future<String?> getPendingAccountabilityMessage() async {
    final prefs = await SharedPreferences.getInstance();
    final message = prefs.getString(_accountabilityKey);
    return message == null || message.isEmpty ? null : message;
  }

  Future<void> clearPendingAccountabilityMessage() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_accountabilityKey);
  }

  void _notifyNativeService() async {
    if (Platform.isAndroid) {
      try {
        await _channel.invokeMethod('refreshLockStatus');
      } catch (e) {
        debugPrint('StudyLock: Native platform service not available on this device');
      }
    }
  }

  Future<bool> isAccessibilityEnabled() async {
    if (!Platform.isAndroid) return false;
    try {
      final result = await _channel.invokeMethod<bool>('isAccessibilityEnabled');
      return result ?? false;
    } catch (e) {
      return false;
    }
  }

  Future<void> requestAccessibilityPermission() async {
    if (!Platform.isAndroid) return;
    try {
      await _channel.invokeMethod('requestAccessibilityPermission');
    } catch (e) {
      try {
        await launchUrl(Uri.parse('android.settings.ACCESSIBILITY_SETTINGS'));
      } catch (e2) {
        debugPrint('StudyLock: Could not open accessibility settings: $e2');
      }
    }
  }

  Future<bool> hasOverlayPermission() async {
    if (!Platform.isAndroid) return true;
    try {
      final result = await _channel.invokeMethod<bool>('hasOverlayPermission');
      return result ?? false;
    } catch (e) {
      return false;
    }
  }

  Future<void> requestOverlayPermission() async {
    if (!Platform.isAndroid) return;
    try {
      await _channel.invokeMethod('openOverlaySettings');
    } catch (e) {
      try {
        await launchUrl(Uri.parse('android.settings.MANAGE_OVERLAY_PERMISSION'));
      } catch (e2) {
        debugPrint('StudyLock: Could not open overlay settings: $e2');
      }
    }
  }

  Future<bool> hasRequiredAndroidPermissions() async {
    if (!Platform.isAndroid) return true;
    final results = await Future.wait<bool>([
      hasOverlayPermission(),
      isAccessibilityEnabled(),
    ]);
    return results.every((isGranted) => isGranted);
  }

  void debugLog(String message) {
    debugPrint('[StudyLock] $message');
  }
}

class AppBlockerService {
  final String packageName;
  final String appName;

  const AppBlockerService({required this.packageName, required this.appName});

  Map<String, dynamic> toJson() => {
        'packageName': packageName,
        'appName': appName,
      };

  factory AppBlockerService.fromJson(Map<String, dynamic> json) {
    return AppBlockerService(
      packageName: json['packageName'] ?? '',
      appName: json['appName'] ?? '',
    );
  }
}

class CommonBlockedApps {
  static const List<AppBlockerService> socialMedia = [
    AppBlockerService(packageName: 'com.instagram.android', appName: 'Instagram'),
    AppBlockerService(packageName: 'com.twitter.android', appName: 'Twitter/X'),
    AppBlockerService(packageName: 'com.facebook.katana', appName: 'Facebook'),
    AppBlockerService(packageName: 'com.snapchat.android', appName: 'Snapchat'),
    AppBlockerService(packageName: 'com.zhiliaoapp.musically', appName: 'TikTok'),
    AppBlockerService(packageName: 'com.reddit.frontpage', appName: 'Reddit'),
  ];

  static const List<AppBlockerService> video = [
    AppBlockerService(packageName: 'com.google.android.youtube', appName: 'YouTube'),
    AppBlockerService(packageName: 'com.netflix.mediaclient', appName: 'Netflix'),
    AppBlockerService(packageName: 'com.amazon.mv6', appName: 'Prime Video'),
  ];

  static List<AppBlockerService> get all => [...socialMedia, ...video];
}
