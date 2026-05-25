// lib/main.dart -- Axon: Edge AI Study Intelligence
// ─────────────────────────────────────────────────────────────────

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart' as firebase_auth;
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'firebase_options.dart';
import 'theme/app_theme.dart';
import 'services/notification_service.dart';
import 'services/secure_credentials_service.dart';
import 'services/app_state.dart';
import 'services/board_exam_service.dart';
import 'services/smart_reminder_service.dart';
import 'services/google_drive_downloader.dart';
import 'services/grok_service.dart';
import 'services/personalization_service.dart';
import 'services/ambient_light_service.dart';
import 'router/app_router.dart';
import 'services/study_lock_service.dart';
import 'services/exam_data_service.dart';
import 'services/study_activity_service.dart';
import 'services/widget_service.dart';
import 'services/curriculum_catalog_service.dart';
import 'services/daily_plan_service.dart';
import 'services/comprehensive_curriculum_service.dart';
import 'services/unified_curriculum_service.dart';
import 'services/adaptive_performance_service.dart';
import 'services/enhanced_security_service.dart';
import 'screens/study/locked_app_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize adaptive performance FIRST - before any heavy operations
  final perfService = AdaptivePerformanceService();
  perfService.initialize();

  unawaited(EnhancedSecurityService().initialize().catchError((e) {
    debugPrint('EnhancedSecurityService init failed: $e');
  }));

  try {
    await dotenv.load(fileName: '.env');
    debugPrint('.env loaded successfully');
  } catch (e) {
    debugPrint('.env file not found or could not be loaded: $e');
  }

  _prefsInstance = await SharedPreferences.getInstance();

  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    ).timeout(const Duration(seconds: 4));
  } catch (e) {
    debugPrint('Firebase init failed: $e');
  }

  try {
    // Initialize secure credentials first
    final credentials = SecureCredentialsService();
    await credentials.initialize().timeout(const Duration(seconds: 2));
    final allCreds = await credentials.getAllCredentials();

    debugPrint('Axon: keys are server-side only');
  }

  runApp(
    ProviderScope(
      child: AxonApp(onInitialized: _runDeferredStartupTasks),
    ),
  );

  unawaited(() async {
    try {
      await Future.delayed(const Duration(seconds: 3));
      await ambientLightService.startMonitoring();
    } catch (e) {
      debugPrint('AmbientLightService start failed: $e');
    }
  }());
}

SharedPreferences? _prefsInstance;

SharedPreferences get prefsInstance {
  if (_prefsInstance != null) return _prefsInstance!;
  throw Exception('SharedPreferences not initialized. Call main() first.');
}

bool get isPrefsInitialized => _prefsInstance != null;

// ─────────────────────────────────────────────────────────────────
// AXON APP - Handles splash + initialization
// ─────────────────────────────────────────────────────────────────

class AxonApp extends StatefulWidget {
  final Future<void> Function() onInitialized;

  const AxonApp({super.key, required this.onInitialized});

  @override
  State<AxonApp> createState() => _AxonAppState();
}

class _AxonAppState extends State<AxonApp> {
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      // Use a timeout to ensure splash screen doesn't hang indefinitely
      await widget.onInitialized().timeout(const Duration(seconds: 10), onTimeout: () {
        debugPrint('Startup tasks timed out. Proceeding to app.');
      });
    } catch (e) {
      debugPrint('Initialization error: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = AxonThemeMode.isDark;
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      themeMode: AxonThemeMode.mode,
      darkTheme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor:
            isDark ? const Color(0xFF000000) : const Color(0xFFFFFFFF),
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          scrolledUnderElevation: 0,
        ),
        colorScheme: ColorScheme.dark(
          surface: isDark ? const Color(0xFF0D0D0D) : const Color(0xFFFFFFFF),
          primary: const Color(0xFF3A86FF),
          secondary: const Color(0xFF6BA3FF),
        ),
      ),
      theme: ThemeData.light().copyWith(
        scaffoldBackgroundColor:
            isDark ? const Color(0xFF000000) : const Color(0xFFFFFFFF),
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          scrolledUnderElevation: 0,
        ),
        colorScheme: ColorScheme.light(
          surface: isDark ? const Color(0xFF0D0D0D) : const Color(0xFFFFFFFF),
          primary: const Color(0xFF3A86FF),
          secondary: const Color(0xFF6BA3FF),
        ),
      ),
      home: _isInitialized
          ? const _AppWithRouter()
          : const Scaffold(backgroundColor: Color(0xFF040B14)),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// APP WITH ROUTER
// ─────────────────────────────────────────────────────────────────

class _AppWithRouter extends ConsumerWidget {
  const _AppWithRouter({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Initialize blocked app overlay detection
    BlockedAppOverlay.init(ref);

    final router = ref.watch(appRouterProvider);
    final greyscaleNotifier = greyscaleModeNotifierProvider;

    return ListenableBuilder(
      listenable: greyscaleNotifier,
      builder: (context, _) {
        final isGreyscale = greyscaleNotifier.enabled;
        final child = MaterialApp.router(
          debugShowCheckedModeBanner: false,
          theme: ThemeData.dark(),
          routerConfig: router,
        );
        if (isGreyscale) {
          return ColorFiltered(
            colorFilter: const ColorFilter.mode(
              Color(0xFF999999),
              BlendMode.saturation,
            ),
            child: child,
          );
        }
        return child;
      },
    );
  }
}

Future<void> _runDeferredStartupTasks() async {
  unawaited(NotificationService.initialize()
      .timeout(const Duration(seconds: 5))
      .then((_) => _checkAndShowStreak())
      .catchError((e) {
    debugPrint('NotificationService init failed or timed out: $e');
  }));

  unawaited(SmartReminderService().onAppStartup().catchError((e) {
    debugPrint('SmartReminderService startup failed: $e');
  }));

  try {
    if (dotenv.env.isEmpty) {
      unawaited(dotenv.load(fileName: '.env').timeout(const Duration(seconds: 2)).catchError((e) {
        debugPrint('.env reload failed: $e');
      }));
    }
  } catch (e) {
    debugPrint('.env reload failed: $e');
  }

  unawaited(BoardExamService.initFromPrefs().timeout(const Duration(seconds: 3)).catchError((e) {
    debugPrint('BoardExamService init failed: $e');
  }));

  unawaited(StudyLockService.instance.initialize().timeout(const Duration(seconds: 2)).catchError((e) {
    debugPrint('StudyLockService init failed: $e');
  }));

  // Pre-load curriculum data locally for fast subject/chapter access
  unawaited(CurriculumCatalogService.instance.initializeLocalData().catchError((e) {
    debugPrint('CurriculumCatalogService init failed: $e');
  }));

  unawaited(ExamDataService().initialize().catchError((e) {
    debugPrint('ExamDataService init failed: $e');
  }));

  unawaited(StudyActivityService().initialize().catchError((e) {
    debugPrint('StudyActivityService init failed: $e');
  }));

  // Initialize comprehensive curriculum service
  unawaited(ComprehensiveCurriculumService.instance.initialize().catchError((e) {
    debugPrint('ComprehensiveCurriculumService init failed: $e');
  }));

  // Initialize unified curriculum service
  unawaited(UnifiedCurriculumService.instance.initialize().catchError((e) {
    debugPrint('UnifiedCurriculumService init failed: $e');
  }));

  Future(() async {
    try {
      await WidgetService().initialize();
      await WidgetService().updateAllWidgets();
    } catch (e) {
      debugPrint('WidgetService startup failed: $e');
    }
  });

  Future(() async {
    try {
      await GrokService().initialize();
    } catch (e) {
      debugPrint('GrokService init failed: $e');
    }
  });

  Future(() async {
    try {
      await GoogleDriveDownloader.instance.initialize();
    } catch (e) {
      debugPrint('GoogleDriveDownloader init failed: $e');
    }
  });

  Future(() async {
    try {
      await SmartReminderService().onAppStartup();
    } catch (e) {
      debugPrint('SmartReminderService init failed: $e');
    }
  });

  // Auto-generate daily plan if user has deadlines
  unawaited(_autoGenerateDailyPlan());
}

Future<void> _autoGenerateDailyPlan() async {
  try {
    final user = firebase_auth.FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final service = DailyPlanService();
    final existing = await service.getTasksForDate(
      user.uid,
      DateTime.now().toIso8601String().split('T').first,
    );
    if (existing.isNotEmpty) {
      debugPrint('AutoPlan: plan exists for today, skipping');
      return;
    }

    await service.ensureTodayPlan(user.uid);
    debugPrint('AutoPlan: daily plan generated');
  } catch (e) {
    debugPrint('AutoPlan failed: $e');
  }
}

Future<void> _checkAndShowStreak() async {
  try {
    final prefs = _prefsInstance ?? await SharedPreferences.getInstance();
    final streak = prefs.getInt('study_streak') ?? 0;
    final lastStudyDate = prefs.getString('last_study_date');
    final today = DateTime.now();
    final todayStr = '${today.year}-${today.month}-${today.day}';

    bool studiedToday = lastStudyDate == todayStr;

    final streakEnabled = _prefsInstance?.getBool('streak_notifications_enabled') ?? true;
    if (streak > 0 && streakEnabled) {
      await NotificationService.show(
        id: 7001,
        title: '🔥 $streak Day Streak!',
        body: studiedToday
            ? 'Keep the streak going! You\'re on fire!'
            : 'Don\'t break your $streak day streak!',
        channel: AxonChannel.streakAlert,
        payload: '/home',
      );
    }
  } catch (e) {
    debugPrint('Streak check failed: $e');
  }
}

final deadlineAccentProvider = FutureProvider<AxonAccentScheme>((ref) async {
  final gate = ref.watch(authGateProvider).valueOrNull;
  final board = gate?.profile?.board ?? '';
  if (board.trim().isEmpty) return AxonAccentScheme.focused;

  try {
    final prefs = _prefsInstance ?? await SharedPreferences.getInstance();
    final preferredStudyHour = _preferredStudyHourFromPrefs(prefs);
    final now = DateTime.now();
    final result = await BoardExamService().fetchBoardDates(board);
    DateTime? nearest;
    for (final event in result.events) {
      if (event.startDate.isBefore(now)) continue;
      if (nearest == null || event.startDate.isBefore(nearest)) {
        nearest = event.startDate;
      }
    }
    final days = nearest?.difference(now).inDays;
    final base = AxonAccentScheme.forDaysUntilExam(days);
    return _withStudyWindowGlow(base, preferredStudyHour, now);
  } catch (_) {
    final prefs = _prefsInstance ?? await SharedPreferences.getInstance();
    final preferredStudyHour = _preferredStudyHourFromPrefs(prefs);
    return _withStudyWindowGlow(
      AxonAccentScheme.focused,
      preferredStudyHour,
      DateTime.now(),
    );
  }
});

int? _preferredStudyHourFromPrefs(SharedPreferences prefs) {
  final raw = prefs.getString('timer_history');
  if (raw == null || raw.isEmpty) return null;
  try {
    final decoded = raw;
    final matches = RegExp(r'"date"\s*:\s*"([^"]+)"')
        .allMatches(decoded)
        .map((m) => DateTime.tryParse(m.group(1)!))
        .whereType<DateTime>()
        .toList();
    if (matches.isEmpty) return null;
    final buckets = <int, int>{};
    for (final date in matches) {
      buckets[date.hour] = (buckets[date.hour] ?? 0) + 1;
    }
    final sorted = buckets.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return sorted.first.key;
  } catch (_) {
    return null;
  }
}

AxonAccentScheme _withStudyWindowGlow(
  AxonAccentScheme base,
  int? preferredHour,
  DateTime now,
) {
  if (preferredHour == null) return base;
  final minutesNow = now.hour * 60 + now.minute;
  final preferredMinutes = preferredHour * 60;
  final delta = (minutesNow - preferredMinutes).abs();
  if (delta > 35) return base;
  final t = 1.0 - (delta / 35.0);
  return AxonAccentScheme(
    accent: Color.lerp(base.accent, Colors.white, 0.18 * t)!,
    accentBlue: Color.lerp(base.accentBlue, base.accent, 0.32 * t)!,
    accentPurple: Color.lerp(base.accentPurple, base.accentBlue, 0.22 * t)!,
    warning: Color.lerp(base.warning, Colors.white, 0.12 * t)!,
  );
}
