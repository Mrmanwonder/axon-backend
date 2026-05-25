// lib/router/app_router.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart' hide ChangeNotifierProvider;
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart' show ChangeNotifierProvider;
import '../services/app_state.dart';
import '../screens/auth/login_screen.dart';
import '../screens/auth/register_screen.dart';
import '../screens/auth/onboarding_screen.dart';
import '../screens/auth/board_selection_screen.dart';
import '../screens/auth/subject_selection_screen.dart';
import '../screens/auth/motivation_selection_screen.dart';
import '../screens/dashboard/dashboard_screen.dart';
import '../screens/analysis/analysis_screen.dart';
import '../screens/analysis/admissions_tracker_screen.dart';
import '../screens/exam_calendar_screen.dart';
import '../screens/pdf/pdf_viewer_screen.dart';
import '../screens/study/feynman_technique_screen.dart';
import '../screens/study/leitner_system_screen.dart';
import '../screens/study/interleaved_practice_screen.dart';
import '../screens/study/study_screen.dart';
import '../screens/timer/axon_timer_screen.dart';
import '../screens/study/strict_lock_screen.dart';
import '../screens/study/flash_cards_screen.dart';
import '../screens/study/quiz_wrapper_screen.dart';
import '../screens/study/active_recall_screen.dart';
import '../screens/study/mock_simulation_screen.dart';
import '../screens/settings/settings_screen.dart';
import '../screens/settings/accessibility_settings_screen.dart';
import '../screens/settings/edit_account_screen.dart';
import '../screens/settings/sync_settings_screen.dart';
import '../screens/settings/coaching_persona_screen.dart';
import '../screens/settings/study_lock_settings_screen.dart';
import '../screens/achievements_screen.dart';
import '../screens/settings/subscription/subscription_screen.dart';
import '../screens/settings/subscription/subscription_management_screen.dart';
import '../screens/media/media_viewer_screen.dart';
import '../screens/ai/axon_ai_screen.dart';
import '../screens/video/video_player_screen.dart';
import '../screens/exam/exam_planner_screen.dart';
import '../screens/shell_screen.dart';
import '../models/models.dart';
import '../services/exam_planner_service.dart';
import '../screens/axon_catalog/lib/screens/catalog_screen.dart';
import '../screens/axon_catalog/lib/controllers/catalog_controller.dart';

enum AuthGateStage {
  unknown,
  signedOut,
  onboarding,
  ready,
}

class AppRoutes {
  static const login = '/auth/login';
  static const register = '/auth/register';
  static const onboarding = '/auth/onboarding';
  static const boardSelection = '/auth/board';
  static const subjectSelection = '/auth/subjects';
  static const motivationSelection = '/auth/motivation';
  static const home = '/home';
  static const dashboard = '/dashboard';
  static const analysis = '/analysis';
  static const analytics = '/analytics';
  static const admissions = '/analysis/admissions';
  static const pdfLibrary = '/pdf';
  static const pdfViewer = '/pdf-viewer';
  static const videoPlayer = '/video-player';
  static const study = '/study';
  static const studyTimer = '/study/timer';
  static const timer = '/timer';
  static const ai = '/ai';
  static const studyLock = '/study/lock';
  static const catalog = '/catalog';
  static const quiz = '/study/quiz';
  static const flashCards = '/study/flashcards';
  static const activeRecall = '/study/recall';
  static const studyFeynman = '/study/feynman';
  static const studyLeitner = '/study/leitner';
  static const studyInterleaved = '/study/interleaved';
  static const mockSimulation = '/study/mock-simulation';
  static const settings = '/settings';
  static const settingsAccessibility = '/settings/accessibility';
  static const settingsAccount = '/settings/account';
  static const settingsSync = '/settings/sync';
  static const settingsSubscription = '/settings/subscription';
  static const settingsSubscriptionManage = '/settings/subscription/manage';
  static const examPlanner = '/exam/planner';
  static const calendar = '/calendar';
  static const coachingPersona = '/settings/coaching-persona';
  static const studyLockSettings = '/settings/study-lock';
  static const achievements = '/achievements';
  static const mediaViewer = '/media-viewer';
}

MotivationStyle _parseMotivationStyle(String? style) {
  switch (style) {
    case 'toughLove':
      return MotivationStyle.toughLove;
    case 'positiveReinforcement':
      return MotivationStyle.positiveReinforcement;
    case 'logicBased':
      return MotivationStyle.logicBased;
    default:
      return MotivationStyle.positiveReinforcement;
  }
}

final appRouterProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    initialLocation: AppRoutes.login,
    debugLogDiagnostics: true,
    refreshListenable:
        _GoRouterRefreshStream(ref.watch(authStateProvider.notifier).stream),
    redirect: (context, state) {
      final auth = ref.read(authStateProvider);
      final path = state.uri.path;
      final isAuthRoute = path.startsWith('/auth');
      final isOnboarding = path == AppRoutes.onboarding;
      final onboardingIncomplete = auth.isAuthenticated &&
          auth.user != null &&
          !auth.user!.onboardingComplete;

      if (!auth.isAuthenticated) {
        if (!isAuthRoute) return AppRoutes.login;
        return null;
      }

      if (onboardingIncomplete && !isOnboarding) return AppRoutes.onboarding;
      if (onboardingIncomplete && isOnboarding) return null;

      if (isAuthRoute) return AppRoutes.home;

      return null;
    },
    routes: [
      GoRoute(
        path: AppRoutes.login,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const LoginScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.register,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const RegisterScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.onboarding,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const OnboardingScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.boardSelection,
        pageBuilder: (context, state) {
          final extra = state.extra as Map<String, dynamic>? ?? {};
          final board = extra['board'] as String? ?? 'CAIE IGCSE';
          return NoTransitionPage(
            key: state.pageKey,
            child: BoardSelectionScreen(initialBoard: board),
          );
        },
      ),
      GoRoute(
        path: AppRoutes.subjectSelection,
        pageBuilder: (context, state) {
          final extra = state.extra as Map<String, dynamic>? ?? {};
          final board = extra['board'] as String? ?? 'IGCSE';
          return NoTransitionPage(
            key: state.pageKey,
            child: SubjectSelectionScreen(board: board),
          );
        },
      ),
      GoRoute(
        path: AppRoutes.motivationSelection,
        pageBuilder: (context, state) {
          final extra = state.extra as Map<String, dynamic>? ?? {};
          return NoTransitionPage(
            key: state.pageKey,
            child: MotivationSelectionScreen(
              board: extra['board'] ?? 'IGCSE',
              subjects: List<String>.from(extra['subjects'] ?? ['Mathematics']),
            ),
          );
        },
      ),
      ShellRoute(
        builder: (context, state, child) {
          return ShellScreen(child: child);
        },
        routes: [
          GoRoute(
            path: AppRoutes.home,
            pageBuilder: (context, state) => NoTransitionPage(
              key: state.pageKey,
              child: const DashboardScreen(),
            ),
          ),
          GoRoute(
            path: AppRoutes.analysis,
            pageBuilder: (context, state) => NoTransitionPage(
              key: state.pageKey,
              child: const AnalysisScreen(),
            ),
          ),
          GoRoute(
            path: AppRoutes.pdfLibrary,
            pageBuilder: (context, state) => NoTransitionPage(
              key: state.pageKey,
              child: const ExamPlannerScreen(),
            ),
          ),
          GoRoute(
            path: AppRoutes.study,
            pageBuilder: (context, state) => NoTransitionPage(
              key: state.pageKey,
              child: const StudyScreen(),
            ),
          ),
          GoRoute(
            path: AppRoutes.mockSimulation,
            pageBuilder: (context, state) {
              final extra = state.extra as Map<String, dynamic>? ?? {};
              final paper = extra['paper'];
              if (paper is! PastPaperPack) {
                return NoTransitionPage(
                  key: state.pageKey,
                  child: const Scaffold(
                    body: Center(
                      child: Text('Missing mock simulation payload'),
                    ),
                  ),
                );
              }
              return NoTransitionPage(
                key: state.pageKey,
                child: MockSimulationScreen(paper: paper),
              );
            },
          ),
          GoRoute(
            path: AppRoutes.settings,
            pageBuilder: (context, state) => NoTransitionPage(
              key: state.pageKey,
              child: const SettingsScreen(),
            ),
          ),
        ],
      ),
      GoRoute(
        path: AppRoutes.pdfViewer,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: PdfViewerScreen(
            url: state.uri.queryParameters['url'],
            title: state.uri.queryParameters['title'],
          ),
        ),
      ),
      GoRoute(
        path: AppRoutes.videoPlayer,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: VideoPlayerScreen(
            url: state.uri.queryParameters['url'],
            title: state.uri.queryParameters['title'],
          ),
        ),
      ),
      GoRoute(
        path: AppRoutes.studyTimer,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const TimerScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.studyLock,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const StudyLockSetupScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.catalog,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: ChangeNotifierProvider(
            create: (_) => CatalogController(),
            child: const CatalogScreen(),
          ),
        ),
      ),
      GoRoute(
        path: AppRoutes.quiz,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: QuizWrapperScreen(
            subject: state.uri.queryParameters['subject'] ?? '',
            chapter: state.uri.queryParameters['chapter'] ?? '',
            questions: const [],
          ),
        ),
      ),
      GoRoute(
        path: AppRoutes.activeRecall,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: ActiveRecallScreen(
            subject: state.uri.queryParameters['subject'],
          ),
        ),
      ),
      GoRoute(
        path: AppRoutes.flashCards,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: FlashCardsScreen(
            subject: state.uri.queryParameters['subject'] ?? '',
          ),
        ),
      ),

      GoRoute(
        path: AppRoutes.examPlanner,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const ExamPlannerScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.calendar,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const ExamCalendarScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.mediaViewer,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: MediaViewerScreen(
            url: state.uri.queryParameters['url'],
            title: state.uri.queryParameters['title'],
          ),
        ),
      ),
      GoRoute(
        path: AppRoutes.settingsAccessibility,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const AccessibilitySettingsScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.settingsAccount,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const EditAccountScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.settingsSync,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const SyncSettingsScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.settingsSubscription,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const SubscriptionScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.settingsSubscriptionManage,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const SubscriptionManagementScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.studyFeynman,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const FeynmanTechniqueScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.studyLeitner,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const LeitnerSystemScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.studyInterleaved,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const InterleavedPracticeScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.ai,
        pageBuilder: (context, state) {
          final extra = state.extra as Map<String, dynamic>? ?? {};
          MotivationStyle style = MotivationStyle.positiveReinforcement;
          final rawStyle = extra['motivationStyle'];
          if (rawStyle is String) {
            style = _parseMotivationStyle(rawStyle);
          } else if (rawStyle is MotivationStyle) {
            style = rawStyle;
          }
          return NoTransitionPage(
            key: state.pageKey,
            child: AxonAiScreen(
              title: extra['title'] as String? ?? 'Ask Axon',
              initialQuestion: extra['initialQuestion'] as String?,
              contextFuture: extra['contextFuture'] != null
                  ? (extra['contextFuture'] as Future<dynamic>)
                      .then((v) => v?.toString() ?? '')
                  : null,
              motivationStyle: style,
            ),
          );
        },
      ),
      GoRoute(
        path: AppRoutes.timer,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const TimerScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.admissions,
        pageBuilder: (context, state) {
          return NoTransitionPage(
            key: state.pageKey,
            child: const AdmissionsTrackerScreen(),
          );
        },
      ),
      GoRoute(
        path: AppRoutes.analytics,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const AnalysisScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.coachingPersona,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const CoachingPersonaScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.studyLockSettings,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const StudyLockSettingsScreen(),
        ),
      ),
      GoRoute(
        path: AppRoutes.achievements,
        pageBuilder: (context, state) => NoTransitionPage(
          key: state.pageKey,
          child: const AchievementsScreen(),
        ),
      ),
    ],
  );
});

class _GoRouterRefreshStream extends ChangeNotifier {
  _GoRouterRefreshStream(Stream<dynamic> stream) {
    notifyListeners();
    _subscription = stream.asBroadcastStream().listen(
          (dynamic _) => notifyListeners(),
        );
  }

  late final StreamSubscription<dynamic> _subscription;

  @override
  void dispose() {
    _subscription.cancel();
    super.dispose();
  }
}
