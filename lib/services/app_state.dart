// lib/services/app_state.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import '../models/models.dart';

import '../services/app_usage_service.dart';
import '../services/backend_health_service.dart';
import '../services/firestore_service.dart';
import '../services/prediction_service.dart';
import '../services/profile_service.dart';
import '../services/study_catalog.dart';
import '../services/notification_service.dart';
import '../services/axon_feedback_service.dart';
import '../services/coach_report_service.dart';
import '../services/gamification_service.dart';
import '../services/sync_service.dart';
import '../services/offline_sync_service.dart';
import '../services/smart_reminder_service.dart';
import '../services/grok_service.dart';
import '../services/axon_auto_crawl_service.dart';
import '../services/sync_manager.dart';
import '../services/notion_service.dart';
import '../services/google_calendar_service.dart';
import '../services/obsidian_service.dart';
import '../services/curriculum_catalog_service.dart';
import '../services/daily_plan_service.dart';
import '../models/daily_plan_task.dart';

class SubjectCache {
  static List<String>? _cachedSubjects;
  static Map<String, List<String>>? _cachedChapters;
  static DateTime? _lastLoadTime;
  static String? _cachedBoard;

  // LRU cache bounds - max board+subject combinations to cache
  static const int _maxChaptersEntries = 20;

  static Future<List<String>> getSubjects(String board) async {
    final now = DateTime.now();
    if (_cachedSubjects != null &&
        _cachedChapters != null &&
        _cachedBoard == board &&
        _lastLoadTime != null &&
        now.difference(_lastLoadTime!).inHours < 24) {
      return _cachedSubjects!;
    }
    final subjects =
        await CurriculumCatalogService.instance.subjectsForBoard(board);
    _cachedSubjects = subjects;
    _cachedBoard = board;
    _lastLoadTime = now;
    return subjects;
  }

  static Future<List<String>> getChapters(String board, String subject) async {
    final key = '${board}_$subject';
    if (_cachedChapters?.containsKey(key) ?? false) {
      // LRU: move to end by removing and re-adding
      final chapters = _cachedChapters![key]!;
      _cachedChapters!.remove(key);
      _cachedChapters![key] = chapters;
      return chapters;
    }
    final chapters = await CurriculumCatalogService.instance.chapterTitles(
      board: board,
      subject: subject,
    );
    _cachedChapters ??= {};

    // LRU eviction: remove oldest (first) if at capacity
    if (_cachedChapters!.length >= _maxChaptersEntries) {
      final oldestKey = _cachedChapters!.keys.first;
      _cachedChapters!.remove(oldestKey);
    }

    _cachedChapters![key] = chapters;
    return chapters;
  }

  static void clear() {
    _cachedSubjects = null;
    _cachedChapters = null;
    _lastLoadTime = null;
    _cachedBoard = null;
  }
}

final dailyPlanServiceProvider = Provider((ref) => DailyPlanService());

final todayPlanProvider = StreamProvider<List<DailyPlanTask>>((ref) {
  final service = ref.watch(dailyPlanServiceProvider);
  final auth = ref.watch(authStateProvider);
  if (!auth.isAuthenticated) return const Stream.empty();
  return service.watchTodayPlan(auth.user!.uid);
});

final generatePlanProvider =
    Provider<Future<void> Function(String? focusAreas)>((ref) {
  final service = ref.watch(dailyPlanServiceProvider);
  return (String? focusAreas) =>
      service.generateTodayPlan(focusAreas: focusAreas);
});

final ensureTodayPlanProvider =
    Provider<Future<List<DailyPlanTask>> Function(String? focusAreas)>((ref) {
  final service = ref.watch(dailyPlanServiceProvider);
  final auth = ref.watch(authStateProvider);
  if (!auth.isAuthenticated) return (_) async => [];
  return (String? focusAreas) =>
      service.ensureTodayPlan(auth.user!.uid, focusAreas: focusAreas);
});

//  Auth Provider
final authStateProvider = StateNotifierProvider<AuthNotifier, AuthState>(
  (ref) => AuthNotifier(),
);

final authDraftProvider =
    StateNotifierProvider<AuthDraftNotifier, AuthDraftState>(
  (ref) => AuthDraftNotifier(),
);

class AuthDraftState {
  final String name;
  final String email;
  final String password;

  const AuthDraftState({
    this.name = '',
    this.email = '',
    this.password = '',
  });

  AuthDraftState copyWith({
    String? name,
    String? email,
    String? password,
  }) {
    return AuthDraftState(
      name: name ?? this.name,
      email: email ?? this.email,
      password: password ?? this.password,
    );
  }
}

class AuthDraftNotifier extends StateNotifier<AuthDraftState> {
  AuthDraftNotifier() : super(const AuthDraftState());

  void setName(String value) {
    state = state.copyWith(name: value);
  }

  void setEmail(String value) {
    state = state.copyWith(email: value);
  }

  void setPassword(String value) {
    state = state.copyWith(password: value);
  }

  void clear() {
    state = const AuthDraftState();
  }
}

enum AuthGateStage { loading, signedOut, onboarding, ready }

class AuthGateState {
  final AuthGateStage stage;
  final User? firebaseUser;
  final UserProfile? profile;

  const AuthGateState({
    required this.stage,
    this.firebaseUser,
    this.profile,
  });
}

final profileServiceProvider =
    Provider<ProfileService>((ref) => ProfileService());

bool get _hasFirebaseApp {
  try {
    return Firebase.apps.isNotEmpty;
  } catch (_) {
    return false;
  }
}

FirebaseAuth? get _firebaseAuthOrNull {
  if (!_hasFirebaseApp) return null;
  try {
    return FirebaseAuth.instance;
  } catch (_) {
    return null;
  }
}

final authGateProvider = StreamProvider<AuthGateState>((ref) async* {
  final auth = _firebaseAuthOrNull;
  final profileService = ref.watch(profileServiceProvider);

  if (auth == null) {
    yield const AuthGateState(stage: AuthGateStage.signedOut);
    return;
  }

  // Load cached auth state with short timeout — don't hang
  try {
    final cached =
        await _loadCachedAuthGate().timeout(const Duration(seconds: 1));
    if (cached != null) {
      yield cached;
    }
  } catch (_) {
    yield const AuthGateState(stage: AuthGateStage.signedOut);
    return; // Exit early if cached load fails
  }

  // Fallback: if Firebase doesn't respond in 5 seconds, assume signed out
  final fallbackTimer = Timer(const Duration(seconds: 5), () {
    // We can't easily yield from a Timer callback in a StreamProvider,
    // but we can use a separate stream or just let the stream handle it.
  });

  try {
    // Use a timeout for the first emission of authStateChanges
    final authStream = auth.authStateChanges();

    // Create a stream that emits the first value or timeouts
    await authStream.first.timeout(
      const Duration(seconds: 5),
      onTimeout: () => null,
    );

    // Now listen to the rest of the stream
    await for (final user in authStream) {
      fallbackTimer.cancel();

      if (user == null) {
        const signedOut = AuthGateState(stage: AuthGateStage.signedOut);
        await _cacheAuthGate(signedOut);
        yield signedOut;
        continue;
      }
      final loading =
          AuthGateState(stage: AuthGateStage.loading, firebaseUser: user);
      await _cacheAuthGate(loading);
      yield loading;
      final prefs = await SharedPreferences.getInstance();
      final fallbackProfile = _fallbackUserProfileFromPrefs(prefs, user);
      try {
        final initialProfile = await profileService
            .getProfile(user)
            .timeout(const Duration(seconds: 2));
        final initialState =
            _resolveAuthGateState(user, initialProfile ?? fallbackProfile);
        await _cacheAuthGate(initialState);
        yield initialState;
      } catch (_) {
        final fallbackState = _resolveAuthGateState(user, fallbackProfile);
        await _cacheAuthGate(fallbackState);
        yield fallbackState;
      }

      try {
        await for (final profile in profileService.streamProfile(user)) {
          final resolved = _resolveAuthGateState(
            user,
            profile ?? fallbackProfile,
          );
          await _cacheAuthGate(resolved);
          yield resolved;
        }
      } catch (_) {
        final fallbackState = _resolveAuthGateState(user, fallbackProfile);
        await _cacheAuthGate(fallbackState);
        yield fallbackState;
      }
    }
  } catch (e) {
    fallbackTimer.cancel();
    yield const AuthGateState(stage: AuthGateStage.signedOut);
  }
});

UserProfile _fallbackUserProfileFromPrefs(
  SharedPreferences prefs,
  User user,
) {
  final cachedSubjects =
      prefs.getStringList('userSubjects') ?? const <String>[];
  final cachedBoard = prefs.getString('userBoard') ?? '';
  final cachedTargetHours = prefs.getDouble('userTargetHours') ??
      prefs.getInt('userTargetHours')?.toDouble() ??
      4;
  final cachedOnboarding = (prefs.getBool('userOnboardingComplete') ?? false) ||
      (cachedBoard.isNotEmpty && cachedSubjects.isNotEmpty);
  final cachedStyleIndex = prefs.getInt('motivationStyle');
  final cachedStyle = cachedStyleIndex == null
      ? MotivationStyle.positiveReinforcement
      : MotivationStyle
          .values[cachedStyleIndex.clamp(0, MotivationStyle.values.length - 1)];

  return UserProfile(
    uid: user.uid,
    displayName:
        user.displayName ?? prefs.getString('userName') ?? user.email ?? 'User',
    email: user.email ?? prefs.getString('userEmail') ?? '',
    photoUrl: prefs.getString('userPhoto') ?? user.photoURL,
    board: cachedBoard,
    subjects: cachedSubjects,
    targetStudyHours: cachedTargetHours,
    onboardingComplete: cachedOnboarding,
    motivationStyle: cachedStyle,
    createdAt: DateTime.now(),
  );
}

AuthGateState _resolveAuthGateState(
  User user,
  UserProfile? profile,
) {
  if (profile == null || !profile.onboardingComplete) {
    return AuthGateState(
      stage: AuthGateStage.onboarding,
      firebaseUser: user,
      profile: profile,
    );
  }
  return AuthGateState(
    stage: AuthGateStage.ready,
    firebaseUser: user,
    profile: profile,
  );
}

Future<AuthGateState?> _loadCachedAuthGate() async {
  SharedPreferences prefs;
  try {
    prefs = await SharedPreferences.getInstance();
  } catch (_) {
    return null;
  }
  final stageRaw = prefs.getString('authGateStage');
  if (stageRaw == null) return null;
  final stage = AuthGateStage.values.firstWhere(
    (s) => s.name == stageRaw,
    orElse: () => AuthGateStage.loading,
  );
  final uid = prefs.getString('userUid') ?? '';
  final name = prefs.getString('userName') ?? '';
  final email = prefs.getString('userEmail') ?? '';
  final photoUrl = prefs.getString('userPhoto');
  final board = prefs.getString('userBoard') ?? '';
  final subjects = prefs.getStringList('userSubjects') ?? const <String>[];
  final targetStudyHours = prefs.getDouble('userTargetHours') ??
      prefs.getInt('userTargetHours')?.toDouble() ??
      4;
  final onboardingComplete =
      (prefs.getBool('userOnboardingComplete') ?? false) ||
          (board.isNotEmpty && subjects.isNotEmpty);
  final profile = uid.isEmpty
      ? null
      : UserProfile(
          uid: uid,
          displayName: name.isEmpty ? email : name,
          email: email,
          photoUrl: photoUrl,
          board: board,
          subjects: subjects,
          targetStudyHours: targetStudyHours,
          onboardingComplete: onboardingComplete,
          createdAt: DateTime.now(),
        );
  return AuthGateState(stage: stage, profile: profile);
}

Future<void> _cacheAuthGate(AuthGateState state) async {
  final prefs = await SharedPreferences.getInstance();
  await prefs.setString('authGateStage', state.stage.name);
  final profile = state.profile;
  if (profile != null) {
    await prefs.setString('userUid', profile.uid);
    await prefs.setString('userName', profile.displayName);
    await prefs.setString('userEmail', profile.email);
    if (profile.photoUrl != null) {
      await prefs.setString('userPhoto', profile.photoUrl!);
    }
    await prefs.setString('userBoard', profile.board);
    await prefs.setStringList('userSubjects', profile.subjects);
    await prefs.setDouble('userTargetHours', profile.targetStudyHours);
    await prefs.setBool('userOnboardingComplete', profile.onboardingComplete);
  }
}

class AuthState {
  final bool isAuthenticated;
  final UserProfile? user;
  final bool isLoading;
  final String? error;

  const AuthState({
    this.isAuthenticated = false,
    this.user,
    this.isLoading = false,
    this.error,
  });

  AuthState copyWith({
    bool? isAuthenticated,
    UserProfile? user,
    bool? isLoading,
    String? error,
  }) {
    return AuthState(
      isAuthenticated: isAuthenticated ?? this.isAuthenticated,
      user: user ?? this.user,
      isLoading: isLoading ?? this.isLoading,
      error: error,
    );
  }
}

class AuthNotifier extends StateNotifier<AuthState> {
  static const String _googleServerClientId =
      '650295450237-h7mmvojo7g2jnsajnt9ukm3clb2a8g9m.apps.googleusercontent.com';
  final GoogleSignIn _googleSignIn = GoogleSignIn(
    clientId: _googleServerClientId,
    scopes: [
      'email',
      'profile',
      'https://www.googleapis.com/auth/calendar',
      'https://www.googleapis.com/auth/calendar.events',
    ],
  );
  final ProfileService _profileService = ProfileService();
  StreamSubscription<User?>? _authSub;
  StreamSubscription<UserProfile?>? _profileSub;
  String? _currentSyncUid;

  AuthNotifier() : super(const AuthState()) {
    _init();
  }

  FirebaseAuth? get _auth => _firebaseAuthOrNull;

  UserProfile _fallbackProfileFromPrefs(
    SharedPreferences prefs,
    User user,
  ) {
    final cachedSubjects =
        prefs.getStringList('userSubjects') ?? const <String>[];
    final cachedBoard = prefs.getString('userBoard') ?? '';
    final cachedTargetHours = prefs.getDouble('userTargetHours') ??
        prefs.getInt('userTargetHours')?.toDouble() ??
        4;
    final cachedOnboarding =
        (prefs.getBool('userOnboardingComplete') ?? false) ||
            (cachedBoard.isNotEmpty && cachedSubjects.isNotEmpty);
    final cachedStyleIndex = prefs.getInt('motivationStyle');
    final cachedStyle = cachedStyleIndex == null
        ? MotivationStyle.positiveReinforcement
        : MotivationStyle.values[
            cachedStyleIndex.clamp(0, MotivationStyle.values.length - 1)];

    return UserProfile(
      uid: user.uid,
      displayName: user.displayName ??
          prefs.getString('userName') ??
          user.email ??
          'User',
      email: user.email ?? prefs.getString('userEmail') ?? '',
      photoUrl: prefs.getString('userPhoto') ?? user.photoURL,
      board: cachedBoard,
      subjects: cachedSubjects,
      targetStudyHours: cachedTargetHours,
      onboardingComplete: cachedOnboarding,
      motivationStyle: cachedStyle,
      createdAt: DateTime.now(),
    );
  }

  Future<void> _syncUserDocument(
    User user,
    SharedPreferences prefs, {
    UserProfile? profile,
  }) async {
    final firestoreAvailable =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (!firestoreAvailable) return;
    final source = profile ?? _fallbackProfileFromPrefs(prefs, user);
    final payload = <String, dynamic>{
      'uid': user.uid,
      'display_name': source.displayName,
      'email': source.email.isNotEmpty ? source.email : (user.email ?? ''),
      'photo_url': source.photoUrl ?? user.photoURL,
      'motivation_style': source.motivationStyle.index,
      'board': source.board,
      'subjects': source.subjects,
      'target_hours': source.targetStudyHours,
      'onboarding_complete': source.onboardingComplete,
    };

    await AxonFirestore.instance
        .collection(AxonCollections.usersPrivate)
        .doc(user.uid)
        .set(payload, SetOptions(merge: true));
  }

  UserProfile _fastResolvedProfile(User user, UserProfile? profile) {
    return profile ??
        UserProfile(
          uid: user.uid,
          displayName: user.displayName ?? user.email ?? 'User',
          email: user.email ?? '',
          photoUrl: user.photoURL,
          board: '',
          subjects: const [],
          onboardingComplete: false,
          createdAt: DateTime.now(),
        );
  }

  Future<void> applyOnboardingProfile(UserProfile profile) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('userUid', profile.uid);
    await prefs.setString('userName', profile.displayName);
    await prefs.setString('userEmail', profile.email);
    await prefs.setString('userBoard', profile.board);
    await prefs.setStringList('userSubjects', profile.subjects);
    await prefs.setBool('userOnboardingComplete', profile.onboardingComplete);
    await prefs.setDouble('userTargetHours', profile.targetStudyHours);
    await prefs.setInt('motivationStyle', profile.motivationStyle.index);
    if ((profile.photoUrl ?? '').trim().isNotEmpty) {
      await prefs.setString('userPhoto', profile.photoUrl!);
    }

    final catalog = StudyCatalog();
    for (final subject in profile.subjects) {
      await catalog.addSubject(subject);
    }

    state = state.copyWith(
      isAuthenticated: true,
      isLoading: false,
      user: profile,
      error: null,
    );
  }

  Future<void> _persistSignedInUserInBackground(
    User user, {
    UserProfile? profile,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isLoggedIn', true);
    await prefs.setString('userEmail', user.email ?? '');
    await prefs.setString('userName', user.displayName ?? user.email ?? 'User');
    await prefs.setString('userUid', user.uid);
    await prefs.setString('userPhoto', user.photoURL ?? '');

    if (profile != null) {
      await prefs.setString('userBoard', profile.board);
      await prefs.setStringList('userSubjects', profile.subjects);
      await prefs.setBool('userOnboardingComplete', profile.onboardingComplete);
      await prefs.setDouble('userTargetHours', profile.targetStudyHours);
      await prefs.setInt('motivationStyle', profile.motivationStyle.index);

      final catalog = StudyCatalog();
      for (final subject in profile.subjects) {
        await catalog.addSubject(subject);
      }
    }

    if (profile == null) {
      try {
        await _syncUserDocument(user, prefs, profile: profile);
      } catch (_) {
        // Keep auth responsive even if cloud sync is slow or unavailable.
      }
    }

    if (!mounted) return;
    final resolved = profile ??
        _fallbackProfileFromPrefs(prefs, user).copyWith(
          displayName: user.displayName ?? user.email ?? 'User',
          email: user.email ?? '',
          photoUrl: prefs.getString('userPhoto') ?? user.photoURL,
        );
    state = state.copyWith(
      isAuthenticated: true,
      isLoading: false,
      user: resolved,
      error: null,
    );

    // Initialize sync services to determine connection statuses
    await NotionService().initialize();
    await GoogleCalendarService().initialize();
    await ObsidianService().initialize();

    // Start automatic sync manager if not already watching this user
    if (_currentSyncUid != user.uid) {
      _currentSyncUid = user.uid;
      unawaited(SyncManager().startWatching(user.uid));
    }
  }

  void _init() async {
    final auth = _auth;
    if (auth == null) {
      state = const AuthState(isAuthenticated: false);
      return;
    }
    _authSub?.cancel();
    _authSub = auth.authStateChanges().listen((user) async {
      await _profileSub?.cancel();
      if (user == null) {
        state = const AuthState(isAuthenticated: false);
        return;
      }
      _profileSub = _profileService.streamProfile(user).listen((profile) async {
        // Only update if profile actually changed (skip minor updates like photo URL changes)
        final currentUser = state.user;
        final newUser = _fastResolvedProfile(user, profile);

        // Check if meaningful data changed (not just listening to all changes)
        if (currentUser == null ||
            currentUser.board != newUser.board ||
            currentUser.subjects.join(',') != newUser.subjects.join(',') ||
            currentUser.displayName != newUser.displayName) {
          state = AuthState(
            isAuthenticated: true,
            isLoading: false,
            user: newUser,
          );

          // FIX: Sync to prefs so services like ExamScheduleGenerator can see the latest subjects
          final prefs = await SharedPreferences.getInstance();
          await prefs.setStringList('userSubjects', newUser.subjects);
          await prefs.setString('userBoard', newUser.board);
          await prefs.setBool(
              'userOnboardingComplete', newUser.onboardingComplete);
        }

        // Still persist in background but don't trigger full rebuild
        unawaited(_persistSignedInUserInBackground(user, profile: profile));
      });
    });
  }

  Future<void> _loadProfileFromFirestore(
      String uid, SharedPreferences prefs) async {
    final firestoreAvailable =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (!firestoreAvailable) return;
    try {
      final doc = await AxonPaths.privateUserDoc(uid).get();
      if (doc.exists && doc.data() != null) {
        final data = doc.data()!;
        await prefs.setBool(
            'userOnboardingComplete', data['onboarding_complete'] ?? false);
        await prefs.setString('userBoard', data['board'] ?? '');
        final rawTargetHours = data['target_hours'];
        final targetHours = rawTargetHours is num
            ? rawTargetHours.toDouble()
            : double.tryParse('${rawTargetHours ?? ''}') ?? 4;
        await prefs.setDouble('userTargetHours', targetHours);
        await prefs.setInt(
          'motivationStyle',
          (data['motivation_style'] ??
              MotivationStyle.positiveReinforcement.index) as int,
        );
        final subjects = StudyCatalog.extractSubjects(
            data['subjects'] ?? data['study_catalog']);
        if (subjects.isNotEmpty) {
          await prefs.setStringList('userSubjects', subjects);
        }
      }
    } catch (e) {
      // If firestore fails, keep default values
    }
  }

  void _loadProfileInBackground(String uid) async {
    final prefs = await SharedPreferences.getInstance();
    await _loadProfileFromFirestore(uid, prefs);
  }

  Future<bool> signInWithEmail(String email, String password) async {
    state = state.copyWith(isLoading: true, error: null);
    final auth = _auth;
    if (auth == null) {
      state = state.copyWith(
        isLoading: false,
        error: 'Authentication is unavailable in this Windows build.',
      );
      return false;
    }
    try {
      debugPrint('AuthNotifier: Starting email/password sign-in for: $email');
      final cred = await auth.signInWithEmailAndPassword(
        email: email.trim(),
        password: password,
      );

      debugPrint(
          'AuthNotifier: Email sign-in successful, user: ${cred.user?.uid}');

      final user = cred.user!;
      _loadProfileInBackground(user.uid);
      final prefs = await SharedPreferences.getInstance();
      final cachedBoard = prefs.getString('userBoard') ?? '';
      final cachedSubjects =
          prefs.getStringList('userSubjects') ?? const <String>[];
      final cachedOnboarding =
          (prefs.getBool('userOnboardingComplete') ?? false) ||
              (cachedBoard.isNotEmpty && cachedSubjects.isNotEmpty);
      state = AuthState(
        isAuthenticated: true,
        isLoading: false,
        user: UserProfile(
          uid: user.uid,
          displayName: user.displayName ?? email.split('@')[0],
          email: user.email ?? email,
          photoUrl: user.photoURL,
          board: cachedBoard,
          subjects: cachedSubjects,
          onboardingComplete: cachedOnboarding,
          createdAt: DateTime.now(),
        ),
      );
      unawaited(_persistSignedInUserInBackground(user));
      unawaited(Future.microtask(() async {
        _startResourceCrawling(user.uid);
        OfflineSyncService.instance.init(user.uid);
      }));
      return true;
    } on FirebaseAuthException catch (e) {
      debugPrint(
          'AuthNotifier: FirebaseAuthException: ${e.code} - ${e.message}');
      String errorMsg;
      switch (e.code) {
        case 'invalid-email':
        case 'INVALID_EMAIL':
          errorMsg = 'The email address is not valid.';
          break;
        case 'user-disabled':
        case 'USER_DISABLED':
          errorMsg = 'This user account has been disabled.';
          break;
        case 'user-not-found':
        case 'USER_NOT_FOUND':
          errorMsg = 'No account found with this email. Please register first.';
          break;
        case 'wrong-password':
        case 'WRONG_PASSWORD':
          errorMsg = 'Incorrect password. Please try again.';
          break;
        case 'invalid-credential':
        case 'INVALID_LOGIN_CREDENTIALS':
        case 'INVALID_CREDENTIAL':
          errorMsg =
              'Invalid email or password. If you do not have an account yet, register first.';
          break;
        case 'too-many-requests':
        case 'TOO_MANY_ATTEMPTS_TRY_LATER':
          errorMsg = 'Too many failed attempts. Please try again later.';
          break;
        case 'network-request-failed':
          errorMsg =
              'Network error. Check your internet connection and try again.';
          break;
        case 'RECAPTCHA_VERIFICATION_FAILED':
          errorMsg = 'Verification failed. Please try again.';
          break;
        default:
          errorMsg = (e.message == null || e.message!.trim().isEmpty)
              ? 'Sign-in failed: ${e.code}'
              : e.message!;
      }
      state = state.copyWith(isLoading: false, error: errorMsg);
      return false;
    } catch (e) {
      debugPrint('AuthNotifier: Email sign-in error: $e');
      state = state.copyWith(
        isLoading: false,
        error: e.toString().replaceFirst('Exception: ', ''),
      );
      return false;
    }
  }

  Future<bool> signInWithGoogle() async {
    state = state.copyWith(isLoading: true, error: null);
    final auth = _auth;
    if (auth == null) {
      state = state.copyWith(
        isLoading: false,
        error: 'Google sign-in is unavailable in this Windows build.',
      );
      return false;
    }
    try {
      debugPrint('AuthNotifier: Starting Google Sign-In...');
      final googleUser = await _googleSignIn.signIn();
      debugPrint('AuthNotifier: Google user result: ${googleUser?.id}');

      if (googleUser == null) {
        state = state.copyWith(
            isLoading: false, error: 'Google sign-in was cancelled');
        return false;
      }

      debugPrint('AuthNotifier: Getting Google auth...');
      final googleAuth = await googleUser.authentication;

      final credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );
      debugPrint('AuthNotifier: Signing in with Firebase...');

      final cred = await auth.signInWithCredential(credential);
      final user = cred.user!;
      debugPrint('AuthNotifier: Firebase user sign-in success');
      _loadProfileInBackground(user.uid);
      final prefs = await SharedPreferences.getInstance();
      final cachedBoard = prefs.getString('userBoard') ?? '';
      final cachedSubjects =
          prefs.getStringList('userSubjects') ?? const <String>[];
      final cachedOnboarding =
          (prefs.getBool('userOnboardingComplete') ?? false) ||
              (cachedBoard.isNotEmpty && cachedSubjects.isNotEmpty);
      state = AuthState(
        isAuthenticated: true,
        isLoading: false,
        user: UserProfile(
          uid: user.uid,
          displayName: user.displayName ?? user.email ?? 'User',
          email: user.email ?? '',
          photoUrl: user.photoURL,
          board: cachedBoard,
          subjects: cachedSubjects,
          onboardingComplete: cachedOnboarding,
          createdAt: DateTime.now(),
        ),
      );
      unawaited(_persistSignedInUserInBackground(user));
      // Background initialization - don't block sign-in response
      unawaited(Future.microtask(() async {
        _startResourceCrawling(user.uid);
        OfflineSyncService.instance.init(user.uid);
      }));
      return true;
    } on FirebaseAuthException catch (e) {
      debugPrint(
          'AuthNotifier: FirebaseAuthException: ${e.code} - ${e.message}');
      state = state.copyWith(isLoading: false, error: e.message);
      return false;
    } catch (e) {
      debugPrint('AuthNotifier: Google sign-in generic error: $e');
      String errorMsg;
      final raw = e.toString();
      if (raw.contains('MissingPluginException')) {
        errorMsg =
            'Google sign-in is not available on Windows in this build. Use email and password.';
      } else if (raw.contains('channel-error')) {
        errorMsg =
            'Google sign-in could not start on this device. Use email and password.';
      } else if (raw.contains('DEVELOPER_ERROR') ||
          raw.contains('Unknown calling package name')) {
        errorMsg = 'Google sign-in misconfigured. Please contact support.';
      } else if (raw.contains('network_error') ||
          raw.contains('SocketException') ||
          raw.contains('UnknownHostException')) {
        errorMsg = 'Network error. Check your internet connection.';
      } else if (raw.contains('sign_in_failed')) {
        errorMsg = 'Google Sign-In failed. Please try again.';
      } else if (raw.contains('CANCELED')) {
        errorMsg = 'Sign-in was cancelled.';
      } else {
        errorMsg = 'An unexpected error occurred. Please try again.';
      }
      state = state.copyWith(isLoading: false, error: errorMsg);
      return false;
    }
  }

  Future<bool> register(
    String name,
    String email,
    String password, {
    MotivationStyle style = MotivationStyle.positiveReinforcement,
  }) async {
    state = state.copyWith(isLoading: true);
    final auth = _auth;
    if (auth == null) {
      state = state.copyWith(
        isLoading: false,
        error: 'Registration is unavailable in this Windows build.',
      );
      return false;
    }
    try {
      final cred = await auth.createUserWithEmailAndPassword(
        email: email.trim(),
        password: password,
      );
      await cred.user?.updateDisplayName(name);
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool('isLoggedIn', true);
      await prefs.setString('userEmail', email);
      await prefs.setString('userName', name);
      await prefs.setString('userUid', cred.user?.uid ?? 'new_user');
      await prefs.setBool('userOnboardingComplete', false);
      await prefs.setInt('motivationStyle', style.index);
      final photoPath = prefs.getString('userPhoto');
      if (cred.user != null) {
        try {
          await _syncUserDocument(
            cred.user!,
            prefs,
            profile: UserProfile(
              uid: cred.user!.uid,
              displayName: name,
              email: email,
              photoUrl: photoPath,
              motivationStyle: style,
              board: '',
              subjects: const [],
              onboardingComplete: false,
              createdAt: DateTime.now(),
            ),
          );
        } catch (_) {}
      }
      state = AuthState(
        isAuthenticated: true,
        isLoading: false,
        user: UserProfile(
          uid: cred.user?.uid ?? 'new_user',
          displayName: name,
          email: email,
          photoUrl: photoPath,
          motivationStyle: style,
          board: '',
          subjects: const [],
          onboardingComplete: false,
          createdAt: DateTime.now(),
        ),
      );

      _startResourceCrawling(cred.user!.uid);
      unawaited(_persistSignedInUserInBackground(cred.user!));

      return true;
    } on FirebaseAuthException catch (e) {
      if (e.code == 'email-already-in-use') {
        final signedIn = await signInWithEmail(email.trim(), password);
        if (!signedIn) {
          state = state.copyWith(
            isLoading: false,
            error:
                'An account already exists for this email. Sign in with your password instead.',
          );
        }
        return signedIn;
      }
      state = state.copyWith(isLoading: false, error: e.message);
      return false;
    } catch (e) {
      state = state.copyWith(isLoading: false, error: e.toString());
      return false;
    }
  }

  Future<void> _startResourceCrawling(String uid) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final board = prefs.getString('userBoard');
      if (board == null || board.isEmpty) return;

      // Get user's selected subjects - handle both String and List formats
      final subjectsRaw = prefs.get('userSubjects');
      List<String> subjects = [];

      if (subjectsRaw is String) {
        if (subjectsRaw.isNotEmpty) {
          try {
            subjects = List<String>.from(jsonDecode(subjectsRaw));
          } catch (_) {
            subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology'];
          }
        }
      } else if (subjectsRaw is List) {
        subjects = List<String>.from(subjectsRaw);
      } else {
        subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology'];
      }

      if (subjects.isEmpty) {
        subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology'];
      }

      // Initialize and start auto-crawl service
      final autoCrawl = AxonAutoCrawlService();
      await autoCrawl.initialize(AxonAutoCrawlConfig(
        board: board,
        subjects: subjects,
        dailyDateSheetCheck: true,
        maxResourcesPerDay: 50,
        gemmaVerificationBatchSize: 5,
      ));

      // Register Gemma verifier callback
      autoCrawl.registerGemmaVerifier(_verifyWithGemma);

      // Start crawling - runs in background until app closes
      await autoCrawl.start();

      // Store reference for later access
      _autoCrawlService = autoCrawl;

      debugPrint(
          'Started auto-crawl for new user: $uid with board: $board, subjects: $subjects');
    } catch (e) {
      debugPrint('Error starting auto-crawl: $e');
    }
  }

  AxonAutoCrawlService? _autoCrawlService;

  // Gemma verification callback - called by auto-crawl to verify resources
  Future<Map<String, dynamic>?> _verifyWithGemma(
      String url, Map<String, dynamic>? context) async {
    // This would integrate with your existing Gemma/offline AI service
    // Return null if Gemma is busy or unavailable
    // The auto-crawl service will add resources without verification if this returns null

    try {
      try {
        final grok = GrokService();
        if (grok.isReady) {
          final result = await grok.chat(
            'Verify if this resource URL is relevant for studying: $url',
            systemPrompt: 'Reply only YES or NO.',
          );
          final isValid = result.trim().toUpperCase() == 'YES';
          if (isValid) {
            return {'verified': true, 'source': 'grok'};
          }
        }
      } catch (e) {
        debugPrint('AI resource verification failed: $e');
      }
      return null;
    } catch (_) {
      return null;
    }
  }

  Future<bool> resetPassword(String email) async {
    state = state.copyWith(isLoading: true, error: null);
    final auth = _auth;
    if (auth == null) {
      state = state.copyWith(
        isLoading: false,
        error: 'Password reset is unavailable in this Windows build.',
      );
      return false;
    }
    try {
      await auth.sendPasswordResetEmail(email: email.trim());
      state = state.copyWith(isLoading: false);
      return true;
    } on FirebaseAuthException catch (e) {
      state = state.copyWith(isLoading: false, error: e.message);
      return false;
    } catch (e) {
      state = state.copyWith(isLoading: false, error: e.toString());
      return false;
    }
  }

  Future<void> updateMotivationStyle(MotivationStyle style) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('motivationStyle', style.index);
    final current = _auth?.currentUser;
    if (current != null) {
      await _profileService.updateMotivationStyle(current.uid, style);
    }
    if (state.user != null) {
      state = state.copyWith(
        user: state.user!.copyWith(motivationStyle: style),
      );
    }
  }

  Future<void> updateStudySubjects(List<String> subjects) async {
    final normalized = StudyCatalog.normalizeSubjects(subjects);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('userSubjects', normalized);
    final current = _auth?.currentUser;
    await StudyCatalog().replaceSubjects(normalized);
    if (current != null) {
      await _profileService.updateSubjects(current.uid, normalized);
    }
    if (state.user != null) {
      state = state.copyWith(
        user: state.user!.copyWith(subjects: normalized),
      );
    }
  }

  Future<void> updateProfile({
    required String displayName,
    String? photoUrl,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('userName', displayName);
    final current = _auth?.currentUser;
    final currentEmail = current?.email ?? state.user?.email ?? '';

    String? photoUrlToSave = photoUrl;
    String? remotePhotoUrl;

    if (photoUrl != null && photoUrl.isNotEmpty) {
      if (photoUrl.startsWith('http://') || photoUrl.startsWith('https://')) {
        photoUrlToSave = photoUrl;
        remotePhotoUrl = photoUrl;
      } else {
        final file = File(photoUrl);
        if (await file.exists()) {
          photoUrlToSave = photoUrl;
          await prefs.setString('userPhoto', photoUrl);
          if (current != null) {
            remotePhotoUrl =
                await _profileService.uploadProfilePhoto(current.uid, file);
            if (remotePhotoUrl != null) {
              photoUrlToSave = remotePhotoUrl;
              await prefs.setString('userPhoto', remotePhotoUrl);
            }
          }
        } else {
          photoUrlToSave = photoUrl;
        }
      }
    }

    final firestoreAvailable =
        await BackendHealthService.instance.isFirestoreAvailable();
    if (current != null) {
      try {
        await current.updateDisplayName(displayName);
      } catch (_) {}
      if (remotePhotoUrl != null) {
        try {
          await current.updatePhotoURL(remotePhotoUrl);
        } catch (_) {}
      }
      if (firestoreAvailable) {
        try {
          await _profileService.updateProfile(current.uid, {
            'display_name': displayName,
            if (remotePhotoUrl != null) 'photo_url': remotePhotoUrl,
          });
        } catch (_) {}
      }
    }
    if (state.user != null) {
      state = state.copyWith(
        user: state.user!.copyWith(
          displayName: displayName,
          email: currentEmail,
          photoUrl: photoUrlToSave ?? state.user!.photoUrl,
        ),
      );
    }
  }

  Future<String?> updateProfilePhoto(String path) async {
    final prefs = await SharedPreferences.getInstance();
    final current = _auth?.currentUser;
    if (current == null) return null;

    try {
      final file = File(path);
      if (await file.exists()) {
        await prefs.setString('userPhoto', path);
        final remoteUrl =
            await _profileService.uploadProfilePhoto(current.uid, file);
        final nextPhoto = remoteUrl ?? path;
        await prefs.setString('userPhoto', nextPhoto);

        if (state.user != null) {
          state = state.copyWith(
            user: state.user!.copyWith(photoUrl: nextPhoto),
          );
        }

        if (remoteUrl != null) {
          try {
            await current.updatePhotoURL(remoteUrl);
          } catch (_) {}
        }

        return nextPhoto;
      }
    } catch (e) {
      await prefs.setString('userPhoto', path);
      if (state.user != null) {
        state = state.copyWith(
          user: state.user!.copyWith(photoUrl: path),
        );
      }
      return path;
    }

    return path;
  }

  Future<String?> pickAndUploadProfilePhoto(ImageSource source) async {
    final current = _auth?.currentUser;
    if (current == null) return null;

    try {
      final imageFile = await _profileService.selectAndCropProfilePhoto(source);
      if (imageFile == null) {
        debugPrint('No image selected or returned null');
        return null;
      }

      if (!imageFile.existsSync()) {
        debugPrint('Image file does not exist: ${imageFile.path}');
        return null;
      }

      final tempLocalPath = imageFile.path;
      unawaited(_uploadProfilePhotoInBackground(tempLocalPath));
      return tempLocalPath;
    } catch (e) {
      debugPrint('Error in pickAndUploadProfilePhoto: $e');
      return null;
    }
  }

  Future<void> _uploadProfilePhotoInBackground(String localPath) async {
    try {
      await Future.delayed(const Duration(milliseconds: 500));
      final uploadedUrl = await updateProfilePhoto(localPath);
      if (uploadedUrl != null) {
        debugPrint('Background pfp upload succeeded: $uploadedUrl');
      } else {
        debugPrint('Background pfp upload returned null');
      }
    } catch (e) {
      debugPrint('Background pfp upload failed: $e');
    }
  }

  Future<void> removeProfilePhoto() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('userPhoto');
    if (state.user != null) {
      state = state.copyWith(
        user: state.user!.copyWith(photoUrl: null),
      );
    }
  }

  Future<void> signOut() async {
    try {
      // Stop auto-crawl service
      _autoCrawlService?.stop();
      _autoCrawlService = null;

      // Disconnect sync services - wrap each in try-catch to avoid blocking
      try {
        await NotionService().disconnect();
      } catch (e) {
        debugPrint('SignOut: Notion disconnect failed: $e');
      }
      try {
        await GoogleCalendarService().signOut();
      } catch (e) {
        debugPrint('SignOut: Google Calendar signOut failed: $e');
      }
      try {
        await ObsidianService().disconnect();
      } catch (e) {
        debugPrint('SignOut: Obsidian disconnect failed: $e');
      }

      // Stop sync manager
      _currentSyncUid = null;
      try {
        await SyncManager().stopWatching();
      } catch (e) {
        debugPrint('SignOut: SyncManager stop failed: $e');
      }
      try {
        OfflineSyncService.instance.dispose();
      } catch (e) {
        debugPrint('SignOut: OfflineSyncService dispose failed: $e');
      }

      // Important: clear state before slow async tasks if possible,
      // but here we clear it after basic cleanup.
      state = const AuthState(isAuthenticated: false);

      final prefs = await SharedPreferences.getInstance();
      await prefs.clear();
      await _profileSub?.cancel();

      final auth = _auth;
      if (auth != null) {
        await auth.signOut();
      }
      await _googleSignIn.signOut();
    } catch (e) {
      debugPrint('SignOut: Overall sign out error: $e');
      // Force state change even if something fails
      state = const AuthState(isAuthenticated: false);
    }
  }

  Future<void> updateSubjectsOptimistically(List<String> newSubjects) async {
    final currentUser = state.user;
    if (currentUser == null) return;

    final normalizedSubjects = StudyCatalog.normalizeSubjects(newSubjects);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('userSubjects', normalizedSubjects);
    await StudyCatalog().replaceSubjects(normalizedSubjects);

    final updatedUser = currentUser.copyWith(subjects: normalizedSubjects);
    state = state.copyWith(user: updatedUser);

    unawaited(_syncSubjectsToFirestore(currentUser.uid, normalizedSubjects));
  }

  Future<void> _syncSubjectsToFirestore(
      String uid, List<String> subjects) async {
    try {
      final available =
          await BackendHealthService.instance.isFirestoreAvailable();
      if (!available) return;
      await _profileService.updateSubjects(uid, subjects);
    } catch (e) {
      debugPrint('Failed to sync subjects to Firestore: $e');
    }
  }

  Future<void> updateProfileOptimistically({
    String? displayName,
    String? board,
    double? targetStudyHours,
    MotivationStyle? motivationStyle,
  }) async {
    final currentUser = state.user;
    if (currentUser == null) return;

    final updatedUser = currentUser.copyWith(
      displayName: displayName,
      board: board,
      targetStudyHours: targetStudyHours,
      motivationStyle: motivationStyle,
    );
    state = state.copyWith(user: updatedUser);

    unawaited(_syncProfileToFirestore(
      uid: currentUser.uid,
      displayName: displayName,
      board: board,
      targetStudyHours: targetStudyHours,
      motivationStyle: motivationStyle,
    ));
  }

  Future<void> _syncProfileToFirestore({
    required String uid,
    String? displayName,
    String? board,
    double? targetStudyHours,
    MotivationStyle? motivationStyle,
  }) async {
    try {
      final available =
          await BackendHealthService.instance.isFirestoreAvailable();
      if (!available) return;
      final payload = <String, dynamic>{};
      if (displayName != null) payload['display_name'] = displayName;
      if (board != null) payload['board'] = board;
      if (targetStudyHours != null) payload['target_hours'] = targetStudyHours;
      if (motivationStyle != null) {
        payload['motivation_style'] = motivationStyle.index;
      }
      if (payload.isEmpty) return;
      await AxonPaths.privateUserDoc(uid).set(
        payload,
        SetOptions(merge: true),
      );
    } catch (e) {
      debugPrint('Failed to sync profile to Firestore: $e');
    }
  }

  @override
  void dispose() {
    _authSub?.cancel();
    _profileSub?.cancel();
    super.dispose();
  }
}

//  Metrics Provider
final metricsProvider = StateNotifierProvider<MetricsNotifier, MetricsState>(
  (ref) => MetricsNotifier(ref),
);

class MetricsState {
  final bool hasData;
  final double sleepHours;
  final double screenTimeHours;
  final double subjectDifficulty;
  final double mockScore;
  final double predictedPerformance;
  final double sevenDayAvgSleep;
  final double sevenDayAvgFocus;
  final List<DailyMetrics> weekHistory;
  final String delta;
  final String primarySubject;
  final double syllabusCoverage;
  final double activeStudyHours;
  final double targetStudyHours;
  final double focusRatio;
  final int consistencyStreak;
  final double stressLevel;
  final bool showStreakHighlight;

  int get streak => consistencyStreak;

  const MetricsState({
    this.hasData = true,
    this.sleepHours = 0.0,
    this.screenTimeHours = 0.0,
    this.subjectDifficulty = 0.0,
    this.mockScore = 0.0,
    this.predictedPerformance = 0.0,
    this.sevenDayAvgSleep = 0.0,
    this.sevenDayAvgFocus = 0.0,
    this.weekHistory = const [],
    this.delta = '',
    this.primarySubject = '',
    this.syllabusCoverage = 0.0,
    this.activeStudyHours = 0.0,
    this.targetStudyHours = 6.0,
    this.focusRatio = 0.0,
    this.consistencyStreak = 0,
    this.stressLevel = 0.0,
    this.showStreakHighlight = false,
  });

  MetricsState copyWith({
    bool? hasData,
    double? sleepHours,
    double? screenTimeHours,
    double? subjectDifficulty,
    double? mockScore,
    double? predictedPerformance,
    double? sevenDayAvgSleep,
    double? sevenDayAvgFocus,
    List<DailyMetrics>? weekHistory,
    String? delta,
    String? primarySubject,
    double? syllabusCoverage,
    double? activeStudyHours,
    double? targetStudyHours,
    double? focusRatio,
    int? consistencyStreak,
    double? stressLevel,
    bool? showStreakHighlight,
  }) {
    return MetricsState(
      hasData: hasData ?? this.hasData,
      sleepHours: sleepHours ?? this.sleepHours,
      screenTimeHours: screenTimeHours ?? this.screenTimeHours,
      subjectDifficulty: subjectDifficulty ?? this.subjectDifficulty,
      mockScore: mockScore ?? this.mockScore,
      predictedPerformance: predictedPerformance ?? this.predictedPerformance,
      sevenDayAvgSleep: sevenDayAvgSleep ?? this.sevenDayAvgSleep,
      sevenDayAvgFocus: sevenDayAvgFocus ?? this.sevenDayAvgFocus,
      weekHistory: weekHistory ?? this.weekHistory,
      delta: delta ?? this.delta,
      primarySubject: primarySubject ?? this.primarySubject,
      syllabusCoverage: syllabusCoverage ?? this.syllabusCoverage,
      activeStudyHours: activeStudyHours ?? this.activeStudyHours,
      targetStudyHours: targetStudyHours ?? this.targetStudyHours,
      focusRatio: focusRatio ?? this.focusRatio,
      consistencyStreak: consistencyStreak ?? this.consistencyStreak,
      stressLevel: stressLevel ?? this.stressLevel,
      showStreakHighlight: showStreakHighlight ?? this.showStreakHighlight,
    );
  }
}

class MetricsNotifier extends StateNotifier<MetricsState> {
  final Ref _ref;
  final AppUsageService _appUsageService = AppUsageService();
  final PredictionService _predictionService = PredictionService();
  Timer? _screenTimeTimer;

  MetricsNotifier(this._ref) : super(const MetricsState()) {
    _loadFromPrefs();
    _startScreenTimePolling();
  }

  void syncWithProfile(UserProfile profile) {
    if (profile.targetStudyHours > 0 &&
        state.targetStudyHours != profile.targetStudyHours) {
      updateTargetStudyHours(profile.targetStudyHours.toDouble());
    }
    if (profile.subjects.isNotEmpty &&
        (state.primarySubject.isEmpty ||
            !profile.subjects.contains(state.primarySubject))) {
      updateSubject(profile.subjects.first);
    }
    if (state.consistencyStreak != profile.currentStreak) {
      updateConsistencyStreak(profile.currentStreak);
    }
  }

  Future<void> refreshMetrics() async {
    await syncScreenTimeFromDevice();
    await _computeSevenDayAvgFocus();
  }

  Future<void> _computeSevenDayAvgFocus() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString('timer_history');
      if (raw == null || raw.isEmpty) return;

      final history = jsonDecode(raw) as List;
      final sevenDaysAgo = DateTime.now().subtract(const Duration(days: 7));
      double totalFocus = 0;
      int count = 0;

      for (final entry in history) {
        if (entry is! Map) continue;
        final dateStr = entry['date']?.toString();
        if (dateStr == null) continue;
        final date = DateTime.tryParse(dateStr);
        if (date == null || date.isBefore(sevenDaysAgo)) continue;

        final quality = (entry['focusQuality'] ?? entry['intensityIndex']);
        if (quality is num) {
          totalFocus += quality.toDouble();
          count++;
        }
      }

      if (count > 0) {
        final avg = totalFocus / count;
        state = state.copyWith(sevenDayAvgFocus: avg.clamp(0.0, 1.0));
        await _saveToPrefs(state);
      }
    } catch (e) {
      debugPrint('Failed to compute 7-day avg focus: $e');
    }
  }

  Future<void> syncScreenTimeFromDevice() async {
    final hasAccess = await _appUsageService.hasUsageAccess();
    if (!hasAccess) {
      debugPrint('Screen time: Usage access denied');
      return;
    }
    final hours = await _appUsageService.getDailyScreenTimeHours();
    if (hours < 0) {
      debugPrint('Screen time fetch failed: $hours');
      return;
    }
    debugPrint('Screen time fetched: ${hours.toStringAsFixed(1)}h');
    await updateScreenTime(hours);
  }

  void _startScreenTimePolling() {
    _screenTimeTimer?.cancel();
    _screenTimeTimer = Timer.periodic(
      const Duration(minutes: 15),
      (_) => syncScreenTimeFromDevice(),
    );
  }

  Future<void> _loadFromPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('metrics_state');
    if (raw == null) return;
    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      final history = (map['weekHistory'] as List?)
              ?.map((e) => DailyMetrics.fromJson(Map<String, dynamic>.from(e)))
              .toList() ??
          const [];
      state = MetricsState(
        hasData: map['hasData'] ?? true,
        sleepHours: (map['sleepHours'] ?? 0).toDouble(),
        screenTimeHours: (map['screenTimeHours'] ?? 0).toDouble(),
        subjectDifficulty: (map['subjectDifficulty'] ?? 0).toDouble(),
        mockScore: (map['mockScore'] ?? 0).toDouble(),
        predictedPerformance: (map['predictedPerformance'] ?? 0).toDouble(),
        sevenDayAvgSleep: (map['sevenDayAvgSleep'] ?? 0).toDouble(),
        sevenDayAvgFocus: (map['sevenDayAvgFocus'] ?? 0).toDouble(),
        weekHistory: history,
        delta: (map['delta'] ?? '').toString(),
        primarySubject: (map['primarySubject'] ?? '').toString(),
        syllabusCoverage: (map['syllabusCoverage'] ?? 0).toDouble(),
        activeStudyHours: (map['activeStudyHours'] ?? 0).toDouble(),
        targetStudyHours: (map['targetStudyHours'] ?? 6).toDouble(),
        focusRatio: (map['focusRatio'] ?? 0).toDouble(),
        consistencyStreak: (map['consistencyStreak'] ?? 0).toInt(),
        stressLevel: (map['stressLevel'] ?? 0).toDouble(),
      );
    } catch (_) {
      // Ignore corrupt state
    }
  }

  Future<void> _saveToPrefs(MetricsState s) async {
    final prefs = await SharedPreferences.getInstance();
    final map = {
      'hasData': s.hasData,
      'sleepHours': s.sleepHours,
      'screenTimeHours': s.screenTimeHours,
      'subjectDifficulty': s.subjectDifficulty,
      'mockScore': s.mockScore,
      'predictedPerformance': s.predictedPerformance,
      'sevenDayAvgSleep': s.sevenDayAvgSleep,
      'sevenDayAvgFocus': s.sevenDayAvgFocus,
      'weekHistory': s.weekHistory.map((e) => e.toJson()).toList(),
      'delta': s.delta,
      'primarySubject': s.primarySubject,
      'syllabusCoverage': s.syllabusCoverage,
      'activeStudyHours': s.activeStudyHours,
      'targetStudyHours': s.targetStudyHours,
      'focusRatio': s.focusRatio,
      'consistencyStreak': s.consistencyStreak,
      'stressLevel': s.stressLevel,
    };
    await prefs.setString('metrics_state', jsonEncode(map));
  }

  double _computeScore({
    required double mockScore,
    required double studyHours,
    required double targetHours,
    required double focusRatio,
    required double sleepHours,
    required double screenTimeHours,
    double? syllabusCoverage,
    int? consistencyStreak,
    double? stressLevel,
  }) {
    return _predictionService.predict(
      mockScore: mockScore,
      studyHours: studyHours,
      targetHours: targetHours,
      focusRatio: focusRatio,
      sleepHours: sleepHours,
      screenTimeHours: screenTimeHours,
      syllabusCoverage: syllabusCoverage ?? state.syllabusCoverage,
      consistencyStreak: consistencyStreak ?? state.consistencyStreak,
      stressLevel: stressLevel ?? state.stressLevel,
    );
  }

  String _buildDelta({
    required double nextScore,
    required double sleepHours,
    required double screenTimeHours,
  }) {
    return _predictionService.explain(
      currentScore: nextScore,
      previousScore: state.predictedPerformance,
      sleepHours: sleepHours,
      sevenDayAvgSleep: state.sevenDayAvgSleep,
      screenTimeHours: screenTimeHours,
    );
  }

  void updateSleep(double hours) {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: hours,
      screenTimeHours: state.screenTimeHours,
    );
    final next = state.copyWith(
        hasData: true,
        sleepHours: hours,
        sevenDayAvgSleep:
            state.sevenDayAvgSleep == 0.0 ? hours : state.sevenDayAvgSleep,
        predictedPerformance: score,
        delta: _buildDelta(
          nextScore: score,
          sleepHours: hours,
          screenTimeHours: state.screenTimeHours,
        ));
    state = next;
    _saveToPrefs(next);
  }

  Future<void> updateScreenTime(double hours) async {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: hours,
    );
    final next = state.copyWith(
        hasData: true,
        screenTimeHours: hours,
        predictedPerformance: score,
        delta: _buildDelta(
          nextScore: score,
          sleepHours: state.sleepHours,
          screenTimeHours: hours,
        ));
    state = next;
    await _saveToPrefs(next);

    // Cloud sync if backend healthy
    final healthy = await BackendHealthService.instance.isFirestoreAvailable();
    if (healthy && _ref.read(authStateProvider).isAuthenticated) {
      final todayMetrics = DailyMetrics(
        date: DateTime.now(),
        sleepHours: next.sleepHours,
        screenTimeHours: next.screenTimeHours,
        primarySubject: next.primarySubject,
        subjectDifficulty: next.subjectDifficulty,
        studyIntensity: next.focusRatio, // approximate
        mockScore: next.mockScore,
        predictedPerformance: next.predictedPerformance,
        syllabusCoverage: next.syllabusCoverage,
        activeStudyHours: next.activeStudyHours,
        targetStudyHours: next.targetStudyHours,
        focusRatio: next.focusRatio,
        consistencyStreak: next.consistencyStreak,
        stressLevel: next.stressLevel,
      );
      unawaited(SyncService().syncDailyMetrics(todayMetrics));
      debugPrint('Screen time synced to cloud: ${hours.toStringAsFixed(1)}h');
    }
  }

  void updateMockScore(double score) {
    updateMockScoreWithChapter(
        score: score, subject: state.primarySubject, chapter: '');
  }

  Future<void> updateMockScoreWithChapter({
    required double score,
    required String subject,
    String? chapter,
  }) async {
    final chapterStr = chapter ?? '';
    if (chapterStr.isNotEmpty) {
      await _appendMockScore(subject, chapterStr, score.round());
      await SmartReminderService().onMockScoreRecorded(
        subject: subject,
        chapter: chapterStr,
        score: score.round(),
      );
    }
    final predictedScore = _computeScore(
      mockScore: score,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
    );
    final next = state.copyWith(
      hasData: true,
      mockScore: score,
      predictedPerformance: predictedScore,
      delta: _buildDelta(
        nextScore: predictedScore,
        sleepHours: state.sleepHours,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    await _saveToPrefs(next);
  }

  Future<void> _appendMockScore(
      String subject, String chapter, int score) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('mock_scores_history');
    List<dynamic> scores = [];
    if (raw != null && raw.isNotEmpty) {
      try {
        scores = jsonDecode(raw) as List;
      } catch (_) {}
    }
    scores.insert(0, {
      'subject': subject,
      'chapter': chapter,
      'score': score,
      'date': DateTime.now().toIso8601String(),
    });
    if (scores.length > 100) scores.removeLast();
    await prefs.setString('mock_scores_history', jsonEncode(scores));
  }

  void updateSyllabusCoverage(double value) {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
      syllabusCoverage: value,
    );
    final next = state.copyWith(
      hasData: true,
      syllabusCoverage: value,
      predictedPerformance: score,
      delta: _buildDelta(
        nextScore: score,
        sleepHours: state.sleepHours,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    _saveToPrefs(next);
  }

  void updateSubjectDifficulty(double value) {
    final next = state.copyWith(hasData: true, subjectDifficulty: value);
    state = next;
    _saveToPrefs(next);
  }

  void updateStressLevel(double value) {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
      stressLevel: value,
    );
    final next = state.copyWith(
      hasData: true,
      stressLevel: value,
      predictedPerformance: score,
      delta: _buildDelta(
        nextScore: score,
        sleepHours: state.sleepHours,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    _saveToPrefs(next);
  }

  void updateTargetStudyHours(double value) {
    final next = state.copyWith(hasData: true, targetStudyHours: value);
    state = next;
    _saveToPrefs(next);
  }

  void updateConsistencyStreak(int value) {
    final score = _computeScore(
      mockScore: state.mockScore,
      studyHours: state.activeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: state.focusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
      consistencyStreak: value,
    );
    final next = state.copyWith(
      hasData: true,
      consistencyStreak: value,
      predictedPerformance: score,
      showStreakHighlight: value > state.consistencyStreak,
      delta: _buildDelta(
        nextScore: score,
        sleepHours: state.sleepHours,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    _saveToPrefs(next);
  }

  void setShowStreakHighlight(bool value) {
    state = state.copyWith(showStreakHighlight: value);
    _saveToPrefs(state);
  }

  void updateSubject(String subject) {
    final next = state.copyWith(hasData: true, primarySubject: subject);
    state = next;
    _saveToPrefs(next);
  }

  Future<void> addStudySession(
      double intensityIndex, double studyHours, double focusRatio) async {
    final now = DateTime.now();
    final streak = await GamificationService.instance
        .logStudySession((studyHours * 60).round());
    final existingToday = state.weekHistory.cast<DailyMetrics?>().firstWhere(
          (h) => h != null && _sameDay(h.date, now),
          orElse: () => null,
        );
    final previousDayHours = existingToday?.activeStudyHours ?? 0.0;
    final cumulativeStudyHours = previousDayHours + studyHours;
    final previousActiveHours = state.activeStudyHours;
    final previousFocusRatio = existingToday?.focusRatio ?? 0.0;
    final previousIntensity = existingToday?.studyIntensity ?? 0.0;
    final weightedFocusRatio = cumulativeStudyHours <= 0
        ? focusRatio
        : ((previousFocusRatio * previousDayHours) +
                (focusRatio * studyHours)) /
            cumulativeStudyHours;
    final weightedIntensity = cumulativeStudyHours <= 0
        ? intensityIndex
        : ((previousIntensity * previousDayHours) +
                (intensityIndex * studyHours)) /
            cumulativeStudyHours;
    final newScore = _computeScore(
      mockScore: state.mockScore,
      studyHours: cumulativeStudyHours,
      targetHours: state.targetStudyHours,
      focusRatio: weightedFocusRatio,
      sleepHours: state.sleepHours,
      screenTimeHours: state.screenTimeHours,
      consistencyStreak: streak.currentStreak,
    );
    final updated = _upsertHistory(
        state.weekHistory,
        DailyMetrics(
          date: now,
          sleepHours: state.sleepHours,
          screenTimeHours: state.screenTimeHours,
          primarySubject:
              state.primarySubject.isEmpty ? 'General' : state.primarySubject,
          subjectDifficulty: state.subjectDifficulty,
          studyIntensity: weightedIntensity,
          mockScore: state.mockScore,
          predictedPerformance: newScore,
          syllabusCoverage: state.syllabusCoverage,
          activeStudyHours: cumulativeStudyHours,
          targetStudyHours: state.targetStudyHours,
          focusRatio: weightedFocusRatio,
          consistencyStreak: streak.currentStreak,
          stressLevel: state.stressLevel,
        ));

    final avgSleep = updated.isEmpty
        ? 0.0
        : updated.map((h) => h.sleepHours).reduce((a, b) => a + b) /
            updated.length;
    final avgFocus = updated.isEmpty
        ? 0.0
        : updated.map((h) => h.studyIntensity).reduce((a, b) => a + b) /
            updated.length;

    final next = state.copyWith(
      hasData: true,
      weekHistory: updated,
      sevenDayAvgSleep: avgSleep,
      sevenDayAvgFocus: avgFocus,
      predictedPerformance: newScore,
      activeStudyHours: cumulativeStudyHours,
      focusRatio: weightedFocusRatio,
      consistencyStreak: streak.currentStreak,
      showStreakHighlight: streak.currentStreak > state.consistencyStreak,
      delta: _predictionService.explain(
        currentScore: newScore,
        previousScore: state.predictedPerformance,
        sleepHours: state.sleepHours,
        sevenDayAvgSleep: avgSleep,
        screenTimeHours: state.screenTimeHours,
      ),
    );
    state = next;
    await _saveToPrefs(next);
    await _maybeCelebrateDailyGoal(
      previousHours: previousActiveHours,
      currentHours: cumulativeStudyHours,
      targetHours: next.targetStudyHours,
      date: now,
    );
    await _maybeRewardMilestones(
      streak: streak,
      sessionMinutes: (studyHours * 60).round(),
      intensityIndex: intensityIndex,
      targetHours: next.targetStudyHours,
      currentHours: cumulativeStudyHours,
      focusRatio: weightedFocusRatio,
      screenTimeHours: next.screenTimeHours,
      date: now,
    );
    await CoachReportService.instance.maybeSendLossAversionReminder(
      metrics: next,
      style: _ref.read(authStateProvider).user?.motivationStyle ??
          MotivationStyle.positiveReinforcement,
      currentStreak: streak.currentStreak,
      board: _ref.read(authStateProvider).user?.board ?? '',
      subject: next.primarySubject,
    );
  }

  Future<void> _maybeCelebrateDailyGoal({
    required double previousHours,
    required double currentHours,
    required double targetHours,
    required DateTime date,
  }) async {
    if (targetHours <= 0) return;
    if (previousHours >= targetHours || currentHours < targetHours) return;

    final prefs = await SharedPreferences.getInstance();
    final todayKey = '${date.year}-${date.month}-${date.day}';
    if (prefs.getString('goal_unlock_chime_day') == todayKey) return;

    await prefs.setString('goal_unlock_chime_day', todayKey);
    await AxonFeedbackService.playGoalUnlockChime();
  }

  Future<void> _maybeRewardMilestones({
    required StudyStreak streak,
    required int sessionMinutes,
    required double intensityIndex,
    required double targetHours,
    required double currentHours,
    required double focusRatio,
    required double screenTimeHours,
    required DateTime date,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final xp = GamificationService.instance.calculateSessionXp(
      sessionMinutes,
      intensityIndex,
    );
    await GamificationService.instance.addXp(xp, reason: 'study_session');
    final unlocked = await GamificationService.instance.checkAndUnlockBadges();
    var milestoneTriggered = unlocked.isNotEmpty;

    final dayKey = '${date.year}-${date.month}-${date.day}';
    final perfectDay = targetHours > 0 &&
        currentHours >= targetHours &&
        focusRatio >= 0.78 &&
        screenTimeHours <= 5.5;
    if (perfectDay && prefs.getString('perfect_day_badge_day') != dayKey) {
      await prefs.setString('perfect_day_badge_day', dayKey);
      await GamificationService.instance.unlockBadge('perfect_day');
      milestoneTriggered = true;
    }

    if ((streak.currentStreak == 7 ||
            streak.currentStreak == 30 ||
            streak.currentStreak == 90) &&
        prefs.getString('streak_milestone_day') != dayKey) {
      await prefs.setString('streak_milestone_day', dayKey);
      milestoneTriggered = true;
    }

    if (milestoneTriggered) {
      await AxonFeedbackService.playMilestoneChime();
    }
  }

  List<DailyMetrics> _upsertHistory(
      List<DailyMetrics> history, DailyMetrics entry) {
    final next = List<DailyMetrics>.from(history);
    final idx = next.indexWhere((h) => _sameDay(h.date, entry.date));
    if (idx >= 0) {
      next[idx] = entry;
    } else {
      next.add(entry);
    }
    next.sort((a, b) => a.date.compareTo(b.date));
    if (next.length > 7) {
      return next.sublist(next.length - 7);
    }
    return next;
  }

  bool _sameDay(DateTime a, DateTime b) =>
      a.year == b.year && a.month == b.month && a.day == b.day;

  @override
  void dispose() {
    _screenTimeTimer?.cancel();
    super.dispose();
  }
}

enum StudySessionMode {
  pomodoro,
  examSimulation,
  recallSprint,
  timedPaper,
}

extension StudySessionModeX on StudySessionMode {
  String get id => name;

  String get label {
    switch (this) {
      case StudySessionMode.pomodoro:
        return 'Pomodoro';
      case StudySessionMode.examSimulation:
        return 'Exam Simulation';
      case StudySessionMode.recallSprint:
        return 'Recall Sprint';
      case StudySessionMode.timedPaper:
        return 'Timed Paper';
    }
  }

  String get description {
    switch (this) {
      case StudySessionMode.pomodoro:
        return 'Structured work with frequent resets.';
      case StudySessionMode.examSimulation:
        return 'Long unbroken focus with low interruption tolerance.';
      case StudySessionMode.recallSprint:
        return 'Short, intense memory retrieval bursts.';
      case StudySessionMode.timedPaper:
        return 'Exam-paper pacing with stricter timing pressure.';
    }
  }

  int get fatigueThresholdMinutes {
    switch (this) {
      case StudySessionMode.pomodoro:
        return 25;
      case StudySessionMode.examSimulation:
        return 55;
      case StudySessionMode.recallSprint:
        return 18;
      case StudySessionMode.timedPaper:
        return 45;
    }
  }

  static StudySessionMode fromId(String raw) {
    return StudySessionMode.values.firstWhere(
      (mode) => mode.id == raw,
      orElse: () => StudySessionMode.pomodoro,
    );
  }
}

enum SessionReflection {
  easy,
  draining,
  confusing,
  productive,
}

extension SessionReflectionX on SessionReflection {
  String get label {
    switch (this) {
      case SessionReflection.easy:
        return 'Easy';
      case SessionReflection.draining:
        return 'Draining';
      case SessionReflection.confusing:
        return 'Confusing';
      case SessionReflection.productive:
        return 'Productive';
    }
  }
}

class SessionSummary {
  final Map<String, dynamic> record;
  final double focusQuality;
  final double fatigueIndex;
  final int recommendedBreakMinutes;

  const SessionSummary({
    required this.record,
    required this.focusQuality,
    required this.fatigueIndex,
    required this.recommendedBreakMinutes,
  });
}

class TimerState {
  final bool isRunning;
  final Duration elapsed;
  final int breakCount;
  final List<Duration> focusSegments;
  final String subject;
  final String chapter;
  final double intensityIndex;
  final int pings;
  final DateTime? startedAt;
  final bool isRestoring;
  final String templateId;
  final String templateName;
  final Duration? targetDuration;
  final StudySessionMode mode;
  final bool antiDistractionLock;
  final int interruptionCount;
  final int distractionNudges;
  final double fatigueIndex;
  final double focusQuality;
  final int recommendedBreakMinutes;
  final bool pomodoroEnabled;
  final String? lastOpenedResource;
  final String? lastOpenedResourceType;
  final String activityType;

  const TimerState({
    this.isRunning = false,
    this.elapsed = Duration.zero,
    this.breakCount = 0,
    this.focusSegments = const [],
    this.subject = '',
    this.chapter = '',
    this.intensityIndex = 0.0,
    this.pings = 0,
    this.startedAt,
    this.isRestoring = false,
    this.templateId = '',
    this.templateName = '',
    this.targetDuration,
    this.mode = StudySessionMode.pomodoro,
    this.antiDistractionLock = false,
    this.interruptionCount = 0,
    this.distractionNudges = 0,
    this.fatigueIndex = 0.0,
    this.focusQuality = 1.0,
    this.recommendedBreakMinutes = 0,
    this.pomodoroEnabled = true,
    this.lastOpenedResource,
    this.lastOpenedResourceType,
    this.activityType = 'timer',
  });

  Duration get remaining {
    final target = targetDuration;
    if (target == null) return Duration.zero;
    final delta = target - elapsed;
    return delta.isNegative ? Duration.zero : delta;
  }

  bool get hasTargetDuration =>
      targetDuration != null && targetDuration! > Duration.zero;

  TimerState copyWith({
    bool? isRunning,
    Duration? elapsed,
    int? breakCount,
    List<Duration>? focusSegments,
    String? subject,
    String? chapter,
    double? intensityIndex,
    int? pings,
    DateTime? startedAt,
    bool clearStartedAt = false,
    bool? isRestoring,
    String? templateId,
    String? templateName,
    Duration? targetDuration,
    bool clearTargetDuration = false,
    StudySessionMode? mode,
    bool? antiDistractionLock,
    int? interruptionCount,
    int? distractionNudges,
    double? fatigueIndex,
    double? focusQuality,
    int? recommendedBreakMinutes,
    bool? pomodoroEnabled,
    String? lastOpenedResource,
    String? lastOpenedResourceType,
    String? activityType,
  }) {
    return TimerState(
      isRunning: isRunning ?? this.isRunning,
      elapsed: elapsed ?? this.elapsed,
      breakCount: breakCount ?? this.breakCount,
      focusSegments: focusSegments ?? this.focusSegments,
      subject: subject ?? this.subject,
      chapter: chapter ?? this.chapter,
      intensityIndex: intensityIndex ?? this.intensityIndex,
      pings: pings ?? this.pings,
      startedAt: clearStartedAt ? null : (startedAt ?? this.startedAt),
      isRestoring: isRestoring ?? this.isRestoring,
      templateId: templateId ?? this.templateId,
      templateName: templateName ?? this.templateName,
      targetDuration:
          clearTargetDuration ? null : (targetDuration ?? this.targetDuration),
      mode: mode ?? this.mode,
      antiDistractionLock: antiDistractionLock ?? this.antiDistractionLock,
      interruptionCount: interruptionCount ?? this.interruptionCount,
      distractionNudges: distractionNudges ?? this.distractionNudges,
      fatigueIndex: fatigueIndex ?? this.fatigueIndex,
      focusQuality: focusQuality ?? this.focusQuality,
      recommendedBreakMinutes:
          recommendedBreakMinutes ?? this.recommendedBreakMinutes,
      pomodoroEnabled: pomodoroEnabled ?? this.pomodoroEnabled,
      lastOpenedResource: lastOpenedResource ?? this.lastOpenedResource,
      lastOpenedResourceType:
          lastOpenedResourceType ?? this.lastOpenedResourceType,
      activityType: activityType ?? this.activityType,
    );
  }
}

class TimerNotifier extends StateNotifier<TimerState> {
  final Ref _ref;
  Timer? _ticker;
  Timer? _cloudSyncTimer;
  static const _prefsKey = 'active_timer_state';
  final SyncService _syncService = SyncService();

  TimerNotifier(this._ref) : super(const TimerState(isRestoring: true)) {
    _restore();
  }

  Future<void> _restore() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_prefsKey);

    final recoverableSession = await _syncService.getRecoverableSession();

    if (recoverableSession != null && raw != null && raw.isNotEmpty) {
      try {
        final map = Map<String, dynamic>.from(jsonDecode(raw));
        final startedAtRaw = map['startedAt']?.toString();
        final startedAt = startedAtRaw == null || startedAtRaw.isEmpty
            ? null
            : DateTime.tryParse(startedAtRaw);

        state = TimerState(
          isRunning: map['isRunning'] == true,
          elapsed:
              Duration(seconds: (map['elapsedSeconds'] as num?)?.toInt() ?? 0),
          breakCount: (map['breakCount'] as num?)?.toInt() ?? 0,
          subject: (map['subject'] ?? recoverableSession.subject).toString(),
          chapter: (map['chapter'] ?? recoverableSession.chapter).toString(),
          intensityIndex: (map['intensityIndex'] as num?)?.toDouble() ?? 0.0,
          pings: (map['pings'] as num?)?.toInt() ?? 0,
          startedAt: startedAt,
          isRestoring: false,
          templateId: (map['templateId'] ?? '').toString(),
          templateName: (map['templateName'] ?? '').toString(),
          targetDuration: (map['targetDurationSeconds'] as num?) == null
              ? null
              : Duration(
                  seconds: (map['targetDurationSeconds'] as num).toInt()),
          mode: StudySessionModeX.fromId((map['modeId'] ?? '').toString()),
          antiDistractionLock: map['antiDistractionLock'] == true,
          interruptionCount: (map['interruptionCount'] as num?)?.toInt() ?? 0,
          distractionNudges: (map['distractionNudges'] as num?)?.toInt() ?? 0,
          fatigueIndex: (map['fatigueIndex'] as num?)?.toDouble() ?? 0.0,
          focusQuality: (map['focusQuality'] as num?)?.toDouble() ?? 1.0,
          recommendedBreakMinutes:
              (map['recommendedBreakMinutes'] as num?)?.toInt() ?? 0,
          pomodoroEnabled: map['pomodoroEnabled'] != false,
        );
        if (state.isRunning && state.startedAt != null) {
          state = state.copyWith(isRunning: false);
          await prefs.remove(_prefsKey);
        }
      } catch (_) {
        state = const TimerState();
        await prefs.remove(_prefsKey);
      }
    } else {
      state = const TimerState();
    }
  }

  void start(
    String subject,
    String chapter, {
    String templateId = '',
    String templateName = '',
    Duration? targetDuration,
    StudySessionMode mode = StudySessionMode.pomodoro,
    bool antiDistractionLock = false,
    bool pomodoroEnabled = true,
  }) {
    final startedAt = DateTime.now().subtract(state.elapsed);
    state = state.copyWith(
      isRunning: true,
      subject: subject,
      chapter: chapter,
      startedAt: startedAt,
      isRestoring: false,
      templateId: templateId,
      templateName: templateName,
      targetDuration: targetDuration,
      mode: mode,
      antiDistractionLock: antiDistractionLock,
      pomodoroEnabled: pomodoroEnabled,
      lastOpenedResource: null,
      lastOpenedResourceType: null,
    );
    _saveState();
    _syncActiveSession();
    AxonFeedbackService.startActiveLoop();
    _ticker?.cancel();
    _ticker = Timer.periodic(const Duration(seconds: 1), (t) => tick());
  }

  void trackOpenedResource(String resourceUrl, String resourceType) {
    state = state.copyWith(
      lastOpenedResource: resourceUrl,
      lastOpenedResourceType: resourceType,
    );
    _saveState();
  }

  void pause({bool countAsBreak = true}) {
    _syncElapsedFromClock();
    final signals = _computeSessionSignals(
      elapsed: state.elapsed,
      breakCount: state.breakCount + (countAsBreak ? 1 : 0),
      interruptionCount: state.interruptionCount,
      pings: state.pings,
      mode: state.mode,
    );
    state = state.copyWith(
      isRunning: false,
      breakCount: state.breakCount + (countAsBreak ? 1 : 0),
      clearStartedAt: true,
      intensityIndex: signals.intensity,
      focusQuality: signals.focusQuality,
      fatigueIndex: signals.fatigueIndex,
      recommendedBreakMinutes: signals.recommendedBreakMinutes,
    );
    _ticker?.cancel();
    AxonFeedbackService.stopActiveLoop();
    _saveState();
  }

  void tick() {
    if (state.isRunning) {
      final nextElapsed = _elapsedFromClock();
      var nextPings = state.pings;
      if (nextElapsed.inSeconds > 0 &&
          nextElapsed.inSeconds % 300 == 0 &&
          nextElapsed.inSeconds != state.elapsed.inSeconds) {
        nextPings++;
      }
      final signals = _computeSessionSignals(
        elapsed: nextElapsed,
        breakCount: state.breakCount,
        interruptionCount: state.interruptionCount,
        pings: nextPings,
        mode: state.mode,
      );
      state = state.copyWith(
        elapsed: nextElapsed,
        pings: nextPings,
        intensityIndex: signals.intensity,
        focusQuality: signals.focusQuality,
        fatigueIndex: signals.fatigueIndex,
        recommendedBreakMinutes: signals.recommendedBreakMinutes,
      );
      _saveState();
      if (state.hasTargetDuration && nextElapsed >= state.targetDuration!) {
        unawaited(stop());
      }
    }
  }

  void markInterruption() {
    if (!state.isRunning) return;
    final nextInterruptions = state.interruptionCount + 1;
    final nextNudges = state.antiDistractionLock
        ? state.distractionNudges + 1
        : state.distractionNudges;
    final signals = _computeSessionSignals(
      elapsed: _elapsedFromClock(),
      breakCount: state.breakCount,
      interruptionCount: nextInterruptions,
      pings: state.pings,
      mode: state.mode,
    );
    state = state.copyWith(
      elapsed: _elapsedFromClock(),
      interruptionCount: nextInterruptions,
      distractionNudges: nextNudges,
      intensityIndex: signals.intensity,
      focusQuality: signals.focusQuality,
      fatigueIndex: signals.fatigueIndex,
      recommendedBreakMinutes: signals.recommendedBreakMinutes,
    );
    _saveState();
  }

  Future<SessionSummary> stop() async {
    _syncElapsedFromClock();
    _ticker?.cancel();
    _cloudSyncTimer?.cancel();
    await AxonFeedbackService.stopActiveLoop();
    final signals = _computeSessionSignals(
      elapsed: state.elapsed,
      breakCount: state.breakCount,
      interruptionCount: state.interruptionCount,
      pings: state.pings,
      mode: state.mode,
    );
    final intensity = signals.intensity;
    final studyHours = state.elapsed.inSeconds / 3600.0;

    await _ref.read(metricsProvider.notifier).addStudySession(
          intensity,
          studyHours,
          signals.focusQuality,
        );

    final record = {
      'subject': state.subject,
      'chapter': state.chapter,
      'duration': state.elapsed.inSeconds,
      'date': DateTime.now().toIso8601String(),
      'intensity': intensity,
      'focusQuality': signals.focusQuality,
      'fatigueIndex': signals.fatigueIndex,
      'breaks': state.breakCount,
      'interruptions': state.interruptionCount,
      'distractionNudges': state.distractionNudges,
      'recommendedBreakMinutes': signals.recommendedBreakMinutes,
      'templateId': state.templateId,
      'templateName': state.templateName,
      'targetDurationSeconds': state.targetDuration?.inSeconds,
      'modeId': state.mode.id,
      'modeName': state.mode.label,
      'antiDistractionLock': state.antiDistractionLock,
      'reflection': '',
      'activityType': state.activityType,
    };
    await _syncSessionToCloud(record);

    // Write-through cache: persist cloud-facing session first, then update
    // local history for offline reads and fast UI restores.
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString('timer_history');
    List<dynamic> history = [];
    if (data != null) {
      try {
        history = jsonDecode(data);
      } catch (_) {}
    }
    history.insert(0, record);
    if (history.length > 50) history.removeLast();
    await prefs.setString('timer_history', jsonEncode(history));
    await prefs.remove(_prefsKey);

    await _syncService.clearActiveSession();

    await SmartReminderService().onSessionComplete(
      subject: state.subject,
      chapter: state.chapter,
      durationMinutes: state.elapsed.inMinutes,
      focusQuality: signals.focusQuality,
    );

    await _maybeNotifyIntensityDrop();

    state = const TimerState();
    return SessionSummary(
      record: record,
      focusQuality: signals.focusQuality,
      fatigueIndex: signals.fatigueIndex,
      recommendedBreakMinutes: signals.recommendedBreakMinutes,
    );
  }

  void reset() {
    _ticker?.cancel();
    _cloudSyncTimer?.cancel();
    AxonFeedbackService.stopActiveLoop();
    _clearSavedState();
    _syncService.clearActiveSession();
    state = const TimerState();
  }

  void updateSubject(String subject) {
    state = state.copyWith(subject: subject);
  }

  Future<void> _syncActiveSession() async {
    if (state.subject.isEmpty || state.startedAt == null) return;

    final sessionPings = <SessionPing>[];
    for (int i = 0; i < state.pings; i++) {
      sessionPings.add(SessionPing(
        timestamp: state.startedAt!.add(Duration(minutes: 5 * (i + 1))),
        isActive: true,
      ));
    }

    final session = ActiveSession(
      sessionId: 'session_${DateTime.now().millisecondsSinceEpoch}',
      subject: state.subject,
      chapter: state.chapter,
      startedAt: state.startedAt!,
      elapsed: state.elapsed,
      breakCount: state.breakCount,
      pings: sessionPings,
      deviceId: '',
      isSyncing: false,
      metadata: {
        'modeId': state.mode.id,
        'templateId': state.templateId,
        'antiDistractionLock': state.antiDistractionLock,
      },
    );

    await _syncService.saveActiveSession(session);

    _cloudSyncTimer?.cancel();
    _cloudSyncTimer = Timer.periodic(const Duration(seconds: 30), (_) async {
      if (state.isRunning) {
        final updatedSession = session.copyWith(
          elapsed: state.elapsed,
          breakCount: state.breakCount,
        );
        await _syncService.saveActiveSession(updatedSession);
      }
    });
  }

  Future<void> _syncSessionToCloud(Map<String, dynamic> record) async {
    final sessionId =
        record['id'] ?? 'session_${DateTime.now().millisecondsSinceEpoch}';
    final session = StudySession(
      id: sessionId,
      date: DateTime.now(),
      durationMinutes: ((record['duration'] ?? 0) / 60).round(),
      subject: record['subject'] ?? '',
      breakCount: record['breaks'] ?? 0,
      intensityIndex: (record['intensity'] ?? 0.0).toDouble(),
      pings: const [],
    );
    await _syncService.syncSession(session);

    final user = _firebaseAuthOrNull?.currentUser;
    if (user == null) return;

    final objectiveId = _objectiveIdFromRecord(record);
    await AxonPaths.privateUserCollection(user.uid, 'study_events')
        .doc(sessionId)
        .set({
      'id': sessionId,
      'subject': (record['subject'] ?? '').toString(),
      'chapter': (record['chapter'] ?? '').toString(),
      'objective_id': objectiveId,
      'topic_id': objectiveId,
      'type': 'study_session',
      'duration_minutes': ((record['duration'] ?? 0) / 60).round(),
      'occurred_at': DateTime.now().toIso8601String(),
      'accuracy_score': (record['focusQuality'] ?? 0.0).toDouble(),
      'intensity': (record['intensity'] ?? 0.0).toDouble(),
    }, SetOptions(merge: true));

    try {
      final token = await user.getIdToken();
      if (token != null && token.isNotEmpty) {
        await http
            .post(
              Uri.parse('https://bhavu.up.railway.app/analyze-study-pulse'),
              headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer $token',
              },
              body: jsonEncode({
                'user_id': user.uid,
                'session_id': sessionId,
              }),
            )
            .timeout(const Duration(seconds: 8));
      }
    } catch (_) {}
  }

  String _objectiveIdFromRecord(Map<String, dynamic> record) {
    final existing = (record['objective_id'] ?? '').toString().trim();
    if (existing.isNotEmpty) {
      return existing;
    }
    final chapter = (record['chapter'] ?? '').toString().trim();
    final subject = (record['subject'] ?? '').toString().trim();
    final board = (record['board'] ?? '').toString().trim().toUpperCase();
    final raw = chapter.isNotEmpty ? chapter : subject;
    final slug = raw
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^_|_$'), '');
    if (board.isNotEmpty && subject.isNotEmpty) {
      return '${board}_${subject.toUpperCase().replaceAll(' ', '_')}_$slug';
    }
    return slug;
  }

  Duration _elapsedFromClock() {
    final startedAt = state.startedAt;
    if (startedAt == null) {
      return state.elapsed;
    }
    return DateTime.now().difference(startedAt);
  }

  void _syncElapsedFromClock() {
    if (!state.isRunning || state.startedAt == null) return;
    state = state.copyWith(elapsed: _elapsedFromClock());
  }

  Future<void> _saveState() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _prefsKey,
      jsonEncode({
        'isRunning': state.isRunning,
        'elapsedSeconds': state.elapsed.inSeconds,
        'breakCount': state.breakCount,
        'subject': state.subject,
        'chapter': state.chapter,
        'intensityIndex': state.intensityIndex,
        'pings': state.pings,
        'startedAt': state.startedAt?.toIso8601String(),
        'templateId': state.templateId,
        'templateName': state.templateName,
        'targetDurationSeconds': state.targetDuration?.inSeconds,
        'modeId': state.mode.id,
        'antiDistractionLock': state.antiDistractionLock,
        'interruptionCount': state.interruptionCount,
        'distractionNudges': state.distractionNudges,
        'fatigueIndex': state.fatigueIndex,
        'focusQuality': state.focusQuality,
        'recommendedBreakMinutes': state.recommendedBreakMinutes,
        'pomodoroEnabled': state.pomodoroEnabled,
      }),
    );
  }

  _SessionSignals _computeSessionSignals({
    required Duration elapsed,
    required int breakCount,
    required int interruptionCount,
    required int pings,
    required StudySessionMode mode,
  }) {
    final totalMinutes = elapsed.inSeconds / 60.0;
    if (totalMinutes <= 0) {
      return const _SessionSignals(
        intensity: 0.0,
        focusQuality: 1.0,
        fatigueIndex: 0.0,
        recommendedBreakMinutes: 0,
      );
    }

    final breakPenalty = breakCount * 0.1;
    final interruptionPenalty = interruptionCount * 0.13;
    final threshold = mode.fatigueThresholdMinutes.toDouble();
    final overtimeRatio = totalMinutes <= threshold
        ? 0.0
        : ((totalMinutes - threshold) / threshold);
    final fatigueIndex = (overtimeRatio * 0.75 +
            (breakCount * 0.05) +
            (interruptionCount * 0.08))
        .clamp(0.0, 1.0);
    final pingBonus = pings == 0
        ? 0.0
        : (pings / ((totalMinutes / 5.0).ceil().clamp(1, 9999))) * 0.05;
    final focusQuality = (1.0 -
            breakPenalty -
            interruptionPenalty -
            (fatigueIndex * 0.22) +
            pingBonus)
        .clamp(0.0, 1.0);
    final intensity = (1.0 - (breakPenalty * 0.7) - (interruptionPenalty * 0.9))
        .clamp(0.0, 1.0);

    final recommendedBreakMinutes = fatigueIndex >= 0.75
        ? 15
        : fatigueIndex >= 0.5
            ? 10
            : fatigueIndex >= 0.3
                ? 5
                : 0;

    return _SessionSignals(
      intensity: intensity,
      focusQuality: focusQuality,
      fatigueIndex: fatigueIndex,
      recommendedBreakMinutes: recommendedBreakMinutes,
    );
  }

  Future<void> _maybeNotifyIntensityDrop() async {
    final metrics = _ref.read(metricsProvider);
    final history = metrics.weekHistory;
    if (history.length < 3) return;
    final last3 = history.sublist(history.length - 3);
    final drop = last3.first.studyIntensity - last3.last.studyIntensity;
    if (drop < 0.12) return;

    final prefs = await SharedPreferences.getInstance();
    final lastAlert = prefs.getString('intensity_alert_date');
    final today = DateTime.now();
    final todayKey = '${today.year}-${today.month}-${today.day}';
    if (lastAlert == todayKey) return;

    final style = _ref.read(authStateProvider).user?.motivationStyle ??
        MotivationStyle.positiveReinforcement;
    final message = _buildMotivationMessage(style, drop);
    await NotificationService.show(
      id: 101,
      title: 'Axon Focus Alert',
      body: message,
    );
    await prefs.setString('intensity_alert_date', todayKey);
  }

  String _buildMotivationMessage(MotivationStyle style, double drop) {
    final pct = (drop * 100).round();
    switch (style) {
      case MotivationStyle.toughLove:
        return 'Your efficiency is down $pct%. At this rate, you miss your target. Reset and push now.';
      case MotivationStyle.logicBased:
        return 'Your focus trend dipped $pct% over 3 days. A 15-minute reset will lift your score.';
      case MotivationStyle.positiveReinforcement:
        return 'You are a bit tired. A short break can lift your focus by about $pct%.';
    }
  }

  Future<void> _clearSavedState() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_prefsKey);
  }
}

class _SessionSignals {
  final double intensity;
  final double focusQuality;
  final double fatigueIndex;
  final int recommendedBreakMinutes;

  const _SessionSignals({
    required this.intensity,
    required this.focusQuality,
    required this.fatigueIndex,
    required this.recommendedBreakMinutes,
  });
}

// Global UI State
final navbarVisibleProvider = StateProvider<bool>((ref) => true);
final examPlannerFlipProvider = StateProvider<double>((ref) => 0.0);
final navigationLockProvider = StateProvider<bool>((ref) => false);

// Force show navbar on navigation
void ensureNavbarVisible(WidgetRef ref) {
  ref.read(navigationLockProvider.notifier).state = false;
  Future.microtask(() {
    ref.read(navbarVisibleProvider.notifier).state = true;
  });
}

final studyStreakProvider = FutureProvider<int>((ref) async {
  final prefs = await SharedPreferences.getInstance();
  final raw = prefs.getString('timer_history');
  if (raw == null || raw.isEmpty) return 0;

  try {
    final decoded = jsonDecode(raw);
    if (decoded is! List) return 0;

    final days = decoded
        .whereType<Map>()
        .map((entry) => DateTime.tryParse((entry['date'] ?? '').toString()))
        .whereType<DateTime>()
        .map((date) => DateTime(date.year, date.month, date.day))
        .toSet()
        .toList()
      ..sort((a, b) => b.compareTo(a));

    if (days.isEmpty) return 0;

    var streak = 0;
    var cursor = DateTime.now();
    cursor = DateTime(cursor.year, cursor.month, cursor.day);

    for (final day in days) {
      final diff = cursor.difference(day).inDays;
      if (diff > 1) break;
      streak++;
      cursor = day.subtract(const Duration(days: 1));
    }
    return streak;
  } catch (_) {
    return 0;
  }
});

class PdfWorkspaceState {
  final String title;
  final List<PdfQuestion> questions;
  final Map<String, String> answers;
  final String markingSchemeText;
  final String filePath;

  const PdfWorkspaceState({
    required this.title,
    required this.questions,
    required this.answers,
    required this.markingSchemeText,
    required this.filePath,
  });

  PdfWorkspaceState copyWith({
    String? title,
    List<PdfQuestion>? questions,
    Map<String, String>? answers,
    String? markingSchemeText,
    String? filePath,
  }) {
    return PdfWorkspaceState(
      title: title ?? this.title,
      questions: questions ?? this.questions,
      answers: answers ?? this.answers,
      markingSchemeText: markingSchemeText ?? this.markingSchemeText,
      filePath: filePath ?? this.filePath,
    );
  }
}

final pdfWorkspaceProvider = AsyncNotifierProvider.family<PdfWorkspaceNotifier,
    PdfWorkspaceState, String>(PdfWorkspaceNotifier.new);

class PdfWorkspaceNotifier
    extends FamilyAsyncNotifier<PdfWorkspaceState, String> {
  @override
  PdfWorkspaceState build(String arg) {
    return PdfWorkspaceState(
      title: '',
      questions: const [],
      answers: const {},
      markingSchemeText: '',
      filePath: arg,
    );
  }

  void initialize({
    required String title,
    required List<PdfQuestion> questions,
    required Map<String, String> answers,
    required String? markingSchemeText,
  }) {
    final current = state.valueOrNull;
    final next = PdfWorkspaceState(
      title: title,
      questions: List<PdfQuestion>.from(questions),
      answers: Map<String, String>.from(answers),
      markingSchemeText: markingSchemeText ?? '',
      filePath: arg,
    );
    if (current != null &&
        current.title == next.title &&
        current.filePath == next.filePath &&
        current.questions.length == next.questions.length &&
        current.answers.length == next.answers.length &&
        current.markingSchemeText == next.markingSchemeText) {
      return;
    }
    state = AsyncData(next);
  }

  void updateAnswer(String key, String value) {
    final current = state.valueOrNull;
    if (current == null) return;
    final updatedAnswers = Map<String, String>.from(current.answers)
      ..[key] = value;
    state = AsyncData(current.copyWith(answers: updatedAnswers));
  }

  void updateMarkingScheme(String text) {
    final current = state.valueOrNull;
    if (current == null) return;
    state = AsyncData(current.copyWith(markingSchemeText: text));
  }
}

Future<void> safeNavigate(BuildContext context, String path) async {
  final container = ProviderScope.containerOf(context);
  final lock = container.read(navigationLockProvider);
  if (lock) return;

  container.read(navigationLockProvider.notifier).state = true;
  try {
    context.go(path);
  } finally {
    await Future.delayed(const Duration(milliseconds: 500));
    if (context.mounted) {
      container.read(navigationLockProvider.notifier).state = false;
    }
  }
}

Future<void> safePop(BuildContext context) async {
  final container = ProviderScope.containerOf(context);
  final lock = container.read(navigationLockProvider);
  if (lock) return;

  container.read(navigationLockProvider.notifier).state = true;
  try {
    if (Navigator.canPop(context)) {
      Navigator.of(context).pop();
    } else {
      context.go('/home');
    }
  } finally {
    await Future.delayed(const Duration(milliseconds: 300));
    if (context.mounted) {
      container.read(navigationLockProvider.notifier).state = false;
    }
  }
}
