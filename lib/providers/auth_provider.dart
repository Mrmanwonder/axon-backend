// lib/providers/auth_provider.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/models.dart';
import '../services/profile_service.dart';
import '../services/backend_health_service.dart';
import '../services/firestore_service.dart';
import '../services/notification_service.dart';
import '../services/study_catalog.dart';
import '../services/sync_service.dart';
import '../services/offline_sync_service.dart';
import '../services/sync_manager.dart';
import '../services/notion_service.dart';
import '../services/google_calendar_service.dart';
import '../services/obsidian_service.dart';
import '../services/grok_service.dart';
import '../services/axon_auto_crawl_service.dart';

// Helper getters for Firebase
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

// Auth Draft State and Notifier
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

// Auth Gate related
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

// Main Auth State and Notifier
final authStateProvider = StateNotifierProvider<AuthNotifier, AuthState>(
  (ref) => AuthNotifier(),
);

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
  AxonAutoCrawlService? _autoCrawlService;

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
