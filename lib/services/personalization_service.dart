// lib/services/personalization_service.dart
// ─────────────────────────────────────────────────────────────────
// Personalization Service
// Manages AxonMode, CoachingPersona, AccessibilityPreset, StudyGraph
// ─────────────────────────────────────────────────────────────────

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

// ─────────────────────────────────────────────────────────────────
// AXON MODE — UI operating mode
// ─────────────────────────────────────────────────────────────────

enum AxonMode {
  calm,
  intense,
  examCrunch,
}

extension AxonModeExt on AxonMode {
  String get displayName {
    switch (this) {
      case AxonMode.calm:
        return 'Calm';
      case AxonMode.intense:
        return 'Intense';
      case AxonMode.examCrunch:
        return 'Exam Crunch';
    }
  }

  String get description {
    switch (this) {
      case AxonMode.calm:
        return 'Gentle nudges, relaxed pace, steady progress';
      case AxonMode.intense:
        return 'Full power mode — aggressive coaching, more reminders';
      case AxonMode.examCrunch:
        return 'Maximum urgency — daily alerts, chapter targets, exam countdown';
    }
  }

  String get icon {
    switch (this) {
      case AxonMode.calm:
        return 'CALM';
      case AxonMode.intense:
        return 'INTENSE';
      case AxonMode.examCrunch:
        return 'CRUNCH';
    }
  }

  Color get accentColor {
    switch (this) {
      case AxonMode.calm:
        return const Color(0xFF10B981);
      case AxonMode.intense:
        return const Color(0xFFFF6B35);
      case AxonMode.examCrunch:
        return const Color(0xFFEF4444);
    }
  }

  Duration get reminderInterval {
    switch (this) {
      case AxonMode.calm:
        return const Duration(hours: 8);
      case AxonMode.intense:
        return const Duration(hours: 4);
      case AxonMode.examCrunch:
        return const Duration(hours: 2);
    }
  }

  bool get showAnimations {
    switch (this) {
      case AxonMode.calm:
        return false;
      case AxonMode.intense:
        return true;
      case AxonMode.examCrunch:
        return true;
    }
  }

  bool get showExamCountdown {
    switch (this) {
      case AxonMode.calm:
        return false;
      case AxonMode.intense:
        return true;
      case AxonMode.examCrunch:
        return true;
    }
  }

  bool get aggressiveNudges {
    switch (this) {
      case AxonMode.calm:
        return false;
      case AxonMode.intense:
        return true;
      case AxonMode.examCrunch:
        return true;
    }
  }

  static AxonMode fromString(String value) {
    return AxonMode.values.firstWhere(
      (m) => m.name == value,
      orElse: () => AxonMode.calm,
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// COACHING PERSONA — 5 distinct personalities
// ─────────────────────────────────────────────────────────────────

enum CoachingPersona {
  mentor,
  drillSergeant,
  cheerleader,
  scientist,
  buddy,
}

extension CoachingPersonaExt on CoachingPersona {
  String get displayName {
    switch (this) {
      case CoachingPersona.mentor:
        return 'The Mentor';
      case CoachingPersona.drillSergeant:
        return 'The Sergeant';
      case CoachingPersona.cheerleader:
        return 'The Cheerleader';
      case CoachingPersona.scientist:
        return 'The Scientist';
      case CoachingPersona.buddy:
        return 'The Study Buddy';
    }
  }

  String get description {
    switch (this) {
      case CoachingPersona.mentor:
        return 'Wise, steady guidance. Challenges you to think deeply, celebrates growth.';
      case CoachingPersona.drillSergeant:
        return 'Takes no excuses. Brutally honest, expects excellence every session.';
      case CoachingPersona.cheerleader:
        return 'Unstoppable positivity. Every effort is a win. You can do this!';
      case CoachingPersona.scientist:
        return 'Cold, hard data. Shows you the numbers, lets logic drive improvement.';
      case CoachingPersona.buddy:
        return 'Casual, relatable. Like a friend who actually wants to see you succeed.';
    }
  }

  String get avatarEmoji {
    switch (this) {
      case CoachingPersona.mentor:
        return 'M';
      case CoachingPersona.drillSergeant:
        return 'S';
      case CoachingPersona.cheerleader:
        return 'C';
      case CoachingPersona.scientist:
        return 'D';
      case CoachingPersona.buddy:
        return 'B';
    }
  }

  String get voice {
    switch (this) {
      case CoachingPersona.mentor:
        return 'measured, encouraging, insightful';
      case CoachingPersona.drillSergeant:
        return 'direct, demanding, no fluff';
      case CoachingPersona.cheerleader:
        return 'energetic, warm, celebratory';
      case CoachingPersona.scientist:
        return 'analytical, precise, data-driven';
      case CoachingPersona.buddy:
        return 'casual, friendly, supportive';
    }
  }

  String getPreferredMotivationStyle() {
    switch (this) {
      case CoachingPersona.mentor:
        return 'supportive';
      case CoachingPersona.drillSergeant:
        return 'aggressive';
      case CoachingPersona.cheerleader:
        return 'encouraging';
      case CoachingPersona.scientist:
        return 'logical';
      case CoachingPersona.buddy:
        return 'casual';
    }
  }

  static CoachingPersona fromString(String value) {
    return CoachingPersona.values.firstWhere(
      (p) => p.name == value,
      orElse: () => CoachingPersona.mentor,
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// ACCESSIBILITY PRESET — reading density, motion, contrast
// ─────────────────────────────────────────────────────────────────

enum ReadingDensity {
  compact,
  normal,
  spacious,
  ultraSpacious,
}

extension ReadingDensityExt on ReadingDensity {
  String get displayName {
    switch (this) {
      case ReadingDensity.compact:
        return 'Compact';
      case ReadingDensity.normal:
        return 'Normal';
      case ReadingDensity.spacious:
        return 'Spacious';
      case ReadingDensity.ultraSpacious:
        return 'Ultra Spacious';
    }
  }

  double get lineHeight {
    switch (this) {
      case ReadingDensity.compact:
        return 1.2;
      case ReadingDensity.normal:
        return 1.5;
      case ReadingDensity.spacious:
        return 1.75;
      case ReadingDensity.ultraSpacious:
        return 2.0;
    }
  }

  double get fontSizeMultiplier {
    switch (this) {
      case ReadingDensity.compact:
        return 0.85;
      case ReadingDensity.normal:
        return 1.0;
      case ReadingDensity.spacious:
        return 1.15;
      case ReadingDensity.ultraSpacious:
        return 1.3;
    }
  }

  double get cardPadding {
    switch (this) {
      case ReadingDensity.compact:
        return 8.0;
      case ReadingDensity.normal:
        return 16.0;
      case ReadingDensity.spacious:
        return 24.0;
      case ReadingDensity.ultraSpacious:
        return 32.0;
    }
  }

  static ReadingDensity fromString(String value) {
    return ReadingDensity.values.firstWhere(
      (d) => d.name == value,
      orElse: () => ReadingDensity.normal,
    );
  }
}

enum MotionPreference {
  full,
  reduced,
  minimal,
  none,
}

extension MotionPreferenceExt on MotionPreference {
  String get displayName {
    switch (this) {
      case MotionPreference.full:
        return 'Full Animations';
      case MotionPreference.reduced:
        return 'Reduced';
      case MotionPreference.minimal:
        return 'Minimal';
      case MotionPreference.none:
        return 'None';
    }
  }

  String get description {
    switch (this) {
      case MotionPreference.full:
        return 'All animations and transitions';
      case MotionPreference.reduced:
        return 'Essential animations only';
      case MotionPreference.minimal:
        return 'Only page transitions';
      case MotionPreference.none:
        return 'No animations';
    }
  }

  Duration get animationDuration {
    switch (this) {
      case MotionPreference.full:
        return const Duration(milliseconds: 300);
      case MotionPreference.reduced:
        return const Duration(milliseconds: 150);
      case MotionPreference.minimal:
        return const Duration(milliseconds: 100);
      case MotionPreference.none:
        return Duration.zero;
    }
  }

  bool get enableParallax => this == MotionPreference.full;
  bool get enableCardAnimations => this != MotionPreference.none;
  bool get enablePulseEffects => this == MotionPreference.full;
  bool get enableSlideTransitions => index <= MotionPreference.minimal.index;

  static MotionPreference fromString(String value) {
    return MotionPreference.values.firstWhere(
      (m) => m.name == value,
      orElse: () => MotionPreference.full,
    );
  }
}

enum ContrastMode {
  standard,
  high,
  inverted,
}

extension ContrastModeExt on ContrastMode {
  String get displayName {
    switch (this) {
      case ContrastMode.standard:
        return 'Standard';
      case ContrastMode.high:
        return 'High Contrast';
      case ContrastMode.inverted:
        return 'Inverted';
    }
  }

  String get description {
    switch (this) {
      case ContrastMode.standard:
        return 'Default contrast for light/dark themes';
      case ContrastMode.high:
        return 'Maximum contrast — easier to read';
      case ContrastMode.inverted:
        return 'Full light-on-dark in both themes';
    }
  }

  double get textOpacity {
    switch (this) {
      case ContrastMode.standard:
        return 1.0;
      case ContrastMode.high:
        return 1.0;
      case ContrastMode.inverted:
        return 1.0;
    }
  }

  double get borderWidth {
    switch (this) {
      case ContrastMode.standard:
        return 1.0;
      case ContrastMode.high:
        return 2.0;
      case ContrastMode.inverted:
        return 1.5;
    }
  }

  static ContrastMode fromString(String value) {
    return ContrastMode.values.firstWhere(
      (c) => c.name == value,
      orElse: () => ContrastMode.standard,
    );
  }
}

class AccessibilityPreset {
  final ReadingDensity density;
  final MotionPreference motion;
  final ContrastMode contrast;
  final bool reduceTransparency;
  final bool boldText;
  final double textScaleFactor;
  final bool greyscaleMode;

  const AccessibilityPreset({
    this.density = ReadingDensity.normal,
    this.motion = MotionPreference.full,
    this.contrast = ContrastMode.standard,
    this.reduceTransparency = false,
    this.boldText = false,
    this.textScaleFactor = 1.0,
    this.greyscaleMode = false,
  });

  AccessibilityPreset copyWith({
    ReadingDensity? density,
    MotionPreference? motion,
    ContrastMode? contrast,
    bool? reduceTransparency,
    bool? boldText,
    double? textScaleFactor,
    bool? greyscaleMode,
  }) {
    return AccessibilityPreset(
      density: density ?? this.density,
      motion: motion ?? this.motion,
      contrast: contrast ?? this.contrast,
      reduceTransparency: reduceTransparency ?? this.reduceTransparency,
      boldText: boldText ?? this.boldText,
      textScaleFactor: textScaleFactor ?? this.textScaleFactor,
      greyscaleMode: greyscaleMode ?? this.greyscaleMode,
    );
  }

  Map<String, dynamic> toJson() => {
        'density': density.name,
        'motion': motion.name,
        'contrast': contrast.name,
        'reduceTransparency': reduceTransparency,
        'boldText': boldText,
        'textScaleFactor': textScaleFactor,
        'greyscaleMode': greyscaleMode,
      };

  factory AccessibilityPreset.fromJson(Map<String, dynamic> json) {
    return AccessibilityPreset(
      density: ReadingDensityExt.fromString(json['density'] ?? 'normal'),
      motion: MotionPreferenceExt.fromString(json['motion'] ?? 'full'),
      contrast: ContrastModeExt.fromString(json['contrast'] ?? 'standard'),
      reduceTransparency: json['reduceTransparency'] ?? false,
      boldText: json['boldText'] ?? false,
      textScaleFactor: (json['textScaleFactor'] ?? 1.0).toDouble(),
      greyscaleMode: json['greyscaleMode'] ?? false,
    );
  }

  static const AccessibilityPreset defaultPreset = AccessibilityPreset();

  static const AccessibilityPreset dyslexia = AccessibilityPreset(
    density: ReadingDensity.spacious,
    motion: MotionPreference.reduced,
    contrast: ContrastMode.standard,
    reduceTransparency: false,
    boldText: false,
    textScaleFactor: 1.1,
  );

  static const AccessibilityPreset visualImpairment = AccessibilityPreset(
    density: ReadingDensity.ultraSpacious,
    motion: MotionPreference.none,
    contrast: ContrastMode.high,
    reduceTransparency: true,
    boldText: true,
    textScaleFactor: 1.3,
  );

  static const AccessibilityPreset motionSensitive = AccessibilityPreset(
    density: ReadingDensity.normal,
    motion: MotionPreference.none,
    contrast: ContrastMode.standard,
    reduceTransparency: true,
    boldText: false,
    textScaleFactor: 1.0,
  );

  static const AccessibilityPreset fastReader = AccessibilityPreset(
    density: ReadingDensity.compact,
    motion: MotionPreference.full,
    contrast: ContrastMode.standard,
    reduceTransparency: false,
    boldText: false,
    textScaleFactor: 1.0,
  );
}

// ─────────────────────────────────────────────────────────────────
// PERSONALIZATION STATE
// ─────────────────────────────────────────────────────────────────

class PersonalizationState {
  final AxonMode axonMode;
  final CoachingPersona coachingPersona;
  final AccessibilityPreset accessibility;
  final bool onboardingAssessmentComplete;
  final DateTime? lastModeChange;

  const PersonalizationState({
    this.axonMode = AxonMode.calm,
    this.coachingPersona = CoachingPersona.mentor,
    this.accessibility = const AccessibilityPreset(),
    this.onboardingAssessmentComplete = false,
    this.lastModeChange,
  });

  PersonalizationState copyWith({
    AxonMode? axonMode,
    CoachingPersona? coachingPersona,
    AccessibilityPreset? accessibility,
    bool? onboardingAssessmentComplete,
    DateTime? lastModeChange,
  }) {
    return PersonalizationState(
      axonMode: axonMode ?? this.axonMode,
      coachingPersona: coachingPersona ?? this.coachingPersona,
      accessibility: accessibility ?? this.accessibility,
      onboardingAssessmentComplete:
          onboardingAssessmentComplete ?? this.onboardingAssessmentComplete,
      lastModeChange: lastModeChange ?? this.lastModeChange,
    );
  }

  Map<String, dynamic> toJson() => {
        'axonMode': axonMode.name,
        'coachingPersona': coachingPersona.name,
        'accessibility': accessibility.toJson(),
        'onboardingAssessmentComplete': onboardingAssessmentComplete,
        'lastModeChange': lastModeChange?.toIso8601String(),
      };

  factory PersonalizationState.fromJson(Map<String, dynamic> json) {
    return PersonalizationState(
      axonMode: AxonModeExt.fromString(json['axonMode'] ?? 'calm'),
      coachingPersona:
          CoachingPersonaExt.fromString(json['coachingPersona'] ?? 'mentor'),
      accessibility: json['accessibility'] != null
          ? AccessibilityPreset.fromJson(json['accessibility'])
          : const AccessibilityPreset(),
      onboardingAssessmentComplete:
          json['onboardingAssessmentComplete'] ?? false,
      lastModeChange: json['lastModeChange'] != null
          ? DateTime.tryParse(json['lastModeChange']?.toString() ?? '')
          : null,
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// PERSONALIZATION SERVICE
// ─────────────────────────────────────────────────────────────────

class PersonalizationService {
  static final PersonalizationService _instance =
      PersonalizationService._internal();
  factory PersonalizationService() => _instance;
  PersonalizationService._internal();

  static const String _key = 'personalization_state';
  static final ValueNotifier<PersonalizationState> notifier =
      ValueNotifier(const PersonalizationState());

  Future<PersonalizationState> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_key);
    if (raw == null || raw.isEmpty) {
      notifier.value = const PersonalizationState();
      return const PersonalizationState();
    }

    try {
      final state = PersonalizationState.fromJson(jsonDecode(raw));
      notifier.value = state;
      return state;
    } catch (_) {
      notifier.value = const PersonalizationState();
      return const PersonalizationState();
    }
  }

  Future<void> save(PersonalizationState state) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(state.toJson()));
    notifier.value = state;
  }

  Future<void> setAxonMode(AxonMode mode) async {
    final state = await load();
    await save(state.copyWith(
      axonMode: mode,
      lastModeChange: DateTime.now(),
    ));
  }

  Future<void> setCoachingPersona(CoachingPersona persona) async {
    final state = await load();
    await save(state.copyWith(coachingPersona: persona));
  }

  Future<void> setAccessibility(AccessibilityPreset preset) async {
    final state = await load();
    await save(state.copyWith(accessibility: preset));
  }

  Future<void> setOnboardingComplete() async {
    final state = await load();
    await save(state.copyWith(onboardingAssessmentComplete: true));
  }

  Future<AxonMode> getAxonMode() async {
    final state = await load();
    return state.axonMode;
  }

  Future<CoachingPersona> getCoachingPersona() async {
    final state = await load();
    return state.coachingPersona;
  }

  Future<AccessibilityPreset> getAccessibility() async {
    final state = await load();
    return state.accessibility;
  }

  Future<void> applyPreset(String presetName) async {
    switch (presetName) {
      case 'dyslexia':
        await setAccessibility(AccessibilityPreset.dyslexia);
        break;
      case 'visual':
        await setAccessibility(AccessibilityPreset.visualImpairment);
        break;
      case 'motion':
        await setAccessibility(AccessibilityPreset.motionSensitive);
        break;
      case 'fast':
        await setAccessibility(AccessibilityPreset.fastReader);
        break;
      default:
        await setAccessibility(AccessibilityPreset.defaultPreset);
    }
  }

  Future<void> resetToDefaults() async {
    await save(const PersonalizationState());
  }
}

final personalizationServiceProvider = PersonalizationService();

// ─────────────────────────────────────────────────────────────────
// GREYSCALE MODE NOTIFIER — ambient light detection & toggle
// ─────────────────────────────────────────────────────────────────

class GreyscaleModeNotifier extends ChangeNotifier {
  static final GreyscaleModeNotifier _instance =
      GreyscaleModeNotifier._internal();
  factory GreyscaleModeNotifier() => _instance;
  GreyscaleModeNotifier._internal();

  bool _enabled = false;
  bool _autoDetectedDark = false;
  bool _suggestedForUser = false;

  bool get enabled => _enabled;
  bool get autoDetectedDark => _autoDetectedDark;
  bool get suggestedForUser => _suggestedForUser;

  void setEnabled(bool value) {
    if (_enabled != value) {
      _enabled = value;
      notifyListeners();
    }
  }

  void setAutoDetectedDark(bool value) {
    if (_autoDetectedDark != value) {
      _autoDetectedDark = value;
      notifyListeners();
    }
  }

  void setSuggestedForUser(bool value) {
    if (_suggestedForUser != value) {
      _suggestedForUser = value;
      notifyListeners();
    }
  }

  void dismissSuggestion() {
    if (_suggestedForUser) {
      _suggestedForUser = false;
      notifyListeners();
    }
  }
}

final greyscaleModeNotifierProvider = GreyscaleModeNotifier();
