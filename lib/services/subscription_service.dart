// lib/services/subscription_service.dart
// ─────────────────────────────────────────────────────────────────
// Subscription Service
// Manages tier definitions, feature gates, and entitlement checks.
// Reads real subscription state from Firestore (populated by Stripe
// Firebase Extension) and syncs locally for offline access.
// ─────────────────────────────────────────────────────────────────

import 'dart:async';
import 'dart:convert';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'firestore_service.dart';

const bool kForceAllUsersProPlan = true;

// ─────────────────────────────────────────────────────────────────
// TIER DEFINITIONS
// ─────────────────────────────────────────────────────────────────

enum SubscriptionTier {
  free,
  plus,
  premium,
  pro,
}

extension SubscriptionTierExt on SubscriptionTier {
  String get displayName {
    switch (this) {
      case SubscriptionTier.free:
        return 'Free';
      case SubscriptionTier.plus:
        return 'Plus';
      case SubscriptionTier.premium:
        return 'Premium';
      case SubscriptionTier.pro:
        return 'Pro';
    }
  }

  String get subtitle {
    switch (this) {
      case SubscriptionTier.free:
        return 'Get started with the basics';
      case SubscriptionTier.plus:
        return 'Level up your study game';
      case SubscriptionTier.premium:
        return 'Full AI-powered learning';
      case SubscriptionTier.pro:
        return 'Everything Axon has to offer';
    }
  }

  String get price {
    switch (this) {
      case SubscriptionTier.free:
        return 'Free';
      case SubscriptionTier.plus:
        return '\$4.99';
      case SubscriptionTier.premium:
        return '\$9.99';
      case SubscriptionTier.pro:
        return '\$19.99';
    }
  }

  String get priceMonthly {
    switch (this) {
      case SubscriptionTier.free:
        return 'Free';
      case SubscriptionTier.plus:
        return '\$4.99/mo';
      case SubscriptionTier.premium:
        return '\$9.99/mo';
      case SubscriptionTier.pro:
        return '\$19.99/mo';
    }
  }

  String get priceYearly {
    switch (this) {
      case SubscriptionTier.free:
        return 'Free';
      case SubscriptionTier.plus:
        return '\$47.88/yr'; // ~20% off
      case SubscriptionTier.premium:
        return '\$95.88/yr';
      case SubscriptionTier.pro:
        return '\$191.88/yr';
    }
  }

  double get monthlyPriceUsd {
    switch (this) {
      case SubscriptionTier.free:
        return 0;
      case SubscriptionTier.plus:
        return 4.99;
      case SubscriptionTier.premium:
        return 9.99;
      case SubscriptionTier.pro:
        return 19.99;
    }
  }

  double get yearlyPriceUsd {
    switch (this) {
      case SubscriptionTier.free:
        return 0;
      case SubscriptionTier.plus:
        return 47.88;
      case SubscriptionTier.premium:
        return 95.88;
      case SubscriptionTier.pro:
        return 191.88;
    }
  }

  int get yearlySavingPercent {
    switch (this) {
      case SubscriptionTier.free:
        return 0;
      case SubscriptionTier.plus:
        return 20;
      case SubscriptionTier.premium:
        return 20;
      case SubscriptionTier.pro:
        return 20;
    }
  }

  bool get isPaid => this != SubscriptionTier.free;

  bool get hasYearlyDiscount => this != SubscriptionTier.free;

  static SubscriptionTier fromString(String value) {
    return SubscriptionTier.values.firstWhere(
      (t) => t.name == value,
      orElse: () => SubscriptionTier.free,
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// FEATURE DEFINITIONS
// ─────────────────────────────────────────────────────────────────

enum Feature {
  studyTimerAdvanced,
  sessionHistory,
  analyticsDashboard,
  smartReminders,
  studyGraph,
  examPlannerUnlimited,
  offlinePacks,
  peerChallenges,
  motivationEngineAdvanced,
  aiCoachReports,
  pdfLibraryUnlimited,
  aiStudyAssistant,
  pastPapers,
  resourceRecommendationsAll,
  gamificationFull,
  accountabilityPartners,
  dataExport,
  prioritySupport,
  betaFeatures,
  leaderboard,
  streakProtection,
  customGoals,
  tutoringAccess,
  flashcardDecks,
  flashcardAI,
  weeklyReports,
  dailyReports,
  realTimeAIReports,
  resourceQualityRanking,
  weakTopicDetection,
  examCountdown,
  chapterPrerequisites,
  studyStreakRecovery,
}

extension FeatureExt on Feature {
  String get displayName {
    switch (this) {
      case Feature.studyTimerAdvanced:
        return 'Advanced Study Timer';
      case Feature.sessionHistory:
        return 'Full Session History';
      case Feature.analyticsDashboard:
        return 'Analytics Dashboard';
      case Feature.smartReminders:
        return 'Smart Reminders';
      case Feature.studyGraph:
        return 'Study Graph';
      case Feature.examPlannerUnlimited:
        return 'Unlimited Exam Planner';
      case Feature.offlinePacks:
        return 'Offline Resource Packs';
      case Feature.peerChallenges:
        return 'Peer Challenges';
      case Feature.motivationEngineAdvanced:
        return 'Advanced Motivation Engine';
      case Feature.aiCoachReports:
        return 'AI Coach Reports';
      case Feature.pdfLibraryUnlimited:
        return 'Unlimited PDF Library';
      case Feature.aiStudyAssistant:
        return 'AI Study Assistant';
      case Feature.pastPapers:
        return 'Full Past Paper Access';
      case Feature.resourceRecommendationsAll:
        return 'All Resource Recommendations';
      case Feature.gamificationFull:
        return 'Full Gamification';
      case Feature.accountabilityPartners:
        return 'Accountability Partners';
      case Feature.dataExport:
        return 'Data Export';
      case Feature.prioritySupport:
        return 'Priority Support';
      case Feature.betaFeatures:
        return 'Beta Features';
      case Feature.leaderboard:
        return 'Leaderboard';
      case Feature.streakProtection:
        return 'Streak Protection';
      case Feature.customGoals:
        return 'Custom Goals';
      case Feature.tutoringAccess:
        return 'Live Tutoring Access';
      case Feature.flashcardDecks:
        return 'Flashcard Decks';
      case Feature.flashcardAI:
        return 'AI Flashcard Generation';
      case Feature.weeklyReports:
        return 'Weekly AI Reports';
      case Feature.dailyReports:
        return 'Daily AI Reports';
      case Feature.realTimeAIReports:
        return 'Real-time AI Coach';
      case Feature.resourceQualityRanking:
        return 'Resource Quality Ranking';
      case Feature.weakTopicDetection:
        return 'Weak Topic Detection';
      case Feature.examCountdown:
        return 'Exam Countdown';
      case Feature.chapterPrerequisites:
        return 'Chapter Prerequisites';
      case Feature.studyStreakRecovery:
        return 'Streak Recovery';
    }
  }

  String get description {
    switch (this) {
      case Feature.studyTimerAdvanced:
        return 'Pomodoro, exam simulation, recall sprint, timed papers';
      case Feature.sessionHistory:
        return 'Access unlimited session history and trends';
      case Feature.analyticsDashboard:
        return 'Deep insights, correlations, and trend analysis';
      case Feature.smartReminders:
        return 'Behavior-driven nudges and daily briefings';
      case Feature.studyGraph:
        return 'Visual knowledge graph with mastery tracking';
      case Feature.examPlannerUnlimited:
        return 'Plan all subjects, chapters, and past papers';
      case Feature.offlinePacks:
        return 'Download resources for offline study';
      case Feature.peerChallenges:
        return 'Compete and collaborate with other students';
      case Feature.motivationEngineAdvanced:
        return 'Personalized coaching with 5 AI personas';
      case Feature.aiCoachReports:
        return 'Weekly and daily AI-powered study analysis';
      case Feature.pdfLibraryUnlimited:
        return 'Store and access unlimited PDFs';
      case Feature.aiStudyAssistant:
        return 'Ask questions, get explanations instantly';
      case Feature.pastPapers:
        return 'Access all exam board past papers';
      case Feature.resourceRecommendationsAll:
        return 'Board-specific curated resources ranked by quality';
      case Feature.gamificationFull:
        return 'XP, badges, levels, trophies, leaderboards';
      case Feature.accountabilityPartners:
        return 'Study buddies to keep you accountable';
      case Feature.dataExport:
        return 'Export your data in multiple formats';
      case Feature.prioritySupport:
        return 'Skip the queue with dedicated support';
      case Feature.betaFeatures:
        return 'Early access to experimental features';
      case Feature.leaderboard:
        return 'Compete with students globally';
      case Feature.streakProtection:
        return 'Shield your streak from missed days';
      case Feature.customGoals:
        return 'Set and track custom study targets';
      case Feature.tutoringAccess:
        return 'Book live 1-on-1 tutoring sessions';
      case Feature.flashcardDecks:
        return 'Create and study flashcard decks';
      case Feature.flashcardAI:
        return 'AI auto-generates flashcards from notes';
      case Feature.weeklyReports:
        return 'Comprehensive weekly performance digest';
      case Feature.dailyReports:
        return 'Daily AI briefing and recommendations';
      case Feature.realTimeAIReports:
        return 'Real-time coaching during study sessions';
      case Feature.resourceQualityRanking:
        return 'AI-ranked resources based on your behavior';
      case Feature.weakTopicDetection:
        return 'Auto-detect and prioritize weak topics';
      case Feature.examCountdown:
        return 'Countdown timer with chapter targets';
      case Feature.chapterPrerequisites:
        return 'See which chapters you need first';
      case Feature.studyStreakRecovery:
        return 'Recover lost streaks with missions';
    }
  }

  String get icon {
    switch (this) {
      case Feature.studyTimerAdvanced:
        return 'TIME';
      case Feature.sessionHistory:
        return 'LOG';
      case Feature.analyticsDashboard:
        return 'DATA';
      case Feature.smartReminders:
        return 'ALERT';
      case Feature.studyGraph:
        return 'GRAPH';
      case Feature.examPlannerUnlimited:
        return 'PLAN';
      case Feature.offlinePacks:
        return 'OFFLINE';
      case Feature.peerChallenges:
        return 'GOAL';
      case Feature.motivationEngineAdvanced:
        return 'BOOST';
      case Feature.aiCoachReports:
        return 'AI';
      case Feature.pdfLibraryUnlimited:
        return 'PDF';
      case Feature.aiStudyAssistant:
        return 'IDEA';
      case Feature.pastPapers:
        return 'PAPER';
      case Feature.resourceRecommendationsAll:
        return 'MATCH';
      case Feature.gamificationFull:
        return 'PLAY';
      case Feature.accountabilityPartners:
        return 'TEAM';
      case Feature.dataExport:
        return 'EXPORT';
      case Feature.prioritySupport:
        return 'PRIORITY';
      case Feature.betaFeatures:
        return 'BETA';
      case Feature.leaderboard:
        return 'RANK';
      case Feature.streakProtection:
        return 'SHIELD';
      case Feature.customGoals:
        return 'TARGET';
      case Feature.tutoringAccess:
        return 'TUTOR';
      case Feature.flashcardDecks:
        return 'CARDS';
      case Feature.flashcardAI:
        return 'AUTO';
      case Feature.weeklyReports:
        return 'WEEK';
      case Feature.dailyReports:
        return 'DAY';
      case Feature.realTimeAIReports:
        return 'LIVE';
      case Feature.resourceQualityRanking:
        return 'TOP';
      case Feature.weakTopicDetection:
        return 'SCAN';
      case Feature.examCountdown:
        return 'COUNT';
      case Feature.chapterPrerequisites:
        return 'LINK';
      case Feature.studyStreakRecovery:
        return 'RESET';
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// TIER → FEATURE MATRIX
// ─────────────────────────────────────────────────────────────────

class TierFeatureMatrix {
  static Set<Feature> getFeatures(SubscriptionTier tier) {
    switch (tier) {
      case SubscriptionTier.free:
        return const {
          Feature.studyTimerAdvanced, // Basic timer modes
          Feature.smartReminders, // Basic only
          Feature.motivationEngineAdvanced, // Basic tone only
          Feature.gamificationFull, // XP + streaks only
          Feature.examCountdown, // Basic countdown
        };
      case SubscriptionTier.plus:
        return const {
          Feature.studyTimerAdvanced,
          Feature.sessionHistory, // 30 days
          Feature.analyticsDashboard, // Basic
          Feature.smartReminders,
          Feature.studyGraph,
          Feature.examPlannerUnlimited, // 3 subjects
          Feature.offlinePacks, // 1 pack/month
          Feature.peerChallenges, // 3/week
          Feature.motivationEngineAdvanced, // All tones
          Feature.aiCoachReports, // Weekly only
          Feature.pdfLibraryUnlimited, // 20 PDFs
          Feature.resourceRecommendationsAll, // All sources
          Feature.gamificationFull,
          Feature.leaderboard,
          Feature.streakProtection, // 1 shield/month
          Feature.studyStreakRecovery,
          Feature.weeklyReports,
          Feature.examCountdown,
          Feature.chapterPrerequisites,
        };
      case SubscriptionTier.premium:
        return const {
          Feature.studyTimerAdvanced,
          Feature.sessionHistory, // 1 year
          Feature.analyticsDashboard,
          Feature.smartReminders,
          Feature.studyGraph,
          Feature.examPlannerUnlimited, // All subjects
          Feature.offlinePacks, // 5 packs/month
          Feature.peerChallenges, // Unlimited
          Feature.motivationEngineAdvanced,
          Feature.aiCoachReports, // Weekly + Daily
          Feature.pdfLibraryUnlimited, // 50 PDFs
          Feature.aiStudyAssistant, // 50 queries/month
          Feature.pastPapers, // All years
          Feature.resourceRecommendationsAll,
          Feature.gamificationFull,
          Feature.accountabilityPartners, // 5
          Feature.dataExport, // PDF only
          Feature.prioritySupport, // Email
          Feature.leaderboard,
          Feature.streakProtection, // 3 shields/month
          Feature.customGoals,
          Feature.flashcardDecks,
          Feature.flashcardAI, // 20 cards/month
          Feature.weeklyReports,
          Feature.dailyReports,
          Feature.realTimeAIReports, // Limited
          Feature.resourceQualityRanking,
          Feature.weakTopicDetection,
          Feature.examCountdown,
          Feature.chapterPrerequisites,
          Feature.studyStreakRecovery,
        };
      case SubscriptionTier.pro:
        return const {
          Feature.studyTimerAdvanced,
          Feature.sessionHistory, // Unlimited
          Feature.analyticsDashboard,
          Feature.smartReminders,
          Feature.studyGraph,
          Feature.examPlannerUnlimited, // All + tutor
          Feature.offlinePacks, // Unlimited
          Feature.peerChallenges, // Unlimited
          Feature.motivationEngineAdvanced,
          Feature.aiCoachReports, // Real-time
          Feature.pdfLibraryUnlimited, // Unlimited
          Feature.aiStudyAssistant, // Unlimited
          Feature.pastPapers, // All + mark schemes
          Feature.resourceRecommendationsAll,
          Feature.gamificationFull,
          Feature.accountabilityPartners, // Unlimited
          Feature.dataExport, // All formats
          Feature.prioritySupport, // Dedicated
          Feature.betaFeatures,
          Feature.leaderboard,
          Feature.streakProtection, // Unlimited
          Feature.customGoals,
          Feature.tutoringAccess, // 1 session/month
          Feature.flashcardDecks,
          Feature.flashcardAI, // Unlimited
          Feature.weeklyReports,
          Feature.dailyReports,
          Feature.realTimeAIReports, // Full
          Feature.resourceQualityRanking,
          Feature.weakTopicDetection,
          Feature.examCountdown,
          Feature.chapterPrerequisites,
          Feature.studyStreakRecovery,
        };
    }
  }

  static bool hasFeature(SubscriptionTier tier, Feature feature) {
    return getFeatures(tier).contains(feature);
  }

  static List<Feature> getNewFeaturesAtTier(SubscriptionTier tier) {
    if (tier == SubscriptionTier.free) return getFeatures(tier).toList();
    final previous = getFeatures(_previousTier(tier));
    final current = getFeatures(tier);
    return current.difference(previous).toList();
  }

  static SubscriptionTier _previousTier(SubscriptionTier tier) {
    switch (tier) {
      case SubscriptionTier.free:
        return SubscriptionTier.free;
      case SubscriptionTier.plus:
        return SubscriptionTier.free;
      case SubscriptionTier.premium:
        return SubscriptionTier.plus;
      case SubscriptionTier.pro:
        return SubscriptionTier.premium;
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// SUBSCRIPTION STATE
// ─────────────────────────────────────────────────────────────────

enum SubscriptionStatus {
  active,
  pastDue,
  canceled,
  trialing,
  unpaid,
  none,
}

extension SubscriptionStatusExt on SubscriptionStatus {
  bool get isActive =>
      this == SubscriptionStatus.active || this == SubscriptionStatus.trialing;

  bool get isCanceled => this == SubscriptionStatus.canceled;

  bool get isPastDue => this == SubscriptionStatus.pastDue;

  bool get hasAccess => isActive;

  static SubscriptionStatus fromString(String? value) {
    if (value == null) return SubscriptionStatus.none;
    return SubscriptionStatus.values.firstWhere(
      (s) => s.name == value,
      orElse: () => SubscriptionStatus.none,
    );
  }
}

class SubscriptionState {
  final SubscriptionTier tier;
  final SubscriptionStatus status;
  final DateTime? currentPeriodStart;
  final DateTime? currentPeriodEnd;
  final DateTime? trialEnd;
  final bool isYearly;
  final String? stripeCustomerId;
  final String? stripeSubscriptionId;
  final int? offlinePacksUsed;
  final int? offlinePacksLimit;
  final int? aiQueriesUsed;
  final int? aiQueriesLimit;
  final int? flashcardsGenerated;
  final int? flashcardsLimit;
  final int streakShieldsRemaining;
  final DateTime? lastSynced;

  const SubscriptionState({
    this.tier = SubscriptionTier.free,
    this.status = SubscriptionStatus.none,
    this.currentPeriodStart,
    this.currentPeriodEnd,
    this.trialEnd,
    this.isYearly = false,
    this.stripeCustomerId,
    this.stripeSubscriptionId,
    this.offlinePacksUsed = 0,
    this.offlinePacksLimit,
    this.aiQueriesUsed = 0,
    this.aiQueriesLimit,
    this.flashcardsGenerated = 0,
    this.flashcardsLimit,
    this.streakShieldsRemaining = 0,
    this.lastSynced,
  });

  bool get hasAccess => kForceAllUsersProPlan ? true : status.hasAccess;
  bool get isFree => kForceAllUsersProPlan ? false : tier == SubscriptionTier.free;
  bool get isPaid => kForceAllUsersProPlan ? true : tier.isPaid;

  bool get isYearlyPlan => isYearly;

  int get daysRemaining {
    if (currentPeriodEnd == null) return 0;
    return currentPeriodEnd!.difference(DateTime.now()).inDays;
  }

  bool get isInTrial {
    if (trialEnd == null) return false;
    return DateTime.now().isBefore(trialEnd!) &&
        status == SubscriptionStatus.trialing;
  }

  bool canUseFeature(Feature feature) {
    if (kForceAllUsersProPlan) return true;
    if (!hasAccess) {
      // Free tier gets limited features
      return TierFeatureMatrix.hasFeature(SubscriptionTier.free, feature);
    }
    return TierFeatureMatrix.hasFeature(tier, feature);
  }

  int? get offlinePacksRemaining {
    if (offlinePacksLimit == null) return null;
    return offlinePacksLimit! - (offlinePacksUsed ?? 0);
  }

  int? get aiQueriesRemaining {
    if (aiQueriesLimit == null) return null;
    return aiQueriesLimit! - (aiQueriesUsed ?? 0);
  }

  int? get flashcardsRemaining {
    if (flashcardsLimit == null) return null;
    return flashcardsLimit! - (flashcardsGenerated ?? 0);
  }

  SubscriptionState copyWith({
    SubscriptionTier? tier,
    SubscriptionStatus? status,
    DateTime? currentPeriodStart,
    DateTime? currentPeriodEnd,
    DateTime? trialEnd,
    bool? isYearly,
    String? stripeCustomerId,
    String? stripeSubscriptionId,
    int? offlinePacksUsed,
    int? offlinePacksLimit,
    int? aiQueriesUsed,
    int? aiQueriesLimit,
    int? flashcardsGenerated,
    int? flashcardsLimit,
    int? streakShieldsRemaining,
    DateTime? lastSynced,
  }) {
    return SubscriptionState(
      tier: tier ?? this.tier,
      status: status ?? this.status,
      currentPeriodStart: currentPeriodStart ?? this.currentPeriodStart,
      currentPeriodEnd: currentPeriodEnd ?? this.currentPeriodEnd,
      trialEnd: trialEnd ?? this.trialEnd,
      isYearly: isYearly ?? this.isYearly,
      stripeCustomerId: stripeCustomerId ?? this.stripeCustomerId,
      stripeSubscriptionId: stripeSubscriptionId ?? this.stripeSubscriptionId,
      offlinePacksUsed: offlinePacksUsed ?? this.offlinePacksUsed,
      offlinePacksLimit: offlinePacksLimit ?? this.offlinePacksLimit,
      aiQueriesUsed: aiQueriesUsed ?? this.aiQueriesUsed,
      aiQueriesLimit: aiQueriesLimit ?? this.aiQueriesLimit,
      flashcardsGenerated: flashcardsGenerated ?? this.flashcardsGenerated,
      flashcardsLimit: flashcardsLimit ?? this.flashcardsLimit,
      streakShieldsRemaining:
          streakShieldsRemaining ?? this.streakShieldsRemaining,
      lastSynced: lastSynced ?? this.lastSynced,
    );
  }

  factory SubscriptionState.fromFirestore(Map<String, dynamic>? data) {
    if (data == null) return const SubscriptionState();

    final planId = data['plan_id'] as String? ?? '';
    final tier = _parseTierFromPlanId(planId);
    final statusStr = data['subscription_status'] as String? ?? '';
    final status = SubscriptionStatusExt.fromString(statusStr);
    final isYearly = (data['is_yearly'] as bool?) ?? false;

    return SubscriptionState(
      tier: tier,
      status: status,
      currentPeriodStart: _parseDate(data['current_period_start']),
      currentPeriodEnd: _parseDate(data['current_period_end']),
      trialEnd: _parseDate(data['trial_end']),
      isYearly: isYearly,
      stripeCustomerId: data['stripe_customer_id'] as String?,
      stripeSubscriptionId: data['subscription_id'] as String?,
      offlinePacksUsed: data['offline_packs_used'] as int?,
      offlinePacksLimit: _getOfflinePacksLimit(tier),
      aiQueriesUsed: data['ai_queries_used'] as int?,
      aiQueriesLimit: _getAiQueriesLimit(tier),
      flashcardsGenerated: data['flashcards_generated'] as int?,
      flashcardsLimit: _getFlashcardsLimit(tier),
      streakShieldsRemaining:
          data['streak_shields_remaining'] as int? ?? _getShieldLimit(tier),
      lastSynced: DateTime.now(),
    );
  }

  static SubscriptionTier _parseTierFromPlanId(String planId) {
    if (planId.contains('pro') || planId.contains('ax_pro')) {
      return SubscriptionTier.pro;
    }
    if (planId.contains('premium') || planId.contains('ax_premium')) {
      return SubscriptionTier.premium;
    }
    if (planId.contains('plus') || planId.contains('ax_plus')) {
      return SubscriptionTier.plus;
    }
    return SubscriptionTier.free;
  }

  static DateTime? _parseDate(dynamic value) {
    if (value == null) return null;
    if (value is Timestamp) return value.toDate();
    if (value is String) return DateTime.tryParse(value);
    return null;
  }

  static int? _getOfflinePacksLimit(SubscriptionTier tier) {
    switch (tier) {
      case SubscriptionTier.free:
        return 0;
      case SubscriptionTier.plus:
        return 1;
      case SubscriptionTier.premium:
        return 5;
      case SubscriptionTier.pro:
        return null; // unlimited
    }
  }

  static int? _getAiQueriesLimit(SubscriptionTier tier) {
    switch (tier) {
      case SubscriptionTier.free:
        return 0;
      case SubscriptionTier.plus:
        return 0;
      case SubscriptionTier.premium:
        return 50;
      case SubscriptionTier.pro:
        return null; // unlimited
    }
  }

  static int? _getFlashcardsLimit(SubscriptionTier tier) {
    switch (tier) {
      case SubscriptionTier.free:
        return 0;
      case SubscriptionTier.plus:
        return 0;
      case SubscriptionTier.premium:
        return 20;
      case SubscriptionTier.pro:
        return null;
    }
  }

  static int _getShieldLimit(SubscriptionTier tier) {
    switch (tier) {
      case SubscriptionTier.free:
        return 0;
      case SubscriptionTier.plus:
        return 1;
      case SubscriptionTier.premium:
        return 3;
      case SubscriptionTier.pro:
        return 99; // effectively unlimited
    }
  }

  Map<String, dynamic> toJson() => {
        'tier': tier.name,
        'status': status.name,
        'currentPeriodStart': currentPeriodStart?.toIso8601String(),
        'currentPeriodEnd': currentPeriodEnd?.toIso8601String(),
        'trialEnd': trialEnd?.toIso8601String(),
        'isYearly': isYearly,
        'stripeCustomerId': stripeCustomerId,
        'stripeSubscriptionId': stripeSubscriptionId,
        'offlinePacksUsed': offlinePacksUsed,
        'offlinePacksLimit': offlinePacksLimit,
        'aiQueriesUsed': aiQueriesUsed,
        'aiQueriesLimit': aiQueriesLimit,
        'flashcardsGenerated': flashcardsGenerated,
        'flashcardsLimit': flashcardsLimit,
        'streakShieldsRemaining': streakShieldsRemaining,
        'lastSynced': lastSynced?.toIso8601String(),
      };
}

// ─────────────────────────────────────────────────────────────────
// SUBSCRIPTION SERVICE
// ─────────────────────────────────────────────────────────────────

class SubscriptionService {
  static final SubscriptionService _instance = SubscriptionService._internal();
  factory SubscriptionService() => _instance;
  SubscriptionService._internal();

  static const String _cacheKey = 'subscription_cache_v2';

  SubscriptionState _cachedState = const SubscriptionState();
  StreamSubscription<DocumentSnapshot>? _firestoreSub;
  final _stateController = StreamController<SubscriptionState>.broadcast();
  Timer? _syncTimer;

  Stream<SubscriptionState> get stateStream => _stateController.stream;
  SubscriptionState get current => _effectiveState(_cachedState);

  // Firebase Extension Stripe plan IDs (update with your actual Stripe Price IDs)
  static const Map<SubscriptionTier, Map<bool, String>> stripePriceIds = {
    SubscriptionTier.plus: {
      false:
          'stripe_plus_monthly_price_id', // REPLACE with real Stripe Price ID
      true: 'stripe_plus_yearly_price_id', // REPLACE with real Stripe Price ID
    },
    SubscriptionTier.premium: {
      false:
          'stripe_premium_monthly_price_id', // REPLACE with real Stripe Price ID
      true:
          'stripe_premium_yearly_price_id', // REPLACE with real Stripe Price ID
    },
    SubscriptionTier.pro: {
      false: 'stripe_pro_monthly_price_id', // REPLACE with real Stripe Price ID
      true: 'stripe_pro_yearly_price_id', // REPLACE with real Stripe Price ID
    },
  };

  // Cloud Function base URL (auto-deployed via Firebase Extension)
  // Format: https://us-central1-{projectId}.cloudfunctions.net/stripeRecurring...
  static const String _cloudFunctionBase =
      'https://us-central1-axon-study.cloudfunctions.net'; // REPLACE with your project ID

  Future<void> initialize() async {
    await _loadLocalCache();
    _cachedState = _effectiveState(_cachedState);
    _stateController.add(_cachedState);
    await _subscribeToFirestore();
    _startPeriodicSync();
  }

  Future<void> _loadLocalCache() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_cacheKey);
    if (raw != null && raw.isNotEmpty) {
      try {
        final data = jsonDecode(raw) as Map<String, dynamic>;
        _cachedState = SubscriptionState(
          tier: SubscriptionTierExt.fromString(data['tier'] ?? 'free'),
          status: SubscriptionStatusExt.fromString(data['status']),
          currentPeriodStart: data['currentPeriodStart'] != null
              ? DateTime.parse(data['currentPeriodStart'])
              : null,
          currentPeriodEnd: data['currentPeriodEnd'] != null
              ? DateTime.parse(data['currentPeriodEnd'])
              : null,
          trialEnd: data['trialEnd'] != null
              ? DateTime.parse(data['trialEnd'])
              : null,
          isYearly: data['isYearly'] ?? false,
          offlinePacksUsed: data['offlinePacksUsed'],
          offlinePacksLimit: data['offlinePacksLimit'],
          aiQueriesUsed: data['aiQueriesUsed'],
          aiQueriesLimit: data['aiQueriesLimit'],
          flashcardsGenerated: data['flashcardsGenerated'],
          flashcardsLimit: data['flashcardsLimit'],
          streakShieldsRemaining: data['streakShieldsRemaining'] ?? 0,
          lastSynced: data['lastSynced'] != null
              ? DateTime.parse(data['lastSynced'])
              : null,
        );
        _cachedState = _effectiveState(_cachedState);
        _stateController.add(_cachedState);
      } catch (e) {
        debugPrint('[SubscriptionService] Failed to load local cache: $e');
      }
    }
  }

  Future<void> _saveLocalCache() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_cacheKey, jsonEncode(_cachedState.toJson()));
  }

  Future<void> _subscribeToFirestore() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final docRef = AxonFirestore.instance
        .collection('users')
        .doc(user.uid)
        .collection('subscription')
        .doc('current');

    try {
      _firestoreSub?.cancel();
      _firestoreSub = docRef.snapshots().listen((snap) async {
        if (!snap.exists) return;

        final data = snap.data()!;
        final newState = _effectiveState(SubscriptionState.fromFirestore(data));

        if (newState.tier != _cachedState.tier ||
            newState.status != _cachedState.status) {
          _cachedState = newState;
          await _saveLocalCache();
          _stateController.add(_cachedState);
        }
      });
    } catch (e) {
      debugPrint('Subscription sync error: $e');
    }
  }

  void _startPeriodicSync() {
    _syncTimer?.cancel();
    _syncTimer = Timer.periodic(const Duration(minutes: 30), (_) {
      _refreshFromFirestore();
    });
  }

  Future<void> _refreshFromFirestore() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    try {
      final doc = await AxonFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('subscription')
          .doc('current')
          .get();

      if (doc.exists) {
        _cachedState = _effectiveState(SubscriptionState.fromFirestore(doc.data()));
        await _saveLocalCache();
        _stateController.add(_cachedState);
      }
    } catch (e) {
      debugPrint('[SubscriptionService] Failed to refresh from Firestore: $e');
    }
  }

  // ── Entitlement checks ────────────────────────────────────────

  bool hasAccess(Feature feature) => _cachedState.canUseFeature(feature);

  Future<bool> canUseFeature(Feature feature) async {
    await _refreshFromFirestore();
    return _cachedState.canUseFeature(feature);
  }

  Future<bool> enforceFeature(Feature feature) async {
    if (_cachedState.canUseFeature(feature)) return true;
    return false;
  }

  bool get canUseAdvancedTimer => hasAccess(Feature.studyTimerAdvanced);

  bool get canUseStudyGraph => hasAccess(Feature.studyGraph);

  bool get canUseAnalytics => hasAccess(Feature.analyticsDashboard);

  bool get canUseSmartReminders => hasAccess(Feature.smartReminders);

  bool get canUseOfflinePacks => hasAccess(Feature.offlinePacks);

  bool get canUsePeerChallenges => hasAccess(Feature.peerChallenges);

  bool get canUseAIReports => hasAccess(Feature.aiCoachReports);

  bool get canUseAIAssistant => hasAccess(Feature.aiStudyAssistant);

  bool get canUseFlashcards => hasAccess(Feature.flashcardDecks);

  bool get canUseFlashcardAI => hasAccess(Feature.flashcardAI);

  bool get canUseLeaderboard => hasAccess(Feature.leaderboard);

  bool get canUseBeta => hasAccess(Feature.betaFeatures);

  // ── Usage tracking ────────────────────────────────────────────

  Future<void> incrementOfflinePackUsed() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final newUsed = (_cachedState.offlinePacksUsed ?? 0) + 1;
    final limit = _cachedState.offlinePacksLimit;
    if (limit != null && newUsed > limit) return;

    try {
      await AxonFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('subscription')
          .doc('current')
          .update({'offline_packs_used': newUsed});

      _cachedState = _cachedState.copyWith(offlinePacksUsed: newUsed);
      await _saveLocalCache();
      _stateController.add(_cachedState);
    } catch (e) {
      debugPrint('[SubscriptionService] Failed to increment offline pack usage: $e');
    }
  }

  Future<void> incrementAIQueryUsed() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final newUsed = (_cachedState.aiQueriesUsed ?? 0) + 1;
    final limit = _cachedState.aiQueriesLimit;
    if (limit != null && newUsed > limit) return;

    try {
      await AxonFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('subscription')
          .doc('current')
          .update({'ai_queries_used': newUsed});

      _cachedState = _cachedState.copyWith(aiQueriesUsed: newUsed);
      await _saveLocalCache();
      _stateController.add(_cachedState);
    } catch (e) {
      debugPrint('[SubscriptionService] Failed to increment AI query usage: $e');
    }
  }

  Future<void> incrementFlashcardGenerated() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final newUsed = (_cachedState.flashcardsGenerated ?? 0) + 1;
    final limit = _cachedState.flashcardsLimit;
    if (limit != null && newUsed > limit) return;

    try {
      await AxonFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('subscription')
          .doc('current')
          .update({'flashcards_generated': newUsed});

      _cachedState = _cachedState.copyWith(flashcardsGenerated: newUsed);
      await _saveLocalCache();
      _stateController.add(_cachedState);
    } catch (e) {
      debugPrint('[SubscriptionService] Failed to increment flashcard generation: $e');
    }
  }

  Future<void> useStreakShield() async {
    if (_cachedState.streakShieldsRemaining <= 0) return;
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    try {
      await AxonFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('subscription')
          .doc('current')
          .update({'streak_shields_remaining': FieldValue.increment(-1)});

      _cachedState = _cachedState.copyWith(
        streakShieldsRemaining:
            (_cachedState.streakShieldsRemaining - 1).clamp(0, 99),
      );
      await _saveLocalCache();
      _stateController.add(_cachedState);
    } catch (e) {
      debugPrint('[SubscriptionService] Failed to use streak shield: $e');
    }
  }

  // ── Stripe Checkout (calls Cloud Function) ─────────────────────

  Future<String?> startCheckout({
    required SubscriptionTier tier,
    required bool isYearly,
    required String successUrl,
    required String cancelUrl,
  }) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return null;

    final priceId = stripePriceIds[tier]?[isYearly] ?? '';
    if (priceId.isEmpty || priceId.contains('price_id')) {
      debugPrint(
          'ERROR: Replace Stripe Price IDs in subscription_service.dart');
      return null;
    }

    try {
      final response = await http.post(
        Uri.parse('$_cloudFunctionBase/stripeCreateCheckout'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'uid': user.uid,
          'price_id': priceId,
          'tier': tier.name,
          'is_yearly': isYearly,
          'success_url': successUrl,
          'cancel_url': cancelUrl,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['url'] as String?;
      }
    } catch (e) {
      debugPrint('Checkout error: $e');
    }
    return null;
  }

  Future<String?> openCustomerPortal() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return null;
    final customerId = _cachedState.stripeCustomerId;
    if (customerId == null || customerId.isEmpty) return null;

    try {
      final response = await http.post(
        Uri.parse('$_cloudFunctionBase/stripeCustomerPortal'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'uid': user.uid,
          'customer_id': customerId,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['url'] as String?;
      }
    } catch (e) {
      debugPrint('Portal error: $e');
    }
    return null;
  }

  // ── Force refresh ─────────────────────────────────────────────

  Future<void> refresh() async {
    await _refreshFromFirestore();
  }

  void dispose() {
    _firestoreSub?.cancel();
    _syncTimer?.cancel();
    _stateController.close();
  }

  SubscriptionState _effectiveState(SubscriptionState state) {
    if (!kForceAllUsersProPlan) return state;
    return state.copyWith(
      tier: SubscriptionTier.pro,
      status: SubscriptionStatus.active,
      offlinePacksLimit: null,
      aiQueriesLimit: null,
      flashcardsLimit: null,
      streakShieldsRemaining: 99,
      lastSynced: DateTime.now(),
    );
  }
}

// Singleton provider
final subscriptionServiceProvider = SubscriptionService();
