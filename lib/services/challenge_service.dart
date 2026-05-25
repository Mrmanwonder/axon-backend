// lib/services/challenge_service.dart
// ─────────────────────────────────────────────────────────────────
// Peer Challenge System & Accountability Partners
// ─────────────────────────────────────────────────────────────────

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'firestore_service.dart';
import 'notification_service.dart';

enum ChallengeType {
  questionCount,
  timeSpent,
  chapterComplete,
  mockScore,
  streakMaintain,
  custom,
}

enum ChallengeStatus {
  pending,
  inProgress,
  completed,
  failed,
  expired,
}

class Challenge {
  final String id;
  final String creatorId;
  final String creatorName;
  final String? creatorPhoto;
  final String title;
  final String description;
  final ChallengeType type;
  final int targetValue;
  final int currentValue;
  final String subject;
  final String? chapter;
  final DateTime deadline;
  final DateTime createdAt;
  final ChallengeStatus status;
  final bool isPublic;
  final List<String> participants;
  final List<ChallengeCompletion> completions;
  final int xpReward;
  final String? partnerId;

  Challenge({
    required this.id,
    required this.creatorId,
    required this.creatorName,
    this.creatorPhoto,
    required this.title,
    required this.description,
    required this.type,
    required this.targetValue,
    this.currentValue = 0,
    required this.subject,
    this.chapter,
    required this.deadline,
    required this.createdAt,
    this.status = ChallengeStatus.pending,
    this.isPublic = true,
    this.participants = const [],
    this.completions = const [],
    this.xpReward = 50,
    this.partnerId,
  });

  double get progress =>
      targetValue > 0 ? (currentValue / targetValue).clamp(0.0, 1.0) : 0.0;
  bool get isExpired => DateTime.now().isAfter(deadline);
  Duration get timeRemaining => deadline.difference(DateTime.now());
  bool get isCompleted => status == ChallengeStatus.completed;

  Map<String, dynamic> toJson() => {
        'id': id,
        'creatorId': creatorId,
        'creatorName': creatorName,
        'creatorPhoto': creatorPhoto,
        'title': title,
        'description': description,
        'type': type.index,
        'targetValue': targetValue,
        'currentValue': currentValue,
        'subject': subject,
        'chapter': chapter,
        'deadline': deadline.toIso8601String(),
        'createdAt': createdAt.toIso8601String(),
        'status': status.index,
        'isPublic': isPublic,
        'participants': participants,
        'completions': completions.map((c) => c.toJson()).toList(),
        'xpReward': xpReward,
        'partnerId': partnerId,
      };

  factory Challenge.fromJson(Map<String, dynamic> json) {
    DateTime deadline;
    DateTime createdAt;
    try {
      deadline = DateTime.parse(json['deadline']);
    } catch (e) {
      debugPrint('[ChallengeService] Failed to parse deadline: $e');
      deadline = DateTime.now().add(const Duration(days: 7));
    }
    try {
      createdAt = DateTime.parse(json['createdAt']);
    } catch (e) {
      debugPrint('[ChallengeService] Failed to parse createdAt: $e');
      createdAt = DateTime.now();
    }
    return Challenge(
      id: json['id'],
      creatorId: json['creatorId'],
      creatorName: json['creatorName'],
      creatorPhoto: json['creatorPhoto'],
      title: json['title'],
      description: json['description'],
      type: ChallengeType.values[json['type'] ?? 0],
      targetValue: json['targetValue'] ?? 0,
      currentValue: json['currentValue'] ?? 0,
      subject: json['subject'] ?? '',
      chapter: json['chapter'],
      deadline: deadline,
      createdAt: createdAt,
      status: ChallengeStatus.values[json['status'] ?? 0],
      isPublic: json['isPublic'] ?? true,
      participants: List<String>.from(json['participants'] ?? []),
      completions: (json['completions'] as List?)
              ?.map((c) => ChallengeCompletion.fromJson(c))
              .toList() ??
          [],
      xpReward: json['xpReward'] ?? 50,
      partnerId: json['partnerId'],
    );
  }
}

class ChallengeCompletion {
  final String oderId;
  final DateTime completedAt;
  final int value;

  ChallengeCompletion({
    required this.oderId,
    required this.completedAt,
    required this.value,
  });

  factory ChallengeCompletion.fromJson(Map<String, dynamic> json) {
    DateTime completedAt;
    try {
      completedAt = DateTime.parse(json['completedAt']);
    } catch (e) {
      debugPrint('[ChallengeService] Failed to parse completion date: $e');
      completedAt = DateTime.now();
    }
    return ChallengeCompletion(
      oderId: json['oderId'],
      completedAt: completedAt,
      value: json['value'],
    );
  }

  Map<String, dynamic> toJson() => {
        'oderId': oderId,
        'completedAt': completedAt.toIso8601String(),
        'value': value,
      };
}

class AccountabilityPartner {
  final String id;
  final String oderId;
  final String partnerName;
  final String? partnerPhoto;
  final DateTime connectedAt;
  final DateTime? lastCheckIn;
  final int totalChallengesTogether;
  final int challengesWon;
  final bool isActive;

  AccountabilityPartner({
    required this.id,
    required this.oderId,
    required this.partnerName,
    this.partnerPhoto,
    required this.connectedAt,
    this.lastCheckIn,
    this.totalChallengesTogether = 0,
    this.challengesWon = 0,
    this.isActive = true,
  });

  factory AccountabilityPartner.fromJson(Map<String, dynamic> json) {
    DateTime connectedAt;
    try {
      connectedAt = DateTime.parse(json['connectedAt']);
    } catch (e) {
      debugPrint('[ChallengeService] Failed to parse connectedAt: $e');
      connectedAt = DateTime.now();
    }
    DateTime? lastCheckIn;
    if (json['lastCheckIn'] != null) {
      try {
        lastCheckIn = DateTime.parse(json['lastCheckIn']);
      } catch (e) {
        debugPrint('[ChallengeService] Failed to parse lastCheckIn: $e');
      }
    }
    return AccountabilityPartner(
      id: json['id'],
      oderId: json['oderId'],
      partnerName: json['partnerName'],
      partnerPhoto: json['partnerPhoto'],
      connectedAt: connectedAt,
      lastCheckIn: lastCheckIn,
      totalChallengesTogether: json['totalChallengesTogether'] ?? 0,
      challengesWon: json['challengesWon'] ?? 0,
      isActive: json['isActive'] ?? true,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'oderId': oderId,
        'partnerName': partnerName,
        'partnerPhoto': partnerPhoto,
        'connectedAt': connectedAt.toIso8601String(),
        'lastCheckIn': lastCheckIn?.toIso8601String(),
        'totalChallengesTogether': totalChallengesTogether,
        'challengesWon': challengesWon,
        'isActive': isActive,
      };
}

class CheckIn {
  final String id;
  final String oderId;
  final String partnerId;
  final DateTime timestamp;
  final String message;
  final int streak;
  final String mood;

  CheckIn({
    required this.id,
    required this.oderId,
    required this.partnerId,
    required this.timestamp,
    required this.message,
    required this.streak,
    required this.mood,
  });

  factory CheckIn.fromJson(Map<String, dynamic> json) {
    DateTime timestamp;
    try {
      timestamp = DateTime.parse(json['timestamp']);
    } catch (e) {
      debugPrint('[ChallengeService] Failed to parse check-in timestamp: $e');
      timestamp = DateTime.now();
    }
    return CheckIn(
      id: json['id'],
      oderId: json['oderId'],
      partnerId: json['partnerId'],
      timestamp: timestamp,
      message: json['message'],
      streak: json['streak'],
      mood: json['mood'],
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'oderId': oderId,
        'partnerId': partnerId,
        'timestamp': timestamp.toIso8601String(),
        'message': message,
        'streak': streak,
        'mood': mood,
      };
}

class ChallengeService {
  static final ChallengeService _instance = ChallengeService._internal();
  factory ChallengeService() => _instance;
  ChallengeService._internal();

  static const String _challengesKey = 'axon_challenges';
  static const String _partnersKey = 'axon_partners';
  static const String _checkInsKey = 'axon_checkins';

  final FirebaseFirestore _db = AxonFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  Future<Challenge> createChallenge({
    required String title,
    required String description,
    required ChallengeType type,
    required int targetValue,
    required String subject,
    String? chapter,
    required DateTime deadline,
    bool isPublic = true,
    String? partnerId,
    int xpReward = 50,
  }) async {
    final user = _auth.currentUser;
    final challenge = Challenge(
      id: 'challenge_${DateTime.now().millisecondsSinceEpoch}',
      creatorId: user?.uid ?? 'anonymous',
      creatorName: user?.displayName ?? 'Anonymous',
      creatorPhoto: user?.photoURL,
      title: title,
      description: description,
      type: type,
      targetValue: targetValue,
      subject: subject,
      chapter: chapter,
      deadline: deadline,
      createdAt: DateTime.now(),
      isPublic: isPublic,
      partnerId: partnerId,
      xpReward: xpReward,
    );

    await _saveChallengeLocally(challenge);

    try {
      await _db
          .collection('challenges')
          .doc(challenge.id)
          .set(challenge.toJson());
    } catch (e) {
      debugPrint('[ChallengeService] Failed to sync challenge to cloud: $e');
    }

    return challenge;
  }

  Future<List<Challenge>> getActiveChallenges() async {
    final local = await _getLocalChallenges();
    final now = DateTime.now();

    try {
      final snapshot = await _db
          .collection('challenges')
          .where('deadline', isGreaterThan: now)
          .orderBy('deadline')
          .limit(20)
          .get();

      final cloudChallenges =
          snapshot.docs.map((doc) => Challenge.fromJson(doc.data())).toList();
      final allChallenges = {
        ...local,
        ...Map.fromEntries(cloudChallenges.map((c) => MapEntry(c.id, c)))
      };

      return allChallenges.values
          .where((c) => !c.isExpired && c.status != ChallengeStatus.completed)
          .toList()
        ..sort((a, b) => a.deadline.compareTo(b.deadline));
    } catch (e) {
      debugPrint('[ChallengeService] Failed to fetch active challenges from cloud: $e');
      return local.values
          .where((c) => !c.isExpired && c.status != ChallengeStatus.completed)
          .toList()
        ..sort((a, b) => a.deadline.compareTo(b.deadline));
    }
  }

  Future<List<Challenge>> getPublicChallenges() async {
    final user = _auth.currentUser;
    if (user == null) return [];

    try {
      final snapshot = await _db
          .collection('challenges')
          .where('isPublic', isEqualTo: true)
          .where('status', isEqualTo: ChallengeStatus.pending.index)
          .where('deadline', isGreaterThan: DateTime.now())
          .orderBy('deadline')
          .limit(30)
          .get();

      return snapshot.docs
          .map((doc) => Challenge.fromJson(doc.data()))
          .where((c) =>
              c.creatorId != user.uid && !c.participants.contains(user.uid))
          .toList();
    } catch (e) {
      debugPrint('[ChallengeService] Failed to fetch public challenges: $e');
      return [];
    }
  }

  Future<void> joinChallenge(String challengeId) async {
    final user = _auth.currentUser;
    if (user == null) return;

    final challenge = await _getChallenge(challengeId);
    if (challenge == null) return;

    if (challenge.participants.contains(user.uid)) return;

    final updatedParticipants = [...challenge.participants, user.uid];
    final updatedChallenge = Challenge(
      id: challenge.id,
      creatorId: challenge.creatorId,
      creatorName: challenge.creatorName,
      creatorPhoto: challenge.creatorPhoto,
      title: challenge.title,
      description: challenge.description,
      type: challenge.type,
      targetValue: challenge.targetValue,
      currentValue: challenge.currentValue,
      subject: challenge.subject,
      chapter: challenge.chapter,
      deadline: challenge.deadline,
      createdAt: challenge.createdAt,
      status: ChallengeStatus.inProgress,
      isPublic: challenge.isPublic,
      participants: updatedParticipants,
      completions: challenge.completions,
      xpReward: challenge.xpReward,
      partnerId: challenge.partnerId,
    );

    await _saveChallengeLocally(updatedChallenge);

    try {
      await _db.collection('challenges').doc(challengeId).update({
        'participants': updatedParticipants,
        'status': ChallengeStatus.inProgress.index,
      });
    } catch (e) {
      debugPrint('[ChallengeService] Failed to join challenge: $e');
    }
  }

  Future<void> updateProgress(String challengeId, int value) async {
    final challenge = await _getChallenge(challengeId);
    if (challenge == null) return;

    final newValue = challenge.currentValue + value;
    final isCompleted = newValue >= challenge.targetValue;

    final updatedChallenge = Challenge(
      id: challenge.id,
      creatorId: challenge.creatorId,
      creatorName: challenge.creatorName,
      creatorPhoto: challenge.creatorPhoto,
      title: challenge.title,
      description: challenge.description,
      type: challenge.type,
      targetValue: challenge.targetValue,
      currentValue: newValue,
      subject: challenge.subject,
      chapter: challenge.chapter,
      deadline: challenge.deadline,
      createdAt: challenge.createdAt,
      status:
          isCompleted ? ChallengeStatus.completed : ChallengeStatus.inProgress,
      isPublic: challenge.isPublic,
      participants: challenge.participants,
      completions: challenge.completions,
      xpReward: challenge.xpReward,
      partnerId: challenge.partnerId,
    );

    await _saveChallengeLocally(updatedChallenge);

    if (isCompleted) {
      await _notifyChallengeComplete(updatedChallenge);
    }

    try {
      await _db.collection('challenges').doc(challengeId).update({
        'currentValue': newValue,
        'status': isCompleted
            ? ChallengeStatus.completed.index
            : ChallengeStatus.inProgress.index,
      });
    } catch (e) {
      debugPrint('[ChallengeService] Failed to update progress: $e');
    }
  }

  Future<Challenge?> _getChallenge(String challengeId) async {
    final local = await _getLocalChallenges();
    if (local.containsKey(challengeId)) {
      return local[challengeId];
    }

    try {
      final doc = await _db.collection('challenges').doc(challengeId).get();
      if (doc.exists) {
        return Challenge.fromJson(doc.data()!);
      }
    } catch (e) {
      debugPrint('[ChallengeService] Failed to get challenge from cloud: $e');
    }

    return null;
  }

  // ACCOUNTABILITY PARTNERS
  Future<List<AccountabilityPartner>> getPartners() async {
    final local = await _getLocalPartners();
    return local.values.toList()
      ..sort((a, b) {
        if (a.lastCheckIn == null && b.lastCheckIn == null) return 0;
        if (a.lastCheckIn == null) return 1;
        if (b.lastCheckIn == null) return -1;
        return b.lastCheckIn!.compareTo(a.lastCheckIn!);
      });
  }

  Future<AccountabilityPartner?> connectWithPartner(
      String oderId, String partnerName, String? partnerPhoto) async {
    final user = _auth.currentUser;
    if (user == null) return null;

    final partner = AccountabilityPartner(
      id: 'partner_${DateTime.now().millisecondsSinceEpoch}',
      oderId: oderId,
      partnerName: partnerName,
      partnerPhoto: partnerPhoto,
      connectedAt: DateTime.now(),
    );

    await _savePartnerLocally(partner);
    return partner;
  }

  Future<CheckIn> sendCheckIn({
    required String partnerId,
    required String message,
    required int streak,
    required String mood,
  }) async {
    final user = _auth.currentUser;
    if (user == null) throw Exception('Not authenticated');

    final checkIn = CheckIn(
      id: 'checkin_${DateTime.now().millisecondsSinceEpoch}',
      oderId: user.uid,
      partnerId: partnerId,
      timestamp: DateTime.now(),
      message: message,
      streak: streak,
      mood: mood,
    );

    await _saveCheckInLocally(checkIn);
    await _notifyPartnerCheckIn(checkIn);

    return checkIn;
  }

  Future<List<CheckIn>> getCheckInsForPartner(String partnerId) async {
    final local = await _getLocalCheckIns();
    return local.values.where((c) => c.partnerId == partnerId).toList()
      ..sort((a, b) => b.timestamp.compareTo(a.timestamp));
  }

  Future<int> getPartnerStreak(String partnerId) async {
    final checkIns = await getCheckInsForPartner(partnerId);
    if (checkIns.isEmpty) return 0;

    int streak = 0;
    DateTime? lastDate;

    for (final checkIn in checkIns) {
      final checkDate = DateTime(
        checkIn.timestamp.year,
        checkIn.timestamp.month,
        checkIn.timestamp.day,
      );

      if (lastDate == null) {
        streak = 1;
        lastDate = checkDate;
      } else {
        final diff = lastDate.difference(checkDate).inDays;
        if (diff == 1) {
          streak++;
          lastDate = checkDate;
        } else if (diff == 0) {
          continue;
        } else {
          break;
        }
      }
    }

    return streak;
  }

  Future<void> _notifyChallengeComplete(Challenge challenge) async {
    await NotificationService.show(
      id: 7001,
      title: 'Challenge Complete',
      body: 'You completed: ${challenge.title}',
      channel: AxonChannel.achievement,
    );
  }

  Future<void> _notifyPartnerCheckIn(CheckIn checkIn) async {
    final partners = await getPartners();
    final partner =
        partners.where((p) => p.id == checkIn.partnerId).firstOrNull;

    if (partner != null) {
      await NotificationService.show(
        id: 7010,
        title: 'Check-in from ${partner.partnerName}',
        body: checkIn.message.length > 50
            ? '${checkIn.message.substring(0, 50)}...'
            : checkIn.message,
        channel: AxonChannel.streakAlert,
      );
    }
  }

  Future<void> sendPartnerReminder(String partnerId) async {
    final partners = await getPartners();
    final partner = partners.where((p) => p.id == partnerId).firstOrNull;

    if (partner != null) {
      await NotificationService.show(
        id: 7020,
        title: 'Partner Check-in Reminder',
        body: 'Time to check in with ${partner.partnerName}!',
        channel: AxonChannel.studyReminder,
      );
    }
  }

  // LOCAL STORAGE
  Future<void> _saveChallengeLocally(Challenge challenge) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_challengesKey);
    final Map<String, dynamic> challenges = {};

    if (raw != null && raw.isNotEmpty) {
      try {
        challenges.addAll(jsonDecode(raw) as Map<String, dynamic>);
      } catch (e) {
        debugPrint('[ChallengeService] Failed to decode local challenges: $e');
      }
    }

    challenges[challenge.id] = challenge.toJson();
    await prefs.setString(_challengesKey, jsonEncode(challenges));
  }

  Future<Map<String, Challenge>> _getLocalChallenges() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_challengesKey);
    if (raw == null || raw.isEmpty) return {};

    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return map.map((k, v) => MapEntry(k, Challenge.fromJson(v)));
    } catch (e) {
      debugPrint('[ChallengeService] Failed to load local challenges: $e');
      return {};
    }
  }

  Future<void> _savePartnerLocally(AccountabilityPartner partner) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_partnersKey);
    final Map<String, dynamic> partners = {};

    if (raw != null && raw.isNotEmpty) {
      try {
        partners.addAll(jsonDecode(raw) as Map<String, dynamic>);
      } catch (e) {
        debugPrint('[ChallengeService] Failed to decode local partners: $e');
      }
    }

    partners[partner.id] = partner.toJson();
    await prefs.setString(_partnersKey, jsonEncode(partners));
  }

  Future<Map<String, AccountabilityPartner>> _getLocalPartners() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_partnersKey);
    if (raw == null || raw.isEmpty) return {};

    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return map.map((k, v) => MapEntry(k, AccountabilityPartner.fromJson(v)));
    } catch (e) {
      debugPrint('[ChallengeService] Failed to load local partners: $e');
      return {};
    }
  }

  Future<void> _saveCheckInLocally(CheckIn checkIn) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_checkInsKey);
    final Map<String, dynamic> checkIns = {};

    if (raw != null && raw.isNotEmpty) {
      try {
        checkIns.addAll(jsonDecode(raw) as Map<String, dynamic>);
      } catch (e) {
        debugPrint('[ChallengeService] Failed to decode local checkIns: $e');
      }
    }

    checkIns[checkIn.id] = checkIn.toJson();
    await prefs.setString(_checkInsKey, jsonEncode(checkIns));
  }

  Future<Map<String, CheckIn>> _getLocalCheckIns() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_checkInsKey);
    if (raw == null || raw.isEmpty) return {};

    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return map.map((k, v) => MapEntry(k, CheckIn.fromJson(v)));
    } catch (e) {
      debugPrint('[ChallengeService] Failed to load local checkIns: $e');
      return {};
    }
  }
}

final challengeServiceProvider = ChallengeService();
