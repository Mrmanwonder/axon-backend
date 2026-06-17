import 'package:flutter/material.dart';
import '../services/study_catalog.dart';

class PristinePalette {
  static const Color accent = Color(0xFF3A86FF);
  static const Color backgroundDark = Color(0xFF0A0A0A);
  static const Color glassWhite = Color(0xA6FFFFFF);
  static const Color glassDark = Color(0xB30A0A0A);
}

class StudySession {
  final String id;
  final DateTime date;
  final int durationMinutes;
  final String subject;
  final int breakCount;
  final double intensityIndex;
  final List<SessionPing> pings;

  StudySession({
    required this.id,
    required this.date,
    required this.durationMinutes,
    required this.subject,
    required this.breakCount,
    required this.intensityIndex,
    required this.pings,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'date': date.toIso8601String(),
        'durationMinutes': durationMinutes,
        'subject': subject,
        'breakCount': breakCount,
        'intensityIndex': intensityIndex,
        'pings': pings.map((p) => p.toJson()).toList(),
      };

  factory StudySession.fromJson(Map<String, dynamic> json) {
    DateTime date;
    try {
      date = DateTime.parse(json['date']);
    } catch (_) {
      date = DateTime.now();
    }
    return StudySession(
      id: json['id'],
      date: date,
      durationMinutes: json['durationMinutes'],
      subject: json['subject'],
      breakCount: json['breakCount'],
      intensityIndex: json['intensityIndex'].toDouble(),
      pings:
          (json['pings'] as List).map((p) => SessionPing.fromJson(p)).toList(),
    );
  }
}

class SessionPing {
  final DateTime timestamp;
  final bool isActive;

  SessionPing({required this.timestamp, required this.isActive});

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp.toIso8601String(),
        'isActive': isActive,
      };

  factory SessionPing.fromJson(Map<String, dynamic> json) {
    DateTime timestamp;
    try {
      timestamp = DateTime.parse(json['timestamp']);
    } catch (_) {
      timestamp = DateTime.now();
    }
    return SessionPing(
      timestamp: timestamp,
      isActive: json['isActive'],
    );
  }
}

class DailyMetrics {
  final DateTime date;
  final double sleepHours;
  final double screenTimeHours;
  final String primarySubject;
  final double subjectDifficulty;
  final double studyIntensity;
  final double mockScore;
  final double predictedPerformance;
  final double syllabusCoverage;
  final double activeStudyHours;
  final double targetStudyHours;
  final double focusRatio;
  final int consistencyStreak;
  final double stressLevel;

  DailyMetrics({
    required this.date,
    required this.sleepHours,
    required this.screenTimeHours,
    required this.primarySubject,
    required this.subjectDifficulty,
    required this.studyIntensity,
    this.mockScore = 0,
    this.predictedPerformance = 0.5,
    this.syllabusCoverage = 0,
    this.activeStudyHours = 0,
    this.targetStudyHours = 6,
    this.focusRatio = 0,
    this.consistencyStreak = 0,
    this.stressLevel = 0,
  });

  Map<String, dynamic> toJson() => {
        'date': date.toIso8601String(),
        'sleepHours': sleepHours,
        'screenTimeHours': screenTimeHours,
        'primarySubject': primarySubject,
        'subjectDifficulty': subjectDifficulty,
        'studyIntensity': studyIntensity,
        'mockScore': mockScore,
        'predictedPerformance': predictedPerformance,
        'syllabusCoverage': syllabusCoverage,
        'activeStudyHours': activeStudyHours,
        'targetStudyHours': targetStudyHours,
        'focusRatio': focusRatio,
        'consistencyStreak': consistencyStreak,
        'stressLevel': stressLevel,
      };

  factory DailyMetrics.fromJson(Map<String, dynamic> json) {
    DateTime date;
    try {
      date = DateTime.parse(json['date']);
    } catch (_) {
      date = DateTime.now();
    }
    return DailyMetrics(
      date: date,
      sleepHours: (json['sleepHours'] ?? 0).toDouble(),
      screenTimeHours: (json['screenTimeHours'] ?? 0).toDouble(),
      primarySubject: (json['primarySubject'] ?? '').toString(),
      subjectDifficulty: (json['subjectDifficulty'] ?? 0).toDouble(),
      studyIntensity: (json['studyIntensity'] ?? 0).toDouble(),
      mockScore: (json['mockScore'] ?? 0).toDouble(),
      predictedPerformance: (json['predictedPerformance'] ?? 0).toDouble(),
      syllabusCoverage: (json['syllabusCoverage'] ?? 0).toDouble(),
      activeStudyHours: (json['activeStudyHours'] ?? 0).toDouble(),
      targetStudyHours: (json['targetStudyHours'] ?? 6).toDouble(),
      focusRatio: (json['focusRatio'] ?? 0).toDouble(),
      consistencyStreak: (json['consistencyStreak'] ?? 0).toInt(),
      stressLevel: (json['stressLevel'] ?? 0).toDouble(),
    );
  }
}

enum MotivationStyle {
  toughLove,
  positiveReinforcement,
  logicBased,
}

extension MotivationStyleExt on MotivationStyle {
  String get displayName {
    switch (this) {
      case MotivationStyle.toughLove:
        return 'Tough Love';
      case MotivationStyle.positiveReinforcement:
        return 'Positive Reinforcement';
      case MotivationStyle.logicBased:
        return 'Logic-Based';
    }
  }

  String get description {
    switch (this) {
      case MotivationStyle.toughLove:
        return 'Direct, no-nonsense feedback that challenges you to push harder';
      case MotivationStyle.positiveReinforcement:
        return 'Encouraging messages that celebrate progress and build confidence';
      case MotivationStyle.logicBased:
        return 'Data-driven insights that explain exactly why and how to improve';
    }
  }

  String get icon {
    switch (this) {
      case MotivationStyle.toughLove:
        return 'TL';
      case MotivationStyle.positiveReinforcement:
        return 'PR';
      case MotivationStyle.logicBased:
        return 'LB';
    }
  }

  String get matrixKey {
    switch (this) {
      case MotivationStyle.toughLove:
        return 'tough_love';
      case MotivationStyle.positiveReinforcement:
        return 'positive';
      case MotivationStyle.logicBased:
        return 'logic';
    }
  }
}

class UserProfile {
  final String uid;
  final String displayName;
  final String email;
  final String? photoUrl;
  final MotivationStyle motivationStyle;
  final String board;
  final List<String> subjects;
  final double targetStudyHours;
  final bool onboardingComplete;
  final DateTime createdAt;
  final Map<String, dynamic> preferences;
  final int xp;
  final int level;
  final List<String> badges;
  final int currentStreak;
  final int longestStreak;
  final int totalSessions;
  final int perfectQuizzes;

  UserProfile({
    required this.uid,
    required this.displayName,
    required this.email,
    this.photoUrl,
    this.motivationStyle = MotivationStyle.positiveReinforcement,
    this.board = '',
    this.subjects = const [
      'Mathematics',
      'Physics',
      'Chemistry',
      'Biology',
      'Computer Science',
      'Economics',
      'History',
      'Geography',
      'English Literature',
      'French',
      'Spanish',
      'German',
    ],
    this.targetStudyHours = 4,
    this.onboardingComplete = false,
    required this.createdAt,
    this.preferences = const {},
    this.xp = 0,
    this.level = 1,
    this.badges = const [],
    this.currentStreak = 0,
    this.longestStreak = 0,
    this.totalSessions = 0,
    this.perfectQuizzes = 0,
  });

  UserProfile copyWith({
    String? displayName,
    String? email,
    String? photoUrl,
    MotivationStyle? motivationStyle,
    String? board,
    List<String>? subjects,
    double? targetStudyHours,
    bool? onboardingComplete,
    Map<String, dynamic>? preferences,
    int? xp,
    int? level,
    List<String>? badges,
    int? currentStreak,
    int? longestStreak,
    int? totalSessions,
    int? perfectQuizzes,
  }) {
    return UserProfile(
      uid: uid,
      displayName: displayName ?? this.displayName,
      email: email ?? this.email,
      photoUrl: photoUrl ?? this.photoUrl,
      motivationStyle: motivationStyle ?? this.motivationStyle,
      board: board ?? this.board,
      subjects: subjects ?? this.subjects,
      targetStudyHours: targetStudyHours ?? this.targetStudyHours,
      onboardingComplete: onboardingComplete ?? this.onboardingComplete,
      createdAt: createdAt,
      preferences: preferences ?? this.preferences,
      xp: xp ?? this.xp,
      level: level ?? this.level,
      badges: badges ?? this.badges,
      currentStreak: currentStreak ?? this.currentStreak,
      longestStreak: longestStreak ?? this.longestStreak,
      totalSessions: totalSessions ?? this.totalSessions,
      perfectQuizzes: perfectQuizzes ?? this.perfectQuizzes,
    );
  }

  factory UserProfile.fromFirestore(Map<String, dynamic> map,
      {required String uid, String? fallbackEmail}) {
    final styleIndex = (map['motivation_style'] is num)
        ? (map['motivation_style'] as num).toInt()
        : int.tryParse('${map['motivation_style']}') ??
            MotivationStyle.positiveReinforcement.index;
    final rawCreatedAt = map['created_at'];
    DateTime createdAt;
    if (rawCreatedAt is DateTime) {
      createdAt = rawCreatedAt;
    } else if (rawCreatedAt != null &&
        rawCreatedAt.toString().contains('Timestamp')) {
      try {
        createdAt = rawCreatedAt.toDate() as DateTime;
      } catch (_) {
        createdAt = DateTime.now();
      }
    } else {
      createdAt = DateTime.tryParse('${rawCreatedAt ?? ''}') ?? DateTime.now();
    }
    return UserProfile(
      uid: uid,
      displayName:
          (map['display_name'] ?? map['displayName'] ?? 'Student').toString(),
      email: (map['email'] ?? fallbackEmail ?? '').toString(),
      photoUrl: map['photo_url']?.toString(),
      motivationStyle: MotivationStyle
          .values[styleIndex.clamp(0, MotivationStyle.values.length - 1)],
      board: (map['board'] ?? '').toString(),
      subjects: StudyCatalog.extractSubjects(
        map['subjects'] ?? map['study_catalog'],
      ),
      targetStudyHours: (map['target_hours'] is num)
          ? (map['target_hours'] as num).toDouble()
          : double.tryParse('${map['target_hours']}') ?? 4,
      onboardingComplete: map['onboarding_complete'] == true,
      createdAt: createdAt,
      preferences:
          Map<String, dynamic>.from(map['preferences'] as Map? ?? const {}),
      xp: (map['xp'] is num) ? (map['xp'] as num).toInt() : 0,
      level: (map['level'] is num) ? (map['level'] as num).toInt() : 1,
      badges: (map['badges'] as List? ?? const [])
          .map((e) => e.toString())
          .toList(),
      currentStreak: (map['current_streak'] is num)
          ? (map['current_streak'] as num).toInt()
          : 0,
      longestStreak: (map['longest_streak'] is num)
          ? (map['longest_streak'] as num).toInt()
          : 0,
      totalSessions: (map['total_sessions'] is num)
          ? (map['total_sessions'] as num).toInt()
          : 0,
      perfectQuizzes: (map['perfect_quizzes'] is num)
          ? (map['perfect_quizzes'] as num).toInt()
          : 0,
    );
  }

  Map<String, dynamic> toFirestore() => {
        'uid': uid,
        'display_name': displayName,
        'email': email,
        'photo_url': photoUrl,
        'motivation_style': motivationStyle.index,
        'board': board,
        'subjects': subjects,
        'target_hours': targetStudyHours,
        'onboarding_complete': onboardingComplete,
        'created_at': createdAt.toIso8601String(),
        'preferences': preferences,
        'xp': xp,
        'level': level,
        'badges': badges,
        'current_streak': currentStreak,
        'longest_streak': longestStreak,
        'total_sessions': totalSessions,
        'perfect_quizzes': perfectQuizzes,
      };
}
