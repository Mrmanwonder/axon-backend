// lib/models/models.dart
// Core data models for Axon

import 'package:flutter/material.dart';
import '../services/study_catalog.dart';

/// Pristine Design System Constants
class PristinePalette {
  static const Color accent = Color(0xFF3A86FF);
  static const Color backgroundDark = Color(0xFF0A0A0A);
  static const Color glassWhite = Color(0xA6FFFFFF); // 65% White
  static const Color glassDark = Color(0xB30A0A0A); // 70% Dark
}

class StudySession {
  final String id;
  final DateTime date;
  final int durationMinutes;
  final String subject;
  final int breakCount;
  final double intensityIndex; // Focus quality score 0-1
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
  final double subjectDifficulty; // 0-1
  final double studyIntensity; // From timer pings
  final double mockScore; // 0-100 if taken
  final double predictedPerformance; // 0-1 XGBoost output
  final double syllabusCoverage; // 0-100
  final double activeStudyHours;
  final double targetStudyHours;
  final double focusRatio; // 0-1
  final int consistencyStreak;
  final double stressLevel; // 1-10

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

class _MotivationItem {
  final String situation;
  final String tone;
  final String subject;
  final String title;
  final String message;

  const _MotivationItem({
    required this.situation,
    required this.tone,
    required this.subject,
    required this.title,
    required this.message,
  });
}

class PerformanceState {
  static final List<_MotivationItem> _matrix = [
    // General - Streak Lost
    const _MotivationItem(
        situation: 'STREAK_LOST',
        tone: 'tough_love',
        subject: 'general',
        title: 'Zero days logged.',
        message:
            'Motivation follows action, not the other way around. Start a 15-minute timer now.'),
    const _MotivationItem(
        situation: 'STREAK_LOST',
        tone: 'positive',
        subject: 'general',
        title: 'Rest is part of the process.',
        message: "You're back now. Let's close one small loop today."),
    const _MotivationItem(
        situation: 'STREAK_LOST',
        tone: 'logic',
        subject: 'general',
        title: 'Data shows a clear path.',
        message:
            'Students who resume immediately after a missed day retain 80% more momentum. Open a chapter.'),

    // General - Pre Exam 14 Days
    const _MotivationItem(
        situation: 'PRE_EXAM_14_DAYS',
        tone: 'tough_love',
        subject: 'general',
        title: 'The window is closing.',
        message:
            'Stop reading notes. If you aren\'t doing timed past papers, you are wasting time.'),
    const _MotivationItem(
        situation: 'PRE_EXAM_14_DAYS',
        tone: 'positive',
        subject: 'general',
        title: 'Two weeks left.',
        message:
            'You have enough time to fix your weakest 3 topics. Let\'s lock them in today.'),
    const _MotivationItem(
        situation: 'PRE_EXAM_14_DAYS',
        tone: 'logic',
        subject: 'general',
        title: 'Transition strategy.',
        message:
            '80% active recall and 20% review yields the highest mark variance in the final 14 days.'),

    // General - Pre Exam 24 Hours
    const _MotivationItem(
        situation: 'PRE_EXAM_24_HOURS',
        tone: 'tough_love',
        subject: 'general',
        title: 'No new concepts.',
        message:
            'Review your flagged mistakes, memorize the formulas, and go to sleep. Exhaustion kills grades.'),
    const _MotivationItem(
        situation: 'PRE_EXAM_24_HOURS',
        tone: 'positive',
        subject: 'general',
        title: 'Trust your preparation.',
        message:
            'The hard work is in the vault. Review your highlights and rest your mind.'),

    // General - Post Low Score
    const _MotivationItem(
        situation: 'POST_LOW_SCORE',
        tone: 'tough_love',
        subject: 'general',
        title: 'Raw telemetry.',
        message:
            'A failing score is just raw telemetry. You now know exactly what you don\'t know. Fix the gaps.'),
    const _MotivationItem(
        situation: 'POST_LOW_SCORE',
        tone: 'positive',
        subject: 'general',
        title: 'Mistakes are practice marks.',
        message:
            'Mistakes in practice mean marks in the real exam. Flag those errors and turn them into strengths.'),
    const _MotivationItem(
        situation: 'POST_LOW_SCORE',
        tone: 'logic',
        subject: 'general',
        title: 'Error rate analysis.',
        message:
            'Your error rate is concentrated. Use the Feynman technique on these specific missed questions.'),

    // General - High Volatility
    const _MotivationItem(
        situation: 'HIGH_VOLATILITY',
        tone: 'tough_love',
        subject: 'general',
        title: 'Rollercoaster detected.',
        message:
            'Your scores are a rollercoaster. You are relying on luck and easy topics. Standardize your foundation.'),
    const _MotivationItem(
        situation: 'HIGH_VOLATILITY',
        tone: 'logic',
        subject: 'general',
        title: 'Stabilize with interleaving.',
        message:
            'High volatility detected. Switch to interleaved practice to stabilize your recall across all syllabus branches.'),

    // General - Late Night
    const _MotivationItem(
        situation: 'LATE_NIGHT',
        tone: 'tough_love',
        subject: 'general',
        title: 'Past 11PM. Stop.',
        message:
            "It's past 11PM. Cognitive load limits are breached. Go to sleep. A tired brain cannot encode new data."),
    const _MotivationItem(
        situation: 'LATE_NIGHT',
        tone: 'positive',
        subject: 'general',
        title: 'Incredible dedication.',
        message:
            'Finish this last cycle, then get some rest so your brain can process this.'),

    // General - Streak 7 Days
    const _MotivationItem(
        situation: 'STREAK_7_DAYS',
        tone: 'tough_love',
        subject: 'general',
        title: '7 days. Do not break the chain.',
        message:
            "Don't break the chain now. Complacency sets in at week one. Push harder today."),
    const _MotivationItem(
        situation: 'STREAK_7_DAYS',
        tone: 'positive',
        subject: 'general',
        title: 'One full week.',
        message:
            'One full week of relentless consistency. You are building an undeniable foundation. Keep it up.'),

    // General - Low Efficiency
    const _MotivationItem(
        situation: 'LOW_EFFICIENCY',
        tone: 'tough_love',
        subject: 'general',
        title: 'You paused 4 times in 20 minutes.',
        message:
            'Put your phone in another room. Deep work requires isolation.'),

    // General - Long Session No Break
    const _MotivationItem(
        situation: 'LONG_SESSION_NO_BREAK',
        tone: 'logic',
        subject: 'general',
        title: '90 minutes active.',
        message:
            "You've been active for 90 minutes. Diminishing returns have started. Take a 10-minute visual rest."),

    // General - High Mastery Revision
    const _MotivationItem(
        situation: 'HIGH_MASTERY_REVISION',
        tone: 'logic',
        subject: 'general',
        title: 'Stop reviewing. Pivot.',
        message:
            'Mastery > 85%. Stop reviewing this chapter. Your time is better spent on your weakest sub-topic.'),

    // Physics
    const _MotivationItem(
        situation: 'NEW_CHAPTER',
        tone: 'logic',
        subject: '9702',
        title: 'Physics needs derivation.',
        message:
            "Physics isn't about memorizing equations. Understand the units, the derivations, and the physical constraints first."),
    const _MotivationItem(
        situation: 'LOW_SCORE_MCQ',
        tone: 'tough_love',
        subject: '9702',
        title: 'Stop guessing.',
        message:
            "You're guessing on Paper 1. Draw the free body diagrams. Do the math. Don't rely on intuition."),
    const _MotivationItem(
        situation: 'FORMULA_STRUGGLE',
        tone: 'logic',
        subject: '9702',
        title: 'Derive, not stare.',
        message:
            'Stop staring at the formula sheet. Derive it. If you can derive it, you won\'t forget it under pressure.'),
    const _MotivationItem(
        situation: 'P4_PREP',
        tone: 'positive',
        subject: '9702',
        title: 'Paper 4 is repetitive.',
        message:
            "Paper 4 is heavy, but it's highly repetitive. Master the past 5 years of Quantum and Medical Physics questions."),
    const _MotivationItem(
        situation: 'CALCULATION_ERROR',
        tone: 'tough_love',
        subject: '9702',
        title: 'Check your prefixes.',
        message:
            'Milli, Micro, Nano. You understand the physics but you\'re bleeding marks to bad arithmetic.'),
    const _MotivationItem(
        situation: 'P3_PRACTICAL',
        tone: 'logic',
        subject: '9702',
        title: 'Paper 3 is free marks.',
        message:
            'Format your tables correctly. Decimal places must match the precision of your instrument.'),
    const _MotivationItem(
        situation: 'CONCEPT_BLOCK',
        tone: 'positive',
        subject: '9702',
        title: 'Fields and Waves are abstract.',
        message: 'Try visualizing the lines of force. You\'ll get this.'),
    const _MotivationItem(
        situation: 'TOPIC_KINEMATICS',
        tone: 'logic',
        subject: '9702',
        title: 'Define directions first.',
        message:
            'Always define your positive and negative directions before applying SUVAT equations.'),

    // Computer Science
    const _MotivationItem(
        situation: 'NEW_CHAPTER',
        tone: 'tough_love',
        subject: '9618',
        title: 'CS is not a spectator sport.',
        message:
            'Stop just reading the pseudocode. Write it out. Trace it on paper.'),
    const _MotivationItem(
        situation: 'P2_STRUGGLE',
        tone: 'logic',
        subject: '9618',
        title: 'Trace tables line by line.',
        message:
            'Track every variable line by line. Do not skip steps in your head.'),
    const _MotivationItem(
        situation: 'THEORY_HEAVY',
        tone: 'positive',
        subject: '9618',
        title: 'Terminology stacks up.',
        message:
            'Hardware and Networking have a lot of terminology. Use Leitner flashcards to make it stick permanently.'),
    const _MotivationItem(
        situation: 'P4_PREP',
        tone: 'tough_love',
        subject: '9618',
        title: 'Paper 4 needs coding speed.',
        message: 'Stop planning and start typing. Build the skeleton first.'),
    const _MotivationItem(
        situation: 'SQL_DATABASES',
        tone: 'logic',
        subject: '9618',
        title: 'Keys first.',
        message:
            'Always verify your Primary and Foreign keys before writing the DDL statements.'),
    const _MotivationItem(
        situation: 'BOOLEAN_ALGEBRA',
        tone: 'logic',
        subject: '9618',
        title: 'Draw the circuit first.',
        message:
            'Draw the logic gate circuit first, then derive the boolean expression from left to right.'),
    const _MotivationItem(
        situation: 'HIGH_MASTERY',
        tone: 'positive',
        subject: '9618',
        title: 'Your logic is sharp.',
        message:
            'Try explaining this algorithm using the Feynman technique to ensure deep mastery.'),

    // Mathematics
    const _MotivationItem(
        situation: 'NEW_CHAPTER',
        tone: 'tough_love',
        subject: '9709',
        title: 'Pick up the pen.',
        message:
            'You cannot read mathematics. Solve the first 10 questions of the exercise now.'),
    const _MotivationItem(
        situation: 'CALCULUS_BLOCK',
        tone: 'logic',
        subject: '9709',
        title: 'Integration by parts needs reps.',
        message:
            'Integration by parts is pure muscle memory. If you are stuck, you haven\'t done enough repetitions.'),
    const _MotivationItem(
        situation: 'STATS_PROBABILITY',
        tone: 'positive',
        subject: '9709',
        title: 'Highlight the keywords.',
        message:
            "Probability is tricky because of the wording. Highlight 'AND', 'OR', 'GIVEN THAT' in every question."),
    const _MotivationItem(
        situation: 'MECHANICS',
        tone: 'logic',
        subject: '9709',
        title: 'Resolve forces first.',
        message:
            'Resolve forces parallel and perpendicular to the plane immediately. Do not skip the diagram.'),
    const _MotivationItem(
        situation: 'CARELESS_MISTAKES',
        tone: 'tough_love',
        subject: '9709',
        title: 'You are dropping negative signs.',
        message: 'Slow down. Write out every algebraic step on a new line.'),
    const _MotivationItem(
        situation: 'TRIG_IDENTITIES',
        tone: 'logic',
        subject: '9709',
        title: 'Convert to sines and cosines.',
        message:
            'If a trig proof looks impossible, convert everything to sines and cosines as your baseline.'),

    // English
    const _MotivationItem(
        situation: 'ESSAY_PREP',
        tone: 'tough_love',
        subject: '0500',
        title: 'Never start without a plan.',
        message:
            'A good essay requires a skeleton. Never start writing without a 5-minute structural plan.'),
    const _MotivationItem(
        situation: 'READING_COMPREHENSION',
        tone: 'logic',
        subject: '0500',
        title: 'Read the questions first.',
        message:
            'Read the questions before you read the passage. Know exactly what data you are extracting.'),
    const _MotivationItem(
        situation: 'ANALYSIS_QUESTIONS',
        tone: 'positive',
        subject: '0500',
        title: 'Focus on effect.',
        message:
            "Don't just list techniques; explain *why* the writer used them."),

    // Further Maths
    const _MotivationItem(
        situation: 'NEW_CHAPTER',
        tone: 'tough_love',
        subject: '9231',
        title: 'Further Maths does not forgive passive learning.',
        message:
            'If you don\'t prove the theorem yourself, you don\'t know it.'),
    const _MotivationItem(
        situation: 'MATRICES',
        tone: 'logic',
        subject: '9231',
        title: 'Picture the transformation.',
        message:
            'Matrix transformations are visual. Picture the unit square mapping to verify your algebra.'),
    const _MotivationItem(
        situation: 'DIFFERENTIAL_EQ',
        tone: 'logic',
        subject: '9231',
        title: 'Identify the type first.',
        message:
            'Is it separable? Integrating factor? Homogeneous? Identify before diving in.'),

    // System messages
    const _MotivationItem(
        situation: 'DASHBOARD_EMPTY',
        tone: 'positive',
        subject: 'general',
        title: 'Blank slate.',
        message:
            'Every great grade starts with a single focused session. Tap the timer.'),
    const _MotivationItem(
        situation: 'VAULT_SCROLLING',
        tone: 'tough_love',
        subject: 'general',
        title: 'Stop scrolling.',
        message: 'Pick one. Download it. Solve it under timed conditions.'),
    const _MotivationItem(
        situation: 'MOCK_STARTED',
        tone: 'logic',
        subject: 'general',
        title: 'Simulation active.',
        message: 'Treat this exactly like the exam hall. No notes, no pauses.'),
    const _MotivationItem(
        situation: 'PLANNER_OVERLOAD',
        tone: 'tough_love',
        subject: 'general',
        title: '15 tasks is not realistic.',
        message: 'Pick the 3 most critical ones and ignore the rest.'),
    const _MotivationItem(
        situation: 'GOAL_HIT',
        tone: 'positive',
        subject: 'general',
        title: 'Target acquired.',
        message:
            'You hit your study hours for the day. The algorithm is updating your trajectory upward.'),
    const _MotivationItem(
        situation: 'MULTIPLE_FAILURES',
        tone: 'tough_love',
        subject: 'general',
        title: 'Strategy is broken.',
        message:
            'Repeated failures on the same topic. Switch to Feynman Technique immediately.'),
    const _MotivationItem(
        situation: 'LONG_BREAK',
        tone: 'positive',
        subject: 'general',
        title: 'Break time is over.',
        message:
            'Your brain has encoded the previous data. Time to open the loop again.'),
  ];

  static String getMessage({
    required double score,
    required MotivationStyle style,
    String? situation,
    String? subjectCode,
  }) {
    if (situation != null) {
      final items = _matrix
          .where((i) =>
              i.situation == situation &&
              i.tone == style.matrixKey &&
              (i.subject == 'general' || i.subject == subjectCode))
          .toList();
      if (items.isNotEmpty) {
        items.shuffle();
        return items.first.message;
      }
    }

    String state;
    if (score < 0.5) {
      state = 'poor';
    } else if (score < 0.7) {
      state = 'fair';
    } else if (score < 0.85) {
      state = 'good';
    } else {
      state = 'excellent';
    }

    final fallbackMessages = <String, Map<String, String>>{
      'poor': {
        'tough_love':
            "You've been coasting. With this sleep deficit and study pattern, you're setting yourself up to fail. The numbers don't lie — fix your schedule tonight.",
        'positive':
            "Today was tough, and that's okay. Your brain is absorbing more than you think. Rest well tonight, and tomorrow we rebuild — one focused hour at a time.",
        'logic':
            "Your sleep is 20% below your 7-day average, and screen time exceeded 4h. Axon calculates a 67% correlation between this pattern and poor test scores. Reduce screens by 2h for 48 hours.",
      },
      'fair': {
        'tough_love': "You're building something. Don't slow down now.",
        'positive': " Steady progress. Every hour compounds. Keep going.",
        'logic':
            "Your trajectory is positive. Consistency index: 71%. Maintain current pace.",
      },
      'good': {
        'tough_love': "Good. Now don't stop. Momentum is fragile.",
        'positive':
            "You're on fire! Your focus sessions are getting longer and scores climbing.",
        'logic':
            "7-day intensity index up 34%. Sleep consistency at 88%. One more strong session puts you in the Excellent bracket.",
      },
      'excellent': {
        'tough_love':
            "Peak form. Protect it. One bad night undoes three good ones.",
        'positive':
            "You're in the zone! Every metric is aligned. Remember this feeling.",
        'logic':
            "Sleep: 8.2h (103% of target). Study intensity: 0.87. Confidence: 91%. Peak efficiency.",
      },
    };

    return fallbackMessages[state]?[style.matrixKey] ?? 'Keep pushing forward.';
  }

  static String getTitle({
    required double score,
    required MotivationStyle style,
    String? situation,
    String? subjectCode,
  }) {
    if (situation != null) {
      final items = _matrix
          .where((i) =>
              i.situation == situation &&
              i.tone == style.matrixKey &&
              (i.subject == 'general' || i.subject == subjectCode))
          .toList();
      if (items.isNotEmpty) {
        items.shuffle();
        return items.first.title;
      }
    }

    String state;
    if (score < 0.5) {
      state = 'poor';
    } else if (score < 0.7) {
      state = 'fair';
    } else if (score < 0.85) {
      state = 'good';
    } else {
      state = 'excellent';
    }

    final fallbackTitles = <String, Map<String, String>>{
      'poor': {
        'tough_love': 'Time to get serious.',
        'positive': 'Every expert was once a beginner.',
        'logic': 'Performance gap detected.',
      },
      'fair': {
        'tough_love': 'Building momentum.',
        'positive': 'Steady progress.',
        'logic': 'Trajectory positive.',
      },
      'good': {
        'tough_love': "Good. Don't stop.",
        'positive': "You're on fire!",
        'logic': 'Positive trajectory confirmed.',
      },
      'excellent': {
        'tough_love': 'Peak form. Protect it.',
        'positive': 'Axon sees it: Excellent.',
        'logic': 'All systems nominal.',
      },
    };

    return fallbackTitles[state]?[style.matrixKey] ?? 'Axon Intelligence';
  }
}

class PdfQuestion {
  final int questionNumber;
  final String questionText;
  final List<PdfPart> parts;
  final double yPosition;
  final double xPosition;
  final double width;
  final double height;
  final int pageNumber;
  String? userAnswer;
  String? correctAnswer;
  String? feedback;
  bool? isCorrect;
  int? marksAvailable;
  int? marksAwarded;
  final String? contextText;
  final List<String> figurePaths;
  final List<PdfFigureRegion> figureRegions;
  final List<String>? figureBase64;
  final String subjectTag;
  final String chapterTag;
  final String topicTag;
  final String difficultyTag;
  final String paperType;
  final String boardTag;
  final String paperYear;
  final Map<String, dynamic> spatialMetadata;

  PdfQuestion({
    required this.questionNumber,
    required this.questionText,
    this.parts = const [],
    required this.yPosition,
    required this.xPosition,
    required this.width,
    required this.height,
    required this.pageNumber,
    this.userAnswer,
    this.correctAnswer,
    this.feedback,
    this.isCorrect,
    this.marksAvailable,
    this.marksAwarded,
    this.contextText,
    List<String> figurePaths = const [],
    List<PdfFigureRegion> figureRegions = const [],
    List<String>? figureBase64,
    this.subjectTag = '',
    this.chapterTag = '',
    this.topicTag = '',
    this.difficultyTag = '',
    this.paperType = '',
    this.boardTag = '',
    this.paperYear = '',
    Map<String, dynamic> spatialMetadata = const {},
  })  : figurePaths = List<String>.from(figurePaths),
        figureRegions = List<PdfFigureRegion>.from(figureRegions),
        figureBase64 =
            figureBase64 == null ? null : List<String>.from(figureBase64),
        spatialMetadata = Map<String, dynamic>.from(spatialMetadata);

  String get fullText {
    final partsText = parts.map((part) => part.text).join(' ');
    return [
      contextText ?? '',
      questionText,
      partsText,
    ].where((value) => value.trim().isNotEmpty).join(' ').trim();
  }

  String get toFullLaTeX {
    final segments = <String>[
      if ((contextText ?? '').trim().isNotEmpty) contextText!.trim(),
      if (questionText.trim().isNotEmpty) questionText.trim(),
      ...parts
          .where((part) => part.text.trim().isNotEmpty)
          .map((part) => r'\textbf{(' + part.label + r')} ' + part.text.trim()),
    ];
    return segments.join(r' \\ ').trim();
  }

  PdfQuestion copyWith({
    int? questionNumber,
    String? questionText,
    List<PdfPart>? parts,
    double? yPosition,
    double? xPosition,
    double? width,
    double? height,
    int? pageNumber,
    String? userAnswer,
    String? correctAnswer,
    String? feedback,
    bool? isCorrect,
    int? marksAvailable,
    int? marksAwarded,
    String? contextText,
    List<String>? figurePaths,
    List<PdfFigureRegion>? figureRegions,
    List<String>? figureBase64,
    String? subjectTag,
    String? chapterTag,
    String? topicTag,
    String? difficultyTag,
    String? paperType,
    String? boardTag,
    String? paperYear,
    Map<String, dynamic>? spatialMetadata,
  }) {
    return PdfQuestion(
      questionNumber: questionNumber ?? this.questionNumber,
      questionText: questionText ?? this.questionText,
      parts: parts ?? this.parts,
      yPosition: yPosition ?? this.yPosition,
      xPosition: xPosition ?? this.xPosition,
      width: width ?? this.width,
      height: height ?? this.height,
      pageNumber: pageNumber ?? this.pageNumber,
      userAnswer: userAnswer ?? this.userAnswer,
      correctAnswer: correctAnswer ?? this.correctAnswer,
      feedback: feedback ?? this.feedback,
      isCorrect: isCorrect ?? this.isCorrect,
      marksAvailable: marksAvailable ?? this.marksAvailable,
      marksAwarded: marksAwarded ?? this.marksAwarded,
      contextText: contextText ?? this.contextText,
      figurePaths: figurePaths ?? this.figurePaths,
      figureRegions: figureRegions ?? this.figureRegions,
      figureBase64: figureBase64 ?? this.figureBase64,
      subjectTag: subjectTag ?? this.subjectTag,
      chapterTag: chapterTag ?? this.chapterTag,
      topicTag: topicTag ?? this.topicTag,
      difficultyTag: difficultyTag ?? this.difficultyTag,
      paperType: paperType ?? this.paperType,
      boardTag: boardTag ?? this.boardTag,
      paperYear: paperYear ?? this.paperYear,
      spatialMetadata: spatialMetadata ?? this.spatialMetadata,
    );
  }

  factory PdfQuestion.fromJson(Map<String, dynamic> json) {
    int asInt(dynamic v) {
      if (v is int) return v;
      if (v is num) return v.toInt();
      return int.tryParse(v?.toString() ?? '') ?? 0;
    }

    double asDouble(dynamic v) {
      if (v is double) return v;
      if (v is num) return v.toDouble();
      return double.tryParse(v?.toString() ?? '') ?? 0.0;
    }

    String asString(dynamic v) {
      if (v == null) return '';
      return v.toString();
    }

    return PdfQuestion(
      questionNumber: asInt(
        json['question_number'] ?? json['questionNumber'] ?? json['number'],
      ),
      questionText: asString(
          json['question_text'] ?? json['questionText'] ?? json['text']),
      parts: (json['parts'] as List?)
              ?.map((e) => PdfPart.fromJson(Map<String, dynamic>.from(e)))
              .toList() ??
          const [],
      yPosition: asDouble(json['y_position'] ?? json['yPosition'] ?? json['y']),
      xPosition: asDouble(json['x_position'] ?? json['xPosition'] ?? json['x']),
      width: asDouble(json['width']),
      height: asDouble(json['height']),
      pageNumber:
          asInt(json['page_number'] ?? json['pageNumber'] ?? json['page']),
      contextText: asString(json['context_text'] ?? json['contextText']),
      figurePaths: (json['figure_paths'] as List? ??
              json['figurePaths'] as List? ??
              const [])
          .map((e) => e.toString())
          .where((e) => e.isNotEmpty)
          .toList(),
      figureRegions: (json['figure_regions'] as List? ??
              json['figureRegions'] as List? ??
              const [])
          .whereType<Map>()
          .map((e) => PdfFigureRegion.fromJson(Map<String, dynamic>.from(e)))
          .toList(),
      figureBase64: json['figureBase64'] == null
          ? null
          : (json['figureBase64'] as List).cast<String>(),
      subjectTag: asString(
          json['subjectTag'] ?? json['subject_tag'] ?? json['subject']),
      chapterTag: asString(
          json['chapterTag'] ?? json['chapter_tag'] ?? json['chapter']),
      topicTag:
          asString(json['topicTag'] ?? json['topic_tag'] ?? json['topic']),
      difficultyTag: asString(json['difficultyTag'] ??
          json['difficulty_tag'] ??
          json['difficulty']),
      paperType: asString(
          json['paperType'] ?? json['paper_type'] ?? json['paperType']),
      boardTag:
          asString(json['boardTag'] ?? json['board_tag'] ?? json['board']),
      paperYear:
          asString(json['paperYear'] ?? json['paper_year'] ?? json['year']),
      spatialMetadata: Map<String, dynamic>.from(
        json['spatial_metadata'] as Map? ??
            json['spatialMetadata'] as Map? ??
            const {},
      ),
    );
  }

  Map<String, dynamic> toJson() => {
        'question_number': questionNumber,
        'question_text': questionText,
        'parts': parts.map((p) => p.toJson()).toList(),
        'y_position': yPosition,
        'x_position': xPosition,
        'width': width,
        'height': height,
        'page_number': pageNumber,
        'userAnswer': userAnswer,
        'correctAnswer': correctAnswer,
        'feedback': feedback,
        'isCorrect': isCorrect,
        'marksAvailable': marksAvailable,
        'marksAwarded': marksAwarded,
        'context_text': contextText,
        'figure_paths': figurePaths,
        'figure_regions':
            figureRegions.map((region) => region.toJson()).toList(),
        'subjectTag': subjectTag,
        'chapterTag': chapterTag,
        'topicTag': topicTag,
        'difficultyTag': difficultyTag,
        'paperType': paperType,
        'boardTag': boardTag,
        'paperYear': paperYear,
        'spatial_metadata': spatialMetadata,
      };
}

class PdfFigureRegion {
  final int pageNumber;
  final double x;
  final double top;
  final double width;
  final double height;

  const PdfFigureRegion({
    required this.pageNumber,
    required this.x,
    required this.top,
    required this.width,
    required this.height,
  });

  factory PdfFigureRegion.fromJson(Map<String, dynamic> json) {
    double asDouble(dynamic v) {
      if (v is double) return v;
      if (v is num) return v.toDouble();
      return double.tryParse(v?.toString() ?? '') ?? 0.0;
    }

    int asInt(dynamic v) {
      if (v is int) return v;
      if (v is num) return v.toInt();
      return int.tryParse(v?.toString() ?? '') ?? 0;
    }

    return PdfFigureRegion(
      pageNumber:
          asInt(json['page_number'] ?? json['pageNumber'] ?? json['page']),
      x: asDouble(json['x']),
      top: asDouble(json['top'] ?? json['y']),
      width: asDouble(json['width']),
      height: asDouble(json['height']),
    );
  }

  Map<String, dynamic> toJson() => {
        'page_number': pageNumber,
        'x': x,
        'top': top,
        'width': width,
        'height': height,
      };
}

class PdfPart {
  final String label;
  final String text;
  const PdfPart({required this.label, required this.text});

  factory PdfPart.fromJson(Map<String, dynamic> json) => PdfPart(
        label: (json['label'] ?? '').toString(),
        text: (json['text'] ?? '').toString(),
      );

  Map<String, dynamic> toJson() => {
        'label': label,
        'text': text,
      };
}

class ExamEvent {
  final String board;
  final String subject;
  final String label;
  final DateTime startDate;
  final DateTime endDate;
  final String source;
  final String? loadState;

  const ExamEvent({
    required this.board,
    required this.subject,
    required this.label,
    required this.startDate,
    required this.endDate,
    required this.source,
    this.loadState,
  });

  factory ExamEvent.fromJson(Map<String, dynamic> json) => ExamEvent(
        board: (json['board'] ?? '').toString(),
        subject: (json['subject'] ?? '').toString(),
        label: (json['label'] ?? '').toString(),
        startDate: DateTime.tryParse(
                (json['start_date'] ?? json['startDate'] ?? '').toString()) ??
            DateTime.now(),
        endDate: DateTime.tryParse(
                (json['end_date'] ?? json['endDate'] ?? '').toString()) ??
            DateTime.now(),
        source: (json['source'] ?? '').toString(),
        loadState:
            (json['load_state'] ?? json['loadState'] ?? 'balanced').toString(),
      );

  Map<String, dynamic> toJson() => {
        'board': board,
        'subject': subject,
        'label': label,
        'start_date': startDate.toIso8601String().split('T').first,
        'end_date': endDate.toIso8601String().split('T').first,
        'source': source,
        if (loadState != null) 'load_state': loadState,
      };
}

class DailyExecutionTask {
  final String taskId;
  final String moduleName;
  final Duration estimatedTime;
  final String focusTechnique;
  final bool isHighPriority;
  final String subjectId;
  final String objectiveId;
  final String topic;
  final String startTime;
  final String recoveryNote;

  const DailyExecutionTask({
    required this.taskId,
    required this.moduleName,
    required this.estimatedTime,
    required this.focusTechnique,
    required this.isHighPriority,
    this.subjectId = '',
    this.objectiveId = '',
    this.topic = '',
    this.startTime = '',
    this.recoveryNote = '',
  });

  factory DailyExecutionTask.fromJson(Map<String, dynamic> json) =>
      DailyExecutionTask(
        taskId: (json['task_id'] ?? json['taskId'] ?? '').toString(),
        moduleName:
            (json['module_name'] ?? json['moduleName'] ?? '').toString(),
        estimatedTime: Duration(
          minutes: (json['estimated_minutes'] as num?)?.toInt() ??
              (json['estimatedMinutes'] as num?)?.toInt() ??
              0,
        ),
        focusTechnique: (json['focus_technique'] ??
                json['focusTechnique'] ??
                'Active Recall')
            .toString(),
        isHighPriority:
            json['is_high_priority'] == true || json['isHighPriority'] == true,
        subjectId: (json['subject_id'] ?? json['subjectId'] ?? '').toString(),
        objectiveId:
            (json['objective_id'] ?? json['objectiveId'] ?? '').toString(),
        topic: (json['topic'] ?? '').toString(),
        startTime: (json['start_time'] ?? json['startTime'] ?? '').toString(),
        recoveryNote:
            (json['recovery_note'] ?? json['recoveryNote'] ?? '').toString(),
      );

  Map<String, dynamic> toJson() => {
        'task_id': taskId,
        'module_name': moduleName,
        'estimated_minutes': estimatedTime.inMinutes,
        'focus_technique': focusTechnique,
        'is_high_priority': isHighPriority,
        'subject_id': subjectId,
        'objective_id': objectiveId,
        'topic': topic,
        'start_time': startTime,
        if (recoveryNote.isNotEmpty) 'recovery_note': recoveryNote,
      };
}

class PlannerTrigger {
  final String id;
  final DateTime fireAt;
  final String action;
  final Map<String, dynamic> payload;

  const PlannerTrigger({
    required this.id,
    required this.fireAt,
    required this.action,
    this.payload = const {},
  });

  factory PlannerTrigger.fromJson(Map<String, dynamic> json) => PlannerTrigger(
        id: (json['id'] ?? '').toString(),
        fireAt: DateTime.tryParse(
                (json['fire_at'] ?? json['fireAt'] ?? '').toString()) ??
            DateTime.now(),
        action: (json['action'] ?? '').toString(),
        payload: Map<String, dynamic>.from(json['payload'] as Map? ?? const {}),
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'fire_at': fireAt.toIso8601String(),
        'action': action,
        'payload': payload,
      };
}

class AxonDailyPlan {
  final DateTime date;
  final String subject;
  final String board;
  final String examLabel;
  final bool isExamDay;
  final double intensity;
  final String loadState;
  final double recommendedUnits;
  final String receipt;
  final List<DailyExecutionTask> executionTasks;
  final List<PlannerTrigger> triggers;
  final String focusMantra;
  final String stage;
  final String cognitiveLoadReceipt;

  const AxonDailyPlan({
    required this.date,
    required this.subject,
    required this.board,
    required this.examLabel,
    required this.isExamDay,
    required this.intensity,
    required this.loadState,
    required this.recommendedUnits,
    required this.receipt,
    this.executionTasks = const [],
    this.triggers = const [],
    this.focusMantra = '',
    this.stage = '',
    this.cognitiveLoadReceipt = '',
  });

  factory AxonDailyPlan.fromJson(Map<String, dynamic> json) => AxonDailyPlan(
        date: DateTime.tryParse((json['date'] ?? '').toString()) ??
            DateTime.now(),
        subject: (json['subject'] ?? '').toString(),
        board: (json['board'] ?? '').toString(),
        examLabel: (json['exam_label'] ?? json['examLabel'] ?? '').toString(),
        isExamDay: json['is_exam_day'] == true || json['isExamDay'] == true,
        intensity: (json['intensity'] is num)
            ? (json['intensity'] as num).toDouble()
            : double.tryParse('${json['intensity']}') ?? 0.0,
        loadState:
            (json['load_state'] ?? json['loadState'] ?? 'balanced').toString(),
        recommendedUnits: (json['recommended_units'] is num)
            ? (json['recommended_units'] as num).toDouble()
            : double.tryParse('${json['recommended_units']}') ?? 0.0,
        receipt: (json['receipt'] ?? '').toString(),
        executionTasks: (json['execution_tasks'] as List? ??
                json['executionTasks'] as List? ??
                const [])
            .whereType<Map>()
            .map((item) =>
                DailyExecutionTask.fromJson(Map<String, dynamic>.from(item)))
            .toList(),
        triggers: (json['triggers'] as List? ?? const [])
            .whereType<Map>()
            .map((item) =>
                PlannerTrigger.fromJson(Map<String, dynamic>.from(item)))
            .toList(),
        focusMantra:
            (json['focus_mantra'] ?? json['focusMantra'] ?? '').toString(),
        stage: (json['stage'] ?? '').toString(),
        cognitiveLoadReceipt: (json['cognitive_load_receipt'] ??
                json['cognitiveLoadReceipt'] ??
                '')
            .toString(),
      );

  Map<String, dynamic> toJson() => {
        'date': date.toIso8601String().split('T').first,
        'subject': subject,
        'board': board,
        'exam_label': examLabel,
        'is_exam_day': isExamDay,
        'intensity': intensity,
        'load_state': loadState,
        'recommended_units': recommendedUnits,
        'receipt': receipt,
        'execution_tasks': executionTasks.map((task) => task.toJson()).toList(),
        'triggers': triggers.map((trigger) => trigger.toJson()).toList(),
        if (focusMantra.isNotEmpty) 'focus_mantra': focusMantra,
        if (stage.isNotEmpty) 'stage': stage,
        if (cognitiveLoadReceipt.isNotEmpty)
          'cognitive_load_receipt': cognitiveLoadReceipt,
      };
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
