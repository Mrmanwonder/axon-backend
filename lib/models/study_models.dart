import 'base_models.dart';

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
            'Mastery > 85%. Stop reviewing this chapter. Your time is best spent on your weakest sub-topic.'),

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
        message: "Don't just list techniques; explain *why* the writer used them."),

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
