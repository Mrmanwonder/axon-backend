import 'package:flutter/foundation.dart';

@immutable
class CaiePaper {
  final String id;
  final String subject;
  final String subjectCode;
  final int year;
  final String session;
  final String variant;
  final List<String> topics;
  final bool downloaded;
  final bool bookmarked;
  final bool solved;
  final double accuracy;
  final double difficulty;
  final DateTime? lastOpened;
  final int openCount;

  const CaiePaper({
    required this.id,
    required this.subject,
    required this.subjectCode,
    required this.year,
    required this.session,
    required this.variant,
    this.topics = const [],
    this.downloaded = false,
    this.bookmarked = false,
    this.solved = false,
    this.accuracy = 0.0,
    this.difficulty = 0.5,
    this.lastOpened,
    this.openCount = 0,
  });

  String get displayName => '$subjectCode $session $year $variant';
  String get shortName => '$subjectCode $session$year $variant';

  CaiePaper copyWith({
    String? id,
    String? subject,
    String? subjectCode,
    int? year,
    String? session,
    String? variant,
    List<String>? topics,
    bool? downloaded,
    bool? bookmarked,
    bool? solved,
    double? accuracy,
    double? difficulty,
    DateTime? lastOpened,
    int? openCount,
    bool clearLastOpened = false,
  }) {
    return CaiePaper(
      id: id ?? this.id,
      subject: subject ?? this.subject,
      subjectCode: subjectCode ?? this.subjectCode,
      year: year ?? this.year,
      session: session ?? this.session,
      variant: variant ?? this.variant,
      topics: topics ?? this.topics,
      downloaded: downloaded ?? this.downloaded,
      bookmarked: bookmarked ?? this.bookmarked,
      solved: solved ?? this.solved,
      accuracy: accuracy ?? this.accuracy,
      difficulty: difficulty ?? this.difficulty,
      lastOpened: clearLastOpened ? null : (lastOpened ?? this.lastOpened),
      openCount: openCount ?? this.openCount,
    );
  }

  Map<String, dynamic> toMap() => {
        'id': id,
        'subject': subject,
        'subject_code': subjectCode,
        'year': year,
        'session': session,
        'variant': variant,
        'topics': topics.join(','),
        'downloaded': downloaded ? 1 : 0,
        'bookmarked': bookmarked ? 1 : 0,
        'solved': solved ? 1 : 0,
        'accuracy': accuracy,
        'difficulty': difficulty,
        'last_opened': lastOpened?.millisecondsSinceEpoch,
        'open_count': openCount,
      };

  factory CaiePaper.fromMap(Map<String, dynamic> map) => CaiePaper(
        id: map['id'] as String,
        subject: map['subject'] as String? ?? '',
        subjectCode: map['subject_code'] as String? ?? '',
        year: map['year'] as int? ?? 0,
        session: map['session'] as String? ?? '',
        variant: map['variant'] as String? ?? '',
        topics: (map['topics'] as String? ?? '').split(',').where((t) => t.isNotEmpty).toList(),
        downloaded: (map['downloaded'] as int? ?? 0) == 1,
        bookmarked: (map['bookmarked'] as int? ?? 0) == 1,
        solved: (map['solved'] as int? ?? 0) == 1,
        accuracy: (map['accuracy'] as num?)?.toDouble() ?? 0.0,
        difficulty: (map['difficulty'] as num?)?.toDouble() ?? 0.5,
        lastOpened: map['last_opened'] != null
            ? DateTime.fromMillisecondsSinceEpoch(map['last_opened'] as int)
            : null,
        openCount: map['open_count'] as int? ?? 0,
      );
}

class CaieQuestionRecord {
  final String id;
  final String paperId;
  final String topicId;
  final String topicName;
  final bool correct;
  final double timeSpentSeconds;
  final DateTime attemptedAt;

  const CaieQuestionRecord({
    required this.id,
    required this.paperId,
    required this.topicId,
    required this.topicName,
    required this.correct,
    required this.timeSpentSeconds,
    required this.attemptedAt,
  });

  Map<String, dynamic> toMap() => {
        'id': id,
        'paper_id': paperId,
        'topic_id': topicId,
        'topic_name': topicName,
        'correct': correct ? 1 : 0,
        'time_spent': timeSpentSeconds,
        'attempted_at': attemptedAt.millisecondsSinceEpoch,
      };

  factory CaieQuestionRecord.fromMap(Map<String, dynamic> map) =>
      CaieQuestionRecord(
        id: map['id'] as String,
        paperId: map['paper_id'] as String? ?? '',
        topicId: map['topic_id'] as String? ?? '',
        topicName: map['topic_name'] as String? ?? '',
        correct: (map['correct'] as int? ?? 0) == 1,
        timeSpentSeconds: (map['time_spent'] as num?)?.toDouble() ?? 0.0,
        attemptedAt: DateTime.fromMillisecondsSinceEpoch(
            map['attempted_at'] as int? ?? 0),
      );
}

class CaieTopicPerformance {
  final String topicId;
  final String topicName;
  final String subjectCode;
  final int attempted;
  final int correct;
  final double accuracy;

  const CaieTopicPerformance({
    required this.topicId,
    required this.topicName,
    required this.subjectCode,
    required this.attempted,
    required this.correct,
    required this.accuracy,
  });

  bool get isWeak => accuracy < 0.5;
  bool get isStrong => accuracy >= 0.7;
}

class CaieRecommendation {
  final String paperId;
  final String displayName;
  final String reason;
  final double score;

  const CaieRecommendation({
    required this.paperId,
    required this.displayName,
    required this.reason,
    required this.score,
  });
}

class CaieTopic {
  final String id;
  final String name;
  final String subjectCode;
  final List<String> subtopics;

  const CaieTopic({
    required this.id,
    required this.name,
    required this.subjectCode,
    this.subtopics = const [],
  });
}

const Map<String, List<CaieTopic>> kCaieTopicMap = {
  '0625': [
    CaieTopic(id: '0625_gen', name: 'General Physics', subjectCode: '0625'),
    CaieTopic(id: '0625_kin', name: 'Kinematics', subjectCode: '0625'),
    CaieTopic(id: '0625_dyn', name: 'Dynamics', subjectCode: '0625'),
    CaieTopic(id: '0625_for', name: 'Forces', subjectCode: '0625'),
    CaieTopic(id: '0625_ene', name: 'Energy', subjectCode: '0625'),
    CaieTopic(id: '0625_wav', name: 'Waves', subjectCode: '0625'),
    CaieTopic(id: '0625_ele', name: 'Electricity & Magnetism', subjectCode: '0625'),
    CaieTopic(id: '0625_ato', name: 'Atomic Physics', subjectCode: '0625'),
    CaieTopic(id: '0625_thermal', name: 'Thermal Physics', subjectCode: '0625'),
  ],
  '9702': [
    CaieTopic(id: '9702_phy', name: 'Physical Quantities & Units', subjectCode: '9702'),
    CaieTopic(id: '9702_kin', name: 'Kinematics', subjectCode: '9702'),
    CaieTopic(id: '9702_dyn', name: 'Dynamics', subjectCode: '9702'),
    CaieTopic(id: '9702_for', name: 'Forces, Density & Pressure', subjectCode: '9702'),
    CaieTopic(id: '9702_wor', name: 'Work, Energy & Power', subjectCode: '9702'),
    CaieTopic(id: '9702_def', name: 'Deformation of Solids', subjectCode: '9702'),
    CaieTopic(id: '9702_wav', name: 'Waves', subjectCode: '9702'),
    CaieTopic(id: '9702_sup', name: 'Superposition', subjectCode: '9702'),
    CaieTopic(id: '9702_ele', name: 'Electricity', subjectCode: '9702'),
    CaieTopic(id: '9702_dce', name: 'D.C. Circuits', subjectCode: '9702'),
    CaieTopic(id: '9702_particle', name: 'Particle Physics', subjectCode: '9702'),
    CaieTopic(id: '9702_mag', name: 'Magnetism & Electromagnetism', subjectCode: '9702'),
    CaieTopic(id: '9702_thermal', name: 'Thermal Physics', subjectCode: '9702'),
    CaieTopic(id: '9702_nuclear', name: 'Nuclear Physics', subjectCode: '9702'),
    CaieTopic(id: '9702_quantum', name: 'Quantum Physics', subjectCode: '9702'),
  ],
  '9701': [
    CaieTopic(id: '9701_atomic', name: 'Atomic Structure', subjectCode: '9701'),
    CaieTopic(id: '9701_bond', name: 'Chemical Bonding', subjectCode: '9701'),
    CaieTopic(id: '9701_org', name: 'Organic Chemistry', subjectCode: '9701'),
    CaieTopic(id: '9701_phy_chem', name: 'Physical Chemistry', subjectCode: '9701'),
    CaieTopic(id: '9701_inorg', name: 'Inorganic Chemistry', subjectCode: '9701'),
    CaieTopic(id: '9701_eq', name: 'Equilibria', subjectCode: '9701'),
    CaieTopic(id: '9701_kinetics', name: 'Kinetics', subjectCode: '9701'),
    CaieTopic(id: '9701_thermo', name: 'Thermodynamics', subjectCode: '9701'),
    CaieTopic(id: '9701_analytic', name: 'Analytical Chemistry', subjectCode: '9701'),
  ],
  '9700': [
    CaieTopic(id: '9700_cell', name: 'Cell Structure', subjectCode: '9700'),
    CaieTopic(id: '9700_mol', name: 'Biological Molecules', subjectCode: '9700'),
    CaieTopic(id: '9700_enzymes', name: 'Enzymes', subjectCode: '9700'),
    CaieTopic(id: '9700_transport', name: 'Cell Transport', subjectCode: '9700'),
    CaieTopic(id: '9700_gas', name: 'Gas Exchange', subjectCode: '9700'),
    CaieTopic(id: '9700_genetics', name: 'Genetics', subjectCode: '9700'),
    CaieTopic(id: '9700_eco', name: 'Ecology', subjectCode: '9700'),
    CaieTopic(id: '9700_evo', name: 'Evolution', subjectCode: '9700'),
    CaieTopic(id: '9700_immuno', name: 'Immunology', subjectCode: '9700'),
    CaieTopic(id: '9700_homeo', name: 'Homeostasis', subjectCode: '9700'),
    CaieTopic(id: '9700_reprod', name: 'Reproduction', subjectCode: '9700'),
  ],
  '0580': [
    CaieTopic(id: '0580_num', name: 'Number', subjectCode: '0580'),
    CaieTopic(id: '0580_algebra', name: 'Algebra', subjectCode: '0580'),
    CaieTopic(id: '0580_geo', name: 'Geometry', subjectCode: '0580'),
    CaieTopic(id: '0580_trig', name: 'Trigonometry', subjectCode: '0580'),
    CaieTopic(id: '0580_prob', name: 'Probability', subjectCode: '0580'),
    CaieTopic(id: '0580_stats', name: 'Statistics', subjectCode: '0580'),
    CaieTopic(id: '0580_func', name: 'Functions', subjectCode: '0580'),
    CaieTopic(id: '0580_vec', name: 'Vectors & Transformations', subjectCode: '0580'),
  ],
  '0478': [
    CaieTopic(id: '0478_data', name: 'Data Representation', subjectCode: '0478'),
    CaieTopic(id: '0478_comms', name: 'Communication & Internet', subjectCode: '0478'),
    CaieTopic(id: '0478_hardware', name: 'Hardware & Software', subjectCode: '0478'),
    CaieTopic(id: '0478_sec', name: 'Security', subjectCode: '0478'),
    CaieTopic(id: '0478_algo', name: 'Algorithms', subjectCode: '0478'),
    CaieTopic(id: '0478_prog', name: 'Programming', subjectCode: '0478'),
    CaieTopic(id: '0478_db', name: 'Databases', subjectCode: '0478'),
    CaieTopic(id: '0478_ai', name: 'AI & Ethics', subjectCode: '0478'),
  ],
};
