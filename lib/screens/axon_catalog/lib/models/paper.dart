import 'package:flutter/material.dart';

// ─── Subject Group ─────────────────────────────────────────────────────────

enum SubjectGroup {
  chemistry,
  physics,
  mathematics,
  biology,
  computerScience,
  economics,
  english,
}

extension SubjectGroupExt on SubjectGroup {
  Color get color {
    switch (this) {
      case SubjectGroup.chemistry:      return const Color(0xFF2DD4BF);
      case SubjectGroup.physics:        return const Color(0xFF60A5FA);
      case SubjectGroup.mathematics:    return const Color(0xFFFBBF24);
      case SubjectGroup.biology:        return const Color(0xFF4ADE80);
      case SubjectGroup.computerScience:return const Color(0xFFA78BFA);
      case SubjectGroup.economics:      return const Color(0xFFF97316);
      case SubjectGroup.english:        return const Color(0xFFF472B6);
    }
  }

  Color get dimColor => color.withOpacity(0.18);

  String get label {
    switch (this) {
      case SubjectGroup.chemistry:      return 'Chemistry';
      case SubjectGroup.physics:        return 'Physics';
      case SubjectGroup.mathematics:    return 'Mathematics';
      case SubjectGroup.biology:        return 'Biology';
      case SubjectGroup.computerScience:return 'Comp. Sci.';
      case SubjectGroup.economics:      return 'Economics';
      case SubjectGroup.english:        return 'English';
    }
  }

  String get shortLabel {
    switch (this) {
      case SubjectGroup.chemistry:      return 'CHEM';
      case SubjectGroup.physics:        return 'PHYS';
      case SubjectGroup.mathematics:    return 'MATH';
      case SubjectGroup.biology:        return 'BIO';
      case SubjectGroup.computerScience:return 'CS';
      case SubjectGroup.economics:      return 'ECON';
      case SubjectGroup.english:        return 'ENG';
    }
  }

  String get subjectCode {
    switch (this) {
      case SubjectGroup.chemistry:      return '9701';
      case SubjectGroup.physics:        return '9702';
      case SubjectGroup.mathematics:    return '9709';
      case SubjectGroup.biology:        return '9700';
      case SubjectGroup.computerScience:return '9618';
      case SubjectGroup.economics:      return '9708';
      case SubjectGroup.english:        return '9093';
    }
  }
}

// ─── Paper ─────────────────────────────────────────────────────────────────

class Paper {
  final String id;
  final SubjectGroup subjectGroup;
  final int paperNumber;
  final int variant;
  final String component;
  final int totalMarks;
  final String seriesId;

  const Paper({
    required this.id,
    required this.subjectGroup,
    required this.paperNumber,
    required this.variant,
    required this.component,
    required this.totalMarks,
    required this.seriesId,
  });

  String get paperCode =>
      '${subjectGroup.subjectCode}/${paperNumber.toString().padLeft(2, '0')}$variant';

  String get fullReference => '$paperCode/${seriesId.toUpperCase()}';

  String get subject => subjectGroup.label;
}
