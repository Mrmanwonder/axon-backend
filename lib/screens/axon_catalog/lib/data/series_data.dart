import '../models/paper.dart';
import '../models/paper_series.dart';

// ─── Paper Template ────────────────────────────────────────────────────────
// sessions: 'm' = March only, 's' = Summer only, 'w' = Winter only,
//           'sw' = Summer + Winter, 'msw' = all three

class _PT {
  final SubjectGroup group;
  final int paper;
  final int variant;
  final String component;
  final int marks;
  final String sessions;

  const _PT(
      this.group, this.paper, this.variant, this.component, this.marks, this.sessions);
}

const List<_PT> _templates = [
  // ── CHEMISTRY 9701 ────────────────────────────────────────────────────────
  _PT(SubjectGroup.chemistry, 1, 1, 'Multiple Choice', 40, 'sw'),
  _PT(SubjectGroup.chemistry, 1, 2, 'Multiple Choice', 40, 'sw'),
  _PT(SubjectGroup.chemistry, 1, 3, 'Multiple Choice', 40, 'w'),
  _PT(SubjectGroup.chemistry, 2, 1, 'AS Level Structured Questions', 60, 'msw'),
  _PT(SubjectGroup.chemistry, 2, 2, 'AS Level Structured Questions', 60, 'msw'),
  _PT(SubjectGroup.chemistry, 2, 3, 'AS Level Structured Questions', 60, 'sw'),
  _PT(SubjectGroup.chemistry, 3, 1, 'Advanced Practical Skills 1', 40, 'sw'),
  _PT(SubjectGroup.chemistry, 3, 2, 'Advanced Practical Skills 1', 40, 'sw'),
  _PT(SubjectGroup.chemistry, 4, 1, 'A Level Structured Questions', 100, 'sw'),
  _PT(SubjectGroup.chemistry, 4, 2, 'A Level Structured Questions', 100, 'sw'),
  _PT(SubjectGroup.chemistry, 5, 1, 'Planning, Analysis & Evaluation', 30, 'sw'),
  _PT(SubjectGroup.chemistry, 5, 2, 'Planning, Analysis & Evaluation', 30, 'sw'),
  _PT(SubjectGroup.chemistry, 5, 3, 'Planning, Analysis & Evaluation', 30, 'sw'),

  // ── PHYSICS 9702 ──────────────────────────────────────────────────────────
  _PT(SubjectGroup.physics, 1, 1, 'Multiple Choice', 40, 'sw'),
  _PT(SubjectGroup.physics, 1, 2, 'Multiple Choice', 40, 'sw'),
  _PT(SubjectGroup.physics, 1, 3, 'Multiple Choice', 40, 'w'),
  _PT(SubjectGroup.physics, 2, 1, 'AS Level Structured Questions', 60, 'msw'),
  _PT(SubjectGroup.physics, 2, 2, 'AS Level Structured Questions', 60, 'msw'),
  _PT(SubjectGroup.physics, 2, 3, 'AS Level Structured Questions', 60, 'sw'),
  _PT(SubjectGroup.physics, 3, 1, 'Advanced Practical Skills', 40, 'sw'),
  _PT(SubjectGroup.physics, 3, 2, 'Advanced Practical Skills', 40, 'sw'),
  _PT(SubjectGroup.physics, 4, 1, 'A Level Structured Questions', 100, 'sw'),
  _PT(SubjectGroup.physics, 4, 2, 'A Level Structured Questions', 100, 'sw'),
  _PT(SubjectGroup.physics, 5, 1, 'Planning, Analysis & Evaluation', 30, 'sw'),
  _PT(SubjectGroup.physics, 5, 2, 'Planning, Analysis & Evaluation', 30, 'sw'),
  _PT(SubjectGroup.physics, 5, 3, 'Planning, Analysis & Evaluation', 30, 'sw'),

  // ── MATHEMATICS 9709 ──────────────────────────────────────────────────────
  _PT(SubjectGroup.mathematics, 1, 1, 'Pure Mathematics 1', 75, 'msw'),
  _PT(SubjectGroup.mathematics, 1, 2, 'Pure Mathematics 1', 75, 'sw'),
  _PT(SubjectGroup.mathematics, 1, 3, 'Pure Mathematics 1', 75, 'sw'),
  _PT(SubjectGroup.mathematics, 2, 1, 'Pure Mathematics 2', 50, 'sw'),
  _PT(SubjectGroup.mathematics, 2, 2, 'Pure Mathematics 2', 50, 'sw'),
  _PT(SubjectGroup.mathematics, 3, 1, 'Pure Mathematics 3', 75, 'msw'),
  _PT(SubjectGroup.mathematics, 3, 2, 'Pure Mathematics 3', 75, 'sw'),
  _PT(SubjectGroup.mathematics, 4, 1, 'Mechanics', 50, 'sw'),
  _PT(SubjectGroup.mathematics, 4, 2, 'Mechanics', 50, 'sw'),
  _PT(SubjectGroup.mathematics, 5, 1, 'Probability & Statistics 1', 50, 'msw'),
  _PT(SubjectGroup.mathematics, 5, 2, 'Probability & Statistics 1', 50, 'sw'),
  _PT(SubjectGroup.mathematics, 5, 3, 'Probability & Statistics 1', 50, 'sw'),
  _PT(SubjectGroup.mathematics, 6, 1, 'Probability & Statistics 2', 50, 'sw'),
  _PT(SubjectGroup.mathematics, 6, 2, 'Probability & Statistics 2', 50, 'sw'),

  // ── BIOLOGY 9700 ──────────────────────────────────────────────────────────
  _PT(SubjectGroup.biology, 1, 1, 'Multiple Choice', 40, 'sw'),
  _PT(SubjectGroup.biology, 1, 2, 'Multiple Choice', 40, 'sw'),
  _PT(SubjectGroup.biology, 1, 3, 'Multiple Choice', 40, 'w'),
  _PT(SubjectGroup.biology, 2, 1, 'AS Level Structured Questions', 60, 'sw'),
  _PT(SubjectGroup.biology, 2, 2, 'AS Level Structured Questions', 60, 'sw'),
  _PT(SubjectGroup.biology, 3, 1, 'Practical Exam', 40, 'sw'),
  _PT(SubjectGroup.biology, 3, 2, 'Practical Exam', 40, 'sw'),
  _PT(SubjectGroup.biology, 4, 1, 'A Level Structured Questions', 100, 'sw'),
  _PT(SubjectGroup.biology, 4, 2, 'A Level Structured Questions', 100, 'sw'),
  _PT(SubjectGroup.biology, 5, 1, 'Planning, Analysis & Evaluation', 30, 'sw'),
  _PT(SubjectGroup.biology, 5, 2, 'Planning, Analysis & Evaluation', 30, 'sw'),
  _PT(SubjectGroup.biology, 5, 3, 'Planning, Analysis & Evaluation', 30, 'sw'),

  // ── COMPUTER SCIENCE 9618 ─────────────────────────────────────────────────
  _PT(SubjectGroup.computerScience, 1, 1, 'Theory Fundamentals', 75, 'sw'),
  _PT(SubjectGroup.computerScience, 1, 2, 'Theory Fundamentals', 75, 'sw'),
  _PT(SubjectGroup.computerScience, 2, 1, 'Fundamental Problem Solving', 75, 'sw'),
  _PT(SubjectGroup.computerScience, 2, 2, 'Fundamental Problem Solving', 75, 'sw'),
  _PT(SubjectGroup.computerScience, 3, 1, 'Advanced Theory', 75, 'sw'),
  _PT(SubjectGroup.computerScience, 3, 2, 'Advanced Theory', 75, 'sw'),
  _PT(SubjectGroup.computerScience, 4, 1, 'Practical', 50, 'sw'),
  _PT(SubjectGroup.computerScience, 4, 2, 'Practical', 50, 'sw'),

  // ── ECONOMICS 9708 ────────────────────────────────────────────────────────
  _PT(SubjectGroup.economics, 1, 1, 'Multiple Choice AS Level', 30, 'sw'),
  _PT(SubjectGroup.economics, 1, 2, 'Multiple Choice AS Level', 30, 'sw'),
  _PT(SubjectGroup.economics, 2, 1, 'Data Response & Essays AS', 40, 'sw'),
  _PT(SubjectGroup.economics, 2, 2, 'Data Response & Essays AS', 40, 'sw'),
  _PT(SubjectGroup.economics, 3, 1, 'Multiple Choice A Level', 30, 'sw'),
  _PT(SubjectGroup.economics, 3, 2, 'Multiple Choice A Level', 30, 'sw'),
  _PT(SubjectGroup.economics, 4, 1, 'Data Response & Essays A Level', 70, 'sw'),
  _PT(SubjectGroup.economics, 4, 2, 'Data Response & Essays A Level', 70, 'sw'),

  // ── ENGLISH LANGUAGE 9093 ─────────────────────────────────────────────────
  _PT(SubjectGroup.english, 1, 1, 'Reading & Writing — AS', 50, 'sw'),
  _PT(SubjectGroup.english, 1, 2, 'Reading & Writing — AS', 50, 'sw'),
  _PT(SubjectGroup.english, 2, 1, 'Writing — AS', 50, 'sw'),
  _PT(SubjectGroup.english, 2, 2, 'Writing — AS', 50, 'sw'),
  _PT(SubjectGroup.english, 3, 1, 'Text Analysis — A Level', 50, 'sw'),
  _PT(SubjectGroup.english, 3, 2, 'Text Analysis — A Level', 50, 'sw'),
  _PT(SubjectGroup.english, 4, 1, 'Language Topics — A Level', 50, 'sw'),
  _PT(SubjectGroup.english, 4, 2, 'Language Topics — A Level', 50, 'sw'),
];

// ─── Series Definitions ────────────────────────────────────────────────────

class _SeriesDef {
  final String id;
  final int year;
  final Session session;
  const _SeriesDef(this.id, this.year, this.session);
}

const List<_SeriesDef> _seriesDefs = [
  _SeriesDef('m21', 2021, Session.march),
  _SeriesDef('s21', 2021, Session.summer),
  _SeriesDef('w21', 2021, Session.winter),
  _SeriesDef('m22', 2022, Session.march),
  _SeriesDef('s22', 2022, Session.summer),
  _SeriesDef('w22', 2022, Session.winter),
  _SeriesDef('m23', 2023, Session.march),
  _SeriesDef('s23', 2023, Session.summer),
  _SeriesDef('w23', 2023, Session.winter),
  _SeriesDef('m24', 2024, Session.march),
  _SeriesDef('s24', 2024, Session.summer),
  _SeriesDef('w24', 2024, Session.winter),
  _SeriesDef('m25', 2025, Session.march),
  _SeriesDef('s25', 2025, Session.summer),
  _SeriesDef('w25', 2025, Session.winter),
  _SeriesDef('s26', 2026, Session.summer),
];

// ─── Builder ───────────────────────────────────────────────────────────────

List<PaperSeries> buildAllSeries() {
  return _seriesDefs.map((def) {
    final sessionChar = def.session.shortCode;
    final papers = <Paper>[];

    for (final t in _templates) {
      if (t.sessions.contains(sessionChar)) {
        papers.add(Paper(
          id: '${t.group.subjectCode}_p${t.paper}v${t.variant}_${def.id}',
          subjectGroup: t.group,
          paperNumber: t.paper,
          variant: t.variant,
          component: t.component,
          totalMarks: t.marks,
          seriesId: def.id,
        ));
      }
    }

    return PaperSeries(
      id: def.id,
      year: def.year,
      session: def.session,
      papers: papers,
    );
  }).toList();
}
