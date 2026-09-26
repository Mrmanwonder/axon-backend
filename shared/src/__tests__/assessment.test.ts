import { test } from "node:test";
import assert from "node:assert/strict";
import {
  candidateIsResolvable,
  identityMatchesCandidate,
  normaliseQuestionLabel,
  questionLabelAncestors,
  questionTerms,
  selectScopedCanonicalQuestion,
  resolveAssessmentIdentity,
  resolveSchemeEvidence,
  type AssessmentCandidate,
} from "../assessment.js";

const candidate: AssessmentCandidate = {
  subject_code: "9702",
  level: null,
  exam_year: 2025,
  session: "May/June",
  paper_code: "42",
  component_code: "9702/42",
  variant: "2",
  zone: null,
  assessment_route: null,
  confidence: "high",
};

test("assessment candidates require visible exact discriminators", () => {
  assert.equal(candidateIsResolvable(candidate), true);
  assert.equal(candidateIsResolvable({ ...candidate, confidence: "low" }), false);
  assert.equal(candidateIsResolvable({ ...candidate, subject_code: null }), false);
  assert.equal(candidateIsResolvable({ ...candidate, exam_year: null }), false);
  assert.equal(candidateIsResolvable({ ...candidate, paper_code: null, component_code: null }), false);
  assert.equal(candidateIsResolvable({
    ...candidate,
    paper_code: null,
    component_code: null,
    assessment_route: "sample_paper",
  }), true);
});

test("identity matching compares every stored discriminator, not semantic similarity", () => {
  const identity = {
    level: null,
    exam_year: 2025,
    session: "May June",
    paper_code: "42",
    component_code: "9702-42",
    variant: "2",
    zone: null,
    assessment_route: null,
  };
  assert.equal(identityMatchesCandidate(identity, candidate, null), true);
  assert.equal(identityMatchesCandidate({ ...identity, variant: "3" }, candidate, null), false);
  assert.equal(identityMatchesCandidate({ ...identity, session: "Oct/Nov" }, candidate, null), false);
});

test("student-selected IB level can complete an otherwise exact candidate", () => {
  const ib = {
    level: "HL",
    exam_year: 2026,
    session: "May",
    paper_code: "P1",
    component_code: null,
    variant: null,
    zone: null,
    assessment_route: null,
  };
  const transcribed: AssessmentCandidate = {
    subject_code: "100452",
    level: null,
    exam_year: 2026,
    session: "May",
    paper_code: "P1",
    component_code: null,
    variant: null,
    zone: null,
    assessment_route: null,
    confidence: "high",
  };
  assert.equal(identityMatchesCandidate(ib, transcribed, "HL"), true);
  assert.equal(identityMatchesCandidate(ib, transcribed, "SL"), false);
});

test("question labels are normalized only typographically", () => {
  assert.equal(normaliseQuestionLabel(" 3 (b) (ii). "), "3(B)(II)");
  assert.equal(normaliseQuestionLabel(null), null);
  assert.notEqual(normaliseQuestionLabel("3(b)(ii)"), normaliseQuestionLabel("3(b)(iii)"));
});


test("question-label ancestry is deterministic and only strips trailing parts", () => {
  assert.deepEqual(questionLabelAncestors("29 (b) (ii)"), ["29(B)(II)", "29(B)", "29"]);
  assert.deepEqual(questionLabelAncestors("29"), ["29"]);
  assert.deepEqual(questionLabelAncestors(null), []);
});


test("question text terms drop exam scaffolding but preserve subject content", () => {
  assert.deepEqual(
    [...questionTerms("Explain why the magnetic field changes when current increases.")].sort(),
    ["changes", "current", "field", "increases", "magnetic"].sort(),
  );
});

test("exact and ancestor labels win before any text fallback", () => {
  const rows = [
    { id: "q29", question_label: "29", question_text: "Describe magnetic flux density in the coil", max_marks: 3 },
    { id: "q30", question_label: "30", question_text: "Describe magnetic flux density in the coil", max_marks: 3 },
  ];
  const exact = selectScopedCanonicalQuestion(rows, {
    questionLabel: "30",
    questionText: "completely unrelated OCR text",
    marksAvailable: 3,
  });
  assert.equal(exact?.question.id, "q30");
  assert.equal(exact?.mode, "exact_label");

  const ancestor = selectScopedCanonicalQuestion(rows, {
    questionLabel: "29(a)(ii)",
    questionText: null,
    marksAvailable: 3,
  });
  assert.equal(ancestor?.question.id, "q29");
  assert.equal(ancestor?.mode, "ancestor_label");
});

test("scoped text fallback returns one strong in-assessment match", () => {
  const rows = [
    {
      id: "electric",
      question_label: "12",
      question_text: "Calculate the resistance of the lamp using the current and potential difference.",
      max_marks: 2,
    },
    {
      id: "waves",
      question_label: "13",
      question_text: "Determine the wavelength of the sound wave from its frequency and speed.",
      max_marks: 2,
    },
  ];
  const selected = selectScopedCanonicalQuestion(rows, {
    questionLabel: "Q?",
    questionText: "The potential difference and current of the lamp are shown. Calculate its resistance.",
    marksAvailable: 2,
  });
  assert.equal(selected?.question.id, "electric");
  assert.equal(selected?.mode, "scoped_text");
});

test("scoped text fallback fails closed on mark mismatch or ambiguity", () => {
  const markMismatch = selectScopedCanonicalQuestion([
    {
      id: "q1",
      question_label: "1",
      question_text: "Calculate resistance from current and potential difference for the lamp.",
      max_marks: 3,
    },
  ], {
    questionLabel: null,
    questionText: "Calculate the lamp resistance using current and potential difference.",
    marksAvailable: 2,
  });
  assert.equal(markMismatch, null);

  const ambiguous = selectScopedCanonicalQuestion([
    {
      id: "a",
      question_label: "1",
      question_text: "Explain how current changes when resistance of the circuit increases.",
      max_marks: 2,
    },
    {
      id: "b",
      question_label: "2",
      question_text: "Explain how current changes when resistance in the circuit decreases.",
      max_marks: 2,
    },
  ], {
    questionLabel: null,
    questionText: "Explain how current changes when resistance in the circuit changes.",
    marksAvailable: 2,
  });
  assert.equal(ambiguous, null);

  const tooShort = selectScopedCanonicalQuestion([
    { id: "c", question_label: "3", question_text: "Define momentum", max_marks: 1 },
  ], {
    questionLabel: null,
    questionText: "Define momentum",
    marksAvailable: 1,
  });
  assert.equal(tooShort, null);
});


class SchemeQuery {
  table: string;
  db: SchemeDb;
  filters: [string, unknown][] = [];

  constructor(db: SchemeDb, table: string) {
    this.db = db;
    this.table = table;
  }

  select() { return this; }
  eq(column: string, value: unknown) {
    this.filters.push([column, value]);
    this.db.calls.push({ table: this.table, column, value });
    return this;
  }

  rows() {
    return (this.db.tables[this.table] ?? []).filter(row =>
      this.filters.every(([column, value]) => row[column] === value)
    );
  }

  async maybeSingle() {
    const rows = this.rows();
    if (rows.length > 1) return { data: null, error: new Error("multiple rows") };
    return { data: rows[0] ?? null, error: null };
  }

  then(resolve: (value: any) => unknown, reject: (reason: unknown) => unknown) {
    return Promise.resolve({ data: this.rows(), error: null }).then(resolve, reject);
  }
}

class SchemeDb {
  tables: Record<string, any[]>;
  calls: { table: string; column: string; value: unknown }[] = [];

  constructor(tables: Record<string, any[]>) {
    this.tables = tables;
  }

  from(table: string) {
    return new SchemeQuery(this, table);
  }
}

function schemeDb(overrides: {
  questions?: any[];
  documents?: any[];
  policies?: any[];
} = {}) {
  return new SchemeDb({
    paper: [{ id: "paper-a", assessment_identity_id: "assessment-a" }],
    canonical_question: overrides.questions ?? [{
      id: "q-a",
      assessment_identity_id: "assessment-a",
      question_label: "12",
      question_text: "Calculate resistance from the lamp current and potential difference.",
      max_marks: 2,
      marking_scheme: "Use R = V / I.",
      scheme_source: "https://cbseacademic.nic.in/Physics-MS.pdf",
      scheme_version: "2026-27:test",
      scheme_document_id: "doc-a",
    }],
    scheme_document: overrides.documents ?? [{
      id: "doc-a",
      assessment_identity_id: "assessment-a",
      source_url: "https://cbseacademic.nic.in/Physics-MS.pdf",
      copyright_access_class: "public_official",
      extraction_status: "ready",
      policy_id: "policy-cbse",
      revoked_at: null,
      superseded_by_id: null,
    }],
    scheme_source_policy: overrides.policies ?? [{
      id: "policy-cbse",
      hostname: "cbseacademic.nic.in",
      copyright_access_class: "public_official",
      reproduction_permitted: true,
      active: true,
    }],
  });
}

test("scheme retrieval never crosses the exact assessment scope", async () => {
  const db = schemeDb({
    questions: [
      {
        id: "q-other",
        assessment_identity_id: "assessment-b",
        question_label: "99",
        question_text: "Calculate resistance from the lamp current and potential difference.",
        max_marks: 2,
        marking_scheme: "other assessment",
        scheme_source: "https://cbseacademic.nic.in/Other-MS.pdf",
        scheme_version: "other",
        scheme_document_id: "doc-other",
      },
      {
        id: "q-a",
        assessment_identity_id: "assessment-a",
        question_label: "12",
        question_text: "Determine the wavelength of a sound wave from frequency and speed.",
        max_marks: 2,
        marking_scheme: "assessment A",
        scheme_source: "https://cbseacademic.nic.in/Physics-MS.pdf",
        scheme_version: "2026-27:test",
        scheme_document_id: "doc-a",
      },
    ],
  });

  const result = await resolveSchemeEvidence(db as any, {
    paperId: "paper-a",
    questionLabel: null,
    questionText: "Calculate resistance from the lamp current and potential difference.",
    marksAvailable: 2,
  });

  assert.equal(result, null);
  assert.ok(db.calls.some(call =>
    call.table === "canonical_question"
      && call.column === "assessment_identity_id"
      && call.value === "assessment-a"
  ));
});

test("scheme retrieval returns structured immutable evidence for one scoped match", async () => {
  const db = schemeDb();
  const result = await resolveSchemeEvidence(db as any, {
    paperId: "paper-a",
    questionLabel: null,
    questionText: "The lamp current and potential difference are given. Calculate its resistance.",
    marksAvailable: 2,
  });

  assert.equal(result?.canonicalQuestionId, "q-a");
  assert.equal(result?.schemeDocumentId, "doc-a");
  assert.equal(result?.assessmentIdentityId, "assessment-a");
  assert.equal(result?.retrievalMode, "scoped_text");
  assert.equal(result?.sourceUrl, "https://cbseacademic.nic.in/Physics-MS.pdf");
});

test("scheme retrieval rejects revoked, superseded, or policy-disabled evidence", async () => {
  const revoked = schemeDb({
    documents: [{
      id: "doc-a",
      assessment_identity_id: "assessment-a",
      source_url: "https://cbseacademic.nic.in/Physics-MS.pdf",
      copyright_access_class: "public_official",
      extraction_status: "ready",
      policy_id: "policy-cbse",
      revoked_at: "2026-09-26T00:00:00Z",
      superseded_by_id: null,
    }],
  });
  assert.equal(await resolveSchemeEvidence(revoked as any, {
    paperId: "paper-a", questionLabel: "12", marksAvailable: 2,
  }), null);

  const superseded = schemeDb({
    documents: [{
      id: "doc-a",
      assessment_identity_id: "assessment-a",
      source_url: "https://cbseacademic.nic.in/Physics-MS.pdf",
      copyright_access_class: "public_official",
      extraction_status: "ready",
      policy_id: "policy-cbse",
      revoked_at: null,
      superseded_by_id: "doc-new",
    }],
  });
  assert.equal(await resolveSchemeEvidence(superseded as any, {
    paperId: "paper-a", questionLabel: "12", marksAvailable: 2,
  }), null);

  const inactivePolicy = schemeDb({
    policies: [{
      id: "policy-cbse",
      hostname: "cbseacademic.nic.in",
      copyright_access_class: "public_official",
      reproduction_permitted: true,
      active: false,
    }],
  });
  assert.equal(await resolveSchemeEvidence(inactivePolicy as any, {
    paperId: "paper-a", questionLabel: "12", marksAvailable: 2,
  }), null);
});


class AssessmentQuery {
  table: string;
  db: AssessmentDb;
  filters: ((row: any) => boolean)[] = [];
  patch: Record<string, unknown> | null = null;

  constructor(db: AssessmentDb, table: string) {
    this.db = db;
    this.table = table;
  }

  select() { return this; }
  eq(column: string, value: unknown) {
    this.filters.push(row => row[column] === value);
    return this;
  }
  not(column: string, operator: string, value: unknown) {
    if (operator === "is" && value === null) {
      this.filters.push(row => row[column] !== null && row[column] !== undefined);
      return this;
    }
    throw new Error("unsupported fake not() operation");
  }
  in(column: string, values: unknown[]) {
    this.filters.push(row => values.includes(row[column]));
    return this;
  }
  update(patch: Record<string, unknown>) {
    this.patch = patch;
    return this;
  }

  rows() {
    return (this.db.tables[this.table] ?? []).filter(row =>
      this.filters.every(filter => filter(row))
    );
  }

  result() {
    const rows = this.rows();
    if (this.patch) {
      for (const row of rows) Object.assign(row, this.patch);
    }
    return rows;
  }

  async maybeSingle() {
    const rows = this.result();
    if (rows.length > 1) return { data: null, error: new Error("multiple rows") };
    return { data: rows[0] ?? null, error: null };
  }

  then(resolve: (value: any) => unknown, reject: (reason: unknown) => unknown) {
    return Promise.resolve({ data: this.result(), error: null }).then(resolve, reject);
  }
}

class AssessmentDb {
  tables: Record<string, any[]>;
  constructor(tables: Record<string, any[]>) {
    this.tables = tables;
  }
  from(table: string) {
    return new AssessmentQuery(this, table);
  }
}

function assessmentDb(args: {
  programmeId?: string;
  offeringCode: string;
  selectedLevel?: "SL" | "HL" | null;
  identities: any[];
}) {
  const programmeId = args.programmeId ?? "programme";
  return new AssessmentDb({
    student: [{ id: "student", programme_id: programmeId }],
    student_subject: [{
      student_id: "student",
      subject_offering_id: "offering",
      selected_level: args.selectedLevel ?? null,
    }],
    subject_offering: [{
      id: "offering",
      programme_id: programmeId,
      external_code: args.offeringCode,
    }],
    assessment_identity: args.identities.map(identity => ({
      programme_id: programmeId,
      subject_offering_id: "offering",
      title: "fixture",
      official_source_url: "https://example.invalid/metadata",
      ...identity,
    })),
    paper: [{ id: "paper", student_id: "student", assessment_identity_id: null }],
  });
}

test("Cambridge resolver requires the exact year/session/component/variant identity", async () => {
  const db = assessmentDb({
    offeringCode: "9702",
    identities: [{
      id: "cambridge-42",
      level: null,
      exam_year: 2025,
      session: "May/June",
      paper_code: "42",
      component_code: "9702/42",
      variant: "2",
      zone: null,
      assessment_route: null,
    }],
  });
  const resolved = await resolveAssessmentIdentity(db as any, {
    studentId: "student",
    paperId: "paper",
    candidate,
  });
  assert.equal(resolved?.id, "cambridge-42");
  assert.equal(db.tables.paper[0].assessment_identity_id, "cambridge-42");

  const wrongYear = await resolveAssessmentIdentity(assessmentDb({
    offeringCode: "9702",
    identities: [{
      id: "cambridge-2025",
      level: null,
      exam_year: 2025,
      session: "May/June",
      paper_code: "42",
      component_code: "9702/42",
      variant: "2",
      zone: null,
      assessment_route: null,
    }],
  }) as any, {
    studentId: "student",
    paperId: "paper",
    candidate: { ...candidate, exam_year: 2024 },
  });
  assert.equal(wrongYear, null);

  const wrongSession = await resolveAssessmentIdentity(assessmentDb({
    offeringCode: "9702",
    identities: [{
      id: "cambridge-oct",
      level: null,
      exam_year: 2025,
      session: "Oct/Nov",
      paper_code: "42",
      component_code: "9702/42",
      variant: "2",
      zone: null,
      assessment_route: null,
    }],
  }) as any, {
    studentId: "student",
    paperId: "paper",
    candidate,
  });
  assert.equal(wrongSession, null);
});

test("CBSE resolver uses subject code plus explicit assessment route", async () => {
  const cbseCandidate: AssessmentCandidate = {
    subject_code: "042",
    level: null,
    exam_year: 2027,
    session: null,
    paper_code: null,
    component_code: null,
    variant: null,
    zone: null,
    assessment_route: "sample_paper",
    confidence: "high",
  };
  const db = assessmentDb({
    programmeId: "cbse-senior",
    offeringCode: "042",
    identities: [{
      id: "cbse-physics-sqp",
      level: null,
      exam_year: 2027,
      session: null,
      paper_code: null,
      component_code: null,
      variant: null,
      zone: null,
      assessment_route: "sample_paper",
    }],
  });
  assert.equal((await resolveAssessmentIdentity(db as any, {
    studentId: "student", paperId: "paper", candidate: cbseCandidate,
  }))?.id, "cbse-physics-sqp");

  const wrongSubject = assessmentDb({
    programmeId: "cbse-senior",
    offeringCode: "041",
    identities: [{
      id: "cbse-maths-sqp",
      level: null,
      exam_year: 2027,
      session: null,
      paper_code: null,
      component_code: null,
      variant: null,
      zone: null,
      assessment_route: "sample_paper",
    }],
  });
  assert.equal(await resolveAssessmentIdentity(wrongSubject as any, {
    studentId: "student", paperId: "paper", candidate: cbseCandidate,
  }), null);
});

test("IB resolver uses the selected SL/HL level and never upgrades the wrong level", async () => {
  const ibCandidate: AssessmentCandidate = {
    subject_code: "100452",
    level: null,
    exam_year: 2026,
    session: "May",
    paper_code: "P1",
    component_code: null,
    variant: null,
    zone: null,
    assessment_route: null,
    confidence: "high",
  };
  const hl = assessmentDb({
    programmeId: "ibdp",
    offeringCode: "100452",
    selectedLevel: "HL",
    identities: [{
      id: "ib-physics-hl-p1",
      level: "HL",
      exam_year: 2026,
      session: "May",
      paper_code: "P1",
      component_code: null,
      variant: null,
      zone: null,
      assessment_route: null,
    }],
  });
  assert.equal((await resolveAssessmentIdentity(hl as any, {
    studentId: "student", paperId: "paper", candidate: ibCandidate,
  }))?.id, "ib-physics-hl-p1");

  const slSelection = assessmentDb({
    programmeId: "ibdp",
    offeringCode: "100452",
    selectedLevel: "SL",
    identities: [{
      id: "ib-physics-hl-p1",
      level: "HL",
      exam_year: 2026,
      session: "May",
      paper_code: "P1",
      component_code: null,
      variant: null,
      zone: null,
      assessment_route: null,
    }],
  });
  assert.equal(await resolveAssessmentIdentity(slSelection as any, {
    studentId: "student", paperId: "paper", candidate: ibCandidate,
  }), null);
});

test("ambiguous exact assessment rows remain unbound", async () => {
  const duplicate = {
    level: null,
    exam_year: 2025,
    session: "May/June",
    paper_code: "42",
    component_code: "9702/42",
    variant: "2",
    zone: null,
    assessment_route: null,
  };
  const db = assessmentDb({
    offeringCode: "9702",
    identities: [
      { id: "duplicate-a", ...duplicate },
      { id: "duplicate-b", ...duplicate },
    ],
  });
  assert.equal(await resolveAssessmentIdentity(db as any, {
    studentId: "student", paperId: "paper", candidate,
  }), null);
  assert.equal(db.tables.paper[0].assessment_identity_id, null);
});

test("scheme retrieval rejects missing and restricted documents", async () => {
  const missing = schemeDb({ documents: [] });
  assert.equal(await resolveSchemeEvidence(missing as any, {
    paperId: "paper-a", questionLabel: "12", marksAvailable: 2,
  }), null);

  const restricted = schemeDb({
    documents: [{
      id: "doc-a",
      assessment_identity_id: "assessment-a",
      source_url: "https://ibo.org/restricted-markscheme.pdf",
      copyright_access_class: "metadata_only",
      extraction_status: "ready",
      policy_id: "policy-ib",
      revoked_at: null,
      superseded_by_id: null,
    }],
    policies: [{
      id: "policy-ib",
      hostname: "ibo.org",
      copyright_access_class: "metadata_only",
      reproduction_permitted: false,
      active: true,
    }],
  });
  assert.equal(await resolveSchemeEvidence(restricted as any, {
    paperId: "paper-a", questionLabel: "12", marksAvailable: 2,
  }), null);
});
