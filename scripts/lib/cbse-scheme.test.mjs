import test from "node:test";
import assert from "node:assert/strict";
import {
  assertOfficialCbseUrl,
  discoverCbseIndex,
  detectMarksColumn,
  extractMarkTotal,
  pairOfficialQuestions,
  parseSectionPlan,
  parseCbseHeader,
  parseQuestionBlocks,
} from "../lib/cbse-scheme.mjs";

test("discovers first-party CBSE SQP and MS pairs from an index row", () => {
  const html = [
    "<table><tr><th>Subject</th><th>SQP</th><th>MS</th></tr>",
    "<tr><td>Physics</td>",
    "<td><a href='/web_material/SQP/ClassXII_2026_27/Physics-SQP.pdf'>SQP</a></td>",
    "<td><a href='/web_material/SQP/ClassXII_2026_27/Physics-MS.pdf'>MS</a></td>",
    "</tr></table>",
  ].join("");
  assert.deepEqual(
    discoverCbseIndex(html, "https://cbseacademic.nic.in/SQP_CLASSXII_2026-27.html", 12),
    [{
      classLevel: 12,
      subject: "Physics",
      sqpUrl: "https://cbseacademic.nic.in/web_material/SQP/ClassXII_2026_27/Physics-SQP.pdf",
      msUrl: "https://cbseacademic.nic.in/web_material/SQP/ClassXII_2026_27/Physics-MS.pdf",
    }],
  );
});

test("rejects a non-CBSE source even if the filename looks official", () => {
  assert.throws(
    () => assertOfficialCbseUrl("https://example.com/web_material/SQP/Physics-MS.pdf"),
    /Refusing non-CBSE source/,
  );
});

test("parses CBSE subject code, class and ending year from a session header", () => {
  const header = parseCbseHeader([
    "Sample Question Paper",
    "Subject: Physics (042)",
    "Class – XII",
    "Academic Session 2026-27",
    "There are 33 questions in all.",
  ].join("\n"));
  assert.deepEqual(header, {
    subject: "Physics",
    subjectCode: "042",
    classLevel: 12,
    session: "2026-27",
    examYear: 2027,
    expectedQuestions: 33,
  });
});

test("extracts right-aligned mark totals including compound and fractional allocations", () => {
  assert.equal(extractMarkTotal(["step one                  1", "step two                  1"]), 2);
  assert.equal(extractMarkTotal(["award for method          2 + 1"]), 3);
  assert.equal(extractMarkTotal(["method                    ½", "result                    ½ x 2"]), 1.5);
  assert.equal(extractMarkTotal(["no numeric mark column here"]), null);
});

test("uses the detected Marks column and ignores formula numbers to its left", () => {
  const lines = [
    "Q.No.                 Question                                      Marks",
    "10.                    R = 100 / 60                                  1",
  ];
  const column = detectMarksColumn(lines);
  assert.ok(column !== null);
  assert.equal(extractMarkTotal([
    "                        100",
    "                         60",
    "                                                                     1",
  ], column), 1);
});

test("parses sequential question blocks and preserves alternatives", () => {
  const text = [
    "1. State one observation.                  1",
    "Answer line",
    "2. Calculate the value.                    2",
    "Working",
    "3 (A) Explain the trend.                   3",
    "Reason",
    "3 (B) Describe the graph.                  3",
    "Description",
    "4. Give the final answer.                  1",
  ].join("\n");
  const rows = parseQuestionBlocks(text);
  assert.deepEqual(rows.map(row => row.label), ["1", "2", "3(A)", "3(B)", "4"]);
  assert.deepEqual(rows.map(row => row.maxMarks), [1, 2, 3, 3, 1]);
});

test("accepts a standalone alternative label without losing later questions", () => {
  const rows = parseQuestionBlocks([
    "Q.No.                 Question                                      Marks",
    "1. First question                                                   1",
    "2(A) First choice                                                   2",
    "2(B)",
    "Second choice text                                                  2",
    "3. Final question                                                   1",
  ].join("\n"));
  assert.deepEqual(rows.map(row => row.label), ["1", "2(A)", "2(B)", "3"]);
  assert.deepEqual(rows.map(row => row.maxMarks), [1, 2, 2, 1]);
});

test("pairs only exact SQP/MS labels with matching mark totals", () => {
  const header = [
    "Subject: Physics (042)",
    "Class - XII",
    "Academic Session 2026-27",
    "There are 4 questions in all.",
    "Section A contains one question of 1 mark each. Section B contains one question of 2 marks each.",
    "Section C contains one question of 3 marks each. Section D contains one question of 1 mark each.",
  ].join("\n");

  const sqp = header + "\n" + [
    "1. State one observation.                  1",
    "2. Calculate the value.                    2",
    "3. Explain the trend.                      3",
    "4. Give the final answer.                  1",
  ].join("\n");

  const ms = header + "\n" + [
    "1. Accept the stated observation.           1",
    "2. One mark for method and one for result.     2",
    "3. Three valid linked points.               3",
    "4. Accept the stated result.                1",
  ].join("\n");

  const paired = pairOfficialQuestions(sqp, ms);
  assert.equal(paired.questions.length, 4);
  assert.equal(paired.coverage, 1);
  assert.deepEqual(paired.questions.map(row => row.maxMarks), [1, 2, 3, 1]);
});

test("fails closed when SQP and scheme identities disagree", () => {
  const sqp = [
    "Subject: Physics (042)",
    "Class - XII",
    "Academic Session 2026-27",
    "There are 1 questions in all.",
    "1. State one observation.                  1",
  ].join("\n");
  const ms = [
    "Subject: Chemistry (043)",
    "Class - XII",
    "Academic Session 2026-27",
    "There are 1 questions in all.",
    "1. Accept the observation.                 1",
  ].join("\n");
  assert.throws(() => pairOfficialQuestions(sqp, ms), /identity mismatch/);
});

test("fails closed when verified label/mark coverage is too low", () => {
  const header = [
    "Subject: Physics (042)",
    "Class - XII",
    "Academic Session 2026-27",
    "There are 4 questions in all.",
    "Section A contains four questions of 1 mark each.",
  ].join("\n");
  const sqp = header + "\n" + [
    "1. Question one.                            1",
    "2. Question two.                            1",
    "3. Question three.                          1",
    "4. Question four.                           1",
  ].join("\n");
  const ms = header + "\n" + [
    "1. Scheme one.                              1",
    "2. Scheme two has a conflicting allocation. 2",
  ].join("\n");
  assert.throws(() => pairOfficialQuestions(sqp, ms), /coverage/);
});

// The title blocks below are the first lines of the official 2026-27 CBSE
// SQP/MS PDFs (cbseacademic.nic.in), as `pdftotext -layout` renders them. Only
// the title block is used: no question or scheme text.
for (const [name, lines, want] of [
  ["Chemistry SQP", ["CHEMISTRY (CODE - 043)", "SAMPLE QUESTION PAPER*", "CLASS XII (2026-27)"], ["CHEMISTRY", "043", 12]],
  ["Chemistry MS", ["CHEMISTRY (CODE – 043)", "MARKING SCHEME", "CLASS XII (2026-27)"], ["CHEMISTRY", "043", 12]],
  ["Mathematics SQP", ["SUBJECT: MATHEMATICS (041)", "SAMPLE QUESTION PAPER", "CLASS- XII (2026 - 27)"], ["MATHEMATICS", "041", 12]],
  ["Mathematics MS", ["MATHEMATICS (041)", "MARKING SCHEME", "CLASS XII (2026-27)"], ["MATHEMATICS", "041", 12]],
  ["Biology SQP", ["BIOLOGY – CODE NO. 044", "SAMPLE QUESTION PAPER*", "CLASS – XII (2026-27)"], ["BIOLOGY", "044", 12]],
  ["Biology MS", ["BIOLOGY CODE NO. 044", "MARKING SCHEME", "CLASS – XII (2026–27)"], ["BIOLOGY", "044", 12]],
  ["Science X MS", ["SCIENCE – Code no. 086", "MARKING SCHEME", "CLASS – X (2026-27)"], ["SCIENCE", "086", 10]],
  ["Maths Standard X MS", ["MATHEMATICS STANDARD – Code No. (041)", "MARKING SCHEME", "CLASS X (2026 - 27)"], ["MATHEMATICS STANDARD", "041", 10]],
]) {
  test("parses the official 2026-27 header: " + name, () => {
    const header = parseCbseHeader(lines.join("\n"));
    assert.ok(header, "header not parsed");
    assert.deepEqual([header.subject, header.subjectCode, header.classLevel, header.session, header.examYear],
      [...want, "2026-27", 2027]);
  });
}

test("a subject-like phrase deep in the body is not read as the title", () => {
  const filler = Array.from({ length: 20 }, (_, i) => "Question " + (i + 1) + " text");
  assert.equal(parseCbseHeader([...filler, "A CLASS (2026-27) of 40 students", "PHYSICS (042)"].join("\n")), null);
});

// Section plans: sentences as they appear in the official 2026-27 General
// Instructions (Physics / Chemistry count style, Mathematics range style).
test("reads a count-style section plan", () => {
  const plan = parseSectionPlan([
    "Section A contains sixteen questions, twelve MCQ and four Assertion-Reasoning based of",
    "1mark each, Section B contains five questions of two marks each, Section C contains",
    "seven questions of three marks each, Section D contains two case study-based questions",
    "of four marks each and Section E contains three long answer questions of five marks each.",
  ].join("\n"));
  assert.equal(plan.size, 33);
  assert.deepEqual([plan.get(1), plan.get(16), plan.get(17), plan.get(22), plan.get(29), plan.get(33)], [1, 1, 2, 3, 4, 5]);
});

test("reads a range-style section plan, including 'and' pairs", () => {
  const plan = parseSectionPlan([
    "(iii) In Section A, Question number 1 to 18 are Multiple Choice Questions (MCQs) and Question",
    "number 19 and 20 are Assertion - Reason based questions of 1 mark each.",
    "(iv) In Section B, Question number 21 to 25 are Very Short Answer (VSA)-type questions, carrying",
    "2 marks each.",
    "(v) In Section C, Question number 26 to 31 are Short Answer (SA)-type questions carrying 3 marks",
    "(vi) In Section D, Question number 32 to 35 are Long Answer (LA)-type questions carrying 5 marks",
    "(vii) In Section E, Question number 36 to 38 are case study-based questions carrying 4 marks each.",
  ].join("\n"));
  assert.equal(plan.size, 38);
  assert.deepEqual([plan.get(20), plan.get(21), plan.get(31), plan.get(35), plan.get(36)], [1, 2, 3, 5, 4]);
});

test("a paper whose sections are subjects, not mark bands, has no readable plan", () => {
  assert.equal(parseSectionPlan("This question paper consists of 39 questions in 3 sections. Section A is Biology, Section B is Chemistry and Section C is Physics."), null);
});

test("a record whose marks disagree with the paper's own plan is dropped, never stored", () => {
  const header = [
    "Subject: Mathematics (041)", "Class - X", "Academic Session 2026-27", "There are 2 questions in all.",
    "Section A contains one question of 1 mark each. Section B contains one question of 4 marks each.",
  ].join("\n");
  // Q2's two case-study parts plus an internal-choice alternative summed to 6.
  const sqp = header + "\n" + ["1. One.                                   1", "2. Case study.                            6"].join("\n");
  const ms = header + "\n" + ["1. Scheme one.                            1", "2. Scheme two.                            6"].join("\n");
  assert.throws(() => pairOfficialQuestions(sqp, ms, 0.75), /coverage 50\.0%/);
  const paired = pairOfficialQuestions(sqp, ms, 0.5);
  assert.deepEqual(paired.questions.map(q => q.label), ["1"]);
});
