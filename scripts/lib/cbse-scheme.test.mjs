import test from "node:test";
import assert from "node:assert/strict";
import {
  assertOfficialCbseUrl,
  discoverCbseIndex,
  extractMarkTotal,
  pairOfficialQuestions,
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

test("extracts right-aligned mark totals including compound allocations", () => {
  assert.equal(extractMarkTotal(["step one                  1", "step two                  1"]), 2);
  assert.equal(extractMarkTotal(["award for method          2 + 1"]), 3);
  assert.equal(extractMarkTotal(["no numeric mark column here"]), null);
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

test("pairs only exact SQP/MS labels with matching mark totals", () => {
  const header = [
    "Subject: Physics (042)",
    "Class - XII",
    "Academic Session 2026-27",
    "There are 4 questions in all.",
  ].join("\n");

  const sqp = header + "\n" + [
    "1. State one observation.                  1",
    "2. Calculate the value.                    2",
    "3. Explain the trend.                      3",
    "4. Give the final answer.                  1",
  ].join("\n");

  const ms = header + "\n" + [
    "1. Accept the stated observation.           1",
    "2. One mark for method and one for result.  2",
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
