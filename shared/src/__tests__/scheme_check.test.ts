import { test } from "node:test";
import assert from "node:assert/strict";
import { runSchemeCheckQuestion, failSchemeCheckQuestion } from "../scheme_check.js";

/* The per-question stage writes region_check only, and closes the paper once
   every readable question has a row. Synthetic rows; no model call (a question
   with no scheme section is recorded as an honest gap). */

function fakeDb(regionCount: number) {
  const regionChecks = new Map<string, any>();
  const paperUpdates: any[] = [];
  const touched = new Set<string>();
  const db = {
    regionChecks, paperUpdates, touched,
    from(table: string) {
      touched.add(table);
      const q: any = { filters: [] as any[] };
      const chain: any = {
        select: () => chain, eq: (c: string, v: unknown) => { q.filters.push([c, v]); return chain; },
        or: () => chain, order: () => chain, limit: () => chain,
        maybeSingle: async () => {
          if (table === "question_region") return { data: { id: "r1", paper_id: "p", student_id: "s", question_label: "1(a)", question_text: "Q", student_answer: "A", marks_available: 2 }, error: null };
          return { data: null, error: null };
        },
        upsert: (row: any) => { regionChecks.set(row.region_id, row); return Promise.resolve({ data: null, error: null }); },
        update: (patch: any) => { paperUpdates.push(patch); return { eq: async () => ({ data: null, error: null }) }; },
        then: (resolve: any) => {
          if (table === "question_region") return resolve({ data: Array.from({ length: regionCount }, (_, i) => ({ id: `r${i + 1}` })), error: null });
          if (table === "region_check") return resolve({ data: [...regionChecks.values()].map((r) => ({ can_check: r.can_check })), error: null });
          return resolve({ data: [], error: null });
        },
      };
      return chain;
    },
  };
  return db;
}

test("a question with no scheme section is recorded as an unchecked gap, never a mark", async () => {
  const sb = fakeDb(2);
  const out = await runSchemeCheckQuestion({ env: {} as any, sb: sb as any, msg: { run_id: "run", region_id: "r1", paper: "9231/11", section: null, conventions: null } });
  assert.equal(out.can_check, false);
  const row = sb.regionChecks.get("r1");
  assert.equal(row.can_check, false);
  assert.equal(row.estimated_marks, null);
  assert.equal(row.confidence, "unsure");
  assert.ok(!sb.touched.has("teacher_mark") && !sb.touched.has("region_explanation") && !sb.touched.has("mark_loss_event"));
  assert.equal(sb.paperUpdates.length, 0, "paper stays open until every question has a row");
});

test("the last question closes the paper as done", async () => {
  const sb = fakeDb(1);
  await runSchemeCheckQuestion({ env: {} as any, sb: sb as any, msg: { run_id: "run", region_id: "r1", paper: "9231/11", section: null, conventions: null } });
  assert.equal(sb.paperUpdates.at(-1)?.status, "done");
  assert.equal(sb.paperUpdates.at(-1)?.checked, 0);
});

test("a permanently failed question still gets a row so the paper can finish", async () => {
  const sb = fakeDb(1);
  await failSchemeCheckQuestion(sb as any, { scheme_check_q: { run_id: "run", region_id: "r1", paper: "x", section: "s", conventions: null } }, new Error("boom"));
  assert.equal(sb.regionChecks.get("r1").reason, "This question could not be checked.");
  assert.equal(sb.paperUpdates.at(-1)?.status, "done");
});
