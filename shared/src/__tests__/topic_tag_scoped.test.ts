import { test } from "node:test";
import assert from "node:assert/strict";
import { questionTerms, refinementTerms } from "../syllabus_search.js";
import { validate as validateShortlist } from "../prompts/topic_tag_shortlist.v1.js";
import { tagRegion } from "../topic_tag.js";

// AXO-226: the scoped tagger. The database half (scope, rights, any-term
// search, unavailable state) is proved in Axon-Site tests/db/scoped-topic-tagging.test.mjs.

test("search terms come from the printed question, without command words or bare numbers", () => {
  const t = questionTerms("Calculate the acceleration of the trolleys. Use the equations of motion. (3 marks) 2.5 m/s");
  assert.ok(t.includes("acceleration"));
  assert.ok(t.includes("trolleys") && t.includes("trolley")); // the simple dictionary does not stem
  assert.ok(t.includes("equations") && t.includes("equation"));
  for (const gone of ["calculate", "the", "use", "marks", "3"]) assert.ok(!t.includes(gone), gone);
  assert.ok(questionTerms("H2O and CO2").includes("h2o"));
});

test("search terms are bounded in count and size", () => {
  const long = Array.from({ length: 40 }, (_, i) => `concept${i}word`).join(" ");
  const t = questionTerms(long);
  assert.ok(t.length <= 12);
  assert.ok(new TextEncoder().encode(t.join(" ")).length <= 512);
});

test("refinement terms are filtered to plain search words", () => {
  assert.deepEqual(refinementTerms(["equations of motion", "'); drop table x; --", 7, "Moles"]), ["equations motion", "drop", "moles"]);
  assert.deepEqual(refinementTerms("not a list"), []);
  assert.equal(refinementTerms(Array(20).fill("enthalpy change")).length, 1);
});

test("the shortlist validator keeps listed codes only and asks to refine only when nothing fits", () => {
  const allowed = new Set(["3.1", "3.2"]);
  const r = validateShortlist({ can_tag: true, tags: [{ code: "9.9", strength: "strong", primary: true }, { code: "3.2", strength: "strong", primary: false }], refine_terms: ["x"] }, allowed);
  assert.deepEqual(r.tags.map((x) => [x.code, x.primary]), [["3.2", true]]);
  assert.deepEqual(r.refineTerms, []);
  const none = validateShortlist({ can_tag: false, tags: [], refine_terms: ["moles"] }, allowed);
  assert.equal(none.canTag, false);
  assert.deepEqual(none.refineTerms, ["moles"]);
  assert.throws(() => validateShortlist({ tags: [] }, allowed));
});

// ── the tagger end to end, with a fake database and a fake Gemini ─────────

const PAPER = "p1", REGION = "q1", DOC = "d1";
const cand = (id: string, code: string, title: string) => ({ objective_id: id, code, title, objective_text: `${title} objective`, source_page: 4, text_truncated: false, parent: null });

function harness(o: { scope: any; first?: any[]; second?: any[] }) {
  const rpcs: Array<[string, any]> = [];
  const fromCalls: string[] = [];
  let searches = 0;
  const sb = {
    from(table: string) {
      fromCalls.push(table);
      const rows: Record<string, any> = {
        question_region: { id: REGION, paper_id: PAPER, student_id: "s1", question_label: "2(a)", question_text: "A trolley accelerates down a slope. Calculate its acceleration.", student_answer: "v = u + at", marks_available: 3 },
        syllabus_document: { title: "Physics", syllabus_code: "9702", version_label: "2025-2027" },
        model_route: { stage: "topic_tag", primary_model: "gemini-test", provider: "ai_studio", fallbacks: [], temperature: 0, max_tokens: 512, prompt_version: "topic_tag.v1", thinking_level: "low", allow_training: false, enabled: true },
      };
      const q: any = { select: () => q, eq: () => q, order: () => Promise.resolve({ data: [], error: null }), maybeSingle: async () => ({ data: rows[table] ?? null, error: null }), insert: async () => ({ error: null }) };
      return q;
    },
    async rpc(name: string, args: any) {
      rpcs.push([name, args]);
      if (name === "resolve_syllabus_retrieval_scope") return { data: o.scope, error: null };
      if (name === "search_syllabus_topics_any") {
        searches++;
        return { data: { status: "available", candidates: searches === 1 ? (o.first ?? []) : (o.second ?? []) }, error: null };
      }
      return { data: 1, error: null };
    },
  };
  return { sb, rpcs, fromCalls };
}

function gemini(t: any, answers: unknown[]) {
  const bodies: any[] = [];
  t.mock.method(globalThis, "fetch", async (_u: unknown, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)));
    const answer = answers[Math.min(bodies.length - 1, answers.length - 1)];
    return Response.json({ model: "gemini-test", choices: [{ message: { content: JSON.stringify(answer) } }], usage: { prompt_tokens: 10, completion_tokens: 5 } });
  });
  return bodies;
}

const ON = { GOOGLE_API_KEY: "k", SYLLABUS_SEARCH: "1" };
const AVAILABLE = { status: "available", document_id: DOC, source_sha256: "a".repeat(64) };

test("no approved syllabus: the job ends unavailable with its reason, and no model is called", async (t) => {
  const h = harness({ scope: { status: "unavailable", reason: "unlicensed" } });
  const bodies = gemini(t, [{}]);
  const out = await tagRegion({ env: ON as any, sb: h.sb as any, regionId: REGION, documentId: DOC });
  assert.deepEqual(out, { unavailable: "unlicensed" });
  assert.equal(bodies.length, 0);
  assert.deepEqual(h.rpcs.at(-1), ["finish_topic_tags_unavailable", { p_region_id: REGION, p_document_id: DOC, p_reason: "unlicensed" }]);
});

test("a claimed document that is not the approved one is not used", async (t) => {
  const h = harness({ scope: { ...AVAILABLE, document_id: "other" } });
  const bodies = gemini(t, [{}]);
  assert.deepEqual(await tagRegion({ env: ON as any, sb: h.sb as any, regionId: REGION, documentId: DOC }), { unavailable: "scope_mismatch" });
  assert.equal(bodies.length, 0);
});

test("one search, one judgment on the shortlist, ids from the database, invented codes dropped", async (t) => {
  const h = harness({ scope: AVAILABLE, first: [cand("o31", "3.1", "Kinematics"), cand("o32", "3.2", "Projectiles")] });
  const bodies = gemini(t, [{ can_tag: true, tags: [{ code: "3.1", strength: "strong", primary: true }, { code: "4.4", strength: "strong", primary: false }], refine_terms: [] }]);
  const out: any = await tagRegion({ env: ON as any, sb: h.sb as any, regionId: REGION, documentId: DOC });
  assert.equal(bodies.length, 1);
  assert.equal(out.refined, false);
  const prompt = bodies[0].messages[1].content[0].text as string;
  assert.ok(prompt.includes("3.1 [] Kinematics") && prompt.includes("3.2 [] Projectiles"));
  assert.ok(!h.fromCalls.includes("syllabus_topic")); // the whole syllabus is never read
  const search = h.rpcs.find(([n]) => n === "search_syllabus_topics_any")![1];
  assert.equal(search.p_document_id, DOC);
  assert.ok(!search.p_terms.some((x: string) => x.includes("u + at"))); // the student's answer is not a search term
  const finish = h.rpcs.find(([n]) => n === "finish_topic_tags")![1];
  assert.deepEqual(finish.p_tags, [{ topic_id: "o31", confidence: "likely", is_primary: true }]);
  assert.equal(finish.p_prompt, "topic_tag_shortlist.v1");
});

test("nothing fits: one refinement search and a second judgment, never more", async (t) => {
  const h = harness({ scope: AVAILABLE, first: [cand("o74", "7.4", "Waves")], second: [cand("o31", "3.1", "Kinematics")] });
  const bodies = gemini(t, [
    { can_tag: false, tags: [], refine_terms: ["equations of motion"] },
    { can_tag: false, tags: [], refine_terms: ["still more"] },
  ]);
  const out: any = await tagRegion({ env: ON as any, sb: h.sb as any, regionId: REGION, documentId: DOC });
  assert.equal(bodies.length, 2);
  assert.equal(out.refined, true);
  assert.equal(h.rpcs.filter(([n]) => n === "search_syllabus_topics_any").length, 2);
  assert.ok((bodies[1].messages[1].content[0].text as string).includes("final list"));
  assert.deepEqual(h.rpcs.find(([n]) => n === "finish_topic_tags")![1].p_tags, []);
});

test("flag off: the existing full-syllabus path runs unchanged", async (t) => {
  const h = harness({ scope: AVAILABLE });
  gemini(t, [{ can_tag: false, tags: [] }]);
  await assert.rejects(tagRegion({ env: { GOOGLE_API_KEY: "k" } as any, sb: h.sb as any, regionId: REGION, documentId: DOC }), /no objectives/);
  assert.ok(h.fromCalls.includes("syllabus_topic"));
  assert.ok(!h.rpcs.some(([n]) => n === "resolve_syllabus_retrieval_scope"));
});
