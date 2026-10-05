import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import {
  cambridgeSchemeRef, documentMatches, headerMatches, locateScheme, schemeSections, sectionFor, seriesLetter, topLevelNumber,
} from "../cambridge_scheme.js";
import { validate } from "../prompts/scheme_check.v1.js";

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

const id = (over: Record<string, unknown> = {}) => ({
  subject_code: "9231", exam_year: 2025, session: "October/November 2025", paper_code: "9231/11",
  component_code: null, variant: null, confidence: "high", ...over,
});

test("a printed header resolves to exactly one scheme file", () => {
  assert.deepEqual(cambridgeSchemeRef(id()), {
    code: "9231", component: "11", series: "w", year: 2025, filename: "9231_w25_ms_11.pdf", label: "9231/11/O/N/25",
  });
  assert.equal(cambridgeSchemeRef(id({ paper_code: null, component_code: "12" }))?.filename, "9231_w25_ms_12.pdf");
  assert.equal(cambridgeSchemeRef(id({ paper_code: "1", variant: "3" }))?.filename, "9231_w25_ms_13.pdf");
  assert.equal(cambridgeSchemeRef(id({ subject_code: "0580", session: "May/June", exam_year: 2024, paper_code: "0580/42" }))?.filename, "0580_s24_ms_42.pdf");
  assert.equal(cambridgeSchemeRef(id({ session: "February/March", exam_year: 2023 }))?.filename, "9231_m23_ms_11.pdf");
});

test("anything missing or ambiguous resolves to nothing, never a guess", () => {
  for (const bad of [
    id({ confidence: "low" }), id({ subject_code: "042" }), id({ session: null }), id({ session: "Summer" }),
    id({ paper_code: null, component_code: null }), id({ paper_code: "1", variant: null }), id({ exam_year: null }),
  ]) assert.equal(cambridgeSchemeRef(bad), null, JSON.stringify(bad));
  assert.equal(cambridgeSchemeRef(null), null);
  assert.equal(seriesLetter("Oct/Nov"), "w");
});

test("the page's own reference must agree before a scheme is used", () => {
  const ref = cambridgeSchemeRef(id())!;
  assert.equal(headerMatches("© UCLES 2025 9231/11/O/N/25", ref), true);
  assert.equal(headerMatches("9231/11 October/November 2025", ref), true);
  assert.equal(headerMatches("9231/12/O/N/25", ref), false);
  assert.equal(headerMatches("9231/11/M/J/25", ref), false);
  assert.equal(headerMatches("9231/11 May/June 2025", ref), false);
  assert.equal(headerMatches("9231/11/O/N/24", ref), false);
  assert.equal(headerMatches(null, ref), false);
});

test("the scheme PDF must itself name the paper", () => {
  const ref = cambridgeSchemeRef(id())!;
  const head = "# Cambridge International AS & A Level\n## FURTHER MATHEMATICS\n**9231/11**\nPaper 1\nMARK SCHEME\nMaximum Mark: 75\nPublished\nOctober/November 2025";
  assert.equal(documentMatches(head, ref), true);
  assert.equal(documentMatches(head.replace("9231/11", "9231/12"), ref), false);
  assert.equal(documentMatches(head.replace("2025", "2024"), ref), false);
});

test("sections split on question rows and ignore the preamble", () => {
  const md = [
    "Generic marking principles", "Abbreviations: M1 method",
    "| Question | Answer | Marks | Guidance |", "|---|---|---|---|",
    "| 1(a) | x = 2 | B1 | |", "| 1(b) | y | M1 | |", "| 1(b) | | 2 | |",
    "Page 7 of 17",
    "| Question | Answer | Marks | Guidance |",
    "| 2 | z | B1 | |",
    "| 3(a)(i) | w | A1 | |",
  ].join("\n");
  const s = schemeSections(md);
  assert.deepEqual([...s.keys()], [1, 2, 3]);
  assert.match(s.get(1)!, /1\(b\)/);
  assert.doesNotMatch(s.get(1)!, /Generic/);
  assert.equal(sectionFor(s, "3(a)(ii)")?.includes("3(a)(i)"), true);
  assert.equal(sectionFor(s, "9"), null);
  assert.equal(topLevelNumber("Q12(b)"), 12);
  assert.equal(topLevelNumber("(b)"), null);
});

test("locate asks Tavily first and stops when it finds the exact file", async () => {
  const hosts: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    hosts.push(new URL(String(input)).hostname);
    return new Response(JSON.stringify({ results: [
      { url: "https://bestexamhelp.com/exam/cambridge-international-a-level/mathematics-further-9231/2025/9231_w25_ms_11.pdf" },
      { url: "https://example.org/9231_w25_ms_12.pdf" },
    ] }), { status: 200 });
  }) as typeof fetch;
  const urls = await locateScheme({ FIRECRAWL_API_KEY: "f", TAVILY_API_KEY: "t" } as any, cambridgeSchemeRef(id())!);
  assert.deepEqual(hosts, ["api.tavily.com"]);
  assert.deepEqual(urls, ["https://bestexamhelp.com/exam/cambridge-international-a-level/mathematics-further-9231/2025/9231_w25_ms_11.pdf"]);
});

test("locate falls back to Firecrawl, keeps only the exact file and derives the raw-PDF mirror", async () => {
  const hosts: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    hosts.push(new URL(String(input)).hostname);
    return new Response(JSON.stringify(String(input).includes("firecrawl")
      ? { data: { web: [
          { url: "https://pastpapers.co/caie/a-level/mathematics-further-9231/2025-oct-nov/9231_w25_ms_11.pdf" },
          { url: "https://example.org/9231_w25_ms_12.pdf" },
        ] } }
      : { results: [] }), { status: 200 });
  }) as typeof fetch;
  const urls = await locateScheme({ FIRECRAWL_API_KEY: "f", TAVILY_API_KEY: "t" } as any, cambridgeSchemeRef(id())!);
  assert.deepEqual(hosts, ["api.tavily.com", "api.firecrawl.dev"]);
  assert.deepEqual(urls, [
    "https://bestexamhelp.com/exam/cambridge-international-a-level/mathematics-further-9231/2025/9231_w25_ms_11.pdf",
    "https://pastpapers.co/caie/a-level/mathematics-further-9231/2025-oct-nov/9231_w25_ms_11.pdf",
  ]);
});

test("plain-text scheme rows (Tavily extract) still split by question", () => {
  const s = schemeSections(["1(a) x = 2 B1", "1(b) y M1", "2(a) z B1"].join("\n"));
  assert.deepEqual([...s.keys()], [1, 2]);
});

const scheme = "| 2(a) | Differentiates using the product rule and simplifies to the stated form | M1 | Allow sign errors in the second term |";

test("an estimate stays within the question's maximum, in whole or half marks", () => {
  const ok = validate({ can_check: true, reason: null, estimated_marks: 2, max_marks: 3, confidence: "likely", what_was_right: "Your method is right.", what_was_missing: ["Show the final simplification."], do_this_next: "Write the simplified form on its own line." }, { scheme, marksAvailable: 3 });
  assert.equal(ok.estimatedMarks, 2);
  assert.throws(() => validate({ can_check: true, estimated_marks: 4, max_marks: 3, confidence: "likely", what_was_missing: [] }, { scheme, marksAvailable: 3 }));
  assert.throws(() => validate({ can_check: true, estimated_marks: 1.3, max_marks: 3, confidence: "likely", what_was_missing: [] }, { scheme, marksAvailable: 3 }));
});

test("nothing the student reads may copy the scheme or use its notation", () => {
  assert.throws(() => validate({ can_check: true, estimated_marks: 1, max_marks: 3, confidence: "likely", what_was_right: "You earned M1 for the method.", what_was_missing: [] }, { scheme, marksAvailable: 3 }), /notation/);
  assert.throws(() => validate({ can_check: true, estimated_marks: 1, max_marks: 3, confidence: "likely", what_was_right: null, what_was_missing: ["Differentiates using the product rule and simplifies to the stated form"] }, { scheme, marksAvailable: 3 }), /copied/);
});

test("can_check false carries a reason and no estimate", () => {
  const r = validate({ can_check: false, reason: "Answer unreadable." }, { scheme, marksAvailable: 3 });
  assert.deepEqual([r.canCheck, r.estimatedMarks, r.reason], [false, null, "Answer unreadable."]);
});
