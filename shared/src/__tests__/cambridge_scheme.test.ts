import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import {
  bestExamHelpUrl, cambridgeSchemeRef, documentMatches, fetchCambridgeScheme, headerMatches, locateScheme, schemeSections, searchScheme, sectionFor, seriesLetter, topLevelNumber,
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

const DIRECT = "https://bestexamhelp.com/exam/cambridge-international-a-level/mathematics-further-9231/2025/9231_w25_ms_11.pdf";
const keys = { FIRECRAWL_API_KEY: "f", TAVILY_API_KEY: "t" } as any;

test("the scheme address is computed from the paper code, with no search", () => {
  const ref = (over: Record<string, unknown>) => cambridgeSchemeRef(id(over))!;
  assert.equal(bestExamHelpUrl(ref({}), "cambridge-international-a-level/mathematics-further-9231"), DIRECT);
  assert.equal(bestExamHelpUrl(ref({ paper_code: "9231/12", exam_year: 2023 }), "cambridge-international-a-level/mathematics-further-9231"),
    "https://bestexamhelp.com/exam/cambridge-international-a-level/mathematics-further-9231/2023/9231_w23_ms_12.pdf");
  assert.equal(bestExamHelpUrl(ref({ paper_code: "9231/12" }), "cambridge-international-a-level/mathematics-further-9231"),
    "https://bestexamhelp.com/exam/cambridge-international-a-level/mathematics-further-9231/2025/9231_w25_ms_12.pdf");
});

test("a known subject resolves with one free HEAD and no provider call", async () => {
  const calls: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push(`${init?.method ?? "GET"} ${new URL(String(input)).hostname}`);
    return new Response(null, { status: 200, headers: { "content-type": "application/pdf" } });
  }) as typeof fetch;
  assert.deepEqual(await locateScheme(keys, cambridgeSchemeRef(id())!), [DIRECT]);
  assert.deepEqual(calls, ["HEAD bestexamhelp.com"]);
});

test("a blocked or slow HEAD keeps the direct address; only a 404 sends us to search", async () => {
  globalThis.fetch = (async () => new Response("", { status: 403 })) as typeof fetch;
  assert.deepEqual(await locateScheme(keys, cambridgeSchemeRef(id())!), [DIRECT]);
});

test("a subject missing from the table is found on the site's own index", async () => {
  const calls: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push(`${init?.method ?? "GET"} ${new URL(url).hostname}`);
    if (url.includes("firecrawl")) {
      const body = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({ data: { links: body.url.includes("a-level")
        ? ["https://bestexamhelp.com/exam/cambridge-international-a-level/marine-science-9693/index.php"] : [] } }), { status: 200 });
    }
    return new Response(null, { status: 200, headers: { "content-type": "application/pdf" } });
  }) as typeof fetch;
  const ref = cambridgeSchemeRef(id({ subject_code: "9693", paper_code: "9693/12" }))!;
  assert.deepEqual(await locateScheme(keys, ref), ["https://bestexamhelp.com/exam/cambridge-international-a-level/marine-science-9693/2025/9693_w25_ms_12.pdf"]);
  assert.deepEqual(calls, ["POST api.firecrawl.dev", "HEAD bestexamhelp.com"]);
});

test("locate searches only after the direct file 404s: Tavily first, stops on a hit", async () => {
  const calls: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push(`${init?.method ?? "GET"} ${new URL(String(input)).hostname}`);
    if (init?.method === "HEAD") return new Response(null, { status: 404 });
    return new Response(JSON.stringify({ results: [
      { url: "https://pastpapers.co/caie/a-level/mathematics-further-9231/2025-oct-nov/9231_w25_ms_11.pdf" },
      { url: "https://example.org/9231_w25_ms_12.pdf" },
    ] }), { status: 200 });
  }) as typeof fetch;
  const urls = await locateScheme(keys, cambridgeSchemeRef(id())!);
  assert.deepEqual(calls, ["HEAD bestexamhelp.com", "POST api.tavily.com"]);
  // The address that just 404'd is not offered again.
  assert.deepEqual(urls, ["https://pastpapers.co/caie/a-level/mathematics-further-9231/2025-oct-nov/9231_w25_ms_11.pdf"]);
});

test("reading: the direct PDF is read once; search runs only if it does not name this paper", async () => {
  const good = "# Cambridge International AS & A Level\n**9231/11**\nMARK SCHEME\nOctober/November 2025";
  const calls: string[] = [];
  const serve = (directText: string) => (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = init?.body ? JSON.parse(String(init.body)) : {};
    calls.push(url.includes("/scrape") ? `scrape ${new URL(body.url).hostname}` : `${init?.method ?? "GET"} ${new URL(url).hostname}`);
    if (url.includes("/scrape")) {
      const text = body.url === DIRECT ? directText : good;
      return new Response(JSON.stringify({ data: { markdown: text, metadata: { contentType: "application/pdf" } } }), { status: 200 });
    }
    return new Response(JSON.stringify({ results: [{ url: "https://pastpapers.co/x/9231_w25_ms_11.pdf" }] }), { status: 200 });
  }) as typeof fetch;

  globalThis.fetch = serve(good);
  const hit = await fetchCambridgeScheme(keys, cambridgeSchemeRef(id())!, [DIRECT]);
  assert.equal(hit?.sourceHost, "bestexamhelp.com");
  assert.deepEqual(calls, ["scrape bestexamhelp.com"]);

  calls.length = 0;
  globalThis.fetch = serve(good.replace("9231/11", "9231/12"));
  const fallback = await fetchCambridgeScheme(keys, cambridgeSchemeRef(id())!, [DIRECT]);
  assert.equal(fallback?.sourceHost, "pastpapers.co");
  assert.deepEqual(calls, ["scrape bestexamhelp.com", "POST api.tavily.com", "scrape pastpapers.co"]);
});

test("search falls back to Firecrawl, keeps only the exact file and derives the raw-PDF mirror", async () => {
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
  const urls = await searchScheme(keys, cambridgeSchemeRef(id())!);
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

test("numbered marking principles before the answer table are not mistaken for questions", () => {
  // The layout of a real 9231/11 scheme: principles 1-6 as table rows, then
  // the answer table with its header (content replaced with X).
  const md = [
    "## Generic Marking Principles",
    "| 1 | X | X |", "| 2 | X | X |", "| 3 | X | X |", "| 4 | X | X |", "| 5 | X | X |", "| 6 | X | X |",
    "Abbreviations: X",
    "| Question | Answer | Marks | Guidance |", "|---|---|---|---|",
    "| 1(a) | X | B1 | X |", "| 1(b) | X | M1 | X |", "| 1(c) | X | A1 | X |",
    "| Question | Answer | Marks | Guidance |",
    "| 2(a) | X | B1 | X |", "| 2(d) | X | B1 | X |",
    "| 3 | X | M1 | X |", "| 3 | X | A1 | X |",
    "| 4(a) | X | B1 | X |", "| 7(e) | X | B1 | X |",
  ].join("\n");
  const s = schemeSections(md);
  assert.deepEqual([...s.keys()], [1, 2, 3, 4]);
  assert.match(s.get(1)!, /1\(a\)[\s\S]*1\(c\)/);
  assert.doesNotMatch(s.get(1)!, /\| 6 \|/);
  assert.match(s.get(3)!, /M1[\s\S]*A1/);
});

test("the footer reference alone resolves the scheme, even when other fields are thin", () => {
  assert.equal(cambridgeSchemeRef({ subject_code: "9231", exam_year: null, session: null, paper_code: "9231/11/O/N/25", component_code: null, variant: null, confidence: "low" })?.filename, "9231_w25_ms_11.pdf");
  assert.equal(cambridgeSchemeRef({ subject_code: null, exam_year: null, session: "0580/42/M/J/24", paper_code: null, component_code: null, variant: null })?.filename, "0580_s24_ms_42.pdf");
  // A footer that disagrees with the printed subject code is not trusted.
  assert.equal(cambridgeSchemeRef({ subject_code: "9709", exam_year: 2025, session: null, paper_code: "9231/11/O/N/25", component_code: null, variant: null, confidence: "low" }), null);
});
