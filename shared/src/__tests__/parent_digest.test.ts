import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DIGEST_SUBJECT,
  renderParentDigest,
  resendMailer,
  runParentDigests,
  settingsUrlFor,
  weekLabel,
  type ClaimedDigest,
  type DigestPayload,
  type DigestStore,
  type Mailer,
  type MailMessage,
} from "../parent_digest.js";

const SETTINGS = "https://app.example.test/settings";

const base: DigestPayload = {
  first_name: "Ana",
  period_start: "2026-10-05",
  period_end: "2026-10-11",
  papers_committed: 2,
  subjects: ["Chemistry", "Physics"],
  questions_counted: 7,
  questions_lost: 4,
  marks_lost: 8,
  papers_counted: 4,
  has_enough_data: true,
  areas: [{ kind: "command_word", name: "Explain", questions: 4, papers: 4, marks_lost: 11 }],
};

// Axon.md section 8 and hard rules: no exclamation marks, no percentages, no "score", no
// prediction, comparison or ranking, no grades.
const BANNED = [/!/, /%/, /\bscores?\b/i, /predict|projected|forecast|on track|behind|ahead|rank|better than|worse than|\bgrades?\b/i];
function assertCopyRules(s: string) {
  for (const rule of BANNED) assert.doesNotMatch(s, rule, `copy rule ${rule} violated in: ${s}`);
}

test("the digest says marks lost with its sample sizes, and names no more than the data supports", () => {
  const m = renderParentDigest(base, SETTINGS);
  assert.equal(m.subject, DIGEST_SUBJECT);
  assert.match(m.text, /Ana had 2 papers read in Axon this week \(Chemistry, Physics\)\./);
  assert.match(m.text, /4 of the 7 questions Axon counted lost marks: 8 marks lost in total\./);
  assert.match(m.text, /What keeps costing marks\n- Explain questions: 11 marks lost · 4 questions · 4 papers/);
  assert.match(m.text, /These are the teacher's marks\. Axon explains where they were lost; it does not change them\./);
  assert.ok(m.text.includes(`turn it off in Settings: ${SETTINGS}`));
  for (const part of [m.subject, m.text, m.html.replace(/<style>[\s\S]*?<\/style>/g, "").replace(/<[^>]+>/g, " ")]) assertCopyRules(part);
});

test("with fewer than four papers overall no pattern is named", () => {
  const m = renderParentDigest({ ...base, has_enough_data: false, papers_counted: 2, areas: [] }, SETTINGS);
  assert.match(m.text, /names a pattern once it has read four papers\. It has read 2 so far/);
  assert.doesNotMatch(m.text, /What keeps costing marks/);
  assertCopyRules(m.text);
});

test("questions Axon left out are reported as unknown, never as zero marks lost", () => {
  const m = renderParentDigest({ ...base, papers_committed: 1, questions_counted: 0, questions_lost: 0, marks_lost: 0 }, SETTINGS);
  assert.match(m.text, /not sure how it read the questions on this paper, so none are counted yet/);
  assert.doesNotMatch(m.text, /No marks were lost/);
  assert.doesNotMatch(m.text, /0 marks/);
});

test("counted questions with nothing lost say exactly that", () => {
  const m = renderParentDigest({ ...base, questions_lost: 0, marks_lost: 0, areas: [] }, SETTINGS);
  assert.match(m.text, /No marks were lost on the 7 questions Axon counted\./);
});

test("singular forms read naturally", () => {
  const m = renderParentDigest(
    { ...base, papers_committed: 1, subjects: [], questions_counted: 3, questions_lost: 1, marks_lost: 1,
      areas: [{ kind: "command_word", name: "State", questions: 3, papers: 2, marks_lost: 1 }] },
    SETTINGS,
  );
  assert.match(m.text, /Ana had 1 paper read in Axon this week\./);
  assert.match(m.text, /1 of the 3 questions Axon counted lost marks: 1 mark lost in total\./);
  assert.match(m.text, /State questions: 1 mark lost · 3 questions · 2 papers/);
});

test("a name is data: markup is stripped, entities are escaped, length is capped", () => {
  const m = renderParentDigest({ ...base, first_name: "<script>x</script> Tom & Jerry" }, SETTINGS);
  assert.doesNotMatch(m.html, /<script/);
  assert.match(m.html, /Tom &amp; Jerry/);
  const long = renderParentDigest({ ...base, first_name: "A".repeat(500) }, SETTINGS);
  assert.ok(long.text.split("\n")[0]!.length < 120);
  assert.match(renderParentDigest({ ...base, first_name: "  " }, SETTINGS).text, /^Your child had 2 papers/);
});

test("the HTML version carries the same words and a working Settings link", () => {
  const m = renderParentDigest(base, SETTINGS);
  assert.match(m.html, new RegExp(`<a class="lnk" href="${SETTINGS}"[^>]*>${SETTINGS}</a>`));
  // The HTML says the same things as the text: the same sentences and the same figures.
  assert.match(m.html, /Ana had 2 papers read in Axon this week \(Chemistry, Physics\)\./);
  assert.match(m.html, /4 of the 7 questions Axon counted lost marks: 8 marks lost in total\./);
  assert.match(m.html, /Explain questions<\/div>/);
  assert.match(m.html, /11 marks lost/);
  assert.match(m.html, /4 questions · 4 papers/);
  assert.match(m.html, /These are the teacher&#39;s marks\./);
});

test("the Settings link is built only from a plain https origin", () => {
  assert.equal(settingsUrlFor("https://app.example.test"), "https://app.example.test/settings");
  assert.equal(settingsUrlFor("https://app.example.test/some/path?x=1"), "https://app.example.test/settings");
  assert.equal(settingsUrlFor("http://app.example.test"), null);
  assert.equal(settingsUrlFor("https://user:pw@app.example.test"), null);
  assert.equal(settingsUrlFor("not a url"), null);
  assert.equal(settingsUrlFor(undefined), null);
  assert.equal(settingsUrlFor(""), null);
});

// ── the run ────────────────────────────────────────────────────────────────

function fixture(digests: ClaimedDigest[]) {
  const calls: string[] = [];
  const finished: Array<{ id: string; sent: boolean; error: string | null }> = [];
  const store: DigestStore = {
    queue: async () => { calls.push("queue"); return digests.length; },
    claim: async (limit) => { calls.push(`claim:${limit}`); return digests; },
    finish: async (id, sent, error) => { finished.push({ id, sent, error }); },
  };
  return { store, calls, finished };
}

const d1: ClaimedDigest = { delivery_id: "d-1", to_email: "parent1@example.test", payload: base };
const d2: ClaimedDigest = { delivery_id: "d-2", to_email: "parent2@example.test", payload: { ...base, first_name: "Ben" } };

test("switched off: with the key, sender and URL all set, nothing is queued, claimed or sent", async () => {
  const f = fixture([d1]);
  let sends = 0;
  const mailer: Mailer = { send: async () => { sends++; return { ok: true }; } };
  const r = await runParentDigests(f.store, { mailer, enabled: false, appUrl: "https://app.example.test" });
  assert.equal(r.configured, false);
  assert.deepEqual(f.calls, []);
  assert.equal(sends, 0);
});

test("each email carries a List-Unsubscribe link to Settings and a per-email reference", async () => {
  const f = fixture([d1]);
  const seenMail: MailMessage[] = [];
  const mailer: Mailer = { send: async (m) => { seenMail.push(m); return { ok: true }; } };
  await runParentDigests(f.store, { mailer, enabled: true, appUrl: "https://app.example.test" });
  assert.deepEqual(seenMail[0]!.headers, {
    "List-Unsubscribe": "<https://app.example.test/settings>", "X-Entity-Ref-ID": "d-1",
  });
});

test("the sweep wrangler.toml has sending switched on (owner's instruction, 13 Oct 2026); the database gate still decides who is eligible", () => {
  const toml = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "../../../workers/sweep/wrangler.toml"), "utf8");
  assert.match(toml, /^DIGEST_SEND_ENABLED = "true"$/m);
});

test("unconfigured: nothing is queued or claimed, and the run says why", async () => {
  const noMailer = fixture([d1]);
  const a = await runParentDigests(noMailer.store, { mailer: null, enabled: true, appUrl: "https://app.example.test" });
  assert.deepEqual(a, { configured: false, reason: "no mail provider key or sender address is set" });
  assert.deepEqual(noMailer.calls, []);

  const noUrl = fixture([d1]);
  const mailer: Mailer = { send: async () => ({ ok: true }) };
  const b = await runParentDigests(noUrl.store, { mailer, enabled: true, appUrl: "http://insecure.example.test" });
  assert.equal(b.configured, false);
  assert.deepEqual(noUrl.calls, []);
});

test("each digest is sent once to its own guardian, and success and failure are reported separately", async () => {
  const { store, calls, finished } = fixture([d1, d2]);
  const sentMail: MailMessage[] = [];
  const mailer: Mailer = {
    send: async (msg) => {
      sentMail.push(msg);
      return msg.to === "parent1@example.test" ? { ok: true } : { ok: false, status: 503 };
    },
  };
  const run = await runParentDigests(store, { mailer, enabled: true, appUrl: "https://app.example.test" }, 10);

  assert.deepEqual(calls, ["queue", "claim:10"]);
  assert.deepEqual(run, { configured: true, queued: 2, claimed: 2, sent: 1, failed: 1 });
  assert.deepEqual(sentMail.map((m) => [m.to, m.idempotencyKey]), [
    ["parent1@example.test", "d-1"],
    ["parent2@example.test", "d-2"],
  ]);
  assert.ok(sentMail[0]!.text.startsWith("Ana had"));
  assert.ok(sentMail[1]!.text.startsWith("Ben had"));
  assert.deepEqual(finished, [
    { id: "d-1", sent: true, error: null },
    { id: "d-2", sent: false, error: "provider status 503" },
  ]);
  // The recorded reason never carries an address or a name.
  for (const f of finished) assert.doesNotMatch(String(f.error), /@|Ana|Ben/);
});

test("a provider that throws or never answers is a failed send, not a crash", async () => {
  const { store, finished } = fixture([d1, d2]);
  let n = 0;
  const mailer: Mailer = {
    send: async () => {
      if (++n === 1) throw new Error("boom with parent1@example.test");
      return { ok: false, status: null };
    },
  };
  const run = await runParentDigests(store, { mailer, enabled: true, appUrl: "https://app.example.test" });
  assert.deepEqual(run, { configured: true, queued: 2, claimed: 2, sent: 0, failed: 2 });
  assert.deepEqual(finished.map((f) => f.error), ["request did not complete", "request did not complete"]);
});

test("if recording the result fails, the run carries on with the next digest", async () => {
  const sent: string[] = [];
  const store: DigestStore = {
    queue: async () => 2,
    claim: async () => [d1, d2],
    finish: async () => { throw new Error("database unreachable"); },
  };
  const mailer: Mailer = { send: async (m) => { sent.push(m.idempotencyKey); return { ok: true }; } };
  const run = await runParentDigests(store, { mailer, enabled: true, appUrl: "https://app.example.test" });
  assert.deepEqual(sent, ["d-1", "d-2"]);
  assert.deepEqual(run, { configured: true, queued: 2, claimed: 2, sent: 2, failed: 0 });
});

// ── the provider ───────────────────────────────────────────────────────────

test("the Resend adapter posts one JSON message with a bearer key and an idempotency key", async () => {
  const seen: Array<{ url: string; init: RequestInit }> = [];
  const fakeFetch = (async (url: string, init: RequestInit) => {
    seen.push({ url, init });
    return new Response("{}", { status: 200 });
  }) as unknown as typeof fetch;
  const mailer = resendMailer("re_test_key", "Axon <digest@mail.example.test>", fakeFetch);
  const result = await mailer.send({ to: "p@example.test", subject: "S", text: "T", html: "<p>H</p>", idempotencyKey: "d-9" });

  assert.deepEqual(result, { ok: true });
  assert.equal(seen.length, 1);
  assert.equal(seen[0]!.url, "https://api.resend.com/emails");
  assert.equal(seen[0]!.init.method, "POST");
  const headers = seen[0]!.init.headers as Record<string, string>;
  assert.equal(headers.authorization, "Bearer re_test_key");
  assert.equal(headers["idempotency-key"], "d-9");
  assert.deepEqual(JSON.parse(String(seen[0]!.init.body)), {
    from: "Axon <digest@mail.example.test>", to: ["p@example.test"], subject: "S", text: "T", html: "<p>H</p>",
  });
  await mailer.send({ to: "p@example.test", subject: "S", text: "T", html: "H", idempotencyKey: "d-10", headers: { "X-Entity-Ref-ID": "d-10" } });
  assert.deepEqual(JSON.parse(String(seen[1]!.init.body)).headers, { "X-Entity-Ref-ID": "d-10" });
});

test("the Resend adapter reports a refusal by status only, and a network failure as null", async () => {
  const refuse = (async () => new Response('{"message":"p@example.test is invalid"}', { status: 422 })) as unknown as typeof fetch;
  const refused = await resendMailer("k", "a@b.test", refuse).send({ to: "p@example.test", subject: "S", text: "T", html: "H", idempotencyKey: "x" });
  assert.deepEqual(refused, { ok: false, status: 422 });

  const down = (async () => { throw new TypeError("network"); }) as unknown as typeof fetch;
  const lost = await resendMailer("k", "a@b.test", down).send({ to: "p@example.test", subject: "S", text: "T", html: "H", idempotencyKey: "x" });
  assert.deepEqual(lost, { ok: false, status: null });
});

// ── wiring ─────────────────────────────────────────────────────────────────

test("the weekly digest cron in wrangler.toml is the one the sweep worker routes on", () => {
  const sweep = join(dirname(fileURLToPath(import.meta.url)), "../../../workers/sweep");
  const toml = readFileSync(join(sweep, "wrangler.toml"), "utf8");
  const src = readFileSync(join(sweep, "src/index.ts"), "utf8");
  const inCode = /const DIGEST_CRON = "([^"]+)"/.exec(src)?.[1];
  const crons = /crons = \[([^\]]*)\]/.exec(toml)?.[1]?.match(/"([^"]+)"/g)?.map((c) => c.slice(1, -1)) ?? [];
  assert.ok(inCode, "DIGEST_CRON is declared in the sweep worker");
  assert.ok(crons.includes(inCode!), `wrangler.toml schedules ${inCode}`);
  assert.equal(crons.filter((c) => c === inCode).length, 1);
  // It must not collide with the other two schedules, or one trigger would run two jobs.
  assert.deepEqual(crons.filter((c) => c !== inCode).sort(), ["* * * * *", "*/15 * * * *"].sort());
  // Weekly, on a Sunday.
  assert.match(inCode!, /^\d+ \d+ \* \* 0$/);
});

test("the email design: no images, no tracking, a dark-mode variant, a generic preview line, and the same copy rules", () => {
  const m = renderParentDigest(base, SETTINGS);
  assert.doesNotMatch(m.html, /<img|<script|<link|url\(|http:\/\//i, "nothing is fetched when the email opens");
  assert.match(m.html, /prefers-color-scheme: dark/);
  assert.match(m.html, /name="color-scheme" content="light dark"/);
  // The preview line shows in notifications and on lock screens, so it names no child and no figure.
  const pre = /display:none[^>]*>([^<]*)</.exec(m.html)![1]!.replace(/&zwnj;|&nbsp;/g, "");
  assert.equal(pre, "Your weekly summary from Axon.");
  assertCopyRules(m.html.replace(/<style>[\s\S]*?<\/style>/g, "").replace(/<[^>]+>/g, " "));
  assert.match(m.html, /Weekly summary/);
  assert.match(m.html, /5–11 Oct 2026/);
});

test("a fixed question for the parent appears only when a pattern is named, and uses only data already shown", () => {
  const withArea = renderParentDigest(base, SETTINGS);
  assert.match(withArea.text, /Ask Ana to pick one Explain question from this week and say what the mark scheme was looking for\./);
  assert.match(withArea.html, /Ask Ana to pick one Explain question/);
  const none = renderParentDigest({ ...base, has_enough_data: false, papers_counted: 1, areas: [] }, SETTINGS);
  assert.doesNotMatch(none.text, /Ask Ana/);
  assertCopyRules(withArea.text);
});

test("unknown is not zero in the figures either: left-out questions show Not yet, not 0", () => {
  const m = renderParentDigest({ ...base, questions_counted: 0, questions_lost: 0, marks_lost: 0 }, SETTINGS);
  assert.match(m.html, />Not yet</);
  assert.doesNotMatch(m.html.replace(/<style>[\s\S]*?<\/style>/g, ""), />0</);
});

test("the week label is built only from plain dates", () => {
  assert.equal(weekLabel("2026-10-05", "2026-10-11"), "5–11 Oct 2026");
  assert.equal(weekLabel("2026-09-28", "2026-10-04"), "28 Sep – 4 Oct 2026");
  assert.equal(weekLabel("2025-12-29", "2026-01-04"), "29 Dec 2025 – 4 Jan 2026");
  assert.equal(weekLabel("<b>", "2026-10-11"), "");
  assert.equal(weekLabel("2026-13-05", "2026-10-11"), "");
});

test("figures from the database are whole numbers and never negative in the page", () => {
  const m = renderParentDigest({ ...base, papers_committed: -3 as number, questions_counted: 7.9, questions_lost: 2.2 }, SETTINGS);
  assert.match(m.text, /Ana had 0 papers read/);
  assert.match(m.text, /2 of the 7 questions/);
});

const richer: DigestPayload = {
  ...base,
  by_subject: [
    { subject: "Chemistry", papers: 1, questions: 4, lost_questions: 2, marks_lost: 5 },
    { subject: "Physics", papers: 1, questions: 3, lost_questions: 1, marks_lost: 3 },
  ],
  earlier_questions: 12,
  earlier_lost: 8,
  trend: "fewer",
};

test("the digest says what was worked on, subject by subject, with counts and no scores", () => {
  const m = renderParentDigest(richer, SETTINGS);
  assert.match(m.text, /What they worked on\n- Chemistry: 1 paper · 2 of 4 questions lost marks · 5 marks lost\n- Physics: 1 paper · 1 of 3 questions lost marks · 3 marks lost/);
  assert.match(m.html, /What they worked on/);
  assert.match(m.html, /2 of 4 questions lost marks/);
  for (const part of [m.text, m.html.replace(/<style>[\s\S]*?<\/style>/g, "").replace(/<[^>]+>/g, " ")]) assertCopyRules(part);
});

test("how it is going compares the week with earlier papers, with both counts and a caveat", () => {
  for (const [trend, phrase] of [
    ["fewer", "A smaller share of questions lost marks than in Ana's earlier papers: 4 of 7 this week, against 8 of 12 in earlier papers."],
    ["more", "A larger share of questions lost marks than in Ana's earlier papers: 4 of 7 this week, against 8 of 12 in earlier papers."],
    ["similar", "About the same share of questions lost marks as in Ana's earlier papers: 4 of 7 this week, against 8 of 12 in earlier papers."],
  ] as const) {
    const m = renderParentDigest({ ...richer, trend }, SETTINGS);
    assert.ok(m.text.includes(`How it is going\n${phrase}\nPapers differ in difficulty, so read this as a pointer, not a verdict.`), trend);
    assertCopyRules(m.text);
    assertCopyRules(m.html.replace(/<style>[\s\S]*?<\/style>/g, "").replace(/<[^>]+>/g, " ").replace(/&#39;/g, "'"));
  }
});

test("without enough earlier work the email says so instead of inventing a trend", () => {
  const m = renderParentDigest({ ...richer, trend: null, earlier_questions: null, earlier_lost: null }, SETTINGS);
  assert.match(m.text, /How it is going\nAxon compares a week with earlier papers once there are 10 questions on each side\./);
  assert.doesNotMatch(m.text, /smaller share|larger share|same share/);
});

test("an inconsistent comparison from the database is not shown", () => {
  for (const bad of [
    { earlier_questions: 5, earlier_lost: 9 },
    { earlier_questions: 0, earlier_lost: 0 },
    { trend: "improving" as unknown as "fewer" },
  ]) {
    const m = renderParentDigest({ ...richer, ...bad }, SETTINGS);
    assert.doesNotMatch(m.text, /smaller share|larger share|same share|improving/);
  }
});

test("older payloads without the new fields still render", () => {
  const m = renderParentDigest(base, SETTINGS);
  assert.doesNotMatch(m.text, /What they worked on/);
  assert.match(m.text, /4 of the 7 questions Axon counted lost marks/);
});
