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
  for (const part of [m.subject, m.text, m.html.replace(/<[^>]+>/g, " ")]) assertCopyRules(part);
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
  assert.ok(m.html.includes(`<a href="${SETTINGS}">${SETTINGS}</a>`));
  assert.match(m.html, /Explain questions: 11 marks lost · 4 questions · 4 papers/);
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

test("unconfigured: nothing is queued or claimed, and the run says why", async () => {
  const noMailer = fixture([d1]);
  const a = await runParentDigests(noMailer.store, { mailer: null, appUrl: "https://app.example.test" });
  assert.deepEqual(a, { configured: false, reason: "no mail provider key or sender address is set" });
  assert.deepEqual(noMailer.calls, []);

  const noUrl = fixture([d1]);
  const mailer: Mailer = { send: async () => ({ ok: true }) };
  const b = await runParentDigests(noUrl.store, { mailer, appUrl: "http://insecure.example.test" });
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
  const run = await runParentDigests(store, { mailer, appUrl: "https://app.example.test" }, 10);

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
  const run = await runParentDigests(store, { mailer, appUrl: "https://app.example.test" });
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
  const run = await runParentDigests(store, { mailer, appUrl: "https://app.example.test" });
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
