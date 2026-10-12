/**
 * AXO-263 weekly parent digest.
 *
 * The database decides who is owed one and what it may say (public.queue_parent_digests,
 * claim_parent_digests, finish_parent_digest in the Axon-Site migration). This module turns a
 * claimed payload into an email, sends it, and reports back. Nothing here reads student data
 * itself, and nothing here logs an address, a name or an email body.
 *
 * Copy follows Axon.md section 8: "marks lost", never a score or percentage; every figure has
 * its sample size beside it; no exclamation marks; no streaks; no prediction or comparison; and
 * unknown is not zero.
 */

export interface DigestArea {
  kind: "command_word";
  name: string;
  questions: number;
  papers: number;
  marks_lost: number;
}

export interface DigestPayload {
  first_name: string;
  period_start: string;
  period_end: string;
  papers_committed: number;
  subjects: string[];
  /** Questions from this week's papers that are eligible for analytics. 0 means "left out", not "no marks lost". */
  questions_counted: number;
  questions_lost: number;
  marks_lost: number;
  /** Papers behind the student's analytics overall. */
  papers_counted: number;
  has_enough_data: boolean;
  areas: DigestArea[];
}

export interface ClaimedDigest {
  delivery_id: string;
  to_email: string;
  payload: DigestPayload;
}

export interface RenderedDigest {
  subject: string;
  text: string;
  html: string;
}

export const DIGEST_SUBJECT = "Your weekly Axon summary";

/** A name is data, not markup or copy: strip control characters and cap the length. */
function cleanName(raw: string): string {
  // eslint-disable-next-line no-control-regex
  const name = raw.replace(/[\u0000-\u001f\u007f<>]/g, "").trim().slice(0, 40);
  return name || "Your child";
}

function count(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

interface Section {
  heading?: string;
  lines: string[];
}

/** The words of the email, once, so the plain-text and HTML versions cannot drift apart. */
function sections(p: DigestPayload, settingsUrl: string): Section[] {
  const name = cleanName(p.first_name);
  const subjects = p.subjects.length ? ` (${p.subjects.join(", ")})` : "";
  const out: Section[] = [];

  out.push({
    lines: [`${name} had ${count(p.papers_committed, "paper", "papers")} read in Axon this week${subjects}.`],
  });

  if (p.questions_counted === 0) {
    out.push({
      lines: [
        `Axon was not sure how it read the questions on ${p.papers_committed === 1 ? "this paper" : "these papers"}, ` +
          `so none are counted yet. They are counted once ${name} confirms them in the app.`,
      ],
    });
  } else if (p.questions_lost === 0) {
    out.push({ lines: [`No marks were lost on the ${p.questions_counted} questions Axon counted.`] });
  } else {
    out.push({
      lines: [
        `${p.questions_lost} of the ${p.questions_counted} questions Axon counted lost marks: ` +
          `${p.marks_lost} ${p.marks_lost === 1 ? "mark" : "marks"} lost in total.`,
      ],
    });
  }

  if (!p.has_enough_data) {
    out.push({
      lines: [
        `Axon names a pattern once it has read four papers. It has read ${p.papers_counted} so far, ` +
          `so there is nothing to name yet.`,
      ],
    });
  } else if (p.areas.length > 0) {
    out.push({
      heading: "What keeps costing marks",
      lines: p.areas.map(
        (a) =>
          `${a.name} questions: ${a.marks_lost} ${a.marks_lost === 1 ? "mark" : "marks"} lost · ` +
          `${count(a.questions, "question", "questions")} · ${count(a.papers, "paper", "papers")}`,
      ),
    });
  }

  out.push({
    lines: ["These are the teacher's marks. Axon explains where they were lost; it does not change them."],
  });
  out.push({
    lines: [`You get this because you turned on the weekly summary. You can turn it off in Settings: ${settingsUrl}`],
  });
  return out;
}

export function renderParentDigest(p: DigestPayload, settingsUrl: string): RenderedDigest {
  const secs = sections(p, settingsUrl);

  const text =
    secs
      .map((s) => (s.heading ? [s.heading, ...s.lines.map((l) => `- ${l}`)].join("\n") : s.lines.join("\n")))
      .join("\n\n") + "\n";

  const body = secs
    .map((s) => {
      const link = (l: string) => {
        const escaped = escapeHtml(l);
        const url = escapeHtml(settingsUrl);
        return escaped.replace(url, `<a href="${url}">${url}</a>`);
      };
      if (s.heading) {
        return (
          `<p style="margin:0 0 6px;font-weight:600">${escapeHtml(s.heading)}</p>` +
          `<ul style="margin:0 0 16px;padding-left:20px">` +
          s.lines.map((l) => `<li>${escapeHtml(l)}</li>`).join("") +
          `</ul>`
        );
      }
      return s.lines.map((l) => `<p style="margin:0 0 16px">${link(l)}</p>`).join("");
    })
    .join("");

  const html =
    `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${escapeHtml(DIGEST_SUBJECT)}</title></head>` +
    `<body style="margin:0;padding:24px;font-family:Onest,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;` +
    `font-size:16px;line-height:1.5;color:#1a1a1a;background:#ffffff"><div style="max-width:520px">${body}</div></body></html>`;

  return { subject: DIGEST_SUBJECT, text, html };
}

// ── delivery ────────────────────────────────────────────────────────────────

export interface MailMessage {
  to: string;
  subject: string;
  text: string;
  html: string;
  /** Stable per digest, so a retry after an unclear failure cannot send twice where the provider honours it. */
  idempotencyKey: string;
}

/** `status` is the provider's HTTP status, or null when the request never completed. No body, no address. */
export type SendResult = { ok: true } | { ok: false; status: number | null };

export interface Mailer {
  send(msg: MailMessage): Promise<SendResult>;
}

/** Transactional email over HTTPS (Cloudflare Workers cannot speak SMTP). Resend's JSON API. */
export function resendMailer(apiKey: string, from: string, fetchFn: typeof fetch = fetch): Mailer {
  return {
    async send(msg) {
      try {
        const res = await fetchFn("https://api.resend.com/emails", {
          method: "POST",
          headers: {
            authorization: `Bearer ${apiKey}`,
            "content-type": "application/json",
            "idempotency-key": msg.idempotencyKey,
          },
          body: JSON.stringify({ from, to: [msg.to], subject: msg.subject, text: msg.text, html: msg.html }),
        });
        return res.ok ? { ok: true } : { ok: false, status: res.status };
      } catch {
        return { ok: false, status: null };
      }
    },
  };
}

export interface DigestStore {
  /** public.queue_parent_digests: how many new digests were queued. */
  queue(): Promise<number>;
  /** public.claim_parent_digests */
  claim(limit: number): Promise<ClaimedDigest[]>;
  /** public.finish_parent_digest. `error` is a short reason with no personal data. */
  finish(id: string, sent: boolean, error: string | null): Promise<void>;
}

export interface DigestConfig {
  mailer: Mailer | null;
  /** The app's origin, e.g. https://example.test (no trailing path). Null: not configured. */
  appUrl: string | null;
}

export type DigestRun =
  | { configured: false; reason: string }
  | { configured: true; queued: number; claimed: number; sent: number; failed: number };

/** Build the Settings link from the configured origin; null if it is not a plain https origin. */
export function settingsUrlFor(appUrl: string | null | undefined): string | null {
  if (!appUrl) return null;
  try {
    const u = new URL(appUrl);
    if (u.protocol !== "https:" || u.username || u.password) return null;
    return `${u.origin}/settings`;
  } catch {
    return null;
  }
}

/**
 * One weekly run. Unconfigured means nothing is queued and nothing is claimed: the run says so
 * (and the caller logs it) rather than succeeding quietly, and no attempt is spent.
 */
export async function runParentDigests(store: DigestStore, cfg: DigestConfig, limit = 25): Promise<DigestRun> {
  const settingsUrl = settingsUrlFor(cfg.appUrl);
  if (!cfg.mailer) return { configured: false, reason: "no mail provider key or sender address is set" };
  if (!settingsUrl) return { configured: false, reason: "DIGEST_APP_URL is not set to an https origin" };

  const queued = await store.queue();
  const batch = await store.claim(limit);
  let sent = 0;
  let failed = 0;

  for (const digest of batch) {
    let result: SendResult;
    try {
      const mail = renderParentDigest(digest.payload, settingsUrl);
      result = await cfg.mailer.send({
        to: digest.to_email,
        subject: mail.subject,
        text: mail.text,
        html: mail.html,
        idempotencyKey: digest.delivery_id,
      });
    } catch {
      result = { ok: false, status: null };
    }

    const reason = result.ok ? null : result.status === null ? "request did not complete" : `provider status ${result.status}`;
    try {
      await store.finish(digest.delivery_id, result.ok, reason);
    } catch {
      // The claim expires in ten minutes and the digest is offered again; the idempotency key
      // stops a provider that honours it from sending the same email twice.
    }
    if (result.ok) sent++;
    else failed++;
  }

  return { configured: true, queued, claimed: batch.length, sent, failed };
}
