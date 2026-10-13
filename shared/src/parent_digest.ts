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

/** A subject or command word is data too: no markup or control characters, capped, never a fallback name. */
function cleanLabel(raw: string): string {
  // eslint-disable-next-line no-control-regex
  return String(raw ?? "").replace(/[\u0000-\u001f\u007f<>]/g, "").trim().slice(0, 60);
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

/** A count from the database is trusted for its type, not its value: whole and never negative. */
function whole(n: unknown): number {
  const v = typeof n === "number" && Number.isFinite(n) ? Math.trunc(n) : 0;
  return v < 0 ? 0 : v;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "5–11 Oct 2026", "28 Sep – 4 Oct 2026". Empty when either date is not a plain YYYY-MM-DD. */
export function weekLabel(start: string, end: string): string {
  const re = /^(\d{4})-(\d{2})-(\d{2})$/;
  const a = re.exec(start ?? "");
  const b = re.exec(end ?? "");
  if (!a || !b) return "";
  const [ay, am, ad] = [Number(a[1]), Number(a[2]), Number(a[3])];
  const [by, bm, bd] = [Number(b[1]), Number(b[2]), Number(b[3])];
  if (am < 1 || am > 12 || bm < 1 || bm > 12 || ad < 1 || ad > 31 || bd < 1 || bd > 31) return "";
  if (ay === by && am === bm) return `${ad}–${bd} ${MONTHS[bm - 1]} ${by}`;
  if (ay === by) return `${ad} ${MONTHS[am - 1]} – ${bd} ${MONTHS[bm - 1]} ${by}`;
  return `${ad} ${MONTHS[am - 1]} ${ay} – ${bd} ${MONTHS[bm - 1]} ${by}`;
}

/** The words of the email, written once so the plain-text and HTML versions cannot drift apart. */
function words(p: DigestPayload) {
  const name = cleanName(p.first_name);
  const papers = whole(p.papers_committed);
  const counted = whole(p.questions_counted);
  const lost = whole(p.questions_lost);
  const marksLost = whole(p.marks_lost);
  const subjects = (p.subjects ?? []).map(cleanLabel).filter(Boolean);
  const subjectList = subjects.length ? ` (${subjects.join(", ")})` : "";

  const papersLine = `${name} had ${count(papers, "paper", "papers")} read in Axon this week${subjectList}.`;

  let lossLine: string;
  if (counted === 0) {
    lossLine =
      `Axon was not sure how it read the questions on ${papers === 1 ? "this paper" : "these papers"}, ` +
      `so none are counted yet. They are counted once ${name} confirms them in the app.`;
  } else if (lost === 0) {
    lossLine = `No marks were lost on the ${counted} questions Axon counted.`;
  } else {
    lossLine = `${lost} of the ${counted} questions Axon counted lost marks: ${marksLost} ${marksLost === 1 ? "mark" : "marks"} lost in total.`;
  }

  const patternNote = !p.has_enough_data
    ? `Axon names a pattern once it has read four papers. It has read ${whole(p.papers_counted)} so far, so there is nothing to name yet.`
    : null;

  const areas = p.has_enough_data
    ? (p.areas ?? []).filter((a) => cleanLabel(a.name)).map((a) => {
        const aName = cleanLabel(a.name);
        const aMarks = whole(a.marks_lost);
        const aQ = whole(a.questions);
        const aP = whole(a.papers);
        return {
          name: aName,
          marksText: `${aMarks} ${aMarks === 1 ? "mark" : "marks"} lost`,
          basis: `${count(aQ, "question", "questions")} · ${count(aP, "paper", "papers")}`,
          line: `${aName} questions: ${aMarks} ${aMarks === 1 ? "mark" : "marks"} lost · ${count(aQ, "question", "questions")} · ${count(aP, "paper", "papers")}`,
        };
      })
    : [];

  // A fixed sentence, not generated: it uses only the first name and the command word already shown.
  const ask = areas[0]
    ? `Ask ${name} to pick one ${areas[0].name} question from this week and say what the mark scheme was looking for.`
    : null;

  return {
    name, papers, counted, lost, marksLost, subjects, papersLine, lossLine, patternNote, areas, ask,
    teacherNote: "These are the teacher's marks. Axon explains where they were lost; it does not change them.",
    why: "You get this because you turned on the weekly summary.",
  };
}

// ── design ──────────────────────────────────────────────────────────────────
// Axon's own palette (src/ui/styles): paper #F4F4F7 and white cards in light, black and #151517
// in dark, one blue accent. Tables and inline styles only, because email clients differ on CSS
// layout; no images, no remote fonts, no tracking pixel. Body copy is 15px or larger, labels 13px.

const FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif";
const C = {
  bg: "#F4F4F7", card: "#FFFFFF", ink: "#0C0C10", mute: "#55555F", rule: "#E3E3E8", wash: "#E8F0FD", link: "#1F5FCC",
};

const DARK_CSS =
  "@media (prefers-color-scheme: dark){" +
  ".bg{background:#000000!important}.card{background:#151517!important}.ink{color:#F5F5F7!important}" +
  ".mute{color:#A6A6B0!important}.wash{background:#14233D!important}.rule{border-color:#2A2A2E!important}" +
  ".lnk{color:#6BA3FF!important}.tile{background:#F5F5F7!important;color:#0C0C10!important}}" +
  "@media screen and (max-width:480px){.pad{padding:20px 16px!important}.num{font-size:26px!important}.h1{font-size:24px!important}}";

function gap(h: number): string {
  return `<tr><td height="${h}" style="height:${h}px;font-size:0;line-height:0">&nbsp;</td></tr>`;
}

function card(inner: string, cls = "card", bg = C.card): string {
  return (
    `<tr><td class="${cls} pad" style="background:${bg};border-radius:16px;padding:24px 22px;font-family:${FONT};color:${C.ink}">` +
    `${inner}</td></tr>`
  );
}

function stat(value: string, label: string, first: boolean): string {
  const small = !/^\d+$/.test(value);
  return (
    `<td width="33%" valign="top" class="rule" style="${first ? "" : `border-left:1px solid ${C.rule};`}padding:${first ? "0 8px 0 0" : "0 8px"};font-family:${FONT}">` +
    `<div class="${small ? "" : "num "}ink" style="font-size:${small ? 18 : 30}px;line-height:${small ? 34 : 36}px;font-weight:700;letter-spacing:-.5px;color:${C.ink}">${escapeHtml(value)}</div>` +
    `<div class="mute" style="margin-top:4px;font-size:13px;line-height:1.35;color:${C.mute}">${escapeHtml(label)}</div></td>`
  );
}

export function renderParentDigest(p: DigestPayload, settingsUrl: string): RenderedDigest {
  const w = words(p);

  // Plain text: the same words, in reading order.
  const textParts: string[] = [w.papersLine, w.lossLine];
  if (w.patternNote) textParts.push(w.patternNote);
  else if (w.areas.length) textParts.push(["What keeps costing marks", ...w.areas.map((a) => `- ${a.line}`)].join("\n"));
  if (w.ask) textParts.push(w.ask);
  textParts.push(w.teacherNote);
  textParts.push(`${w.why} You can turn it off in Settings: ${settingsUrl}`);
  const text = textParts.join("\n\n") + "\n";

  // HTML.
  const week = weekLabel(p.period_start, p.period_end);
  const unknown = w.counted === 0;
  const stats =
    `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top:20px"><tr>` +
    stat(String(w.papers), w.papers === 1 ? "Paper read" : "Papers read", true) +
    stat(unknown ? "Not yet" : String(w.counted), "Questions counted", false) +
    stat(unknown ? "Not yet" : String(w.lost), w.lost === 1 ? "Question lost marks" : "Questions lost marks", false) +
    `</tr></table>`;

  const hero = card(
    `<div class="mute" style="font-size:13px;line-height:1.4;color:${C.mute}">Weekly summary</div>` +
      `<h1 class="h1 ink" style="margin:6px 0 0;font-size:28px;line-height:1.2;font-weight:700;letter-spacing:-.6px;color:${C.ink}">${escapeHtml(w.name)}&#39;s week</h1>` +
      `<p class="mute" style="margin:10px 0 0;font-size:15px;line-height:1.55;color:${C.mute}">${escapeHtml(w.papersLine)}</p>` +
      stats,
  );

  const lossCard = card(
    `<p class="ink" style="margin:0;font-size:17px;line-height:1.5;font-weight:600;color:${C.ink}">${escapeHtml(w.lossLine)}</p>`,
    "wash",
    C.wash,
  );

  let patternCard = "";
  if (w.patternNote) {
    patternCard = card(
      `<p class="mute" style="margin:0;font-size:15px;line-height:1.55;color:${C.mute}">${escapeHtml(w.patternNote)}</p>`,
    );
  } else if (w.areas.length) {
    const rows = w.areas
      .map(
        (a, i) =>
          `<tr><td class="rule" style="padding:${i === 0 ? "12px" : "14px"} 0 14px;${i === 0 ? "" : `border-top:1px solid ${C.rule};`}font-family:${FONT}">` +
          `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>` +
          `<td valign="top"><div class="ink" style="font-size:17px;line-height:1.35;font-weight:600;color:${C.ink}">${escapeHtml(a.name)} questions</div>` +
          `<div class="mute" style="margin-top:3px;font-size:13px;line-height:1.4;color:${C.mute}">${escapeHtml(a.basis)}</div></td>` +
          `<td align="right" valign="top" style="white-space:nowrap"><div class="ink" style="font-size:17px;line-height:1.35;font-weight:700;color:${C.ink}">${escapeHtml(a.marksText)}</div></td>` +
          `</tr></table></td></tr>`,
      )
      .join("");
    patternCard = card(
      `<div class="mute" style="font-size:13px;line-height:1.4;font-weight:600;color:${C.mute}">What keeps costing marks</div>` +
        `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">${rows}</table>` +
        (w.ask
          ? `<p class="ink" style="margin:6px 0 0;font-size:15px;line-height:1.55;color:${C.ink}">${escapeHtml(w.ask)}</p>`
          : ""),
    );
  }

  const url = escapeHtml(settingsUrl);
  const footer =
    `<tr><td style="padding:4px 6px 0;font-family:${FONT}">` +
    `<p class="mute" style="margin:0 0 10px;font-size:13px;line-height:1.6;color:${C.mute}">${escapeHtml(w.teacherNote)}</p>` +
    `<p class="mute" style="margin:0;font-size:13px;line-height:1.6;color:${C.mute}">${escapeHtml(w.why)} You can turn it off in Settings: ` +
    `<a class="lnk" href="${url}" style="color:${C.link};text-decoration:underline">${url}</a></p></td></tr>`;

  const brand =
    `<tr><td style="padding:0 4px 16px;font-family:${FONT}">` +
    `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>` +
    `<td valign="middle"><table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>` +
    `<td class="tile" width="28" height="28" align="center" valign="middle" style="width:28px;height:28px;background:${C.ink};color:#FFFFFF;border-radius:8px;font-size:16px;font-weight:700;font-family:${FONT}">A</td>` +
    `<td class="ink" style="padding-left:9px;font-size:18px;font-weight:700;letter-spacing:-.3px;color:${C.ink}">Axon</td></tr></table></td>` +
    (week ? `<td class="mute" align="right" valign="middle" style="font-size:13px;color:${C.mute}">${escapeHtml(week)}</td>` : "") +
    `</tr></table></td></tr>`;

  const preheader = "Your weekly summary from Axon.";
  const html =
    `<!doctype html><html lang="en"><head><meta charset="utf-8">` +
    `<meta name="viewport" content="width=device-width,initial-scale=1">` +
    `<meta name="color-scheme" content="light dark"><meta name="supported-color-schemes" content="light dark">` +
    `<title>${escapeHtml(DIGEST_SUBJECT)}</title><style>${DARK_CSS}</style></head>` +
    `<body class="bg" style="margin:0;padding:0;background:${C.bg}">` +
    `<div style="display:none;max-height:0;overflow:hidden;opacity:0;font-size:1px;line-height:1px;color:${C.bg}">${preheader}${"&zwnj;&nbsp;".repeat(30)}</div>` +
    `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" class="bg" style="background:${C.bg}"><tr>` +
    `<td align="center" style="padding:24px 12px">` +
    `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:560px">` +
    brand + hero + gap(12) + lossCard + (patternCard ? gap(12) + patternCard : "") + gap(16) + footer +
    `</table></td></tr></table></body></html>`;

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
  /** Extra message headers (List-Unsubscribe, X-Entity-Ref-ID). */
  headers?: Record<string, string>;
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
          body: JSON.stringify({ from, to: [msg.to], subject: msg.subject, text: msg.text, html: msg.html, ...(msg.headers ? { headers: msg.headers } : {}) }),
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
  /** The explicit on-switch (DIGEST_SEND_ENABLED === "true"). Off: nothing is queued, claimed or sent, whatever else is set. */
  enabled: boolean;
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
  if (!cfg.enabled) return { configured: false, reason: "DIGEST_SEND_ENABLED is not \"true\": sending is switched off" };
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
        // A link to Settings where the switch is, and a per-email reference so a weekly
        // message with the same subject is not folded into one thread.
        headers: { "List-Unsubscribe": `<${settingsUrl}>`, "X-Entity-Ref-ID": digest.delivery_id },
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
