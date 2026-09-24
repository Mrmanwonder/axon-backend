import { createHash } from "node:crypto";

export const CBSE_OFFICIAL_HOST = "cbseacademic.nic.in";
export const CBSE_PARSER_VERSION = "cbse-sqp-ms-v1";
export const CBSE_INDEXES = Object.freeze({
  10: "https://cbseacademic.nic.in/SQP_CLASSX_2026-27.html",
  12: "https://cbseacademic.nic.in/SQP_CLASSXII_2026-27.html",
});

const ENTITIES = new Map([
  ["amp", "&"], ["nbsp", " "], ["ndash", "-"], ["mdash", "-"],
  ["quot", "\""], ["apos", "'"], ["#39", "'"],
]);

function decodeHtml(value) {
  return String(value)
    .replace(/&([a-zA-Z]+|#39);/g, (_, key) => ENTITIES.get(key) ?? "&" + key + ";")
    .replace(/\u00a0/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function textOf(html) {
  return decodeHtml(
    String(html)
      .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
      .replace(/<style\b[\s\S]*?<\/style>/gi, " ")
      .replace(/<[^>]+>/g, " "),
  );
}

function linksOf(html, baseUrl) {
  const links = [];
  const re = /<a\b([^>]*)>([\s\S]*?)<\/a>/gi;
  let match;
  while ((match = re.exec(String(html)))) {
    const hrefMatch = /\bhref\s*=\s*(["'])(.*?)\1/i.exec(match[1]);
    if (!hrefMatch) continue;
    links.push({
      label: textOf(match[2]),
      href: new URL(hrefMatch[2], baseUrl).href,
    });
  }
  return links;
}

export function assertOfficialCbseUrl(value) {
  const url = new URL(value);
  const host = url.hostname.replace(/^www\./, "").toLowerCase();
  if (url.protocol !== "https:" || host !== CBSE_OFFICIAL_HOST) {
    throw new Error("Refusing non-CBSE source: " + value);
  }
  return url.href;
}

export function discoverCbseIndex(html, baseUrl, classLevel) {
  const out = [];
  const rowRe = /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;
  let rowMatch;
  while ((rowMatch = rowRe.exec(String(html)))) {
    const cells = [...rowMatch[1].matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/gi)].map(m => m[1]);
    if (cells.length < 3) continue;
    const subject = textOf(cells[0]);
    if (!subject || /^subject$/i.test(subject)) continue;
    const sqp = linksOf(cells[1], baseUrl).find(link => /\.pdf(?:$|\?)/i.test(link.href));
    const ms = linksOf(cells[2], baseUrl).find(link => /\.pdf(?:$|\?)/i.test(link.href));
    if (!sqp || !ms) continue;
    out.push({
      classLevel,
      subject,
      sqpUrl: assertOfficialCbseUrl(sqp.href),
      msUrl: assertOfficialCbseUrl(ms.href),
    });
  }
  return out;
}

function finalAcademicYear(start, tail) {
  const first = Number(start);
  if (!Number.isInteger(first)) return null;
  const raw = String(tail);
  const last = raw.length === 2 ? Number(String(first).slice(0, 2) + raw) : Number(raw);
  return Number.isInteger(last) ? last : null;
}

export function parseCbseHeader(text) {
  const body = String(text).replace(/\r/g, "");
  const subjectMatch =
    /Subject\s*:\s*([^\n(]+?)\s*\((\d{3})\)/i.exec(body)
    ?? /(?:SUBJECT|SUB)\s*[:\-]\s*([^\n]+?)\s+(?:CODE\s*[:\-]?\s*)?(\d{3})\b/i.exec(body);
  const classMatch = /Class\s*[–—-]\s*(XII|X)\b/i.exec(body)
    ?? /CLASS\s*[:\-]?\s*(XII|X)\b/i.exec(body);
  const sessionMatch = /(?:Academic\s+Session|Session)\s*(20\d{2})\s*[–—-]\s*(\d{2,4})/i.exec(body);
  if (!subjectMatch || !classMatch || !sessionMatch) return null;

  const classLevel = classMatch[1].toUpperCase() === "XII" ? 12 : 10;
  const examYear = finalAcademicYear(sessionMatch[1], sessionMatch[2]);
  if (!examYear) return null;
  const end = String(sessionMatch[2]).length === 2
    ? String(sessionMatch[2]).padStart(2, "0")
    : String(sessionMatch[2]).slice(-2);

  return {
    subject: subjectMatch[1].replace(/\s+/g, " ").trim(),
    subjectCode: subjectMatch[2],
    classLevel,
    session: String(sessionMatch[1]) + "-" + end,
    examYear,
    expectedQuestions: Number(/There\s+are\s+(\d+)\s+questions\s+in\s+all/i.exec(body)?.[1] ?? 0) || null,
  };
}

function labelMatch(line) {
  const match = /^\s*(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*)?(\d{1,2})(?:\s*\(([AB])\))?\s*[.)]?\s+(.+)$/i.exec(line);
  if (!match) return null;
  const number = Number(match[1]);
  if (!Number.isInteger(number) || number < 1 || number > 80) return null;
  const alternative = match[2]?.toUpperCase() ?? null;
  return {
    number,
    alternative,
    label: String(number) + (alternative ? "(" + alternative + ")" : ""),
  };
}

function isBoilerplate(line) {
  const value = line.trim();
  return !value
    || /^\*?There is no change in the Question Paper Design/i.test(value)
    || /^Subject\s*:/i.test(value)
    || /^(?:Sample Question Paper|Marking Scheme)$/i.test(value)
    || /^Class\s*[–—-]/i.test(value)
    || /^Academic Session/i.test(value)
    || /^Maximum Marks:/i.test(value)
    || /^Time Allowed:/i.test(value)
    || /^Q\.?\s*No\.?\s+Question\s+Marks$/i.test(value)
    || /^Q\.?\s*No\.?\s+SECTION\s+[A-Z]\s+Marks$/i.test(value)
    || /^SECTION\s+[A-Z](?:\s+\(.*\))?$/i.test(value)
    || /^Page\s+\d+\s+of\s+\d+$/i.test(value);
}

function parseMarkExpression(value) {
  const parts = String(value).split("+").map(part => Number(part.trim()));
  if (!parts.length || parts.some(part => !Number.isFinite(part) || part <= 0)) return null;
  const total = parts.reduce((sum, part) => sum + part, 0);
  return Number.isInteger(total) && total > 0 && total <= 30 ? total : null;
}

export function extractMarkTotal(lines) {
  const candidates = [];
  for (const line of lines) {
    const match = /\s{3,}(\d+(?:\s*\+\s*\d+)*)\s*$/.exec(line);
    if (!match) continue;
    const total = parseMarkExpression(match[1]);
    if (total) candidates.push({ total, compound: match[1].includes("+") });
  }
  if (!candidates.length) return null;
  const compound = candidates.filter(candidate => candidate.compound);
  if (compound.length) {
    const unique = [...new Set(compound.map(candidate => candidate.total))];
    return unique.length === 1 ? unique[0] : null;
  }
  const sum = candidates.reduce((total, candidate) => total + candidate.total, 0);
  return Number.isInteger(sum) && sum > 0 && sum <= 30 ? sum : null;
}

function stripMarksColumn(line) {
  return line.replace(/\s{3,}\d+(?:\s*\+\s*\d+)*\s*$/, "").trimEnd();
}

export function parseQuestionBlocks(text) {
  const lines = String(text).replace(/\r/g, "").split("\n");
  const starts = [];
  let lastBase = 0;
  const seen = new Set();

  for (let index = 0; index < lines.length; index++) {
    const candidate = labelMatch(lines[index]);
    if (!candidate) continue;
    if (!starts.length && candidate.number !== 1) continue;
    const allowed = lastBase === 0
      || candidate.number === lastBase + 1
      || (candidate.number === lastBase && candidate.alternative);
    if (!allowed || seen.has(candidate.label)) continue;
    starts.push({ ...candidate, index });
    seen.add(candidate.label);
    lastBase = Math.max(lastBase, candidate.number);
  }

  return starts.map((start, i) => {
    const end = starts[i + 1]?.index ?? lines.length;
    const rawLines = lines.slice(start.index, end).filter(line => !isBoilerplate(line));
    const maxMarks = extractMarkTotal(rawLines);
    const cleaned = rawLines
      .map(stripMarksColumn)
      .filter(line => line.trim())
      .join("\n")
      .trim();
    return {
      label: start.label,
      number: start.number,
      alternative: start.alternative,
      maxMarks,
      text: cleaned,
    };
  }).filter(block => block.text);
}

export function pairOfficialQuestions(sqpText, msText, minCoverage = 0.75) {
  const sqpHeader = parseCbseHeader(sqpText);
  const msHeader = parseCbseHeader(msText);
  if (!sqpHeader || !msHeader) throw new Error("Could not read CBSE SQP/MS header");

  for (const key of ["subjectCode", "classLevel", "session", "examYear"]) {
    if (sqpHeader[key] !== msHeader[key]) throw new Error("SQP/MS identity mismatch on " + key);
  }

  const sqpBlocks = parseQuestionBlocks(sqpText);
  const msBlocks = parseQuestionBlocks(msText);
  const schemeByLabel = new Map(msBlocks.map(block => [block.label, block]));
  const questions = [];

  for (const question of sqpBlocks) {
    const scheme = schemeByLabel.get(question.label);
    if (!scheme) continue;
    if (!question.maxMarks || !scheme.maxMarks || question.maxMarks !== scheme.maxMarks) continue;
    questions.push({
      label: question.label,
      questionText: question.text,
      markingScheme: scheme.text,
      maxMarks: question.maxMarks,
    });
  }

  const expected = sqpHeader.expectedQuestions ?? Math.max(0, ...sqpBlocks.map(block => block.number));
  const coveredBase = new Set(
    questions.map(question => Number(/^\d+/.exec(question.label)?.[0] ?? 0)).filter(Boolean),
  );
  const coverage = expected ? coveredBase.size / expected : 0;
  if (coverage < minCoverage) {
    throw new Error(
      "Verified question coverage " + (coverage * 100).toFixed(1)
      + "% is below " + (minCoverage * 100).toFixed(0) + "%",
    );
  }

  return {
    header: sqpHeader,
    questions,
    coverage,
    parsed: {
      sqpBlocks: sqpBlocks.length,
      msBlocks: msBlocks.length,
      coveredBaseQuestions: coveredBase.size,
      expectedQuestions: expected,
    },
  };
}

export function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

export function normalizedSubjectName(value) {
  const compact = String(value ?? "")
    .normalize("NFKD")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");

  const aliases = new Map([
    ["engg graphic", "engineering graphics"],
    ["computer application", "computer applications"],
    ["english language and literature", "english language literature"],
  ]);
  return aliases.get(compact) ?? compact;
}
