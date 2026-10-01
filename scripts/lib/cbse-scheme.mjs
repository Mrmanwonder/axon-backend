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
  // The title block only. Subject codes and "Class" also appear inside
  // questions ("a class of 40 students"), so the looser patterns below never
  // look past the first lines of the paper.
  const head = body.split("\n").filter(line => line.trim()).slice(0, 12).join("\n");
  const subjectMatch =
    /Subject\s*:\s*([^\n(]+?)\s*\((\d{3})\)/i.exec(body)
    ?? /(?:SUBJECT|SUB)\s*[:\-]\s*([^\n]+?)\s+(?:CODE\s*[:\-]?\s*)?(\d{3})\b/i.exec(body)
    // "CHEMISTRY (CODE – 043)"
    ?? /^\s*([A-Za-z][A-Za-z &.]*?)\s*\(\s*CODE\s*[–—:\-]?\s*(\d{3})\s*\)/im.exec(head)
    // "BIOLOGY – CODE NO. 044", "SCIENCE – Code no. 086", "MATHEMATICS STANDARD – Code No. (041)"
    ?? /^\s*([A-Za-z][A-Za-z &.]*?)\s*[–—\-]?\s*CODE\s*NO\.?\s*\(?\s*(\d{3})\s*\)?/im.exec(head)
    // A bare title line: "MATHEMATICS (041)"
    ?? /^\s*([A-Za-z][A-Za-z &.]*?)\s*\((\d{3})\)\s*$/m.exec(head);
  const classMatch = /Class\s*[–—-]\s*(XII|X)\b/i.exec(body)
    ?? /CLASS\s*[:\-]?\s*(XII|X)\b/i.exec(body);
  const sessionMatch = /(?:Academic\s+Session|Session)\s*(20\d{2})\s*[–—-]\s*(\d{2,4})/i.exec(body)
    // "CLASS XII (2026-27)", "CLASS- XII (2026 - 27)", "CLASS – X (2026–27)"
    ?? /CLASS\s*[–—:\-]?\s*(?:XII|X)\s*\(\s*(20\d{2})\s*[–—-]\s*(\d{2,4})\s*\)/i.exec(head);
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
  const match = /^\s*(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*)?(\d{1,2})(?:\s*\(([AB])\))?\s*[.)]?\s*(.*)$/i.exec(line);
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

function markNumber(value) {
  const token = String(value).trim();
  if (token === "½") return 0.5;
  if (token === "¼") return 0.25;
  if (token === "¾") return 0.75;
  const number = Number(token);
  return Number.isFinite(number) ? number : null;
}

function parseMarkExpression(value) {
  const terms = String(value).replace(/×/g, "x").split("+");
  let total = 0;
  for (const term of terms) {
    const factors = term.split(/[x*]/i).map(markNumber);
    if (!factors.length || factors.some(factor => factor === null || factor <= 0)) return null;
    total += factors.reduce((product, factor) => product * factor, 1);
  }
  return total > 0 && total <= 30 ? Number(total.toFixed(4)) : null;
}

function markExpressionAtEnd(line) {
  return /(?:^|\s{2,})((?:½|¼|¾|\d+(?:\.\d+)?)(?:\s*(?:\+|[x×*])\s*(?:½|¼|¾|\d+(?:\.\d+)?))*)\s*$/.exec(line);
}

export function detectMarksColumn(lines) {
  for (const line of lines) {
    if (/Q\.?\s*No/i.test(line) && /Marks/i.test(line)) return line.lastIndexOf("Marks");
  }
  for (const line of lines) {
    if (/Question/i.test(line) && /Marks/i.test(line)) return line.lastIndexOf("Marks");
  }
  return null;
}

export function markColumnDiagnostics(text) {
  const lines = String(text).replace(/\r/g, "").split("\n");
  const detected = detectMarksColumn(lines);
  const histogram = new Map();
  let maxLength = 0;
  for (const line of lines) {
    maxLength = Math.max(maxLength, line.length);
    const match = markExpressionAtEnd(line);
    if (!match) continue;
    const expressionIndex = line.lastIndexOf(match[1]);
    const key = String(expressionIndex);
    histogram.set(key, (histogram.get(key) ?? 0) + 1);
  }
  return {
    detected,
    maxLength,
    positions: [...histogram.entries()]
      .map(([index, count]) => ({ index: Number(index), count }))
      .sort((a, b) => b.count - a.count || b.index - a.index)
      .slice(0, 20),
  };
}

export function extractMarkTotal(lines, markColumn = null) {
  const candidates = [];
  for (const line of lines) {
    const match = markExpressionAtEnd(line);
    if (!match) continue;
    const expressionIndex = line.lastIndexOf(match[1]);
    if (markColumn !== null && expressionIndex < Math.max(0, markColumn - 12)) continue;
    const total = parseMarkExpression(match[1]);
    if (total !== null) candidates.push(total);
  }
  if (!candidates.length) return null;
  const total = candidates.reduce((sum, value) => sum + value, 0);
  return total > 0 && total <= 30 ? Number(total.toFixed(4)) : null;
}

function stripMarksColumn(line, markColumn = null) {
  const match = markExpressionAtEnd(line);
  if (!match) return line.trimEnd();
  const expressionIndex = line.lastIndexOf(match[1]);
  if (markColumn !== null && expressionIndex < Math.max(0, markColumn - 12)) return line.trimEnd();
  return line.slice(0, expressionIndex).trimEnd();
}

export function parseQuestionBlocks(text) {
  const lines = String(text).replace(/\r/g, "").split("\n");
  const markColumn = detectMarksColumn(lines);
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
    const maxMarks = extractMarkTotal(rawLines, markColumn);
    const cleaned = rawLines
      .map(line => stripMarksColumn(line, markColumn))
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

const NUMBER_WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17,
  eighteen: 18, nineteen: 19, twenty: 20,
};

function countValue(token) {
  if (/^\d+$/.test(token)) return Number(token);
  return NUMBER_WORDS[String(token).toLowerCase()] ?? null;
}

/**
 * The paper's own statement of how many marks each question carries, read from
 * its General Instructions: "Section B contains five questions of two marks
 * each" or "In Section C, Question number 26 to 31 … carrying 3 marks each".
 *
 * Returns question number → marks, or null when no complete, contiguous plan
 * can be read. A record whose parsed maximum disagrees with the paper's own
 * plan is a parse error (two sub-parts merged, an internal-choice alternative
 * summed), and a wrong maximum is a number a student would see as fact.
 */
export function parseSectionPlan(sqpText) {
  const flat = String(sqpText).replace(/\r/g, "").split("\n").slice(0, 120).join(" ").replace(/\s+/g, " ");
  const chunks = flat.split(/(?=\bSection\s*[–—-]?\s*[A-E]\b)/i);
  const sections = new Map();
  const NUM = "(\\d+|" + Object.keys(NUMBER_WORDS).join("|") + ")";
  for (const chunk of chunks) {
    const letter = /^Section\s*[–—-]?\s*([A-E])\b/i.exec(chunk)?.[1]?.toUpperCase();
    if (!letter || sections.has(letter)) continue;
    const body = chunk.slice(0, 260);
    const marks = countValue(new RegExp("(?:of|carrying)\\s+" + NUM + "\\s*marks?\\b", "i").exec(body)?.[1] ?? "");
    if (!marks) continue;
    const ranges = [...body.matchAll(/(\d{1,2})\s*(?:to|and|-|–)\s*(\d{1,2})/g)].map(m => [Number(m[1]), Number(m[2])]);
    if (/Question/i.test(body) && ranges.length) {
      const nums = ranges.flat();
      sections.set(letter, { marks, from: Math.min(...nums), to: Math.max(...nums) });
      continue;
    }
    const count = countValue(new RegExp("(?:contains|consists of|has)\\s+" + NUM + "\\b", "i").exec(body)?.[1] ?? "");
    if (count) sections.set(letter, { marks, count });
  }
  if (!sections.size) return null;

  const plan = new Map();
  let next = 1;
  for (const letter of ["A", "B", "C", "D", "E"].filter(l => sections.has(l))) {
    const section = sections.get(letter);
    const from = section.from ?? next;
    const to = section.to ?? next + section.count - 1;
    if (from !== next || to < from) return null;
    for (let q = from; q <= to; q++) plan.set(q, section.marks);
    next = to + 1;
  }
  return plan;
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
  const pairingFailures = [];

  for (const question of sqpBlocks) {
    const scheme = schemeByLabel.get(question.label);
    if (!scheme) {
      pairingFailures.push({ label: question.label, reason: "scheme_label_missing" });
      continue;
    }
    if (!question.maxMarks) {
      pairingFailures.push({ label: question.label, reason: "sqp_marks_missing", schemeMarks: scheme.maxMarks });
      continue;
    }
    if (!scheme.maxMarks) {
      pairingFailures.push({ label: question.label, reason: "scheme_marks_missing", sqpMarks: question.maxMarks });
      continue;
    }
    if (question.maxMarks !== scheme.maxMarks) {
      pairingFailures.push({
        label: question.label,
        reason: "mark_mismatch",
        sqpMarks: question.maxMarks,
        schemeMarks: scheme.maxMarks,
      });
      continue;
    }
    questions.push({
      label: question.label,
      questionText: question.text,
      markingScheme: scheme.text,
      maxMarks: question.maxMarks,
    });
  }

  const plan = parseSectionPlan(sqpText);
  if (!plan) throw new Error("Could not read the paper's section mark plan; refusing to certify any maximum mark");
  for (let i = questions.length - 1; i >= 0; i--) {
    const base = Number(/^\d+/.exec(questions[i].label)?.[0] ?? 0);
    const declared = plan.get(base);
    if (declared !== questions[i].maxMarks) {
      pairingFailures.push({ label: questions[i].label, reason: "section_plan_mismatch", parsedMarks: questions[i].maxMarks, declaredMarks: declared ?? null });
      questions.splice(i, 1);
    }
  }

  const expected = sqpHeader.expectedQuestions ?? Math.max(0, ...sqpBlocks.map(block => block.number));
  const sqpBase = new Set(sqpBlocks.map(block => block.number));
  const msBase = new Set(msBlocks.map(block => block.number));
  const coveredBase = new Set(
    questions.map(question => Number(/^\d+/.exec(question.label)?.[0] ?? 0)).filter(Boolean),
  );
  const expectedLabels = expected ? Array.from({ length: expected }, (_, index) => index + 1) : [];
  const sqpMissingBase = expectedLabels.filter(number => !sqpBase.has(number));
  const msMissingBase = expectedLabels.filter(number => !msBase.has(number));
  const coverage = expected ? coveredBase.size / expected : 0;
  if (coverage < minCoverage) {
    const diagnostic = {
      sqpParsed: sqpBlocks.length,
      msParsed: msBlocks.length,
      coveredBaseQuestions: coveredBase.size,
      expectedQuestions: expected,
      sqpMissingBase,
      msMissingBase,
      pairingFailures,
      sqpColumns: markColumnDiagnostics(sqpText),
      msColumns: markColumnDiagnostics(msText),
    };
    throw new Error(
      "Verified question coverage " + (coverage * 100).toFixed(1)
      + "% is below " + (minCoverage * 100).toFixed(0)
      + "%; diagnostics=" + JSON.stringify(diagnostic),
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
