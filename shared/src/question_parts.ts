/**
 * Which other parts of a question this part depends on.
 *
 * Cambridge papers are built out of dependent parts. "Justify your answer given
 * in part (d)(i)" is not a question that can be read on its own: every word of
 * it points at another part, and an explanation written without that part is
 * writing about a question it has not seen.
 *
 * This is not hypothetical. On 2026-09-05 the explain worker produced, for
 * exactly that question on a Computer Science paper about floating-point
 * normalisation, a corrected working reading "the signal-to-noise ratio remains
 * above the threshold required for accurate signal reconstruction". The
 * student's own d(i) — "State whether the floating-point number given in part
 * (c) is normalised or not normalised", answered "Not Normalized", 1/1 — was
 * sitting one row away in `question_region`, in the same run, at the adjacent
 * `order_index`. Nothing fetched it. The model was asked to justify an answer
 * it had never been shown, and it did what a model does with a question it
 * cannot see: it wrote fluent, confident, subject-flavoured filler.
 *
 * So the parsing here has one job with two halves. It finds the parts a
 * question points at so they can be put in front of the model, and — where a
 * part it points at cannot be found — it says so, so the pipeline can decline
 * to write a corrected working rather than inventing one.
 */

/** A part label reduced to a comparable key: "d) (i)" and "(d)(i)" both -> "d(i)". */
export function normalisePartKey(label: string | null | undefined): string | null {
  if (typeof label !== "string") return null;
  // Only whitespace is stripped. Trimming a trailing ")" here ate the closing
  // paren of "d(i)" and sent every sub-part down the null path — the regex
  // below already treats the letter's own bracket as optional, so "d)" and
  // "d(i)" both land without help.
  const cleaned = label.toLowerCase().trim();
  // "2a", "2. a)", "d(i)", "(ii)", "b" — pull the letter and any roman suffix.
  const m = cleaned.match(/^\(?\s*(?:\d+\s*[.)]?\s*)?\(?\s*([a-z])\s*\)?\s*(?:\(\s*([ivx]+)\s*\))?\s*$/);
  if (m) return m[2] ? `${m[1]}(${m[2]})` : m[1];
  // A bare roman part — "(ii)" on its own line, which is how a sub-part often
  // prints once its parent letter is established.
  const roman = cleaned.match(/^\(?\s*([ivx]+)\s*\)?$/);
  if (roman) return `(${roman[1]})`;
  return null;
}

/**
 * Generic exam scaffolding that must never be mistaken for a part reference.
 * "part of the process" is prose; "part (d)" is a pointer.
 */
const REFERENCE_LEAD = /\b(?:part|parts)\b/gi;

/**
 * The parts this question text points at, as normalised keys.
 *
 * Deliberately conservative. A missed reference costs context the model would
 * have liked; a false one makes a question look ungrounded and suppresses a
 * corrected working that was fine. So a bare letter only counts when it is not
 * "a" — "part a" is real but indistinguishable from "part a system", and the
 * article is by far the most common way that guess goes wrong. Parenthesised
 * letters carry no such ambiguity and are always taken.
 */
export function partReferences(questionText: string | null | undefined): string[] {
  if (typeof questionText !== "string" || !questionText.trim()) return [];
  const keys = new Set<string>();
  const text = questionText.toLowerCase();

  // "(d)(i)" written out in full, anywhere — the strongest signal there is, and
  // it does not need the word "part" in front of it.
  for (const m of text.matchAll(/\(\s*([a-z])\s*\)\s*\(\s*([ivx]+)\s*\)/g)) {
    keys.add(`${m[1]}(${m[2]})`);
  }

  // Everything hanging off the word "part"/"parts": "part (c)", "parts (a) and
  // (b)", "part (d) (i)", "part b". Scanned in a short window so a later
  // unrelated parenthesis cannot be swept in.
  for (const lead of text.matchAll(REFERENCE_LEAD)) {
    const window = text.slice(lead.index! + lead[0].length, lead.index! + lead[0].length + 48);
    // Stop at the first sentence break — a reference does not cross one.
    const scope = window.split(/[.;:?]/)[0];
    for (const m of scope.matchAll(/\(\s*([a-z])\s*\)\s*(?:\(\s*([ivx]+)\s*\))?/g)) {
      keys.add(m[2] ? `${m[1]}(${m[2]})` : m[1]);
    }
    // "part b" / "parts b and c", unparenthesised. "a" is excluded on purpose.
    for (const m of scope.matchAll(/(?:^|\s|and\s)([b-z])(?![a-z(])/g)) {
      // Only immediately after the lead word or a conjunction, so "part of the
      // system" cannot contribute its "of"/"the" neighbours.
      if (m.index! <= 6 || /and\s$/.test(scope.slice(0, m.index!))) keys.add(m[1]);
    }
  }

  return [...keys];
}

/** One earlier part of the same question, as the explain prompt will see it. */
export interface PriorPart {
  label: string;
  key: string | null;
  questionText: string | null;
  studentAnswer: string | null;
  marksAwarded: number | null;
  marksAvailable: number | null;
}

export interface DependencyResolution {
  /** The referenced parts we found, in paper order. Goes into the prompt. */
  resolved: PriorPart[];
  /** Referenced parts we could not find. Non-empty means "do not claim to know". */
  unresolved: string[];
  /** Whether this question points at another part at all. */
  dependent: boolean;
}

/**
 * Match a question's references against the other parts of the same run.
 *
 * `siblings` is every other region of the run in paper order. Only parts
 * *before* this one are eligible: "justify your answer to (d)(i)" refers
 * backwards, and a forward match would mean the labels are wrong in a way that
 * should not be papered over by feeding the model a later question.
 */
export function resolveDependencies(
  questionText: string | null | undefined,
  ownOrderIndex: number,
  siblings: PriorPart[] & { orderIndex?: number }[],
): DependencyResolution {
  const refs = partReferences(questionText);
  if (!refs.length) return { resolved: [], unresolved: [], dependent: false };

  const earlier = (siblings as (PriorPart & { orderIndex: number })[])
    .filter((s) => s.orderIndex < ownOrderIndex)
    .sort((a, b) => a.orderIndex - b.orderIndex);

  const resolved: PriorPart[] = [];
  const unresolved: string[] = [];

  for (const ref of refs) {
    // Exact key first. Then the roman-only form: "(ii)" printed alone belongs to
    // whichever letter most recently carried one.
    const hit =
      earlier.find((s) => s.key === ref) ??
      (ref.startsWith("(") ? earlier.find((s) => s.key?.endsWith(ref)) : undefined) ??
      // "part (d)" where the parts are labelled d(i) and d(ii): the letter names
      // the group, and every part of it is the referent.
      undefined;

    if (hit) {
      if (!resolved.some((r) => r.label === hit.label)) resolved.push(hit);
      continue;
    }

    const group = ref.length === 1 ? earlier.filter((s) => s.key?.startsWith(ref)) : [];
    if (group.length) {
      for (const g of group) if (!resolved.some((r) => r.label === g.label)) resolved.push(g);
      continue;
    }

    unresolved.push(ref);
  }

  return { resolved, unresolved, dependent: true };
}
