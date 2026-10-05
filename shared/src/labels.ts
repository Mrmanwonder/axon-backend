/**
 * Question labels, validated structurally rather than trusted.
 *
 * Two failures in the live data, both decidable without a model:
 *
 *   The same paper carries `2a` on some runs and `2. a)` on another, for the
 *   same question — so a duplicate is invisible to any check comparing strings.
 *
 *   An adjudication note on a committed row reads, verbatim: "The pipeline read
 *   3/3 for question c, but looking at the first page, the mark for 1a is 3 and
 *   1b is 1. The pipeline seems to have misidentified the question labels or
 *   order." That row carries marks_awarded = 3.00. The system worked out that
 *   its own labelling was wrong, wrote the finding into the record, and
 *   committed the marks anyway.
 *
 * A label set that repeats a part, or that cannot be ordered, is a structural
 * failure of the extraction. It is not a matter of degree and not something to
 * weigh against other evidence — it is arithmetic on a set, and it belongs
 * here rather than in a prompt.
 */

/** `2. a)` and `2a` are the same part. `d) (i)` and `(d)(i)` are the same part. */
export function canonicalLabel(label: string | null | undefined): string | null {
  if (typeof label !== "string") return null;
  const s = label.toLowerCase().trim();

  // A bare roman sub-part, printed alone under its parent letter — "(ii)" on
  // its own line, which is how Cambridge sets a continued sub-part. Checked
  // first: the general pattern below would read "(ii)" as the letter `i`
  // followed by the roman `i`, which is a different part entirely.
  const bareRoman = s.match(/^\(\s*([ivx]+)\s*\)$|^([ivx]{2,})[.)]?$/);
  if (bareRoman) return `(${bareRoman[1] ?? bareRoman[2]})`;

  // Optional leading question number, then a part letter, then an optional
  // roman sub-part. Everything else — brackets, dots, spaces — is punctuation.
  const m = s.match(/^\(?\s*(?:(\d+)\s*[.)]?\s*)?\(?\s*([a-z])\s*\)?\s*(?:[.)]?\s*\(?\s*([ivx]+)\s*\)?)?\s*$/);
  if (m) {
    const [, num, letter, roman] = m;
    return `${num ? num : ""}${letter}${roman ? `(${roman})` : ""}`;
  }
  // A bare question number, "1", "2".
  const bare = s.match(/^\(?\s*(\d+)\s*[.)]?\s*$/);
  if (bare) return bare[1];
  return null;
}

export interface LabelProblem {
  kind: "duplicate" | "unreadable";
  label: string;
  /** Every region index carrying it, so a reviewer can be sent to the pair. */
  at: number[];
}

export interface LabelCheck {
  ok: boolean;
  problems: LabelProblem[];
}

/**
 * Is this run's label set well-formed?
 *
 * Duplicates are compared on the canonical form, so `2a` and `2. a)` collide as
 * they should. An unreadable label is reported but does not on its own fail the
 * set: a region we could not number is already `unsure` by other means, and
 * refusing the whole paper for one of them would throw away nineteen good
 * questions to flag one.
 *
 * A duplicate does fail the set. Two regions claiming the same part means the
 * marks on at least one of them are attached to the wrong question, and there
 * is no reading of that which is safe to commit.
 */
export function checkLabels(labels: (string | null | undefined)[]): LabelCheck {
  const seen = new Map<string, number[]>();
  const problems: LabelProblem[] = [];
  const keys = placedLabels(labels);

  keys.forEach((key, i) => {
    if (key === null) {
      problems.push({ kind: "unreadable", label: String(labels[i] ?? ""), at: [i] });
      return;
    }
    const at = seen.get(key);
    if (at) at.push(i); else seen.set(key, [i]);
  });

  for (const [key, at] of seen) {
    if (at.length > 1) problems.push({ kind: "duplicate", label: key, at });
  }

  return { ok: !problems.some((p) => p.kind === "duplicate"), problems };
}

/** Every region index that sits in a duplicate pair: the regions whose marks may be on the wrong part. */
export function duplicateIndexes(check: LabelCheck): Set<number> {
  return new Set(check.problems.filter((p) => p.kind === "duplicate").flatMap((p) => p.at));
}

/**
 * Canonical labels in document order, with a bare part placed under the
 * question it continues.
 *
 * Cambridge prints the number once: `2(a)` and `2(b)`, then `(c)` on the next
 * page. Read as text, every later `(c)` on the paper collides with that one,
 * and on 5 Oct 2026 that made all nineteen regions of a cleanly read paper
 * structurally unsound (owner's run c5bc874b). A bare part continues the
 * current question only when it moves forward (a letter after the last one,
 * or a sub-part under the same letter). A part that does not move forward
 * starts an unnumbered question and stays bare, so a genuine repeat is still
 * caught rather than explained away.
 */
export function placedLabels(labels: (string | null | undefined)[]): (string | null)[] {
  let num: string | null = null;
  let letter: string | null = null;
  return labels.map((raw) => {
    const key = canonicalLabel(raw);
    if (key === null) return null;
    const bareRoman = key.match(/^\(([ivx]+)\)$/);
    if (bareRoman) return num !== null && letter !== null ? `${num}${letter}(${bareRoman[1]})` : key;
    const m = key.match(/^(\d+)?([a-z])?(\([ivx]+\))?$/);
    if (!m) return key;
    const [, n, l, roman] = m;
    if (n) {
      num = n;
      letter = l ?? null;
      return key;
    }
    if (!l) return key;
    const forward = letter === null ? l === "a" || num !== null : roman ? l >= letter : l > letter;
    if (num !== null && forward) {
      letter = l;
      return `${num}${l}${roman ?? ""}`;
    }
    num = null;
    letter = l;
    return key;
  });
}

/**
 * Did the adjudicator report a structural problem with this paper?
 *
 * `needs_review` exists for exactly this, and the row quoted above proves it
 * was not being used: the adjudication said the labels were misidentified and
 * the commit went ahead. Any of these findings must block the commit and route
 * the paper to a person.
 */
export function adjudicationBlocksCommit(adjudication: unknown): { blocked: boolean; reason: string | null } {
  if (!adjudication || typeof adjudication !== "object") return { blocked: false, reason: null };
  const a = adjudication as Record<string, unknown>;

  const verdict = typeof a.verdict === "string" ? a.verdict.toLowerCase() : "";
  if (verdict === "reject" || verdict === "needs_review" || verdict === "blocked") {
    return { blocked: true, reason: `adjudication verdict: ${verdict}` };
  }
  if (a.labels_misidentified === true || a.structural_problem === true) {
    return { blocked: true, reason: "adjudication reported a structural problem" };
  }

  // The evidence field is prose written by the adjudicator. It is the only
  // place the label finding appeared on the committed row, so it is read — but
  // only for phrases that state a structural failure outright, never as a
  // general sentiment check, because that would be a model judging a model
  // again and this module exists to stop doing that.
  const evidence = typeof a.evidence === "string" ? a.evidence.toLowerCase() : "";
  for (const phrase of [
    "misidentified the question label",
    "misidentified the question labels",
    "wrong question label",
    "question labels or order",
    "labels or order",
  ]) {
    if (evidence.includes(phrase)) {
      return { blocked: true, reason: "adjudication reported misidentified question labels" };
    }
  }
  return { blocked: false, reason: null };
}
