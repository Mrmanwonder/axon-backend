// The confidence-tier logic behind AXON_FIX_BRIEF.md §4.A4. Originally two of
// these four signals were paper-scoped rather than question-scoped — a
// discrepancy anywhere on the paper, or a layer-fallback page anywhere in the
// booklet, vetoed *every* question's confidence, which is why 0 of 48 live
// regions ever reached "confident" and the bulk-accept path was permanently
// empty. Both are region-scoped now (§6.3):
//
//   - `arithmeticOk` is computed per region by the caller (mastery-reconcile),
//     using reconcile()'s own suspect ranking — only the region(s) actually
//     implicated in the discrepancy fail this signal; a clean question on a
//     paper with one bad total can still reach confident.
//   - a layer-fallback page no longer vetoes the tier outright. The caller
//     downgrades that region's `recognition` by one step instead (see
//     workers/reconcile/src/index.ts), scoped to the pages the region's own
//     spans actually touch.

export type ConfidenceTier = "confident" | "unsure" | "unreadable";
export type Recognition = "high" | "medium" | "low" | null;

export interface AssessInput {
  recognition: Recognition;
  /** Numbering judged on placed labels in page order (shared/src/placement.ts). */
  numberingSound: SignalValue;
  /**
   * Whether this region's own working actually checks out, computed by
   * evaluating it — see shared/src/arithmetic.ts.
   *
   * Three-valued on purpose. `"unknown"` is the normal verdict for a prose
   * answer that contains no arithmetic, and it is not a defect: forcing a
   * boolean here is what produced live rows where "Not Normalized" carried an
   * `arithmetic` signal of false on one run and true on four others. There is
   * no arithmetic in those two words to be true or false about.
   *
   * Note this is no longer the reconciliation-discrepancy signal it was. That
   * was paper-scoped arithmetic about totals; this is the region's own chain.
   * Totals are checked separately and deterministically by reconcile().
   */
  arithmeticOk: boolean | "unknown";
  awarded: number | null;
  available: number | null;
  unreadable: boolean;
  /**
   * The paper is shown to be unmarked (council D1, 7 Oct 2026): triage read it
   * as `ungraded_paper` at `high` confidence. Only then is a missing teacher
   * mark the absence of a check (`unknown`) rather than a failed one. Defaults
   * to false: on a marked paper a missing mark is a failure, and it is the
   * signal that caught all five known misreads (ADDENDUM-01A item 4).
   */
  paperUnmarked?: boolean;
}

/**
 * Every signal may say it does not know.
 *
 * A boolean has no way to express "this check does not apply here", so it
 * guesses, and a forced guess is a hallucination with a schema. See
 * shared/src/signals.ts for how the four are read back and kept apart.
 */
export type SignalValue = boolean | "unknown";

export interface AssessSignals {
  recognition: SignalValue;
  structural: SignalValue;
  arithmetic: SignalValue;
  plausibility: SignalValue;
}

export interface AssessResult {
  tier: ConfidenceTier;
  signals: AssessSignals;
}

export function assess(input: AssessInput): AssessResult {
  const signals: AssessSignals = {
    // 'medium' passes. A pass here is not a claim the reading is right — it is a
    // claim that nothing about the recognition itself was alarming, and the
    // other three signals are what turn that into confidence.
    recognition: input.recognition === null ? "unknown" : input.recognition === "high" || input.recognition === "medium",
    structural: input.numberingSound,
    arithmetic: input.arithmeticOk,
    plausibility: plausible(input.awarded, input.available, input.paperUnmarked === true),
  };
  if (input.unreadable) {
    return { tier: "unreadable", signals };
  }
  if (input.recognition === "low" || input.recognition === null) return { tier: "unsure", signals };

  // A signal that is explicitly false blocks confidence. `unknown` does not:
  // it is the absence of a check, not the failure of one, and an answer with no
  // arithmetic in it must not be made unsure for failing to contain any.
  //
  // The one signal that is never allowed to be unknown-and-confident is
  // recognition, and that is handled above: if we could not read the
  // handwriting, nothing downstream is authoritative however tidy it looked.
  // Live rows carried `recognition: false` with `arithmetic: true` and
  // committed marks anyway; that is the collapse this ordering removes.
  const anyFalse = (Object.values(signals) as SignalValue[]).some((v) => v === false);
  if (anyFalse) return { tier: "unsure", signals };
  return { tier: "confident", signals };
}

/** One step down the recognition ladder — used to fold a page-scoped
    layer-fallback into the recognition signal instead of a paper-wide veto. */
export function downgradeRecognition(recognition: Recognition): Recognition {
  if (recognition === "high") return "medium";
  if (recognition === "medium") return "low";
  return recognition;
}

/**
 * Is the teacher's mark, as read, a mark that can exist?
 *
 * Out-of-range or impossible values are always false. A missing mark is false
 * on a marked paper and `unknown` only on a paper shown to be unmarked, where
 * there is no mark to have read. There is no cap on the marks available: a
 * 40-mark essay question is real (ADDENDUM-01A item 9).
 */
export function plausible(awarded: number | null, available: number | null, paperUnmarked = false): SignalValue {
  if (awarded !== null && (!Number.isFinite(awarded) || awarded < 0)) return false;
  if (available !== null && (!Number.isFinite(available) || available <= 0)) return false;
  if (awarded !== null && available !== null && awarded > available) return false;
  // Half marks are the finest grain CAIE marking actually uses; anything
  // finer than that is a misread, not a real award.
  if (awarded !== null && Math.abs(awarded * 2 - Math.round(awarded * 2)) >= 1e-6) return false;
  if (awarded === null || available === null) return paperUnmarked ? "unknown" : false;
  return true;
}

/** Why a part asks the student. Empty means it does not ask. */
export type AskReason = "unreadable" | "recognition" | "teacher_mark" | "unplaceable";

export interface AskInput {
  unreadable: boolean;
  /** The recognition the tier was judged on (after any page downgrade). */
  recognition: Recognition;
  /** True unless the paper is shown to be unmarked. */
  paperMarked: boolean;
  plausibility: SignalValue;
  /** Placement: unassigned, or still colliding once placed. */
  unplaceable: boolean;
  /** False for an unlabelled region with no mark, answer or question text. */
  counted: boolean;
}

/**
 * Council D1 ask-rule (7 Oct 2026). A part asks the student only when
 *   (a) it is unreadable, or its recognition is low or null;
 *   (b) the paper is marked, and the teacher's mark or the marks available is
 *       missing or impossible;
 *   (c) it cannot be placed (unassigned, or colliding after placement).
 * Arithmetic failing alone does not ask: the part stays `unsure` with "Fix
 * this". A region that is not a part (no label and nothing on it) is asked
 * only under (a).
 */
export function askReasons(input: AskInput): AskReason[] {
  const out: AskReason[] = [];
  if (input.unreadable) out.push("unreadable");
  else if (input.recognition === "low" || input.recognition === null) out.push("recognition");
  if (input.counted && input.paperMarked && input.plausibility === false) out.push("teacher_mark");
  if (input.counted && input.unplaceable) out.push("unplaceable");
  return out;
}

/**
 * Shown to be unmarked: triage classified the paper `ungraded_paper` at `high`
 * confidence. There is no student-facing "this paper is unmarked" field today
 * (only the opposite, `route_override.student_says_marked`, which triage
 * already folds into the stored classification).
 */
export function paperShownUnmarked(tierRouting: unknown): boolean {
  const triage = (tierRouting as { triage?: { classification?: unknown; confidence?: unknown } } | null)?.triage;
  return triage?.classification === "ungraded_paper" && triage?.confidence === "high";
}

/**
 * For each region label in document order, whether its number is consistent
 * with the label immediately before it (same number — a multi-part answer —
 * or exactly one higher). A label the model couldn't read (`null`) neither
 * passes nor fails anything else; it is simply skipped when looking
 * backward for "the previous number".
 *
 * @deprecated Raw labels in stored order. Reconcile judges numbering with
 * `placementVerdicts` (shared/src/placement.ts) on placed labels in page order
 * since council D1; this is kept only for callers outside the pipeline.
 */
export function numberingSoundness(labels: Array<string | null>): boolean[] {
  const numeric = labels.map(mainNumber);
  return labels.map((label, i) => {
    if (label === null) return false;
    const n = numeric[i];
    if (n === null) return true;
    const previous = lastNumberBefore(numeric, i);
    if (previous === null) return true;
    return n === previous || n === previous + 1;
  });
}

function mainNumber(label: string | null): number | null {
  if (!label) return null;
  const m = label.match(/\d+/);
  return m ? Number(m[0]) : null;
}

function lastNumberBefore(numeric: Array<number | null>, i: number): number | null {
  for (let j = i - 1; j >= 0; j--) if (numeric[j] !== null) return numeric[j];
  return null;
}
