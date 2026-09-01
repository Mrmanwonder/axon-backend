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
  numberingSound: boolean;
  /** Region-scoped (§6.3): whether this specific region is implicated in the paper's reconciliation discrepancy, if it has one. True (passes) for a region the discrepancy isn't attributable to, even on an otherwise-unreconciled paper. */
  arithmeticOk: boolean;
  awarded: number | null;
  available: number | null;
  unreadable: boolean;
}

export interface AssessSignals {
  recognition: boolean;
  structural: boolean;
  arithmetic: boolean;
  plausibility: boolean;
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
    recognition: input.recognition === "high" || input.recognition === "medium",
    structural: input.numberingSound,
    arithmetic: input.arithmeticOk,
    plausibility: plausible(input.awarded, input.available),
  };
  if (input.unreadable || input.recognition === null) {
    return { tier: "unreadable", signals };
  }
  if (input.recognition === "low") return { tier: "unsure", signals };
  const allPass = Object.values(signals).every(Boolean);
  if (allPass) return { tier: "confident", signals };
  return { tier: "unsure", signals };
}

/** One step down the recognition ladder — used to fold a page-scoped
    layer-fallback into the recognition signal instead of a paper-wide veto. */
export function downgradeRecognition(recognition: Recognition): Recognition {
  if (recognition === "high") return "medium";
  if (recognition === "medium") return "low";
  return recognition;
}

function plausible(awarded: number | null, available: number | null): boolean {
  if (awarded === null || available === null) return false;
  if (awarded < 0 || available <= 0 || awarded > available) return false;
  if (available > 30) return false;
  // Half marks are the finest grain CAIE marking actually uses; anything
  // finer than that is a misread, not a real award.
  return Math.abs(awarded * 2 - Math.round(awarded * 2)) < 1e-6;
}

/**
 * For each region label in document order, whether its number is consistent
 * with the label immediately before it (same number — a multi-part answer —
 * or exactly one higher). A label the model couldn't read (`null`) neither
 * passes nor fails anything else; it is simply skipped when looking
 * backward for "the previous number".
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
