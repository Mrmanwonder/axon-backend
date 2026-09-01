// The confidence-tier logic behind AXON_FIX_BRIEF.md §4.A4: two of these four
// signals are currently paper-scoped rather than question-scoped, which is
// why 0 of 48 live regions have ever reached "confident" (see the brief for
// why that isn't safe to just relax — §6.3 is the tracked fix, not this file).

export type ConfidenceTier = "confident" | "unsure" | "unreadable";
export type Recognition = "high" | "medium" | "low" | null;

export interface AssessInput {
  recognition: Recognition;
  numberingSound: boolean;
  /** Paper-level today (AXON_FIX_BRIEF.md §4.A4) — whether the whole paper's totals reconcile. */
  paperReconciled: boolean;
  awarded: number | null;
  available: number | null;
  /** Paper-level today (AXON_FIX_BRIEF.md §4.A4) — true if any page on the paper used the non-red-ink or student-wrote-in-red fallback. */
  layerFallback: boolean;
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
    arithmetic: input.paperReconciled,
    plausibility: plausible(input.awarded, input.available),
  };
  if (input.unreadable || input.recognition === null) {
    return { tier: "unreadable", signals };
  }
  if (input.recognition === "low") return { tier: "unsure", signals };
  const allPass = Object.values(signals).every(Boolean);
  if (allPass && !input.layerFallback) return { tier: "confident", signals };
  return { tier: "unsure", signals };
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
