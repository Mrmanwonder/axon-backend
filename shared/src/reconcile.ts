export interface QuestionMarks {
  order_index: number;
  label: string | null;
  awarded: number | null;
  available: number | null;
  recognition: "high" | "medium" | "low";
}

export interface ReconcileChecks {
  awarded_matches_total: boolean | null;
  available_matches_maximum: boolean | null;
  every_question_within_its_maximum: boolean;
}

export interface ReconcileResult {
  reconciled: boolean;
  delta: number | null;
  sum_awarded: number;
  sum_available: number;
  checks: ReconcileChecks;
  suspects: number[];
  message: string | null;
}

const sum = (ns: Array<number | null>) => ns.reduce((t: number, n) => t + (typeof n === "number" ? n : 0), 0);
const nearly = (a: number, b: number) => Math.abs(a - b) < 0.01;
const round2 = (n: number) => Math.round(n * 100) / 100;
const trim = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(1));

/**
 * Checks whether the per-question marks a run extracted add up to the
 * totals the paper itself reports, and ranks which questions are the most
 * likely source of any mismatch — feeding mastery-adjudicate's crop
 * selection when they don't.
 */
export function reconcile(regions: QuestionMarks[], reportedTotal: number | null, statedMaximum: number | null): ReconcileResult {
  const sumAwarded = sum(regions.map((r) => r.awarded));
  const sumAvailable = sum(regions.map((r) => r.available));
  const awardedMatches = reportedTotal === null ? null : nearly(sumAwarded, reportedTotal);
  const availableMatches = statedMaximum === null ? null : nearly(sumAvailable, statedMaximum);
  const withinMax = regions.every((r) => r.awarded === null || r.available === null || r.awarded <= r.available + 1e-6);
  const reconciled =
    withinMax && (awardedMatches ?? true) && (availableMatches ?? true) && (awardedMatches !== null || availableMatches !== null);
  const delta = reportedTotal === null ? null : round2(sumAwarded - reportedTotal);
  return {
    reconciled,
    delta,
    sum_awarded: round2(sumAwarded),
    sum_available: round2(sumAvailable),
    checks: {
      awarded_matches_total: awardedMatches,
      available_matches_maximum: availableMatches,
      every_question_within_its_maximum: withinMax,
    },
    suspects: rankSuspects(regions, delta),
    message: messageFor(regions, reportedTotal, sumAwarded, delta, withinMax),
  };
}

function rankSuspects(regions: QuestionMarks[], delta: number | null): number[] {
  const score = (r: QuestionMarks) => {
    let s = 0;
    if (r.recognition === "low") s += 3;
    else if (r.recognition === "medium") s += 1;
    if (r.awarded === null) s += 3;
    if (r.available === null) s += 2;
    if (r.label === null) s += 1;
    return s;
  };
  const ranked = regions
    .map((r) => ({ index: r.order_index, score: score(r) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((r) => r.index);
  if (delta !== null && delta !== 0) {
    const exact = regions.find((r) => r.awarded !== null && nearly(Math.abs(delta), r.awarded));
    if (exact) return [exact.order_index, ...ranked.filter((i) => i !== exact.order_index)];
  }
  return ranked;
}

function messageFor(regions: QuestionMarks[], reportedTotal: number | null, sumAwarded: number, delta: number | null, withinMax: boolean): string | null {
  if (!withinMax) {
    return "Our reading of this paper gives one question more marks than it was worth — worth checking these.";
  }
  if (reportedTotal === null) {
    return regions.length ? "We could not find this paper's total, so we could not check our reading against it." : null;
  }
  if (delta === null || delta === 0) return null;
  return `Our reading of this paper adds up to ${trim(sumAwarded)}, and the total on the paper is ${trim(reportedTotal)} — worth checking these questions.`;
}
