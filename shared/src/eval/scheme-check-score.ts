/**
 * Scores one scheme_check answer against a golden case (AXO-202). Pure, so the rule is tested
 * without a model.
 *
 * A case says whether the question can be checked and, if so, the range of estimated marks an
 * examiner applying that (invented) scheme could defensibly give. Three controls must be right
 * every time, because each is a promise made to the owner:
 *   blank     nothing written → exactly 0
 *   mismatch  the scheme section is for another question → can_check false (an honest gap, never a
 *             confident mark from the wrong scheme)
 *   unreadable  the answer cannot be read → can_check false
 * A "verbatim" case has long scheme lines; copying them is caught by the stage's own validator,
 * which fails the call (schema_valid false), so it is scored like any other case.
 */
import type { CheckResult } from "../prompts/scheme_check.v1.js";

export interface SchemeCheckExpectation {
  can_check: boolean;
  est: [number, number] | null;
  control: "blank" | "mismatch" | "unreadable" | "verbatim" | null;
}

export interface SchemeCheckScore {
  control: SchemeCheckExpectation["control"];
  can_check_ok: boolean;
  /** Estimate inside the accepted range (null when the case is not checkable). */
  in_range: boolean | null;
  /** Estimate above the accepted maximum: the dangerous direction (tells a student they did better). */
  over_estimate: boolean | null;
  abs_error: number | null;
}

export function scoreSchemeCheck(r: CheckResult, e: SchemeCheckExpectation): SchemeCheckScore {
  const can_check_ok = r.canCheck === e.can_check;
  if (!e.can_check || !e.est) {
    return { control: e.control, can_check_ok, in_range: null, over_estimate: null, abs_error: null };
  }
  if (!r.canCheck || r.estimatedMarks === null) {
    return { control: e.control, can_check_ok, in_range: false, over_estimate: false, abs_error: null };
  }
  const [lo, hi] = e.est;
  const est = r.estimatedMarks;
  return {
    control: e.control,
    can_check_ok,
    in_range: est >= lo && est <= hi,
    over_estimate: est > hi,
    abs_error: est < lo ? lo - est : est > hi ? est - hi : 0,
  };
}

export interface SchemeCheckRow { status: string; score: SchemeCheckScore | null }

export function summariseSchemeCheck(rows: SchemeCheckRow[]) {
  const done = rows.filter((r) => r.status === "done" && r.score) as Array<{ status: string; score: SchemeCheckScore }>;
  const failed = rows.filter((r) => r.status === "failed").length;
  const total = done.length + failed;
  const checkable = done.filter((r) => r.score.in_range !== null);
  const inRange = checkable.filter((r) => r.score.in_range).length;
  const over = checkable.filter((r) => r.score.over_estimate).length;
  const ctl = (kind: SchemeCheckExpectation["control"]) => {
    const c = done.filter((r) => r.score.control === kind);
    const ok = c.filter((r) => kind === "blank" ? r.score.in_range === true : r.score.can_check_ok).length;
    return { ok, of: c.length };
  };
  const blank = ctl("blank");
  const mismatch = ctl("mismatch");
  const unreadable = ctl("unreadable");
  return {
    schema_valid: `${done.length}/${total}`,
    schema_valid_rate: total ? done.length / total : 0,
    can_check_agreement: `${done.filter((r) => r.score.can_check_ok).length}/${done.length}`,
    can_check_rate: done.length ? done.filter((r) => r.score.can_check_ok).length / done.length : 0,
    estimate_in_range: `${inRange}/${checkable.length}`,
    in_range_rate: checkable.length ? inRange / checkable.length : 0,
    over_estimates: over,
    blank_zero: `${blank.ok}/${blank.of}`,
    mismatch_refused: `${mismatch.ok}/${mismatch.of}`,
    unreadable_refused: `${unreadable.ok}/${unreadable.of}`,
    controls_all_ok: blank.ok === blank.of && mismatch.ok === mismatch.of && unreadable.ok === unreadable.of,
  };
}

export const SCHEME_CHECK_THRESHOLDS = {
  schema_valid_min: 0.95,
  can_check_min: 0.95,
  in_range_min: 0.8,
  over_estimates_max_rate: 0.1,
  controls: "all",
} as const;

export function passes(s: ReturnType<typeof summariseSchemeCheck>, checkable: number): boolean {
  return s.schema_valid_rate >= SCHEME_CHECK_THRESHOLDS.schema_valid_min
    && s.can_check_rate >= SCHEME_CHECK_THRESHOLDS.can_check_min
    && s.in_range_rate >= SCHEME_CHECK_THRESHOLDS.in_range_min
    && s.over_estimates <= Math.floor(checkable * SCHEME_CHECK_THRESHOLDS.over_estimates_max_rate)
    && s.controls_all_ok;
}
