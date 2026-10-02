/**
 * Deterministic checks for the explain eval (AXO-41). Nothing here calls a model, so a number from
 * this file means the same thing every time it is computed.
 *
 * Hard rule 1: the model never assigns or disputes marks. The teacher's number is a fact, so
 * whether an explanation contradicts it is checkable without a judge: the phrasings that
 * adjudicate, and any stated mark total that is not the one the teacher gave.
 */
import { clearsTheFloor } from "../quality_floor.js";
import type { ExplainResult } from "../prompts/explain_tier1.v2.js";

export interface MarkContext {
  marksAwarded: number;
  marksAvailable: number;
}

const ADJUDICATING: Array<[RegExp, string]> = [
  [/\bshould (?:have|be) (?:got|gotten|received|awarded|given|scored)\b/i, "should_have_got"],
  [/\bdeserv(?:e|es|ed|ing)\b[^.]{0,40}\b(?:more|full|the mark|a mark|credit)\b/i, "deserves_more"],
  [/\barguably\b/i, "arguably"],
  [/\ba stricter (?:reading|marking|interpretation)\b/i, "stricter_reading"],
  [/\b(?:harsh(?:ly)?|unfair(?:ly)?|generous(?:ly)?|lenient(?:ly)?)\b[^.]{0,30}\b(?:marked|marking|penali[sz]ed|awarded)\b/i, "judges_the_marking"],
  [/\b(?:marked|marking|penali[sz]ed|awarded)\b[^.]{0,30}\b(?:harsh(?:ly)?|unfair(?:ly)?|generous(?:ly)?|lenient(?:ly)?)\b/i, "judges_the_marking"],
  [/\b(?:marking|marker'?s?|grading|teacher'?s?|examiner'?s?)\b[^.]{0,20}\b(?:error|mistake|wrong|incorrect|inconsistent)\b/i, "calls_marking_wrong"],
  [/\b(?:could|might|may) (?:well )?(?:have )?(?:earned|got|gotten|scored|received)\b[^.]{0,20}\b(?:more|full|another|an extra)\b/i, "could_have_earned"],
  [/\bworth (?:full|more) marks?\b/i, "worth_more"],
];

const NUMBER = "(\\d+(?:\\.\\d+)?)";

/** Reasons the text disputes the teacher's mark; empty means it does not. */
export function adjudicationReasons(texts: Array<string | null | undefined>, marks: MarkContext): string[] {
  const reasons = new Set<string>();
  for (const raw of texts) {
    if (!raw) continue;
    for (const [pattern, name] of ADJUDICATING) if (pattern.test(raw)) reasons.add(name);

    // "2 out of 3 marks" / "2/3 marks": the stated numbers must be the teacher's.
    for (const m of raw.matchAll(new RegExp(`${NUMBER}\\s*(?:/|out of)\\s*${NUMBER}\\s*marks?`, "gi"))) {
      if (round(Number(m[1])) !== round(marks.marksAwarded) || round(Number(m[2])) !== round(marks.marksAvailable)) {
        reasons.add("states_other_mark_total");
      }
    }
    // "should score 3 marks" style claims: a mark count that is neither awarded, available nor lost.
    for (const m of raw.matchAll(new RegExp(`\\b(?:should|would|ought to)\\b[^.]{0,30}\\b(?:score|get|earn|receive|be given|be awarded)\\b[^.]{0,12}${NUMBER}\\s*marks?`, "gi"))) {
      const n = round(Number(m[1]));
      if (n !== round(marks.marksAwarded)) reasons.add("claims_a_different_mark");
    }
  }
  return [...reasons];
}

const round = (n: number) => Math.round(n * 100) / 100;

export interface ExplainChecks {
  contradiction_reasons: string[];
  teacher_mark_contradiction: boolean;
  loss_reasons_ok: boolean;
  do_this_next_clears_floor: boolean | null;
  can_explain: boolean;
  cause: string | null;
}

/** The same arithmetic the pipeline applies silently, reported instead of dropped. */
export function checkExplain(parsed: ExplainResult, marks: MarkContext): ExplainChecks {
  const marksLost = round(marks.marksAvailable - marks.marksAwarded);
  const decomposed = (parsed.loss_reasons ?? []).reduce((sum, r) => sum + (Number.isFinite(r.marks) ? r.marks : 0), 0);
  const reasons = adjudicationReasons(
    [parsed.body, parsed.do_this_next, parsed.command_word_note, parsed.model_answer, ...(parsed.loss_reasons ?? []).map((r) => r.note)],
    marks,
  );
  return {
    contradiction_reasons: reasons,
    teacher_mark_contradiction: reasons.length > 0,
    loss_reasons_ok: round(decomposed) <= marksLost,
    do_this_next_clears_floor: parsed.do_this_next == null ? null : clearsTheFloor(parsed.do_this_next),
    can_explain: parsed.can_explain === true,
    cause: parsed.cause ?? null,
  };
}
