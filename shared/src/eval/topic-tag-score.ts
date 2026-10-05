/**
 * Scores one topic_tag answer against a golden case (syllabus map). Pure, so the rule is tested
 * without a model.
 *
 * A case lists the objective codes a teacher would accept as the primary tag, and codes that are
 * fair secondary tags. Only "strong" tags reach the heatmap (they are stored as `likely`; "partial"
 * is stored as `unsure` and never reaches analytics), so precision is measured on strong tags.
 * A control case (off-syllabus, or a part with no readable stem) is answered correctly by placing
 * nothing: any strong tag on it is a wrong statement about a student's topic.
 */
import type { TagResult } from "../prompts/topic_tag.v1.js";

export interface TopicTagExpectation {
  primary: string[];
  also_ok: string[];
  control: "off_syllabus" | "no_context" | null;
}

export interface TopicTagScore {
  control: boolean;
  abstained: boolean;
  /** The model's primary tag is one the case accepts as primary (null on a control). */
  primary_hit: boolean | null;
  /** Strong tags outside primary + also_ok: each one would colour a topic the question never assessed. */
  wrong_strong: string[];
  strong: number;
  /** A control answered by placing nothing strong. */
  control_ok: boolean | null;
}

export function scoreTopicTag(result: TagResult, expected: TopicTagExpectation): TopicTagScore {
  const control = expected.control !== null;
  const strong = result.tags.filter((t) => t.strength === "strong");
  const accepted = new Set([...expected.primary, ...expected.also_ok]);
  const primary = result.tags.find((t) => t.primary) ?? null;
  return {
    control,
    abstained: !result.canTag || result.tags.length === 0,
    primary_hit: control ? null : primary !== null && expected.primary.includes(primary.code),
    wrong_strong: control ? strong.map((t) => t.code) : strong.filter((t) => !accepted.has(t.code)).map((t) => t.code),
    strong: strong.length,
    control_ok: control ? strong.length === 0 : null,
  };
}

export interface TopicTagSummary {
  cases: number;
  scored: number;
  controls: number;
  schema_valid: string;
  primary_hit: string;
  primary_hit_rate: number;
  strong_precision: number;
  wrong_strong_tags: number;
  controls_ok: string;
  abstained_on_real_questions: number;
}

/** Aggregates per-case scores; failures count against schema validity and primary hits. */
export function summariseTopicTag(rows: Array<{ status: string; score: TopicTagScore | null }>): TopicTagSummary {
  const done = rows.filter((r) => r.status === "done" && r.score);
  const real = done.filter((r) => !r.score!.control);
  const controls = done.filter((r) => r.score!.control);
  const realTotal = rows.filter((r) => !(r.score?.control ?? false)).length;
  const hits = real.filter((r) => r.score!.primary_hit).length;
  const strong = done.reduce((n, r) => n + r.score!.strong, 0);
  const wrong = done.reduce((n, r) => n + r.score!.wrong_strong.length, 0);
  return {
    cases: rows.length,
    scored: done.length,
    controls: controls.length,
    schema_valid: `${done.length}/${rows.length}`,
    primary_hit: `${hits}/${realTotal}`,
    primary_hit_rate: realTotal ? hits / realTotal : 0,
    strong_precision: strong ? (strong - wrong) / strong : 1,
    wrong_strong_tags: wrong,
    controls_ok: `${controls.filter((r) => r.score!.control_ok).length}/${controls.length}`,
    abstained_on_real_questions: real.filter((r) => r.score!.abstained).length,
  };
}
