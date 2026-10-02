/**
 * The eval judge (AXO-41). A stronger model reads an explanation next to the facts it was given and
 * scores how faithful it is. It never sees, and is never asked about, whether the teacher's mark was
 * right: that number is a fact. It is used for ranking candidates against each other, and its output
 * is flags and a score only.
 *
 * Deliberately outside shared/src/prompts: this is not a prompt a student's answer is built from, so
 * it does not belong to the prompt-change gate. It is versioned here and recorded on every call.
 */

export const JUDGE_PROMPT_VERSION = "eval_judge.v1";

export const JUDGE_FLAGS = [
  "contradicts_teacher_mark",
  "invents_scheme_or_rubric",
  "claims_unseen_content",
  "generic_advice",
  "wrong_subject_fact",
  "ignores_marks_lost",
  "praises_a_wrong_answer",
] as const;
export type JudgeFlag = (typeof JUDGE_FLAGS)[number];

export const SYSTEM = `
You are checking an explanation written for a student about where they lost marks in an exam answer.
You are not marking the answer. The teacher's mark is a fact you must treat as correct, even if you
think it is harsh or wrong; never comment on it.

Score faithfulness from 1 to 5:
5  Every claim is supported by the question, the student's working, the mark, or the teacher's remark.
   The cause fits what was lost. Advice is specific to this answer.
4  Faithful, with one small unsupported or vague point.
3  Mostly faithful, but one claim goes beyond the evidence, or the advice is generic.
2  Several unsupported claims, or the explanation does not fit the marks lost.
1  Invents content, contradicts the evidence, or explains a different question.

Raise a flag only when it is clearly true:
- contradicts_teacher_mark: the text says or implies the teacher's mark was wrong or should be different.
- invents_scheme_or_rubric: it states what "the mark scheme" or "examiners" require without that being supplied.
- claims_unseen_content: it describes working, a diagram, or an earlier part that was not given.
- generic_advice: the next step could apply to any question ("revise this topic", "practise more").
- wrong_subject_fact: a factual error about the subject itself.
- ignores_marks_lost: it does not account for the marks that were actually lost.
- praises_a_wrong_answer: it calls an answer right where the working shows it is not.

If the case was meant to be declined (the explanation says it cannot explain), score whether declining
was honest, not whether it was helpful.
Return only the JSON requested.
`.trim();

export interface JudgeInput {
  question: string | null;
  studentAnswer: string | null;
  teacherRemark: string | null;
  marksAwarded: number;
  marksAvailable: number;
  priorParts?: Array<{ label: string; questionText: string | null; studentAnswer: string | null }>;
  explanation: {
    can_explain: boolean;
    cause: string | null;
    body: string | null;
    do_this_next: string | null;
    model_answer: string | null;
  };
}

export function instruction(input: JudgeInput): string {
  const lines = [
    `Teacher's mark (a fact): ${input.marksAwarded} out of ${input.marksAvailable}.`,
    `Question: ${input.question ?? "(not available)"}`,
    `Student's answer: ${input.studentAnswer ?? "(not available)"}`,
    `Teacher's remark: ${input.teacherRemark?.trim() ? input.teacherRemark : "(none)"}`,
  ];
  for (const part of input.priorParts ?? []) {
    lines.push(`Earlier part ${part.label}: ${part.questionText ?? ""} | student wrote: ${part.studentAnswer ?? ""}`);
  }
  lines.push(
    "",
    "Explanation to check:",
    `can_explain: ${input.explanation.can_explain}`,
    `cause: ${input.explanation.cause ?? "(none)"}`,
    `body: ${input.explanation.body ?? "(none)"}`,
    `do_this_next: ${input.explanation.do_this_next ?? "(none)"}`,
    `corrected working: ${input.explanation.model_answer ?? "(none)"}`,
  );
  return lines.join("\n");
}

export const SCHEMA = {
  name: "eval_judgement",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["faithfulness", "flags"],
    properties: {
      faithfulness: { type: "integer", minimum: 1, maximum: 5 },
      flags: { type: "array", items: { type: "string", enum: [...JUDGE_FLAGS] } },
    },
  },
};

export interface Judgement {
  faithfulness: number;
  flags: JudgeFlag[];
}

export function validate(parsed: unknown): Judgement {
  const value = parsed as { faithfulness?: unknown; flags?: unknown } | null;
  const score = Number(value?.faithfulness);
  if (!Number.isInteger(score) || score < 1 || score > 5) throw new Error("judge faithfulness must be an integer from 1 to 5");
  const flags = Array.isArray(value?.flags) ? value!.flags : [];
  const bad = flags.filter((f) => !(JUDGE_FLAGS as readonly string[]).includes(String(f)));
  if (bad.length) throw new Error(`judge raised unknown flags: ${bad.join(", ")}`);
  return { faithfulness: score, flags: [...new Set(flags as JudgeFlag[])] };
}
