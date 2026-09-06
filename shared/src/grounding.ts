/**
 * The gate in front of the corrected working.
 *
 * `model_answer` is the highest-trust element Axon renders. It is labelled "see
 * the corrected working" — what the student should have written — and it is the
 * thing they copy into their notes. Every other field on the card describes
 * their answer; this one replaces it. A wrong one is not a bad explanation, it
 * is a false thing taught to someone who by definition cannot catch it, in
 * their own revision notes, under our name.
 *
 * So it gets a bar the rest of the card does not, and the bar is structural
 * rather than a disclaimer. A disclaimer under a fabricated answer transfers our
 * failure to the student's judgement in the one domain where the student has no
 * way to exercise it.
 *
 * Two gates, both deliberately deterministic:
 *
 *   1. Dependency. A question that points at a part we could not supply was
 *      explained without that part, so nothing it says about it is grounded.
 *   2. Topicality. A corrected working that shares no subject vocabulary with
 *      the question, the student's answer, or the parts it depends on is not an
 *      answer to this question.
 *
 * Neither is a correctness check and neither is claimed to be one. They are a
 * net under one specific failure — prose generated *without* the question,
 * which is what a model produces when the question is not in front of it. That
 * failure is not subtle and does not need a subtle detector; what it needed was
 * any detector at all.
 */

/**
 * Words that appear in exam questions regardless of subject. They are real
 * words and they overlap freely between an answer about floating-point
 * normalisation and one about signal-to-noise ratio, so counting them would
 * hide exactly the failure this is looking for.
 */
const SCAFFOLDING = new Set([
  "answer", "answers", "question", "questions", "part", "parts", "mark", "marks",
  "state", "explain", "describe", "calculate", "justify", "show", "give", "write",
  "find", "determine", "suggest", "define", "identify", "outline", "compare",
  "correct", "correctly", "incorrect", "wrong", "right", "student", "teacher",
  "following", "above", "below", "given", "using", "used", "use", "value", "values",
  "number", "numbers", "result", "results", "method", "working", "step", "steps",
  "first", "second", "third", "next", "then", "because", "since", "therefore",
  "would", "could", "should", "must", "this", "that", "these", "those", "there",
  "which", "what", "when", "where", "your", "yours", "have", "has", "had", "been",
  "with", "without", "from", "into", "onto", "each", "both", "more", "less",
  "than", "the", "and", "for", "not", "but", "its", "it", "is", "are", "was",
  "were", "will", "can", "may", "one", "two", "any", "all", "you",
]);

/** Content words of four or more characters, lightly stemmed so bit/bits match. */
export function subjectTerms(text: string | null | undefined): Set<string> {
  const terms = new Set<string>();
  if (typeof text !== "string") return terms;
  for (const raw of text.toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw.length < 3) continue;
    if (SCAFFOLDING.has(raw)) continue;
    // Crude but symmetric: applied to both sides, so "bits" in the question and
    // "bit" in the answer meet in the middle.
    const stem = raw.replace(/(?:ing|ed|es|s)$/, "");
    if (stem.length < 3 || SCAFFOLDING.has(stem)) continue;
    terms.add(stem);
  }
  return terms;
}

export type WithholdReason =
  | "unresolved_dependency"
  | "off_topic"
  | null;

export interface GroundingInput {
  modelAnswer: string | null;
  questionText: string | null;
  studentAnswer: string | null;
  /** Question text and answers of the parts this question depends on. */
  contextText: string[];
  /** Referenced parts that could not be found in the run. */
  unresolvedDependencies: string[];
}

export interface GroundingVerdict {
  /** The corrected working to store, or null. */
  modelAnswer: string | null;
  /** Why it was withheld, for the row. Null when nothing was withheld. */
  withheldReason: WithholdReason;
}

/**
 * How many subject terms a corrected working must share with the question it
 * claims to answer.
 *
 * Two, and only once the question itself has enough vocabulary for the count to
 * mean anything. The production fabrication shares exactly one — "bit", from
 * "bit error rate" meeting "3 bits" — against a question about normalised
 * floating-point representation, so one is demonstrably not enough and two
 * catches it. Set higher and a correct answer that legitimately introduces new
 * vocabulary starts getting suppressed, which costs a real feature to catch a
 * failure the first gate already handles in the common case.
 */
const MIN_SHARED_TERMS = 2;
const MIN_CONTEXT_TERMS = 6;

export function gateModelAnswer(input: GroundingInput): GroundingVerdict {
  const answer = typeof input.modelAnswer === "string" && input.modelAnswer.trim()
    ? input.modelAnswer.trim()
    : null;

  // Nothing to withhold. The model declining to write one is the honest outcome
  // the prompt asks for, not a failure, and it carries no reason.
  if (!answer) return { modelAnswer: null, withheldReason: null };

  // Gate 1. The question points at a part that was never put in front of the
  // model. Whatever it wrote about that part, it wrote blind.
  if (input.unresolvedDependencies.length) {
    return { modelAnswer: null, withheldReason: "unresolved_dependency" };
  }

  // Gate 2. Shared subject vocabulary with the question and everything it
  // depends on.
  const context = subjectTerms(
    [input.questionText, input.studentAnswer, ...input.contextText].filter(Boolean).join(" "),
  );
  if (context.size < MIN_CONTEXT_TERMS) {
    // Too little transcribed to judge topicality either way. The first gate has
    // already passed, so this is not withheld on a test it cannot sit.
    return { modelAnswer: answer, withheldReason: null };
  }

  let shared = 0;
  for (const term of subjectTerms(answer)) if (context.has(term)) shared++;

  if (shared < MIN_SHARED_TERMS) {
    return { modelAnswer: null, withheldReason: "off_topic" };
  }

  return { modelAnswer: answer, withheldReason: null };
}
