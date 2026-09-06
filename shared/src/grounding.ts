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

/**
 * The grounding verdict for a row's corrected working.
 *
 * A closed list, mirrored by a CHECK in
 * 20260906090000_explanation_grounding_provenance.sql, because the question it
 * exists to answer is countable: how often do we withhold, and for which
 * reason. A rising `heuristic_off_topic` means the model is drifting; a rising
 * `missing_dependency` means the scanner is losing pages. Free text answers
 * neither.
 *
 * `heuristic_off_topic` is deliberately named for what it is. It is a coarse
 * net for prose generated *without* the question, not a correctness check and
 * not a grounding verifier — an unrelated answer can share terms, and a
 * plausible-but-wrong one can use the question's vocabulary exactly. It must
 * never be surfaced or counted as verification.
 */
export type GroundingStatus =
  | "complete"
  | "missing_dependency"
  | "missing_question_text"
  | "no_verified_answer_source"
  | "heuristic_off_topic"
  | "generation_failed";

/**
 * Where a shown corrected working came from.
 *
 * `verified_scheme` is reserved and currently unreachable: Cambridge and
 * Pearson refused third-party reproduction, so official CAIE scheme content is
 * not ours to render. Everything we show is our own method, and it is labelled
 * as ours rather than borrowing an authority we do not have.
 */
export type AnswerSource = "axon_method" | "verified_scheme";

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
  /** Whether it was grounded, and if not, why. */
  status: GroundingStatus;
  /** Where it came from. Null exactly when there is no answer to attribute. */
  source: AnswerSource | null;
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

  // Order matters: a missing dependency is reported as a missing dependency even
  // when the model also declined to answer, because the two say different things
  // about the scan. One means a page is missing; the other means the model was
  // honest. Collapsing them would hide the first behind the second.
  if (input.unresolvedDependencies.length) {
    return { modelAnswer: null, status: "missing_dependency", source: null };
  }
  if (!input.questionText || !input.questionText.trim()) {
    return { modelAnswer: null, status: "missing_question_text", source: null };
  }

  // The model declining to write one is the outcome the prompt asks for on a
  // question it cannot work through, and the grounding was still complete.
  // `generation_failed` is for a call that came back with nothing usable, which
  // the worker distinguishes; an honest null is not a failure.
  if (!answer) return { modelAnswer: null, status: "complete", source: null };

  // The net. Shared subject vocabulary with the question and everything it
  // depends on. Not a verifier — see GroundingStatus.
  const context = subjectTerms(
    [input.questionText, input.studentAnswer, ...input.contextText].filter(Boolean).join(" "),
  );

  // Too little transcribed to judge topicality either way. The gates above have
  // passed, so the answer is not withheld on a test it cannot sit.
  if (context.size < MIN_CONTEXT_TERMS) {
    return { modelAnswer: answer, status: "complete", source: "axon_method" };
  }

  let shared = 0;
  for (const term of subjectTerms(answer)) if (context.has(term)) shared++;

  if (shared < MIN_SHARED_TERMS) {
    return { modelAnswer: null, status: "heuristic_off_topic", source: null };
  }

  return { modelAnswer: answer, status: "complete", source: "axon_method" };
}
