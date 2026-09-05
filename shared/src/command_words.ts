// Cambridge command words.
//
// A Cambridge question is built around its command word, and the command word
// is a promise about what an answer has to do: "State" wants a fact and nothing
// else, "Explain" wants the because, "Show that" wants every step to a value
// the paper already told you. Marks lost to reading "Explain" as "State" are
// among the most common and the most fixable, and they are invisible to a
// student who has never been told the word is doing work.
//
// A closed list rather than free text, for the same reason `cause` is a closed
// enum: a tag that is matchable is a tag analytics can count, and a model left
// to phrase it freely will produce "explain", "Explanation", "explain why" and
// "Explain (2 marks)" for one question type.
//
// OPEN: this is the standard CAIE set as it appears across the science, maths
// and computing syllabuses, not a transcription of one syllabus document.
// Spec §5.2 asks for the real per-subject closed list — 9618 first — pulled
// from the syllabuses themselves. Until that happens this is deliberately a
// little wide: a word missing from the list renders nothing, which is honest,
// while a word wrongly in it would put scheme vocabulary in a student's mouth.

export const COMMAND_WORDS = [
  "Analyse", "Annotate", "Calculate", "Comment", "Compare", "Complete",
  "Consider", "Construct", "Contrast", "Convert", "Deduce", "Define",
  "Demonstrate", "Derive", "Describe", "Determine", "Develop", "Discuss",
  "Draw", "Estimate", "Evaluate", "Explain", "Give", "Identify", "Illustrate",
  "Interpret", "Justify", "Label", "List", "Measure", "Name", "Outline",
  "Plot", "Predict", "Recall", "Sketch", "Show that", "Solve", "State",
  "Suggest", "Summarise", "Verify", "Write",
] as const;

export type CommandWord = (typeof COMMAND_WORDS)[number];

const BY_LOWER = new Map(COMMAND_WORDS.map((w) => [w.toLowerCase(), w]));

/**
 * The canonical spelling of a command word, or null.
 *
 * Null rather than a guess: an unrecognised word renders nothing, and nothing
 * is the honest outcome. Tolerates the casing and trailing punctuation a model
 * produces ("explain,", "SHOW THAT") because those are the same word; it does
 * not tolerate a phrase that merely contains one, because "state the explain"
 * is not a command word and pretending otherwise mislabels the question.
 */
export function canonicalCommandWord(value: unknown): CommandWord | null {
  if (typeof value !== "string") return null;
  const cleaned = value.trim().replace(/[.,:;]+$/, "").replace(/\s+/g, " ").toLowerCase();
  return BY_LOWER.get(cleaned) ?? null;
}
