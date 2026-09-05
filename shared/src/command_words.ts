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
// SOURCE. Generated from `cambridge_command_words_master_researched.xlsx`, which
// is built on Cambridge's own command-word guidance:
// https://www.cambridgeinternational.org/exam-administration/what-to-expect-on-exams-day/command-words/
// The 22 words Cambridge publishes for every syllabus revised from 2019 onwards
// are UNIVERSAL, and their meanings below are Cambridge's wording, not ours.
// The rest are subject-specific words drawn from individual syllabuses.
//
// The one thing Cambridge is explicit about, and the reason the meanings are
// split in two below: **a command word can mean something subject-specific.**
// So we only ever put a definition in front of a student for a word Cambridge
// defines the same way everywhere. For a subject-specific word we hold the word
// and let the note be written against the question in hand — the current
// syllabus and mark scheme take precedence over any list, including this one.
//
// A row from the sheet that was not a command word at all ("Mark allocation", a
// note about marks not determining depth) is deliberately not here.

export const COMMAND_WORDS = [
  "Advise", "Analyse", "Annotate", "Argue", "Assess", "Calculate", "Comment",
  "Comment on", "Compare", "Consider", "Contrast", "Deduce", "Define",
  "Describe", "Design", "Determine", "Develop", "Devise", "Discuss",
  "Estimate", "Evaluate", "Experiment", "Explain", "Explore", "Find", "Give",
  "Hence", "How far / To what extent", "Identify", "Infer", "Justify",
  "Locate", "Outline", "Plan", "Predict", "Prove", "Refine", "Reflect",
  "Show that", "Show your working", "Sketch", "State", "Suggest", "Summarise",
  "Test", "Trace", "Verify", "Work out", "Write", "Write down",
] as const;

export type CommandWord = (typeof COMMAND_WORDS)[number];

/**
 * Cambridge's own definition, for the words it defines the same way in every
 * syllabus. Verbatim from the command-word guidance — this is the one place in
 * the explanation a student reads a definition we did not write, so it is not
 * paraphrased and not summarised.
 */
export const UNIVERSAL_MEANING: Partial<Record<CommandWord, string>> = {
  "Analyse": "Examine in detail to show meaning and identify elements and relationships between them.",
  "Assess": "Make an informed judgement.",
  "Calculate": "Work out from given facts, figures or information.",
  "Comment": "Give an informed opinion.",
  "Compare": "Identify/comment on similarities and/or differences.",
  "Consider": "Review and respond to given information.",
  "Contrast": "Identify/comment on differences.",
  "Define": "Give a precise meaning.",
  "Describe": "State the points of a topic / give characteristics and main features.",
  "Develop": "Take forward to a more advanced stage or build upon given information.",
  "Discuss": "Write about issue(s) or topic(s) in depth in a structured way.",
  "Evaluate": "Judge or calculate the quality, importance, amount or value of something.",
  "Explain": "Set out purposes or reasons; make relationships clear; say why/how and support with relevant evidence.",
  "Give": "Produce an answer from a given source or recall/memory.",
  "Identify": "Name/select/recognise.",
  "Justify": "Support a case with evidence/argument.",
  "Outline": "Set out the main points.",
  "Predict": "Suggest what may happen based on available information.",
  "Sketch": "Make a simple freehand drawing showing key features, taking care over proportions.",
  "State": "Express in clear terms.",
  "Suggest": "Apply knowledge and understanding to situations where there are a range of valid responses; make proposals/put forward considerations.",
  "Summarise": "Select and present the main points, without detail.",
};

/**
 * Words that appear only in particular syllabuses. Held so the closed list can
 * recognise them, deliberately without a definition: "Advise", "Annotate", "Argue"
 * carry subject-specific meanings, and putting one subject's reading in front
 * of a student sitting another is the same class of mistake as reconstructing a
 * mark scheme. The note for these is written against the question itself.
 */
export const SUBJECT_SPECIFIC: readonly CommandWord[] = [
  "Advise", "Annotate", "Argue", "Comment on", "Deduce", "Design",
  "Determine", "Devise", "Estimate", "Experiment", "Explore", "Find", "Hence",
  "How far / To what extent", "Infer", "Locate", "Plan", "Prove", "Refine",
  "Reflect", "Show that", "Show your working", "Test", "Trace", "Verify",
  "Work out", "Write", "Write down",
];

const BY_LOWER = new Map<string, CommandWord>(
  COMMAND_WORDS.map((w) => [w.toLowerCase(), w]),
);

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

/**
 * Cambridge's definition of a command word, where one applies everywhere.
 *
 * Null for a subject-specific word — not because we have nothing on file, but
 * because what we have on file is one subject's reading and this question may
 * be another's.
 */
export function universalMeaning(word: CommandWord | null): string | null {
  return word ? UNIVERSAL_MEANING[word] ?? null : null;
}
