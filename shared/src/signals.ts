/**
 * Four signals, kept apart, each allowed to say it does not know.
 *
 * `confidence_signals` has always held four independent readings — arithmetic,
 * structural, recognition, plausibility — and the pipeline has always flattened
 * them into one `confidence_tier`. That collapse is the defect, and the live
 * data shows exactly what it costs:
 *
 *   question `b`: recognition FALSE, arithmetic true → committed marks.
 *   question `a`: five rows of identical text, tier `unsure` on some and
 *                 `confident` on others, decided by a signal that flipped.
 *   question `d(i)`: the answer is the two words "Not Normalized". Its
 *                 arithmetic signal reads false on one row and true on four.
 *                 There is no arithmetic in it to be true or false about.
 *
 * That last one is the argument for `unknown` in one line. A boolean has no way
 * to say "this question does not apply here", so it guesses, and a forced guess
 * is a hallucination with a schema.
 *
 * Two rules follow, and they are the whole point of this module:
 *
 *   No signal may authorise another. `recognition` governs whether a
 *   transcription may be shown as authoritative. `arithmetic` governs whether a
 *   working chain may be reasoned over. A clean arithmetic reading cannot make
 *   unreadable handwriting trustworthy, and it has been doing exactly that.
 *
 *   `unknown` is never quietly promoted to true. It withholds.
 */

export type Signal = true | false | "unknown";

export const SIGNAL_NAMES = ["arithmetic", "structural", "recognition", "plausibility"] as const;
export type SignalName = (typeof SIGNAL_NAMES)[number];

export type Signals = Record<SignalName, Signal>;

/** Anything at all, as it arrives from a jsonb column or a model. */
export function readSignal(v: unknown): Signal {
  if (v === true || v === false) return v;
  if (v === "true") return true;
  if (v === "false") return false;
  return "unknown";
}

export function readSignals(raw: unknown): Signals {
  const o = (raw ?? {}) as Record<string, unknown>;
  return {
    arithmetic: readSignal(o.arithmetic),
    structural: readSignal(o.structural),
    recognition: readSignal(o.recognition),
    plausibility: readSignal(o.plausibility),
  };
}

export type Tier = "confident" | "unsure" | "unreadable";

/**
 * What the student may be shown, derived without letting one signal cover for
 * another.
 *
 * `recognition` is load-bearing and separate: if we could not read the
 * handwriting, nothing downstream is authoritative no matter how tidy the
 * arithmetic looked. `unknown` never reaches `confident` — it is the honest
 * middle, and the middle is `unsure`, which the analytics views already exclude
 * until a student confirms it.
 */
export function tierFrom(s: Signals): Tier {
  if (s.recognition === false) return "unreadable";
  if (s.recognition === "unknown") return "unsure";
  // recognition is true from here: the text is readable. Everything else
  // governs how much may be *reasoned over*, not whether it may be shown.
  if (s.structural === false || s.plausibility === false) return "unsure";
  if (s.arithmetic === false) return "unsure";
  if (s.structural === "unknown" || s.plausibility === "unknown") return "unsure";
  // arithmetic `unknown` is normal and not a defect: most prose answers contain
  // no arithmetic at all, and demanding one would make every essay unsure.
  return "confident";
}

/**
 * May this region's working be reasoned over — a corrected working written
 * against it, a step called wrong?
 *
 * Only when the arithmetic was actually checked and held. `unknown` is not
 * permission; it is the absence of a check.
 */
export function mayReasonOverWorking(s: Signals): boolean {
  return s.recognition === true && s.arithmetic === true;
}

/**
 * May the transcription be presented as what the student wrote?
 *
 * The crop is always the authority. This governs whether the text beside it may
 * be shown flat, or must be shown with the gap named.
 */
export function transcriptionIsAuthoritative(s: Signals): boolean {
  return s.recognition === true;
}

/**
 * An inconsistent arithmetic chain sends the crop back to be re-read; it never
 * touches a mark.
 *
 * Which of the student and the transcription is wrong is not knowable from the
 * text — the live data contains handwritten `8/2` stored as `8+1`, which turns
 * a correct step into a false one — so the only safe action is to look again.
 */
export function needsReread(s: Signals): boolean {
  return s.arithmetic === false;
}
