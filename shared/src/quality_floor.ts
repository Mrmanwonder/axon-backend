// Enforces CLAUDE.md's "do this next quality floor" in code, not just in the
// prompt: an explanation whose action is too short or matches one of the
// known-generic patterns never reaches the student. See
// shared/prompts/explain_tier1.v1.ts for where this gates `do_this_next`.

const GENERIC_ADVICE = [
  /\brevis(e|ing)\b/i,
  /\bpractic(e|ing)\s+(more|regularly|daily)/i,
  /\bstudy\s+(more|harder|the\s+chapter)/i,
  /\bread\s+the\s+(chapter|textbook|ncert)\b/i,
  /\bbe\s+(more\s+)?careful\b/i,
  /\bpay\s+(more\s+)?attention\b/i,
  /\bmanage\s+your\s+time\b/i,
  /\bunderstand\s+the\s+concept\b/i,
  /\bwork\s+on\s+(your|this)\b/i,
];

export function clearsTheFloor(line: string | null | undefined): boolean {
  if (!line) return false;
  const text = line.trim();
  if (text.length < 25) return false;
  return !GENERIC_ADVICE.some((re) => re.test(text));
}
