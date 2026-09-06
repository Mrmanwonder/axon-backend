/**
 * Arithmetic, evaluated rather than judged.
 *
 * On 2026-09-06 the live database held one answer, stored three times, whose
 * `arithmetic` signal read true, true, true, true and false — on byte-identical
 * text. The working it was judging contains `4 + 1/2 = 8 + 1/2`, which is false
 * on every day of the week. And the reverse, on question c: a correct chain
 * marked false, and `3/16 | 2^5` — not a well-formed expression at all, the `|`
 * is not an operator anyone wrote — marked true, four times out of five.
 *
 * There is only one way a signal about arithmetic can disagree with itself
 * about the same string: the arithmetic was being *judged by a model* instead
 * of *evaluated*. A verifier built from the same machinery as the thing it
 * verifies fails in the same way, at the same time, on the same inputs, and no
 * amount of prompt work fixes it, because the defect is that a decidable
 * question was handed to something that guesses.
 *
 * So this module decides it. Parse to an expression tree, evaluate over exact
 * rationals — BigInt numerator and denominator, never floats, so 1/3 is 1/3 and
 * not 0.333… — and compare. `4 + 1/2 == 8 + 1/2` becomes `9/2 == 17/2` becomes
 * false, in microseconds, identically forever.
 *
 * Three things this deliberately does NOT do:
 *
 *   It does not decide whether the student was right. An inconsistent chain
 *   means *something* is wrong — the student's maths, or our transcription of
 *   it — and which one is not knowable from the text. It is a re-read trigger,
 *   never a mark deduction. Hard rule 1 is not ours to bend with a parser.
 *
 *   It does not force a boolean. `unknown` is a first-class verdict, because a
 *   boolean forces a guess and a forced guess is a hallucination with a schema.
 *
 *   It does not read binary. `0.1001 * 2^3` on a floating-point paper means a
 *   binary mantissa, and nothing in the transcription says so; evaluating it as
 *   decimal would be inventing a convention. Radix is honoured only where the
 *   text declares it (`100.1_2`), and is `unknown` otherwise.
 */

// ── exact rationals ─────────────────────────────────────────────────────────

export interface Rational { n: bigint; d: bigint }

function gcd(a: bigint, b: bigint): bigint {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) { const t = a % b; a = b; b = t; }
  return a;
}

export function rat(n: bigint, d: bigint = 1n): Rational {
  if (d === 0n) throw new RangeError("division by zero");
  if (d < 0n) { n = -n; d = -d; }
  const g = gcd(n, d) || 1n;
  return { n: n / g, d: d / g };
}

const add = (a: Rational, b: Rational) => rat(a.n * b.d + b.n * a.d, a.d * b.d);
const sub = (a: Rational, b: Rational) => rat(a.n * b.d - b.n * a.d, a.d * b.d);
const mul = (a: Rational, b: Rational) => rat(a.n * b.n, a.d * b.d);
const div = (a: Rational, b: Rational) => {
  if (b.n === 0n) throw new RangeError("division by zero");
  return rat(a.n * b.d, a.d * b.n);
};
export const ratEq = (a: Rational, b: Rational) => a.n === b.n && a.d === b.d;
export const ratStr = (r: Rational) => (r.d === 1n ? `${r.n}` : `${r.n}/${r.d}`);

/** Integer exponent only. A fractional power is not decidable over rationals. */
function pow(base: Rational, exp: Rational): Rational {
  if (exp.d !== 1n) throw new RangeError("fractional exponent");
  let e = exp.n;
  const negative = e < 0n;
  if (negative) e = -e;
  if (e > 4096n) throw new RangeError("exponent too large");
  let acc = rat(1n);
  for (let i = 0n; i < e; i++) acc = mul(acc, base);
  return negative ? div(rat(1n), acc) : acc;
}

// ── tokens ──────────────────────────────────────────────────────────────────

type Tok =
  | { t: "num"; v: Rational; raw: string }
  | { t: "op"; v: "+" | "-" | "*" | "/" | "^" }
  | { t: "("; } | { t: ")"; } | { t: "="; };

/** Multiplication and division as students actually write them. */
const TIMES = new Set(["*", "×", "x", "·", "⋅"]);
const DIVIDE = new Set(["/", "÷"]);

class Unparseable extends Error {}

/**
 * A number, in the radix the text declares.
 *
 * `100.1_2` is binary and evaluates exactly: 4 + 1/2. `4.5_10` is decimal and
 * says so. A bare `0.1001` is decimal, because that is what the characters mean
 * and guessing that the student meant binary — even on a floating-point paper
 * where they almost certainly did — is exactly the kind of inference this
 * module exists to stop making.
 */
function numberFrom(digits: string, radix: number): Rational {
  const [whole, frac = ""] = digits.split(".");
  const R = BigInt(radix);
  let value = rat(0n);
  for (const ch of whole) {
    const d = parseInt(ch, radix);
    if (Number.isNaN(d)) throw new Unparseable(`digit ${ch} is not base ${radix}`);
    value = add(mul(value, rat(R)), rat(BigInt(d)));
  }
  let scale = rat(1n);
  for (const ch of frac) {
    const d = parseInt(ch, radix);
    if (Number.isNaN(d)) throw new Unparseable(`digit ${ch} is not base ${radix}`);
    scale = div(scale, rat(R));
    value = add(value, mul(rat(BigInt(d)), scale));
  }
  return value;
}

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  const s = src.replace(/−/g, "-").replace(/–|—/g, "-");
  while (i < s.length) {
    const ch = s[i];
    if (/\s/.test(ch)) { i++; continue; }
    if (/[0-9]/.test(ch)) {
      let j = i;
      while (j < s.length && /[0-9a-fA-F.]/.test(s[j])) j++;
      let digits = s.slice(i, j);
      let radix = 10;
      // A declared radix: 100.1_2, 1A_16.
      const sub = s.slice(j).match(/^_([0-9]{1,2})/);
      if (sub) { radix = parseInt(sub[1], 10); j += sub[0].length; }
      // Hex letters without a declared radix are not a number we can read.
      if (radix === 10 && /[a-fA-F]/.test(digits)) throw new Unparseable("letters in a decimal number");
      if (radix < 2 || radix > 16) throw new Unparseable(`radix ${radix}`);
      if ((digits.match(/\./g) ?? []).length > 1) throw new Unparseable("two decimal points");
      out.push({ t: "num", v: numberFrom(digits, radix), raw: digits });
      i = j; continue;
    }
    if (ch === "(") { out.push({ t: "(" }); i++; continue; }
    if (ch === ")") { out.push({ t: ")" }); i++; continue; }
    if (ch === "=") { out.push({ t: "=" }); i++; continue; }
    if (ch === "+" || ch === "-") { out.push({ t: "op", v: ch }); i++; continue; }
    if (ch === "^") { out.push({ t: "op", v: "^" }); i++; continue; }
    if (TIMES.has(ch)) { out.push({ t: "op", v: "*" }); i++; continue; }
    if (DIVIDE.has(ch)) { out.push({ t: "op", v: "/" }); i++; continue; }
    // Anything else — a stray `|`, a word, a bracket we do not know — makes the
    // segment unreadable. It does not make it false.
    throw new Unparseable(`unexpected ${JSON.stringify(ch)}`);
  }
  return out;
}

// ── parse and evaluate ──────────────────────────────────────────────────────

/** Every value produced anywhere in a segment, for the continuation rule. */
interface Evaluated { value: Rational; parts: Rational[] }

function evaluateSegment(toks: Tok[]): Evaluated {
  let p = 0;
  const parts: Rational[] = [];
  const peek = () => toks[p];
  const eat = () => toks[p++];

  function primary(): Rational {
    const t = peek();
    if (!t) throw new Unparseable("expression ended early");
    if (t.t === "num") { eat(); parts.push((t as any).v); return (t as any).v; }
    if (t.t === "(") {
      eat();
      const v = expr();
      const close = eat();
      if (!close || close.t !== ")") throw new Unparseable("unclosed bracket");
      parts.push(v);
      return v;
    }
    throw new Unparseable("expected a number");
  }
  function unary(): Rational {
    const t = peek();
    if (t && t.t === "op" && (t.v === "-" || t.v === "+")) {
      eat();
      const v = unary();
      return t.v === "-" ? sub(rat(0n), v) : v;
    }
    return primary();
  }
  // Right-associative, so 2^3^2 is 2^(3^2) as written maths means it.
  function power(): Rational {
    const base = unary();
    const t = peek();
    if (t && t.t === "op" && t.v === "^") {
      eat();
      const e = power();
      const v = pow(base, e);
      parts.push(v);
      return v;
    }
    return base;
  }
  function term(): Rational {
    let v = power();
    for (;;) {
      const t = peek();
      if (t && t.t === "op" && (t.v === "*" || t.v === "/")) {
        eat();
        const r = power();
        v = t.v === "*" ? mul(v, r) : div(v, r);
        parts.push(v);
      } else return v;
    }
  }
  function expr(): Rational {
    let v = term();
    for (;;) {
      const t = peek();
      if (t && t.t === "op" && (t.v === "+" || t.v === "-")) {
        eat();
        const r = term();
        v = t.v === "+" ? add(v, r) : sub(v, r);
        parts.push(v);
      } else return v;
    }
  }

  const value = expr();
  if (p !== toks.length) throw new Unparseable("trailing input");
  parts.push(value);
  return { value, parts };
}

// ── verdicts ────────────────────────────────────────────────────────────────

export type ChainVerdict =
  | { kind: "consistent"; steps: number }
  /**
   * Something in this chain does not hold. Deliberately not "the student is
   * wrong": the transcription may be what broke it, and the live data contains
   * exactly that case — handwritten `8/2` stored as `8+1`, which turns a
   * correct step into a false one. This routes to a re-read of that segment's
   * crop, and never to a mark.
   */
  | { kind: "inconsistent"; at: number; left: string; right: string; leftValue: string; rightValue: string }
  | { kind: "unknown"; reason: string };

/**
 * Does `later` carry `earlier`'s value forward?
 *
 * Students write running chains: `1/8 + 1/16 = 3/16 * 2^5 = 6`. Read as strict
 * transitive equality that is false, because 3/16 is not 6. Read as what the
 * student meant — take the result, now multiply it — it is correct, and it is
 * how a great deal of real working is written.
 *
 * The rule is decidable rather than charitable: the later segment must actually
 * contain a sub-expression whose exact value equals the earlier segment's. That
 * is checked by evaluation, not by guessing at intent, and a segment that does
 * not carry the value forward is still inconsistent.
 */
function continues(earlier: Rational, later: Evaluated): boolean {
  return later.parts.some((p) => ratEq(p, earlier));
}

/**
 * One `=`-separated chain, already stripped of prose.
 *
 * Each side is parsed on its own, and only adjacent pairs that BOTH parsed are
 * compared. That matters: the live question `a` ends in `010010000000 0011`,
 * two numerals with a space between them, which is not an expression. Parsing
 * the line as a whole made the entire chain unreadable and hid the `4 + 1/2 =
 * 8 + 1/2` sitting five segments earlier — an unknown at the end swallowing a
 * decidable falsehood at the front, which is the exact failure mode this module
 * exists to remove.
 *
 * An unreadable side is never treated as an identity, so a chain reading
 * `ok = junk = ok` yields no comparison at all rather than a comparison across
 * the gap. Conservative on purpose: skipping a check is recoverable, inventing
 * one is not.
 */
export function checkChain(text: string): ChainVerdict {
  const raw = text.split("=").map((p) => p.trim());
  if (raw.length < 2) return { kind: "unknown", reason: "no equality to check" };

  const values: (Evaluated | null)[] = raw.map((side) => {
    if (!side) return null;
    try { return evaluateSegment(tokenize(side)); }
    catch { return null; }
  });

  let compared = 0;
  for (let i = 1; i < values.length; i++) {
    const a = values[i - 1], b = values[i];
    if (!a || !b) continue;
    compared++;
    if (ratEq(a.value, b.value)) continue;
    if (continues(a.value, b)) continue;
    return {
      kind: "inconsistent",
      at: i,
      left: raw[i - 1], right: raw[i],
      leftValue: ratStr(a.value), rightValue: ratStr(b.value),
    };
  }

  if (!compared) return { kind: "unknown", reason: "no side of this equality could be read" };
  // Nothing contradicted, but something in the chain could not be read, so
  // "consistent" would claim more than was checked.
  if (values.some((v) => !v)) {
    return { kind: "unknown", reason: "part of this chain could not be read" };
  }
  return { kind: "consistent", steps: values.length - 1 };
}

/**
 * The whole of a student's answer, which is prose and working mixed together.
 *
 * Lines are split, prose stripped from the front of each ("Working 4.5 = …"),
 * and every line carrying an equality is checked. One inconsistency anywhere
 * makes the answer inconsistent; otherwise a single checkable line makes it
 * consistent; and an answer with nothing checkable in it is `unknown`, which is
 * the honest verdict for "Not Normalised" and for prose generally.
 */
export function checkAnswer(answer: string | null | undefined): ChainVerdict {
  if (typeof answer !== "string" || !answer.trim()) {
    return { kind: "unknown", reason: "nothing to check" };
  }
  let sawConsistent = false;
  let firstUnknown: ChainVerdict | null = null;

  for (const rawLine of answer.split(/[\n\r]+/)) {
    // Drop a leading label or prose run — "Working", "Working...", "Ans:" —
    // up to the first character that could begin an expression.
    const line = rawLine.replace(/^[^0-9(+\-]*/, "").trim();
    if (!line.includes("=")) continue;
    const v = checkChain(line);
    if (v.kind === "inconsistent") return v;
    if (v.kind === "consistent") sawConsistent = true;
    else if (!firstUnknown) firstUnknown = v;
  }
  if (sawConsistent) return { kind: "consistent", steps: 1 };
  return firstUnknown ?? { kind: "unknown", reason: "no equality to check" };
}
