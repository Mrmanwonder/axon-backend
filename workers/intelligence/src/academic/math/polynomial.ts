type Polynomial = number[];
type Token = number | "x" | "+" | "-" | "*" | "/" | "^" | "(" | ")";

const trim = (value: Polynomial): Polynomial => {
  const output = [...value];
  while (output.length > 1 && Math.abs(output.at(-1) ?? 0) < 1e-12) output.pop();
  return output;
};
const add = (left: Polynomial, right: Polynomial, sign = 1): Polynomial => trim(Array.from({ length: Math.max(left.length, right.length) }, (_, index) => (left[index] ?? 0) + sign * (right[index] ?? 0)));
const multiply = (left: Polynomial, right: Polynomial): Polynomial => {
  const output = new Array<number>(left.length + right.length - 1).fill(0);
  for (let i = 0; i < left.length; i += 1) for (let j = 0; j < right.length; j += 1) output[i + j] = (output[i + j] ?? 0) + (left[i] ?? 0) * (right[j] ?? 0);
  return trim(output);
};
const power = (value: Polynomial, exponent: number): Polynomial => {
  if (!Number.isInteger(exponent) || exponent < 0 || exponent > 20) throw new Error("Polynomial exponent must be an integer from 0 to 20");
  let output: Polynomial = [1];
  for (let index = 0; index < exponent; index += 1) output = multiply(output, value);
  return output;
};

function tokens(expression: string): Token[] {
  const output: Token[] = [];
  let index = 0;
  const normalized = expression.replaceAll("²", "^2").replaceAll("³", "^3");
  while (index < normalized.length) {
    const rest = normalized.slice(index);
    const whitespace = rest.match(/^\s+/);
    if (whitespace) { index += whitespace[0].length; continue; }
    const numeric = rest.match(/^(?:\d+(?:\.\d*)?|\.\d+)/);
    if (numeric) { output.push(Number(numeric[0])); index += numeric[0].length; continue; }
    const char = normalized[index];
    if (char === "x" || char === "X") { output.push("x"); index += 1; continue; }
    if (char && "+-*/^()".includes(char)) { output.push(char as Exclude<Token, number | "x">); index += 1; continue; }
    throw new Error(`Unsupported polynomial token at ${index}`);
  }
  const implicit: Token[] = [];
  for (const token of output) {
    const prior = implicit.at(-1);
    const leftFactor = typeof prior === "number" || prior === "x" || prior === ")";
    const rightFactor = typeof token === "number" || token === "x" || token === "(";
    if (leftFactor && rightFactor) implicit.push("*");
    implicit.push(token);
  }
  return implicit;
}

export function polynomialCoefficients(expression: string): Polynomial {
  const input = tokens(expression);
  let position = 0;
  const primary = (): Polynomial => {
    const token = input[position];
    if (token === "+" || token === "-") { position += 1; const value = primary(); return token === "-" ? value.map((coefficient) => -coefficient) : value; }
    if (typeof token === "number") { position += 1; return [token]; }
    if (token === "x") { position += 1; return [0, 1]; }
    if (token === "(") {
      position += 1; const value = additive();
      if (input[position] !== ")") throw new Error("Missing closing polynomial parenthesis");
      position += 1; return value;
    }
    throw new Error("Expected polynomial term");
  };
  const exponential = (): Polynomial => {
    let value = primary();
    if (input[position] === "^") {
      position += 1; const exponent = primary();
      if (exponent.length !== 1) throw new Error("Polynomial exponent cannot contain x");
      value = power(value, exponent[0] ?? 0);
    }
    return value;
  };
  const multiplicative = (): Polynomial => {
    let value = exponential();
    while (input[position] === "*" || input[position] === "/") {
      const operator = input[position]; position += 1; const right = exponential();
      if (operator === "*") value = multiply(value, right);
      else {
        if (right.length !== 1 || right[0] === 0) throw new Error("Polynomial division requires a nonzero constant divisor");
        value = value.map((coefficient) => coefficient / right[0]);
      }
    }
    return value;
  };
  const additive = (): Polynomial => {
    let value = multiplicative();
    while (input[position] === "+" || input[position] === "-") {
      const operator = input[position]; position += 1; value = add(value, multiplicative(), operator === "+" ? 1 : -1);
    }
    return value;
  };
  const result = trim(additive());
  if (position !== input.length) throw new Error("Unexpected trailing polynomial input");
  return result;
}

export function expressionsEquivalent(left: string, right: string): boolean {
  const a = polynomialCoefficients(left); const b = polynomialCoefficients(right);
  return a.length === b.length && a.every((coefficient, index) => Math.abs(coefficient - (b[index] ?? 0)) < 1e-12);
}

export interface LinearSolution { kind: "one" | "none" | "all"; value?: number }
export interface PolynomialSolution { kind: "finite" | "none" | "all"; roots: number[] }

function equationCoefficients(equation: string): Polynomial {
  const sides = equation.split("=");
  if (sides.length !== 2 || !sides[0] || !sides[1]) throw new Error("Equation must contain exactly one equals sign");
  return trim(add(polynomialCoefficients(sides[0]), polynomialCoefficients(sides[1]), -1));
}

export function solveLinearEquation(equation: string): LinearSolution {
  const coefficients = equationCoefficients(equation);
  if (coefficients.length > 2) throw new Error("Equation is not linear");
  const constant = coefficients[0] ?? 0;
  const coefficient = coefficients[1] ?? 0;
  if (Math.abs(coefficient) < 1e-12) return Math.abs(constant) < 1e-12 ? { kind: "all" } : { kind: "none" };
  return { kind: "one", value: -constant / coefficient };
}

export function solvePolynomialEquation(equation: string): PolynomialSolution {
  const coefficients = equationCoefficients(equation);
  if (coefficients.length <= 1) return Math.abs(coefficients[0] ?? 0) < 1e-12 ? { kind: "all", roots: [] } : { kind: "none", roots: [] };
  if (coefficients.length === 2) return { kind: "finite", roots: [-(coefficients[0] ?? 0) / (coefficients[1] ?? 0)] };
  if (coefficients.length !== 3) throw new Error("Only linear and quadratic equations are supported");
  const [c = 0, b = 0, a = 0] = coefficients;
  const discriminant = b ** 2 - 4 * a * c;
  if (discriminant < -1e-12) return { kind: "none", roots: [] };
  if (Math.abs(discriminant) < 1e-12) return { kind: "finite", roots: [-b / (2 * a)] };
  const root = Math.sqrt(discriminant);
  return { kind: "finite", roots: [(-b - root) / (2 * a), (-b + root) / (2 * a)].sort((left, right) => left - right) };
}

export function polynomialEquationsEquivalent(left: string, right: string): boolean {
  const a = equationCoefficients(left);
  const b = equationCoefficients(right);
  if (a.length !== b.length) return false;
  const pivot = a.findIndex((coefficient) => Math.abs(coefficient) >= 1e-12);
  const otherPivot = b.findIndex((coefficient) => Math.abs(coefficient) >= 1e-12);
  if (pivot === -1 || otherPivot === -1) return pivot === otherPivot;
  if (pivot !== otherPivot) return false;
  const scale = (b[pivot] ?? 0) / (a[pivot] ?? 1);
  return a.every((coefficient, index) => Math.abs(coefficient * scale - (b[index] ?? 0)) < 1e-12);
}

export function linearEquationsEquivalent(left: string, right: string): boolean {
  const a = solveLinearEquation(left);
  const b = solveLinearEquation(right);
  return a.kind === b.kind && (a.kind !== "one" || Math.abs((a.value ?? 0) - (b.value ?? 0)) < 1e-12);
}
