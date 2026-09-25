export interface CalculationResult { value: number; expression: string; verified: true }

type Token = { type: "number"; value: number } | { type: "operator"; value: string } | { type: "paren"; value: "(" | ")" };

function tokenize(expression: string, variables: Readonly<Record<string, number>>): Token[] {
  const tokens: Token[] = [];
  let index = 0;
  while (index < expression.length) {
    const rest = expression.slice(index);
    const whitespace = rest.match(/^\s+/);
    if (whitespace) { index += whitespace[0].length; continue; }
    const number = rest.match(/^(?:\d+(?:\.\d*)?|\.\d+)/);
    if (number) { tokens.push({ type: "number", value: Number(number[0]) }); index += number[0].length; continue; }
    const variable = rest.match(/^[A-Za-z]+/);
    if (variable) {
      const value = variables[variable[0]];
      if (value === undefined) throw new Error(`Unknown variable: ${variable[0]}`);
      tokens.push({ type: "number", value }); index += variable[0].length; continue;
    }
    const char = expression[index];
    if (char && "+-*/^%".includes(char)) { tokens.push({ type: "operator", value: char }); index += 1; continue; }
    if (char === "(" || char === ")") { tokens.push({ type: "paren", value: char }); index += 1; continue; }
    throw new Error(`Unsupported token at position ${index}`);
  }
  return tokens;
}

function withImplicitMultiplication(tokens: Token[]): Token[] {
  const output: Token[] = [];
  for (const token of tokens) {
    const previous = output.at(-1);
    if (previous && (previous.type === "number" || (previous.type === "paren" && previous.value === ")")) &&
      (token.type === "number" || (token.type === "paren" && token.value === "("))) {
      output.push({ type: "operator", value: "*" });
    }
    output.push(token);
  }
  return output;
}

export function evaluateExpression(expression: string, variables: Readonly<Record<string, number>> = {}): number {
  const tokens = withImplicitMultiplication(tokenize(expression.replaceAll("²", "^2").replaceAll("³", "^3"), variables));
  let position = 0;
  const parsePrimary = (): number => {
    const token = tokens[position];
    if (!token) throw new Error("Unexpected end of expression");
    if (token.type === "operator" && (token.value === "+" || token.value === "-")) {
      position += 1; const value = parsePrimary(); return token.value === "-" ? -value : value;
    }
    if (token.type === "number") { position += 1; return token.value; }
    if (token.type === "paren" && token.value === "(") {
      position += 1; const value = parseAdditive();
      const close = tokens[position];
      if (!close || close.type !== "paren" || close.value !== ")") throw new Error("Missing closing parenthesis");
      position += 1; return value;
    }
    throw new Error("Expected a number or opening parenthesis");
  };
  const parsePower = (): number => {
    const left = parsePrimary(); const token = tokens[position];
    if (token?.type === "operator" && token.value === "^") { position += 1; return left ** parsePower(); }
    return left;
  };
  const parseMultiplicative = (): number => {
    let value = parsePower();
    while (true) {
      const token = tokens[position];
      if (token?.type !== "operator" || !["*", "/", "%"].includes(token.value)) break;
      position += 1; const right = parsePower();
      value = token.value === "*" ? value * right : token.value === "/" ? value / right : value % right;
    }
    return value;
  };
  const parseAdditive = (): number => {
    let value = parseMultiplicative();
    while (true) {
      const token = tokens[position];
      if (token?.type !== "operator" || !["+", "-"].includes(token.value)) break;
      position += 1; const right = parseMultiplicative(); value = token.value === "+" ? value + right : value - right;
    }
    return value;
  };
  const value = parseAdditive();
  if (position !== tokens.length) throw new Error("Unexpected trailing expression");
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error("Calculation did not produce a finite number");
  return value;
}

export function calculate(expression: string): CalculationResult {
  const value = evaluateExpression(expression);
  return { value, expression, verified: true };
}

export function approximatelyEqual(left: number, right: number, tolerance = 1e-10): boolean {
  return Math.abs(left - right) <= tolerance * Math.max(1, Math.abs(left), Math.abs(right));
}
