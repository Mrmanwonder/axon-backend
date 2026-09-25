export type Dimensions = Readonly<Record<"M" | "L" | "T" | "I" | "Theta" | "N" | "J", number>>;
export interface UnitDefinition { symbol: string; scale: number; dimensions: Dimensions }

const d = (M = 0, L = 0, T = 0, I = 0, Theta = 0, N = 0, J = 0): Dimensions => ({ M, L, T, I, Theta, N, J });
const UNITS: Record<string, UnitDefinition> = {
  "1": { symbol: "1", scale: 1, dimensions: d() },
  m: { symbol: "m", scale: 1, dimensions: d(0, 1) }, cm: { symbol: "cm", scale: 0.01, dimensions: d(0, 1) }, km: { symbol: "km", scale: 1000, dimensions: d(0, 1) },
  s: { symbol: "s", scale: 1, dimensions: d(0, 0, 1) }, min: { symbol: "min", scale: 60, dimensions: d(0, 0, 1) }, h: { symbol: "h", scale: 3600, dimensions: d(0, 0, 1) },
  kg: { symbol: "kg", scale: 1, dimensions: d(1) }, g: { symbol: "g", scale: 0.001, dimensions: d(1) },
  A: { symbol: "A", scale: 1, dimensions: d(0, 0, 0, 1) }, K: { symbol: "K", scale: 1, dimensions: d(0, 0, 0, 0, 1) }, mol: { symbol: "mol", scale: 1, dimensions: d(0, 0, 0, 0, 0, 1) },
  N: { symbol: "N", scale: 1, dimensions: d(1, 1, -2) }, J: { symbol: "J", scale: 1, dimensions: d(1, 2, -2) }, W: { symbol: "W", scale: 1, dimensions: d(1, 2, -3) },
  Pa: { symbol: "Pa", scale: 1, dimensions: d(1, -1, -2) }, C: { symbol: "C", scale: 1, dimensions: d(0, 0, 1, 1) }, V: { symbol: "V", scale: 1, dimensions: d(1, 2, -3, -1) }
};

export function unitDefinition(symbol: string): UnitDefinition {
  const unit = UNITS[symbol];
  if (!unit) throw new Error(`Unknown unit: ${symbol}`);
  return unit;
}

export function parseUnitExpression(expression: string): UnitDefinition {
  const normalized = expression.trim().replaceAll("·", "*").replaceAll("²", "^2").replaceAll("³", "^3");
  const divisions = normalized.split("/");
  let scale = 1;
  const dimensions: Record<keyof Dimensions, number> = { M: 0, L: 0, T: 0, I: 0, Theta: 0, N: 0, J: 0 };
  divisions.forEach((section, sectionIndex) => {
    const factors = section.trim().split(/[\s*]+/).filter(Boolean);
    if (factors.length === 0) throw new Error(`Invalid unit expression: ${expression}`);
    for (const factor of factors) {
      const match = factor.match(/^([A-Za-z]+|1)(?:\^(-?\d+))?$/);
      if (!match?.[1]) throw new Error(`Invalid unit factor: ${factor}`);
      const definition = unitDefinition(match[1]);
      const exponent = (sectionIndex === 0 ? 1 : -1) * Number(match[2] ?? "1");
      scale *= definition.scale ** exponent;
      for (const key of Object.keys(dimensions) as Array<keyof Dimensions>) dimensions[key] += definition.dimensions[key] * exponent;
    }
  });
  return { symbol: expression, scale, dimensions };
}

export function dimensionallyEquivalent(left: string, right: string): boolean {
  const a = parseUnitExpression(left).dimensions; const b = parseUnitExpression(right).dimensions;
  return Object.keys(a).every((key) => a[key as keyof Dimensions] === b[key as keyof Dimensions]);
}

export function convert(value: number, from: string, to: string): number {
  const source = parseUnitExpression(from); const target = parseUnitExpression(to);
  if (!dimensionallyEquivalent(from, to)) throw new Error(`Units ${from} and ${to} are not dimensionally equivalent`);
  return value * source.scale / target.scale;
}

export function significantFigures(value: string): number {
  const normalized = value.trim().replace(/^[+-]/, "").split(/[eE]/)[0] ?? "";
  const digits = normalized.replace(".", "").replace(/^0+/, "").replace(/0+$/, (zeros) => normalized.includes(".") ? zeros : "");
  return digits.length;
}
