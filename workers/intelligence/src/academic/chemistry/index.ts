const ATOMIC_MASS: Readonly<Record<string, number>> = {
  H: 1.008, He: 4.0026, Li: 6.94, Be: 9.0122, B: 10.81, C: 12.011, N: 14.007, O: 15.999,
  F: 18.998, Ne: 20.180, Na: 22.990, Mg: 24.305, Al: 26.982, Si: 28.085, P: 30.974, S: 32.06,
  Cl: 35.45, Ar: 39.948, K: 39.098, Ca: 40.078, Fe: 55.845, Cu: 63.546, Zn: 65.38, Br: 79.904,
  Ag: 107.8682, I: 126.904, Ba: 137.327, Au: 196.967, Hg: 200.592, Pb: 207.2
};

export function parseFormula(formula: string): Record<string, number> {
  let index = 0;
  const parseGroup = (until?: string): Record<string, number> => {
    const counts: Record<string, number> = {};
    while (index < formula.length && formula[index] !== until) {
      if (formula[index] === "(") {
        index += 1; const inner = parseGroup(")");
        if (formula[index] !== ")") throw new Error("Unclosed chemical group");
        index += 1; const multiplier = readNumber();
        for (const [element, count] of Object.entries(inner)) counts[element] = (counts[element] ?? 0) + count * multiplier;
        continue;
      }
      const match = formula.slice(index).match(/^[A-Z][a-z]?/);
      if (!match) throw new Error(`Invalid chemical formula at position ${index}`);
      index += match[0].length; const multiplier = readNumber();
      counts[match[0]] = (counts[match[0]] ?? 0) + multiplier;
    }
    return counts;
  };
  const readNumber = (): number => {
    const match = formula.slice(index).match(/^\d+/);
    if (!match) return 1;
    index += match[0].length; return Number(match[0]);
  };
  const result = parseGroup();
  if (index !== formula.length) throw new Error("Unexpected closing group");
  return result;
}

export function molarMass(formula: string): number {
  return Object.entries(parseFormula(formula)).reduce((sum, [element, count]) => {
    const mass = ATOMIC_MASS[element];
    if (mass === undefined) throw new Error(`Unknown atomic mass: ${element}`);
    return sum + mass * count;
  }, 0);
}

export interface BalancedEquation { reactants: Array<{ formula: string; coefficient: number }>; products: Array<{ formula: string; coefficient: number }> }

export function balanceEquation(equation: string, maxCoefficient = 8): BalancedEquation {
  const sides = equation.split(/->|→/);
  if (sides.length !== 2) throw new Error("Equation must have one reaction arrow");
  const reactants = (sides[0] ?? "").split("+").map((value) => value.trim()).filter(Boolean);
  const products = (sides[1] ?? "").split("+").map((value) => value.trim()).filter(Boolean);
  const formulas = [...reactants, ...products];
  if (formulas.length < 2 || formulas.length > 6) throw new Error("Equation must contain 2 to 6 compounds");
  const parsed = formulas.map(parseFormula);
  const elements = [...new Set(parsed.flatMap((item) => Object.keys(item)))];
  const coefficients = new Array<number>(formulas.length).fill(1);
  let attempts = 0;
  const valid = (): boolean => elements.every((element) => {
    const left = reactants.reduce((sum, _, index) => sum + (parsed[index]?.[element] ?? 0) * (coefficients[index] ?? 0), 0);
    const right = products.reduce((sum, _, index) => {
      const offset = reactants.length + index;
      return sum + (parsed[offset]?.[element] ?? 0) * (coefficients[offset] ?? 0);
    }, 0);
    return left === right;
  });
  const search = (position: number): boolean => {
    if (++attempts > 1_000_000) throw new Error("Equation balancing search limit exceeded");
    if (position === coefficients.length) return valid();
    for (let value = 1; value <= maxCoefficient; value += 1) {
      coefficients[position] = value;
      if (search(position + 1)) return true;
    }
    return false;
  };
  if (!search(0)) throw new Error("No balance found within coefficient limit");
  return {
    reactants: reactants.map((formula, index) => ({ formula, coefficient: coefficients[index] ?? 1 })),
    products: products.map((formula, index) => ({ formula, coefficient: coefficients[reactants.length + index] ?? 1 }))
  };
}

const COMMON_OXIDATION_STATE: Readonly<Record<string, number>> = {
  Li: 1, Na: 1, K: 1, Be: 2, Mg: 2, Ca: 2, Ba: 2, F: -1, Cl: -1, Br: -1, I: -1, O: -2, H: 1
};

export function oxidationState(formula: string, targetElement: string, overallCharge = 0): number {
  const counts = parseFormula(formula);
  const targetCount = counts[targetElement];
  if (!targetCount) throw new Error(`Element ${targetElement} is absent from ${formula}`);
  let knownTotal = 0;
  for (const [element, count] of Object.entries(counts)) {
    if (element === targetElement) continue;
    const state = COMMON_OXIDATION_STATE[element];
    if (state === undefined) throw new Error(`Oxidation state is underdetermined because ${element} has no fixed rule`);
    knownTotal += state * count;
  }
  const result = (overallCharge - knownTotal) / targetCount;
  if (!Number.isInteger(result) || result < -8 || result > 8) throw new Error("Oxidation state is not uniquely supported by the deterministic rules");
  return result;
}

export interface StoichiometricMassResult { balanced: BalancedEquation; knownMoles: number; targetMoles: number; targetMassGrams: number }

export function stoichiometricMass(equation: string, knownFormula: string, knownMassGrams: number, targetFormula: string): StoichiometricMassResult {
  if (!Number.isFinite(knownMassGrams) || knownMassGrams < 0) throw new Error("Known mass must be a non-negative finite number");
  const balanced = balanceEquation(equation);
  const species = [...balanced.reactants, ...balanced.products];
  const known = species.find((item) => item.formula === knownFormula);
  const target = species.find((item) => item.formula === targetFormula);
  if (!known || !target) throw new Error("Known and target formulae must both occur in the equation");
  const knownMoles = knownMassGrams / molarMass(knownFormula);
  const targetMoles = knownMoles / known.coefficient * target.coefficient;
  return { balanced, knownMoles, targetMoles, targetMassGrams: targetMoles * molarMass(targetFormula) };
}
