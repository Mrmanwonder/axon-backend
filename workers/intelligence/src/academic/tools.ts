import { balanceEquation, molarMass, oxidationState, stoichiometricMass } from "./chemistry";
import { calculate } from "./math/calculator";
import { expressionsEquivalent, polynomialEquationsEquivalent, solvePolynomialEquation } from "./math/polynomial";
import { convert, dimensionallyEquivalent } from "./units";
import type { Evidence } from "../schemas";

export interface EvidencePacket { request: string; evidence: readonly Evidence[] }
export interface ToolEvidence { toolId: string; value: unknown; verified: true }
export interface AcademicTool { id: string; supports(input: EvidencePacket): boolean; evaluate(input: EvidencePacket): Promise<ToolEvidence> }

export class CalculatorTool implements AcademicTool {
  readonly id = "axon.calculator.v1";
  supports(input: EvidencePacket): boolean { return /(?:calculate|what is)?\s*[\d.]+\s*[+\-*/^%]/i.test(input.request); }
  evaluate(input: EvidencePacket): Promise<ToolEvidence> {
    const expression = input.request.match(/(?:calculate|what is)?\s*([\d\s()+\-*/.^%]{3,})/i)?.[1]?.trim();
    if (!expression) return Promise.reject(new Error("No supported arithmetic expression found"));
    return Promise.resolve({ toolId: this.id, value: calculate(expression), verified: true });
  }
}

export class SymbolicMathTool implements AcademicTool {
  readonly id = "axon.math.v1";
  supports(input: EvidencePacket): boolean { return /\s(?:vs\.?|equivalent to)\s|\bsolve\s+[^=]+=|\bcheck step\b/i.test(input.request); }
  evaluate(input: EvidencePacket): Promise<ToolEvidence> {
    const step = input.request.match(/\bcheck step\s+(.+?=.+?)\s*(?:->|→)\s*(.+?=.+?)(?:\?|$)/i);
    if (step?.[1] && step[2]) return Promise.resolve({ toolId: this.id, value: { from: step[1], to: step[2], equivalent: polynomialEquationsEquivalent(step[1], step[2]) }, verified: true });
    const solve = input.request.match(/\bsolve\s+(.+?=.+?)(?:\?|$)/i);
    if (solve?.[1]) return Promise.resolve({ toolId: this.id, value: { equation: solve[1], solution: solvePolynomialEquation(solve[1]) }, verified: true });
    const match = input.request.match(/(.+?)\s+(?:vs\.?|equivalent to)\s+(.+?)(?:\?|$)/i);
    if (!match?.[1] || !match[2]) return Promise.reject(new Error("No expression pair found"));
    return Promise.resolve({ toolId: this.id, value: { left: match[1], right: match[2], equivalent: expressionsEquivalent(match[1], match[2]) }, verified: true });
  }
}

export class UnitsTool implements AcademicTool {
  readonly id = "axon.units.v1";
  supports(input: EvidencePacket): boolean { return /\bconvert\s+[+-]?[\d.]+\s*[A-Za-z0-9/*^.-]+\s+(?:to|into)\s+[A-Za-z0-9/*^.-]+/i.test(input.request); }
  evaluate(input: EvidencePacket): Promise<ToolEvidence> {
    const oxidation = input.request.match(/\boxidation state(?: of)?\s+([A-Z][a-z]?)\s+in\s+([A-Z][A-Za-z0-9()]*)/i);
    if (oxidation?.[1] && oxidation[2]) return Promise.resolve({ toolId: this.id, value: { formula: oxidation[2], element: oxidation[1], oxidationState: oxidationState(oxidation[2], oxidation[1]) }, verified: true });
    const stoichiometry = input.request.match(/\bmass of\s+([A-Z][A-Za-z0-9()]*)\s+from\s+([\d.]+)\s*g\s+([A-Z][A-Za-z0-9()]*)\s+in\s+(.+?(?:->|→).+?)(?:\?|$)/i);
    if (stoichiometry?.[1] && stoichiometry[2] && stoichiometry[3] && stoichiometry[4]) return Promise.resolve({ toolId: this.id, value: stoichiometricMass(stoichiometry[4].trim(), stoichiometry[3], Number(stoichiometry[2]), stoichiometry[1]), verified: true });
    const match = input.request.match(/\bconvert\s+([+-]?[\d.]+)\s*([A-Za-z0-9/*^.-]+)\s+(?:to|into)\s+([A-Za-z0-9/*^.-]+)/i);
    if (!match?.[1] || !match[2] || !match[3]) return Promise.reject(new Error("No supported unit conversion found"));
    const numeric = Number(match[1]);
    return Promise.resolve({ toolId: this.id, value: { input: numeric, from: match[2], to: match[3], dimensionallyEquivalent: dimensionallyEquivalent(match[2], match[3]), result: convert(numeric, match[2], match[3]) }, verified: true });
  }
}

export class ChemistryTool implements AcademicTool {
  readonly id = "axon.chem.v1";
  supports(input: EvidencePacket): boolean { return /\b(balance|molar mass)\b/i.test(input.request); }
  evaluate(input: EvidencePacket): Promise<ToolEvidence> {
    const balance = input.request.match(/\bbalance\s+([A-Za-z0-9()+\s]+(?:->|→)[A-Za-z0-9()+\s]+)/i);
    if (balance?.[1]) return Promise.resolve({ toolId: this.id, value: balanceEquation(balance[1].trim()), verified: true });
    const mass = input.request.match(/\bmolar mass(?: of)?\s+([A-Z][A-Za-z0-9()]*)/i);
    if (mass?.[1]) return Promise.resolve({ toolId: this.id, value: { formula: mass[1], gramsPerMole: molarMass(mass[1]) }, verified: true });
    return Promise.reject(new Error("No supported chemistry operation found"));
  }
}

export const ACADEMIC_TOOLS: readonly AcademicTool[] = [new CalculatorTool(), new SymbolicMathTool(), new UnitsTool(), new ChemistryTool()];

export async function runAcademicTools(packet: EvidencePacket, requestedToolIds: ReadonlySet<string>): Promise<Evidence[]> {
  const selected = ACADEMIC_TOOLS.filter((tool) => requestedToolIds.has(tool.id) && tool.supports(packet));
  const outputs = await Promise.all(selected.map((tool) => tool.evaluate(packet)));
  return outputs.map((output): Evidence => ({
    id: `${output.toolId}:${crypto.randomUUID()}`, informationClass: "DERIVED", source: "tool", authority: "derived",
    value: output.value, provenance: { toolId: output.toolId }, verification: "verified", confidence: 1
  }));
}
