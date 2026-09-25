import { describe, expect, it } from "vitest";
import { calculate, evaluateExpression } from "../src/academic/math/calculator";
import { expressionsEquivalent, linearEquationsEquivalent, polynomialEquationsEquivalent, solveLinearEquation, solvePolynomialEquation } from "../src/academic/math/polynomial";
import { convert, dimensionallyEquivalent, significantFigures } from "../src/academic/units";
import { balanceEquation, molarMass, oxidationState, parseFormula, stoichiometricMass } from "../src/academic/chemistry";
import { runAcademicTools } from "../src/academic/tools";

describe("AxonMath", () => {
  it("calculates with precedence without eval", () => {
    expect(calculate("2 + 3 * (4 - 1)").value).toBe(11);
    expect(evaluateExpression("2x + 6", { x: 4 })).toBe(14);
  });
  it("checks algebraic and factorisation equivalence", () => {
    expect(expressionsEquivalent("2(x+3)", "2x+6")).toBe(true);
    expect(expressionsEquivalent("(x-3)(x+3)", "x^2-9")).toBe(true);
    expect(expressionsEquivalent("x+1", "x+2")).toBe(false);
    expect(expressionsEquivalent("(x+7.25)(x+3)(x+0.5)x(x-0.75)(x-2)(x-5.5)(x-11)", "0")).toBe(false);
  });
  it("rejects executable input", () => {
    expect(() => calculate("globalThis.fetch('x')")).toThrow();
  });
  it("solves linear equations and validates transformation steps exactly", () => {
    expect(solveLinearEquation("2(x+3)=14")).toEqual({ kind: "one", value: 4 });
    expect(solveLinearEquation("2x+1=2x+1")).toEqual({ kind: "all" });
    expect(solveLinearEquation("2x+1=2x+2")).toEqual({ kind: "none" });
    expect(linearEquationsEquivalent("2x+6=14", "2x=8")).toBe(true);
    expect(linearEquationsEquivalent("2x+6=14", "2x=10")).toBe(false);
  });
  it("solves quadratic equations and validates equivalent equation transformations", () => {
    expect(solvePolynomialEquation("x^2-5x+6=0")).toEqual({ kind: "finite", roots: [2, 3] });
    expect(solvePolynomialEquation("x^2+1=0")).toEqual({ kind: "none", roots: [] });
    expect(polynomialEquationsEquivalent("x^2-5x+6=0", "2x^2-10x+12=0")).toBe(true);
    expect(polynomialEquationsEquivalent("x^2-5x+6=0", "x^2-4x+3=0")).toBe(false);
  });
});

describe("AxonUnits", () => {
  it("converts and checks dimensions", () => {
    expect(convert(150, "cm", "m")).toBeCloseTo(1.5);
    expect(convert(36, "km/h", "m/s")).toBeCloseTo(10);
    expect(dimensionallyEquivalent("kg*m/s^2", "N")).toBe(true);
    expect(dimensionallyEquivalent("J", "N")).toBe(false);
    expect(significantFigures("0.00450")).toBe(3);
  });
});

describe("AxonChem", () => {
  it("parses grouped formulae and molar mass", () => {
    expect(parseFormula("Ca(OH)2")).toEqual({ Ca: 1, O: 2, H: 2 });
    expect(molarMass("H2O")).toBeCloseTo(18.015, 3);
  });
  it("balances a chemical equation deterministically", () => {
    expect(balanceEquation("H2 + O2 -> H2O")).toEqual({
      reactants: [{ formula: "H2", coefficient: 2 }, { formula: "O2", coefficient: 1 }],
      products: [{ formula: "H2O", coefficient: 2 }]
    });
  });
  it("derives oxidation states and stoichiometric masses from deterministic rules", () => {
    expect(oxidationState("KMnO4", "Mn")).toBe(7);
    expect(oxidationState("H2O2", "O")).toBe(-1);
    expect(stoichiometricMass("H2 + O2 -> H2O", "H2", 4.032, "H2O").targetMassGrams).toBeCloseTo(36.03, 2);
  });
});

describe("shared AcademicTool contract", () => {
  it("turns deterministic unit results into verified evidence", async () => {
    const evidence = await runAcademicTools({ request: "Convert 150 cm to m", evidence: [] }, new Set(["axon.units.v1"]));
    expect(evidence[0]?.provenance.toolId).toBe("axon.units.v1");
    expect(evidence[0]?.verification).toBe("verified");
    expect(evidence[0]?.value).toMatchObject({ result: 1.5 });
  });
});
