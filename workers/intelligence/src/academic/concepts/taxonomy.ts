export interface Concept { id: string; subject: string; parent?: string; aliases: readonly string[]; curricula: readonly string[] }

const CONCEPTS: readonly Concept[] = [
  { id: "physics", subject: "physics", aliases: [], curricula: ["general"] },
  { id: "physics.mechanics", subject: "physics", parent: "physics", aliases: [], curricula: ["general"] },
  { id: "physics.mechanics.dynamics", subject: "physics", parent: "physics.mechanics", aliases: ["dynamics"], curricula: ["general"] },
  { id: "physics.mechanics.dynamics.newton_second_law", subject: "physics", parent: "physics.mechanics.dynamics", aliases: ["newton 2", "newton's second law", "f=ma", "force equation"], curricula: ["general"] },
  { id: "biology", subject: "biology", aliases: [], curricula: ["general"] },
  { id: "biology.cells", subject: "biology", parent: "biology", aliases: [], curricula: ["general"] },
  { id: "biology.cells.cell_division", subject: "biology", parent: "biology.cells", aliases: ["cell division"], curricula: ["general"] },
  { id: "biology.cells.cell_division.mitosis", subject: "biology", parent: "biology.cells.cell_division", aliases: ["mitosis", "cell division"], curricula: ["general"] },
  { id: "mathematics", subject: "mathematics", aliases: ["maths", "math"], curricula: ["general"] },
  { id: "mathematics.algebra", subject: "mathematics", parent: "mathematics", aliases: ["algebra"], curricula: ["general"] },
  { id: "mathematics.algebra.quadratics", subject: "mathematics", parent: "mathematics.algebra", aliases: ["quadratics"], curricula: ["general"] },
  { id: "mathematics.algebra.quadratics.factorisation", subject: "mathematics", parent: "mathematics.algebra.quadratics", aliases: ["quadratic factorisation", "factoring quadratics"], curricula: ["general"] },
  { id: "chemistry", subject: "chemistry", aliases: [], curricula: ["general"] },
  { id: "chemistry.quantitative", subject: "chemistry", parent: "chemistry", aliases: ["quantitative chemistry"], curricula: ["general"] },
  { id: "chemistry.quantitative.moles", subject: "chemistry", parent: "chemistry.quantitative", aliases: ["moles"], curricula: ["general"] },
  { id: "chemistry.quantitative.moles.stoichiometry", subject: "chemistry", parent: "chemistry.quantitative.moles", aliases: ["stoichiometry", "mole ratio"], curricula: ["general"] }
];

export class ConceptTaxonomy {
  readonly #byId = new Map(CONCEPTS.map((concept) => [concept.id, concept]));
  readonly #byAlias = new Map(CONCEPTS.flatMap((concept) => concept.aliases.map((alias) => [alias.toLowerCase(), concept.id] as const)));
  resolve(value: string): Concept | undefined { return this.#byId.get(value) ?? this.#byId.get(this.#byAlias.get(value.toLowerCase()) ?? ""); }
  validate(ids: readonly string[]): { valid: string[]; rejected: string[] } {
    return { valid: ids.filter((id) => this.#byId.has(id)), rejected: ids.filter((id) => !this.#byId.has(id)) };
  }
  all(): Concept[] { return [...this.#byId.values()]; }
}
