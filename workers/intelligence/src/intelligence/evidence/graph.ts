import type { Claim, Evidence } from "../../schemas";

export class EvidenceGraph {
  readonly #evidence = new Map<string, Evidence>();
  readonly #claims = new Map<string, Claim>();

  constructor(evidence: readonly Evidence[] = [], claims: readonly Claim[] = []) {
    for (const item of evidence) this.addEvidence(item);
    for (const claim of claims) this.addClaim(claim);
  }

  addEvidence(evidence: Evidence): void {
    if (this.#evidence.has(evidence.id)) throw new Error(`Duplicate evidence id: ${evidence.id}`);
    this.#evidence.set(evidence.id, structuredClone(evidence));
  }

  addClaim(claim: Claim): void {
    if (this.#claims.has(claim.id)) throw new Error(`Duplicate claim id: ${claim.id}`);
    this.#claims.set(claim.id, structuredClone(claim));
  }

  evidence(id: string): Evidence | undefined { return this.#evidence.get(id); }
  claim(id: string): Claim | undefined { return this.#claims.get(id); }
  allEvidence(): Evidence[] { return [...this.#evidence.values()]; }
  allClaims(): Claim[] { return [...this.#claims.values()]; }

  supportFor(claim: Claim): Evidence[] {
    return claim.evidenceIds.flatMap((id) => {
      const item = this.#evidence.get(id);
      return item ? [item] : [];
    });
  }
}
