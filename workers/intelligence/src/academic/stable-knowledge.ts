import type { Evidence } from "../schemas";

export interface StableFact { id: string; subject: string; aliases: readonly string[]; triggers: RegExp; statement: string; sourceHash: string }
export const STABLE_FACTS: readonly StableFact[] = [
  { id: "biology.cells.cell_division.mitosis", subject: "biology", aliases: ["mitosis"], triggers: /\bmitosis\b/i, statement: "Mitosis produces daughter cells that are genetically similar to the parent cell, barring mutation.", sourceHash: "canonical:biology.cells.cell_division.mitosis:v1" },
  { id: "chemistry.substances.water_formula", subject: "chemistry", aliases: ["water", "h2o"], triggers: /\b(?:water|h2o)\b/i, statement: "Water has the chemical formula H₂O.", sourceHash: "canonical:chemistry.substances.water_formula:v1" },
  { id: "physics.mechanics.dynamics.newton_second_law", subject: "physics", aliases: ["newton's second law", "newtons second law", "f ma"], triggers: /\b(?:newton'?s second law|f\s*=\s*ma)\b/i, statement: "Newton's second law relates net force, mass, and acceleration.", sourceHash: "canonical:physics.mechanics.dynamics.newton_second_law:v1" }
];

const normalizeAlias = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

export function resolveStableKnowledge(request: string): Evidence[] {
  return STABLE_FACTS.filter((fact) => fact.triggers.test(request)).map((fact): Evidence => ({
    id: `stable:${fact.id}`, informationClass: "STABLE_KNOWLEDGE", source: "stable_knowledge", authority: "primary",
    value: fact.statement, provenance: { artifactHash: fact.sourceHash }, verification: "verified", confidence: 1
  }));
}

export async function recordStableKnowledge(db: D1Database, revision: string): Promise<void> {
  const createdAt = new Date().toISOString();
  const statements: D1PreparedStatement[] = [];
  for (const fact of STABLE_FACTS) {
    statements.push(db.prepare("INSERT OR IGNORE INTO stable_knowledge (id, subject, statement, revision, source_hash, active, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)")
      .bind(fact.id, fact.subject, fact.statement, revision, fact.sourceHash, createdAt));
    for (const alias of fact.aliases) statements.push(db.prepare("INSERT OR IGNORE INTO stable_knowledge_alias (knowledge_id, alias_norm) VALUES (?, ?)").bind(fact.id, normalizeAlias(alias)));
  }
  await db.batch(statements);
}

function candidatePhrases(message: string): string[] {
  const words = message.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().split(/\s+/).filter(Boolean).slice(0, 100);
  const phrases = new Set<string>();
  for (let start = 0; start < words.length; start += 1) {
    for (let length = 1; length <= 4 && start + length <= words.length; length += 1) phrases.add(words.slice(start, start + length).join(" "));
  }
  return [...phrases].slice(0, 90);
}

export async function resolveStableKnowledgeFromDb(db: D1Database, message: string): Promise<Evidence[]> {
  const phrases = candidatePhrases(message);
  if (phrases.length === 0) return [];
  const placeholders = phrases.map(() => "?").join(",");
  const rows = await db.prepare(`SELECT DISTINCT sk.id, sk.statement, sk.source_hash
    FROM stable_knowledge sk JOIN stable_knowledge_alias a ON a.knowledge_id = sk.id
    WHERE sk.active = 1 AND a.alias_norm IN (${placeholders}) LIMIT 20`).bind(...phrases).all<{ id: string; statement: string; source_hash: string }>();
  return rows.results.map((fact): Evidence => ({
    id: `stable:${fact.id}`, informationClass: "STABLE_KNOWLEDGE", source: "stable_knowledge", authority: "primary",
    value: fact.statement, provenance: { artifactHash: fact.source_hash }, verification: "verified", confidence: 1
  }));
}
