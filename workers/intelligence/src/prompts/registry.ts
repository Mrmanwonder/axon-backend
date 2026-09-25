import type { TSchema } from "@sinclair/typebox";

export interface PromptArtifact {
  id: string;
  system: string;
  task: string;
  schemaId: string;
  schema: TSchema;
}

export interface CompiledPrompt extends PromptArtifact {
  promptHash: string;
  schemaHash: string;
}

const encoder = new TextEncoder();

async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonical(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export class PromptRegistry {
  readonly #artifacts = new Map<string, PromptArtifact>();

  register(artifact: PromptArtifact): void {
    if (this.#artifacts.has(artifact.id)) throw new Error(`Prompt already registered: ${artifact.id}`);
    this.#artifacts.set(artifact.id, Object.freeze(artifact));
  }

  has(id: string): boolean { return this.#artifacts.has(id); }
  ids(): string[] { return [...this.#artifacts.keys()].sort(); }

  async compileAll(): Promise<CompiledPrompt[]> {
    return Promise.all(this.ids().map((id) => this.compile(id)));
  }

  async compile(id: string): Promise<CompiledPrompt> {
    const artifact = this.#artifacts.get(id);
    if (!artifact) throw new Error(`Unknown prompt: ${id}`);
    const serializedSchema = canonical(artifact.schema);
    const [promptHash, schemaHash] = await Promise.all([
      sha256(`${artifact.system}\n${artifact.task}\n${serializedSchema}`),
      sha256(serializedSchema)
    ]);
    return { ...artifact, promptHash, schemaHash };
  }

  validateRoutes(promptIds: readonly string[]): void {
    const missing = promptIds.filter((id) => !this.#artifacts.has(id));
    if (missing.length) throw new Error(`Routes reference missing prompts: ${missing.join(", ")}`);
  }
}
