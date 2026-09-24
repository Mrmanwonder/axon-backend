export interface CacheIdentity { artifactHash: string; stage: string; pipelineVersion: string; model: string; promptHash: string; schemaHash: string }
export async function cacheKey(identity: CacheIdentity): Promise<string> {
  const canonical = [identity.artifactHash, identity.stage, identity.pipelineVersion, identity.model, identity.promptHash, identity.schemaHash].join("\u001f");
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical));
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}
