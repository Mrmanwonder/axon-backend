/**
 * The paper pipeline's bindings, asserted present (AXO-126).
 *
 * The tutor-only deploy (`wrangler deploy --env tutor`) carries no R2 paper
 * bucket, no paper queue and no document-vision service; its router answers
 * 404 for every paper route, so these paths are unreachable there. If one is
 * ever reached anyway, it fails loudly here instead of with an undefined
 * property read halfway through a write.
 */
export interface PaperBindings {
  artifacts: NonNullable<Env["PAPER_ARTIFACTS"]>;
  queue: NonNullable<Env["PAPER_QUEUE"]>;
  vision: NonNullable<Env["DOCUMENT_VISION"]>;
}

export function requirePaperBindings(env: Env): PaperBindings {
  if (!env.PAPER_ARTIFACTS || !env.PAPER_QUEUE || !env.DOCUMENT_VISION) {
    throw new Error("Paper pipeline bindings are not configured in this deployment profile");
  }
  return { artifacts: env.PAPER_ARTIFACTS, queue: env.PAPER_QUEUE, vision: env.DOCUMENT_VISION };
}
