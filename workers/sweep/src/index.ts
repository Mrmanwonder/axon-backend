import { serviceClient } from "@mastery/shared/http.js";
import { deleteObject, deletePrefix, type BucketKind } from "@mastery/shared/r2.js";
import type { Env } from "@mastery/shared/env.js";

const KEYS_PER_TICK = 200;
const CLAIMS_PER_TICK = 20;

interface DeletionClaim {
  id: string;
  bucket: BucketKind;
  key: string | null;
  prefix: string | null;
}

export default {
  async scheduled(_controller: ScheduledController, env: Env) {
    const sb = serviceClient(env);

    // Stuck-run recovery executes in pg_cron; see
    // db/operations/enable-stuck-run-recovery.sql. The function is private,
    // so calling sb.rpc("sweep_stuck_runs") through the public API cannot work.

    const { data: claims, error: claimError } = await sb.rpc("claim_deletions", { p_limit: CLAIMS_PER_TICK });
    if (claimError) {
      console.error("claim_deletions failed", claimError.message);
      return;
    }

    for (const claim of (claims ?? []) as DeletionClaim[]) {
      try {
        if (claim.key) {
          await deleteObject(env, claim.bucket, claim.key);
          await sb.rpc("finish_deletion", { p_id: claim.id });
        } else if (claim.prefix) {
          const walk = await deletePrefix(env, claim.bucket, claim.prefix, { maxKeys: KEYS_PER_TICK });
          if (walk.done) {
            await sb.rpc("finish_deletion", { p_id: claim.id });
          } else {
            await sb.rpc("finish_deletion", { p_id: claim.id, p_error: `${walk.deleted} deleted, more to go` });
          }
        }
      } catch (cause) {
        await sb.rpc("finish_deletion", { p_id: claim.id, p_error: String(cause).slice(0, 500) });
      }
    }
  },
} satisfies ExportedHandler<Env>;
