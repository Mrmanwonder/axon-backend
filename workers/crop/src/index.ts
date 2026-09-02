// Stage 3.5 · cropping (AXON_FIX_BRIEF.md §8).
//
// Between structure and content. Structure has found the questions and written
// their boxes; content is about to send an image per question and has been
// sending the whole page. This cuts each question out once, so it does not have
// to be sent whole N times.
//
// Content and adjudicate already prefer `question_region.crop_key` when it is
// set — no change was needed in either, and none was made. They take the good
// path the moment the column is populated.
//
// ── the rules this stage runs under ────────────────────────────────────────
//
// **A crop failure never blocks a paper.** Every exit below, including the
// permanent-failure handler, sets a terminal `crop_status` and calls
// `advance_after_crop`, and that function counts only pages still pending or
// running. A page that could not be cropped leaves its regions' `crop_key`
// null, and content falls back to the full page exactly as it does today. The
// worst outcome of this whole stage failing is the latency the system already
// had.
//
// **The subrequest budget is the constraint.** §8.2 names it, and it is the
// failure that took the content stage down once. One invocation is one page:
// two R2 gets, at most CROP_BUDGET x 2 R2 puts, and a single RPC for all the
// key writes rather than one update per region.
//
// **Memory is the other constraint.** A 2400x3200 page is about 30MB of RGBA
// before the codec's own heap, against a Worker's 128MB. The page and the mask
// are therefore decoded one after the other and never held at once, and a page
// larger than MAX_PIXELS is skipped rather than attempted.

import { consumeQueue } from "@mastery/shared/worker.js";
import { objectKey } from "@mastery/shared/r2.js";
import { pageDimensions } from "@mastery/shared/page.js";
import { bandForRegion, cutRegion, type Box, type PageSpan, type RgbaImage } from "@mastery/shared/crop.js";
import { decodeImage, encodeWebp } from "./codecs.js";
import type { Env } from "@mastery/shared/env.js";

interface CropMessage {
  run_id: string;
  page_id: string;
  _retries?: number;
}

/**
 * The most regions one invocation will cut.
 *
 * Each costs two R2 puts, so 60 regions is 120 subrequests plus the reads and
 * the RPC — comfortably inside the ceiling, and comfortably above any real exam
 * page. A page with more regions than this is not truncated silently: the
 * regions past the budget keep `crop_key` null and take the full-page path,
 * which is the same fallback every other unhandled case gets.
 */
const CROP_BUDGET = 60;

/** R2 writes in flight at once. Enough to hide latency, few enough that a
    forty-question page does not open forty sockets at the same moment. */
const WRITE_CONCURRENCY = 8;

/** Above this a page is skipped rather than decoded. 2400x3200 is 7.7M; this
    leaves room for an unusually shaped page without letting a pathological one
    take the isolate down with it. */
const MAX_PIXELS = 16_000_000;

interface RegionRow {
  id: string;
  order_index: number;
  page_spans: PageSpan[];
}

interface CropPlan {
  region: RegionRow;
  band: Box;
}

async function finish(env: Env, sb: any, runId: string, pageId: string, status: string, detail: Record<string, unknown>) {
  await sb.from("paper_page").update({ crop_status: status }).eq("id", pageId);
  const { data: advance } = await sb.rpc("advance_after_crop", { p_run_id: runId });
  if (advance?.advanced) {
    const regionIds: string[] = advance.enqueue_content ?? [];
    if (env.CONTENT_QUEUE && regionIds.length) {
      await env.CONTENT_QUEUE.sendBatch(regionIds.map((regionId) => ({ body: { run_id: runId, region_id: regionId } })));
    }
    if (advance.enqueue_reconcile && env.RECONCILE_QUEUE) {
      await env.RECONCILE_QUEUE.send({ run_id: runId });
    }
  }
  return { detail };
}

/** Cut and store one set of crops off one decoded image. Returns region id ->
    key for the ones that landed; a single failed encode or put costs that one
    crop and nothing else. */
async function writeCrops(
  env: Env,
  image: RgbaImage,
  plans: CropPlan[],
  keyFor: (region: RegionRow) => string,
): Promise<Map<string, string>> {
  const written = new Map<string, string>();
  const bucket = env.DERIVED;
  if (!bucket) throw new Error("crop: DERIVED bucket is not bound");

  for (let i = 0; i < plans.length; i += WRITE_CONCURRENCY) {
    const chunk = plans.slice(i, i + WRITE_CONCURRENCY);
    await Promise.all(chunk.map(async (plan) => {
      try {
        const cut = cutRegion(image, plan.band);
        const bytes = await encodeWebp(cut);
        const key = keyFor(plan.region);
        await bucket.put(key, bytes, { httpMetadata: { contentType: "image/webp" } });
        written.set(plan.region.id, key);
      } catch (cause) {
        // Logged, never swallowed, and never fatal: this region simply keeps
        // the full-page path. §10 — every catch logs and surfaces.
        console.error("crop: region failed", plan.region.id, String(cause));
      }
    }));
  }
  return written;
}

const handler = consumeQueue<CropMessage>(
  async ({ env, sb, msg, beat }) => {
    const runId = msg.run_id;
    const pageId = msg.page_id;

    const { data: page } = await sb
      .from("paper_page")
      .select("id, paper_id, student_id, page_number, r2_bucket, r2_key, mask_key, crop_status, conditioning_meta, quality_signals")
      .eq("id", pageId)
      .single();
    if (!page) return { detail: { skipped: "no such page" } };

    // Re-entry, the same way structure handles it (§3.2): a second pass over an
    // already-cropped page must still advance the run rather than dead-ending it.
    if (["done", "failed", "skipped"].includes(page.crop_status)) {
      return await finish(env, sb, runId, pageId, page.crop_status, { skipped: "already " + page.crop_status });
    }

    const { data: run } = await sb.from("extraction_run").select("status").eq("id", runId).single();
    if (!run || ["failed", "rejected", "committed"].includes(run.status)) {
      return { detail: { skipped: run?.status ?? "no run" } };
    }

    await sb.from("paper_page").update({ crop_status: "running" }).eq("id", pageId);
    await beat();

    const dims = pageDimensions(page as any);
    if (!dims) {
      // Structure already refuses a page it cannot place anything on, so this
      // is belt and braces — but a crop cut against guessed dimensions would
      // land in the wrong place on the paper, and a wrong crop is worse than
      // no crop (§8.5).
      return await finish(env, sb, runId, pageId, "skipped", { skipped: "no page dimensions" });
    }
    if (!page.r2_key) {
      return await finish(env, sb, runId, pageId, "skipped", { skipped: "no page image" });
    }
    if (dims.width * dims.height > MAX_PIXELS) {
      return await finish(env, sb, runId, pageId, "skipped", { skipped: "page too large to decode", pixels: dims.width * dims.height });
    }

    const { data: regions } = await sb
      .from("question_region")
      .select("id, order_index, page_spans")
      .eq("run_id", runId)
      .order("order_index");

    const plans: CropPlan[] = [];
    let multiPage = 0;
    for (const region of (regions ?? []) as RegionRow[]) {
      const spans = region.page_spans ?? [];
      if (spans.length > 1) { multiPage++; continue; }
      const band = bandForRegion(spans, page.page_number, dims.width, dims.height);
      if (band) plans.push({ region, band });
    }
    const budgeted = plans.slice(0, CROP_BUDGET);

    if (!budgeted.length) {
      return await finish(env, sb, runId, pageId, "skipped", { skipped: "nothing to cut on this page", multi_page: multiPage });
    }

    const bucketFor = (name: string | null) => (name === "originals" ? env.ORIGINALS : env.DERIVED);

    // ── the page ─────────────────────────────────────────────────────────────
    const pageObject = await bucketFor(page.r2_bucket)?.get(page.r2_key);
    if (!pageObject) {
      return await finish(env, sb, runId, pageId, "skipped", { skipped: "page image not found in R2" });
    }
    let decoded: RgbaImage | null = await decodeImage(new Uint8Array(await pageObject.arrayBuffer()));

    // Structure scaled every box against `dims`. If the stored image is not
    // that size, the boxes do not refer to these pixels and cutting them would
    // produce confidently wrong crops.
    if (decoded.width !== dims.width || decoded.height !== dims.height) {
      const mismatch = `${decoded.width}x${decoded.height} decoded vs ${dims.width}x${dims.height} recorded`;
      console.error("crop: page image does not match its recorded dimensions", pageId, mismatch);
      return await finish(env, sb, runId, pageId, "skipped", { skipped: "dimension mismatch", detail: mismatch });
    }

    const cropKeys = await writeCrops(env, decoded, budgeted, (region) => objectKey({
      studentId: page.student_id,
      paperId: page.paper_id,
      kind: "crop",
      // Deterministic: a re-run over the same run and region overwrites its own
      // crop rather than leaving an orphan in the bucket for the retention
      // sweep to find later. The run id is in the name so two runs over one
      // paper cannot collide.
      name: `${runId}-${region.id}`,
      extension: "webp",
      unguessable: false,
    }));

    // Released before the mask is decoded. Both at once is roughly 60MB of RGBA
    // plus two codec heaps, and the Worker has 128MB.
    decoded = null;

    // ── the mask ─────────────────────────────────────────────────────────────
    // Best effort, and second on purpose. The mask is where the fine detail
    // lives — a faint one-pixel stroke keeps 12% of itself through the page
    // encoder and all of it here — so it is worth cutting, but a page with
    // crops and no crop masks is still a page content can read, and a page with
    // neither is a wasted decode.
    let maskKeys = new Map<string, string>();
    if (page.mask_key) {
      try {
        const maskObject = await bucketFor(page.r2_bucket)?.get(page.mask_key);
        if (maskObject) {
          const mask = await decodeImage(new Uint8Array(await maskObject.arrayBuffer()));
          if (mask.width === dims.width && mask.height === dims.height) {
            maskKeys = await writeCrops(env, mask, budgeted.filter((p) => cropKeys.has(p.region.id)), (region) => objectKey({
              studentId: page.student_id,
              paperId: page.paper_id,
              kind: "cropmask",
              name: `${runId}-${region.id}`,
              extension: "webp",
              unguessable: false,
            }));
          } else {
            console.error("crop: mask does not match page dimensions", pageId, `${mask.width}x${mask.height}`);
          }
        }
      } catch (cause) {
        console.error("crop: mask pass failed, crops kept", pageId, String(cause));
      }
    }

    // ── one write ────────────────────────────────────────────────────────────
    const rows = [...cropKeys].map(([id, crop_key]) => ({
      id,
      crop_key,
      cropmask_key: maskKeys.get(id) ?? null,
    }));
    if (rows.length) {
      const { error } = await sb.rpc("apply_region_crops", { p_run_id: runId, p_rows: rows });
      // A failed write means the crops are in R2 and nothing points at them.
      // Thrown rather than logged: this is the one step worth a retry, and the
      // deterministic keys above make the retry overwrite rather than duplicate.
      if (error) throw new Error("apply_region_crops failed: " + error.message);
    }

    return await finish(env, sb, runId, pageId, rows.length ? "done" : "failed", {
      regions: rows.length,
      planned: budgeted.length,
      with_mask: maskKeys.size,
      multi_page: multiPage,
      over_budget: Math.max(0, plans.length - CROP_BUDGET),
    });
  },
  async ({ env, sb, msg }, error) => {
    // The whole point of the stage's design: even total failure just means
    // content reads full pages, which is what it does today.
    console.error("mastery-crop permanent failure", msg.page_id, String((error as any)?.stack ?? error));
    if (msg.page_id) {
      await finish(env, sb, msg.run_id, msg.page_id, "failed", { failed: true });
    }
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, CropMessage>;
