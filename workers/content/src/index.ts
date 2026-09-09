import { callModel } from "@mastery/shared/openrouter.js";
import { consumeQueue } from "@mastery/shared/worker.js";
import { imageRef } from "@mastery/shared/r2.js";
import { pageDimensions } from "@mastery/shared/page.js";
import { mapModelBoxToPage, frameForIndex, type ModelFrame } from "@mastery/shared/frames.js";
import { bandForRegion } from "@mastery/shared/crop.js";
import { mustData, mustMaybe } from "@mastery/shared/db.js";
import { SYSTEM, instruction, SCHEMA, validate } from "@mastery/shared/prompts/content.v1.js";
import type { Env } from "@mastery/shared/env.js";

interface ContentMessage {
  run_id: string;
  region_id: string;
  _retries?: number;
}

async function advanceAndEnqueue(env: Env, sb: any, runId: string): Promise<void> {
  const { data: advance } = await sb.rpc("advance_after_content", { p_run_id: runId });
  if (advance?.advanced && advance?.enqueue_reconcile && env.RECONCILE_QUEUE) {
    await env.RECONCILE_QUEUE.send({ run_id: runId });
  }
}

/**
 * A model field, mapped back onto the page it was actually read from.
 *
 * This used to take bare width/height and call `takeBox`, which is the
 * transform for a box drawn on a whole page. Two things were wrong with that:
 *
 *   · when the image was a CROP, the model's coordinates are in the crop's own
 *     grid, so scaling them by the page's height put the box hundreds of
 *     pixels from the ink it describes;
 *   · `page_index` was clamped with Math.min, so a field the model located on
 *     an image we did not send was silently recorded on the last page we did —
 *     and every field's dimensions came from the FIRST page regardless, so a
 *     booklet with differently sized pages mis-scaled the later ones.
 *
 * Both disappear once each image carries a frame. An unresolvable index or an
 * ungroundable box yields null, which routes the region to review rather than
 * storing provenance that points at the wrong pixels.
 */
function field(
  v: { value: unknown; box: unknown; page_index?: number } | null | undefined,
  frames: ModelFrame[],
) {
  if (!v || v.value === null || v.value === undefined) return { value: null, box: null };
  const frame = frameForIndex(frames, v.page_index);
  if (!frame) return { value: null, box: null };
  const box = mapModelBoxToPage(frame, v.box);
  return box ? { value: v.value, box } : { value: null, box: null };
}

const handler = consumeQueue<ContentMessage>(
  async ({ env, sb, msg, attempt, beat }) => {
    const runId = msg.run_id;
    const regionId = msg.region_id;
    const { data: region } = await sb
      .from("question_region")
      .select("id, paper_id, student_id, order_index, question_label, question_label_box, page_spans, extract_status, crop_key, cropmask_key")
      .eq("id", regionId)
      .single();
    if (!region) return { detail: { skipped: "no such question" } };

    if (region.extract_status === "done") {
      await advanceAndEnqueue(env, sb, runId);
      return { detail: { skipped: "already read" } };
    }

    const { data: run } = await sb.from("extraction_run").select("status, route_override").eq("id", runId).single();
    if (!run || ["failed", "rejected", "committed"].includes(run.status)) {
      return { detail: { skipped: run?.status ?? "no run" } };
    }
    const override = run.route_override;

    await sb.from("question_region").update({ extract_status: "running" }).eq("id", regionId);
    await beat();

    const spans: Array<{ page: number }> = region.page_spans ?? [];
    if (!spans.length) throw new Error("a question with no page span has nothing to read");

    // Every page this question touches, with the metadata needed to establish
    // its own dimensions. Fetched BEFORE the images because a frame cannot be
    // built without them, and an image sent without a frame is a box we cannot
    // map back — see @mastery/shared/frames.ts.
    const pageRows = await mustData(
      sb.from("paper_page")
        .select("page_number, r2_bucket, r2_key, mask_key, layer_fallback, conditioning_meta, quality_signals")
        .eq("paper_id", region.paper_id)
        .in("page_number", spans.map((s) => s.page))
        .order("page_number"),
      "paper_page read",
    ) as any[];
    const firstPage = pageRows.find((p) => p.page_number === spans[0].page) ?? null;

    // Prefer a pre-cut crop (§8 of AXON_FIX_BRIEF.md) over sending the whole
    // page. A crop is only used where its band can be recomputed exactly —
    // `bandForRegion` is the same function the crop worker cut with, so the
    // geometry is derived rather than remembered. Without the band there is no
    // way to map the model's coordinates back, and a crop whose provenance we
    // cannot express is worse than a full page: it is cheaper and wrong.
    const firstDims = firstPage ? pageDimensions(firstPage as any) : null;
    const cropBand = region.crop_key && firstDims
      ? bandForRegion(spans as any, spans[0].page, firstDims.width, firstDims.height)
      : null;

    const images: Array<NonNullable<Awaited<ReturnType<typeof imageRef>>>> = [];
    const frames: ModelFrame[] = [];

    if (region.crop_key && cropBand && firstDims) {
      const crop = await imageRef(env, "derived", region.crop_key, "high");
      if (crop) {
        images.push(crop);
        frames.push({
          kind: "crop",
          pageNumber: spans[0].page,
          pageWidth: firstDims.width,
          pageHeight: firstDims.height,
          band: cropBand,
        });
      }
      if (region.cropmask_key) {
        const mask = await imageRef(env, "derived", region.cropmask_key, "low");
        if (mask) {
          images.push(mask);
          frames.push({
            kind: "cropmask",
            pageNumber: spans[0].page,
            pageWidth: firstDims.width,
            pageHeight: firstDims.height,
            band: cropBand,
          });
        }
      }
    }

    if (!images.length) {
      // Full pages, each with its own dimensions. A page whose size cannot be
      // established is left out entirely rather than sent with the first
      // page's shape — every box it produced would be wrong by an unknown
      // amount, which is worse than one fewer image (CLAUDE.md rule 4).
      for (const p of pageRows) {
        if (!p.r2_key) continue;
        const dims = pageDimensions(p as any);
        if (!dims) {
          console.info("page excluded from the model: no dimensions", p.page_number);
          continue;
        }
        const img = await imageRef(env, p.r2_bucket ?? "derived", p.r2_key, "high");
        if (!img) continue;
        images.push(img);
        frames.push({
          kind: "page",
          pageNumber: p.page_number,
          pageWidth: dims.width,
          pageHeight: dims.height,
        });
      }
    }

    if (!images.length) throw new Error("nothing to look at for this question");

    const marks = await mustData(
      sb.from("teacher_mark").select("shape, mark_class, page_number").eq("region_id", regionId),
      "teacher_mark read",
    ) as any[];

    const { parsed } = await callModel({
      env,
      sb,
      stage: "content",
      system: SYSTEM,
      instruction: instruction({
        label: region.question_label,
        pageNumbers: spans.map((s) => s.page),
        layerFallback: firstPage?.layer_fallback ?? null,
        teacherMarks: marks.map((m: any) => ({ shape: m.shape, where: `on page ${m.page_number}` })),
      }),
      images,
      schema: SCHEMA,
      validate,
      runId,
      paperId: region.paper_id,
      regionId,
      studentId: region.student_id,
      attempt,
      routeOverride: override,
    });

    try {
      // See @mastery/shared/page.ts for why there is no default here. A region
      // on a page whose size cannot be established is flagged for review rather
      // than filled in against a guessed page shape — every box it produced
      // would be wrong by an unknown amount, which is worse than an admitted
      // gap (CLAUDE.md rule 4).
      //
      // The condition is now "no frames" rather than "no dimensions on the
      // first page": a frame is only built where its page's size is known, so
      // an empty list is exactly the case where nothing the model returned can
      // be placed anywhere.
      if (!frames.length) {
        await sb
          .from("question_region")
          .update({
            extract_status: "done",
            confidence_tier: "unreadable",
            needs_review: true,
            confidence_signals: { unreadable_reason: "We could not work out the size of this page, so we cannot say where anything on it sits." },
            updated_at: new Date().toISOString(),
          })
          .eq("id", regionId);
        await advanceAndEnqueue(env, sb, runId);
        return { detail: { unreadable: "no page dimensions" } };
      }

      const label = field(parsed.question_label as any, frames);
      const question = field(parsed.question_text, frames);
      const answer = field(parsed.student_answer, frames);
      const awarded = field(parsed.marks_awarded, frames);
      const available = field(parsed.marks_available, frames);
      const remark = field(parsed.teacher_remark, frames);

      if (parsed.unreadable) {
        await sb
          .from("question_region")
          .update({
            extract_status: "done",
            confidence_tier: "unreadable",
            needs_review: true,
            confidence_signals: { unreadable_reason: parsed.unreadable_reason },
            updated_at: new Date().toISOString(),
          })
          .eq("id", regionId);
        await advanceAndEnqueue(env, sb, runId);
        return { detail: { unreadable: parsed.unreadable_reason } };
      }

      const { error: updErr } = await sb
        .from("question_region")
        .update({
          question_label: label.value ?? region.question_label,
          question_label_box: label.box ?? region.question_label_box,
          question_text: question.value,
          question_text_box: question.box,
          student_answer: answer.value,
          student_answer_box: answer.box,
          marks_awarded: awarded.value,
          marks_awarded_box: awarded.box,
          marks_available: available.value,
          marks_available_box: available.box,
          teacher_remark: remark.value,
          teacher_remark_box: remark.box,
          answer_block: (parsed as any).answer_block ?? null,
          // region_type is null on 28 of 76 live regions, so over a third of
          // them do not know whether they are maths or prose and nothing
          // downstream can decide how to typeset them. `unknown` is a value;
          // null is an absence that reads as "nobody asked".
          region_type: parsed.region_type ?? null,
          extract_status: "done",
          updated_at: new Date().toISOString(),
        })
        .eq("id", regionId);
      if (updErr) throw new Error("question_region update failed: " + updErr.message + " | " + JSON.stringify(updErr));

      if (awarded.value !== null && awarded.box) {
        const { error: tmErr } = await sb
          .from("teacher_mark")
          .update({ value: awarded.value })
          .eq("region_id", regionId)
          .eq("mark_class", "marginal_number")
          .is("value", null);
        if (tmErr) throw new Error("teacher_mark update failed: " + tmErr.message + " | " + JSON.stringify(tmErr));
      }

      await advanceAndEnqueue(env, sb, runId);
      return { detail: { awarded: awarded.value, available: available.value } };
    } catch (e) {
      console.error("mastery-content post-model error", regionId, String((e as any)?.stack ?? e));
      throw e;
    }
  },
  async ({ env, sb, msg }) => {
    const regionId = msg.region_id;
    const { data: region } = await sb.from("question_region").select("confidence_signals").eq("id", regionId).maybeSingle();
    await sb
      .from("question_region")
      .update({
        extract_status: "failed",
        confidence_tier: "unreadable",
        needs_review: true,
        confidence_signals: {
          ...(region?.confidence_signals ?? {}),
          unreadable_reason: "We could not finish reading this question. It has been flagged for review.",
        },
        updated_at: new Date().toISOString(),
      })
      .eq("id", regionId);
    await advanceAndEnqueue(env, sb, msg.run_id);
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, ContentMessage>;
