import { callModel } from "@mastery/shared/openrouter.js";
import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { imageRef } from "@mastery/shared/r2.js";
import { SYSTEM, instruction, SCHEMA, validate, REJECTION_REASON, qualityFailureMessage, type QualitySignals } from "@mastery/shared/prompts/triage.v1.js";
import type { Env } from "@mastery/shared/env.js";

const PAGES_TO_LOOK_AT = 6;

interface TriageMessage {
  run_id: string;
  _retries?: number;
}

interface Page {
  page_number: number;
  r2_bucket: string | null;
  r2_key: string;
  thumb_key: string | null;
  quality_verdict: string | null;
  quality_signals: QualitySignals | null;
}

const handler = consumeQueue<TriageMessage>(
  async ({ env, sb, msg, attempt, beat }) => {
    const runId = msg.run_id;
    const { data: run } = await sb
      .from("extraction_run")
      .select("id, paper_id, student_id, status, route_override")
      .eq("id", runId)
      .single();
    if (!run) return { detail: { skipped: "no such run" } };
    // Accepts both "queued" and "triaging": a retryable error used to advance
    // the run to "triaging" before the model call, so a retry after that
    // point permanently found the run in the wrong status and was skipped
    // forever. See AXON_FIX_BRIEF.md §3.1.
    if (!["queued", "triaging"].includes(run.status)) return { detail: { skipped: run.status } };
    const override = run.route_override;

    const { data: pages } = await sb
      .from("paper_page")
      .select("page_number, r2_bucket, r2_key, thumb_key, quality_verdict, quality_signals")
      .eq("paper_id", run.paper_id)
      .not("r2_key", "is", null)
      .order("page_number")
      .limit(PAGES_TO_LOOK_AT);
    if (!pages?.length) {
      await failRun(sb, runId, "We could not find the pages for this paper. Try scanning it again.");
      return { detail: { failed: "no pages" } };
    }
    if (pages.every((p: Page) => p.quality_verdict === "fail")) {
      const message = qualityFailureMessage(pages) ?? "These pages did not come out clearly enough to read. Please retake them and try again.";
      await sb.rpc("run_advance", { p_run_id: runId, p_to: "rejected", p_reason: message });
      return { detail: { rejected: "quality", pages: pages.length } };
    }

    await sb.rpc("run_advance", { p_run_id: runId, p_to: "triaging" });
    await beat();

    // The thumbnail, where there is one. The question at this stage is "is this
    // a marked exam paper", not "what does it say" — the code comment had said
    // so for months while the stage went on sending whole 2400px pages, at
    // 27-43 seconds for a two-page paper (AXON_FIX_BRIEF.md §7.5 / B9). A 512px
    // copy settles the same question, and the payload is what the latency was.
    //
    // The full page is still the fallback, not an error: every page written
    // before the client started producing thumbnails has `thumb_key` null, and
    // triage refusing to read them would turn a latency fix into an outage for
    // the existing library. A thumbnail always lives in `derived` regardless of
    // where the page itself went.
    const images = await Promise.all(
      pages.map((p: Page) => (p.thumb_key
        ? imageRef(env, "derived", p.thumb_key, "low")
        : imageRef(env, (p.r2_bucket as any) ?? "derived", p.r2_key, "low")))
    );
    const onThumbs = pages.filter((p: Page) => !!p.thumb_key).length;

    const { parsed } = await callModel({
      env,
      sb,
      stage: "triage",
      system: SYSTEM,
      instruction: instruction(pages.length),
      images,
      schema: SCHEMA,
      validate,
      runId,
      paperId: run.paper_id,
      studentId: run.student_id,
      attempt,
      routeOverride: override,
    });

    if (parsed.classification !== "graded_exam") {
      const isUncertainReject = parsed.classification === "not_schoolwork" && parsed.confidence === "low";
      const reason = (isUncertainReject && qualityFailureMessage(pages)) || REJECTION_REASON[parsed.classification];
      await sb.rpc("run_advance", { p_run_id: runId, p_to: "rejected", p_reason: reason });
      return { detail: { rejected: parsed.classification } };
    }

    await sb
      .from("extraction_run")
      .update({
        tier_routing: { triage: parsed },
        ...(parsed.ink_colour !== "red" ? { status_reason: null } : {}),
      })
      .eq("id", runId);
    if (parsed.ink_colour !== "red") {
      await sb.from("paper_page").update({ layer_fallback: "non_red_marking" }).eq("paper_id", run.paper_id).is("layer_fallback", null);
    }

    await sb.rpc("run_advance", { p_run_id: runId, p_to: "structure" });
    // Reset every page's structure_status to pending before fan-out — a
    // second run over the same paper otherwise finds every page already
    // "done" from the first run and skips them all. See AXON_FIX_BRIEF.md §3.2.
    // `crop_status` goes with it for exactly the same reason: the crop stage
    // has the same "already done, advance and skip" re-entry path, so leaving
    // it terminal would make a second run's cropping a no-op that reused the
    // first run's crops — cut against the first run's boxes.
    await sb.from("paper_page").update({ structure_status: "pending", crop_status: "pending" }).eq("paper_id", run.paper_id);
    const { data: allPages } = await sb.from("paper_page").select("id").eq("paper_id", run.paper_id).not("r2_key", "is", null);
    if (env.STRUCTURE_QUEUE && allPages?.length) {
      await env.STRUCTURE_QUEUE.sendBatch(allPages.map((page: { id: string }) => ({ body: { run_id: runId, page_id: page.id } })));
    }

    // Recorded so §7.7's "triage latency drops to single-digit seconds" can be
    // read straight off the data: `model_call.latency_ms` alone cannot say
    // whether a given call was on thumbnails or on full pages.
    return {
      detail: {
        classification: parsed.classification,
        pages: allPages?.length ?? 0,
        looked_at: pages.length,
        on_thumbnails: onThumbs,
      },
    };
  },
  async ({ sb, msg }, error) => {
    const runId = msg.run_id;
    let pagesStored = false;
    if (runId) {
      const { data: run } = await sb.from("extraction_run").select("paper_id").eq("id", runId).maybeSingle();
      if (run?.paper_id) {
        const { count } = await sb.from("paper_page").select("id", { count: "exact", head: true }).eq("paper_id", run.paper_id).not("r2_key", "is", null);
        pagesStored = (count ?? 0) > 0;
      }
    }
    console.error("mastery-triage permanent failure", runId, String((error as any)?.stack ?? error));
    await failRun(sb, runId, "We ran into a problem reading this paper. Nothing was lost — please try again.");
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, TriageMessage>;
