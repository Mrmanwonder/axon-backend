import { callModel } from "@mastery/shared/openrouter.js";
import { consumeQueue } from "@mastery/shared/worker.js";
import { imageRef } from "@mastery/shared/r2.js";
import { SYSTEM, instruction, SCHEMA, validate } from "@mastery/shared/prompts/adjudicate.v1.js";
import type { Env } from "@mastery/shared/env.js";

interface AdjudicateMessage {
  run_id: string;
  _retries?: number;
}

const CROPS = 6;
const RANK: Record<string, number> = { unreadable: 0, unsure: 1, confident: 2 };

const handler = consumeQueue<AdjudicateMessage>(
  async ({ env, sb, msg, attempt, beat }) => {
    const runId = msg.run_id;
    const { data: run } = await sb
      .from("extraction_run")
      .select("id, paper_id, student_id, status, reconcile_delta, route_override")
      .eq("id", runId)
      .single();
    if (!run) return { detail: { skipped: "no such run" } };
    if (run.status !== "adjudicating") return { detail: { skipped: run.status } };
    const override = run.route_override;
    await beat();

    const { data: paper } = await sb.from("paper").select("reported_total, total_awarded").eq("id", run.paper_id).single();
    const { data: regions } = await sb
      .from("question_region")
      .select("id, order_index, question_label, marks_awarded, marks_available, confidence_tier, confidence_signals, crop_key, page_spans")
      .eq("run_id", runId)
      .order("order_index");
    if (!regions?.length) {
      await sb.rpc("run_advance", { p_run_id: runId, p_to: "needs_review" });
      return { detail: { skipped: "nothing to adjudicate" } };
    }

    const suspects = [...regions].sort((a: any, b: any) => (RANK[a.confidence_tier] ?? 3) - (RANK[b.confidence_tier] ?? 3)).slice(0, CROPS);
    const [suspectImages, coverPageRow] = await Promise.all([
      Promise.all(suspects.filter((r: any) => r.crop_key).map((r: any) => imageRef(env, "derived", r.crop_key, "high"))),
      sb.from("paper_page").select("r2_bucket, r2_key").eq("paper_id", run.paper_id).order("page_number").limit(1).maybeSingle(),
    ]);
    const images = [...suspectImages];
    const coverPage = coverPageRow.data;
    if (coverPage?.r2_key) {
      images.push(await imageRef(env, coverPage.r2_bucket ?? "derived", coverPage.r2_key, "high"));
    }

    const { parsed } = await callModel({
      env,
      sb,
      stage: "adjudicate",
      system: SYSTEM,
      instruction: instruction({
        reportedTotal: paper?.reported_total === null || paper?.reported_total === undefined ? null : Number(paper.reported_total),
        computedTotal: Number(paper?.total_awarded ?? 0),
        delta: Number(run.reconcile_delta ?? 0),
        regions: regions.map((r: any) => ({
          order_index: r.order_index,
          label: r.question_label,
          marks_awarded: r.marks_awarded === null ? null : Number(r.marks_awarded),
          marks_available: r.marks_available === null ? null : Number(r.marks_available),
          confidence_tier: r.confidence_tier,
        })),
      }),
      images,
      schema: SCHEMA,
      validate,
      runId,
      paperId: run.paper_id,
      studentId: run.student_id,
      attempt,
      routeOverride: override,
    });

    const byIndex = new Map(regions.map((r: any) => [r.order_index, r]));
    let flagged = 0;
    for (const correction of parsed.corrections) {
      const region = byIndex.get(correction.order_index) as any;
      if (!region) continue;
      await sb
        .from("question_region")
        .update({
          confidence_tier: "unsure",
          needs_review: true,
          confidence_signals: {
            ...(region.confidence_signals ?? {}),
            adjudication: { field: correction.field, suggests: correction.corrected_value, evidence: correction.evidence },
          },
          updated_at: new Date().toISOString(),
        })
        .eq("id", region.id);
      flagged += 1;
    }

    await sb.from("extraction_run").update({ adjudication: { cause: parsed.cause, checked: parsed.checked, corrections: parsed.corrections } }).eq("id", runId);

    const reason = parsed.corrections.length
      ? "The marks do not quite add up. We have put the questions to check first."
      : "The marks on this paper do not add up to the total written on it. We could not see why, so nothing was changed.";
    await sb.rpc("run_advance", { p_run_id: runId, p_to: "needs_review", p_reason: reason });

    return { detail: { cause: parsed.cause, flagged } };
  },
  async ({ sb, msg }) => {
    await sb.rpc("run_advance", {
      p_run_id: msg.run_id,
      p_to: "needs_review",
      p_reason: "The marks on this paper do not add up to the total written on it. Check the questions below.",
    });
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, AdjudicateMessage>;
