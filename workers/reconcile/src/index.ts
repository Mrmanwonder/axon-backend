import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { reconcile, type QuestionMarks } from "@mastery/shared/reconcile.js";
import { assess, numberingSoundness } from "@mastery/shared/confidence.js";
import type { Env } from "@mastery/shared/env.js";

interface ReconcileMessage {
  run_id: string;
  _retries?: number;
}

// NOTE: this worker has no SELF_QUEUE binding on the live deployment, so a
// retryable failure here falls through to the queue's native message.retry()
// instead of shared/worker.ts's manual re-enqueue path — see
// AXON_FIX_BRIEF.md §4.D2. §9.2 adds the binding; not changed in this port.
//
// NOTE: the confidence update below is one UPDATE per region inside a single
// Promise.all, in one invocation — the same shape of subrequest-ceiling bug
// that took down mastery-content before batch_size was capped at 1 (see
// AXON_FIX_BRIEF.md §3.3). It is fine at today's ≤7-question papers and will
// not be at ~35+. AXON_FIX_BRIEF.md §9.1 replaces this with a single RPC;
// not changed in this port, which reconstructs current live behaviour.
const handler = consumeQueue<ReconcileMessage>(
  async ({ env, sb, msg }) => {
    const runId = msg.run_id;
    const { data: run } = await sb.from("extraction_run").select("id, paper_id, student_id, status").eq("id", runId).single();
    if (!run) return { detail: { skipped: "no such run" } };
    if (["failed", "rejected", "committed", "needs_review", "ready"].includes(run.status)) {
      return { detail: { skipped: run.status } };
    }

    await sb.rpc("run_advance", { p_run_id: runId, p_to: "reconciliation" });

    const { data: paper } = await sb.from("paper").select("reported_total, stated_maximum").eq("id", run.paper_id).single();
    const { data: regions } = await sb
      .from("question_region")
      .select("id, order_index, question_label, marks_awarded, marks_available, confidence_tier, confidence_signals, extract_status")
      .eq("run_id", runId)
      .order("order_index");
    if (!regions?.length) {
      await failRun(sb, runId, "We could not find any questions on this paper. Try scanning it again in better light.");
      return { detail: { failed: "no questions" } };
    }

    const marks: QuestionMarks[] = regions.map((r: any) => ({
      order_index: r.order_index,
      label: r.question_label,
      awarded: r.marks_awarded === null ? null : Number(r.marks_awarded),
      available: r.marks_available === null ? null : Number(r.marks_available),
      recognition: r.confidence_tier === "unreadable" ? "low" : "medium",
    }));

    const result = reconcile(
      marks,
      paper?.reported_total === null || paper?.reported_total === undefined ? null : Number(paper.reported_total),
      paper?.stated_maximum === null || paper?.stated_maximum === undefined ? null : Number(paper.stated_maximum)
    );

    const sound = numberingSoundness(marks.map((m) => m.label));
    const { count: fallbackPages } = await sb
      .from("paper_page")
      .select("id", { count: "exact", head: true })
      .eq("paper_id", run.paper_id)
      .not("layer_fallback", "is", null);

    await Promise.all(
      regions.map((region: any, i: number) => {
        const { tier, signals } = assess({
          recognition: marks[i].recognition,
          numberingSound: sound[i] ?? false,
          paperReconciled: result.reconciled,
          awarded: marks[i].awarded,
          available: marks[i].available,
          layerFallback: (fallbackPages ?? 0) > 0,
          unreadable: region.confidence_tier === "unreadable" || region.extract_status === "failed",
        });
        return sb
          .from("question_region")
          .update({
            confidence_tier: tier,
            confidence_signals: { ...(region.confidence_signals ?? {}), ...signals },
            needs_review: true,
            updated_at: new Date().toISOString(),
          })
          .eq("id", region.id);
      })
    );

    await sb.from("extraction_run").update({ reconciled: result.reconciled, reconcile_delta: result.delta }).eq("id", runId);
    await sb
      .from("paper")
      .update({ total_awarded: result.sum_awarded, total_available: result.sum_available || null, reconciled: result.reconciled })
      .eq("id", run.paper_id);

    if (!result.reconciled) {
      await sb.rpc("run_advance", { p_run_id: runId, p_to: "adjudicating" });
      if (env.ADJUDICATE_QUEUE) await env.ADJUDICATE_QUEUE.send({ run_id: runId });
      return { detail: { reconciled: false, delta: result.delta } };
    }

    await sb.rpc("run_advance", { p_run_id: runId, p_to: "needs_review", p_reason: result.message });
    return { detail: { reconciled: true, questions: regions.length } };
  },
  async ({ sb, msg }) => {
    const runId = msg.run_id;
    let pagesStored = false;
    if (runId) {
      const { data: run } = await sb.from("extraction_run").select("paper_id").eq("id", runId).maybeSingle();
      if (run?.paper_id) {
        const { count } = await sb.from("paper_page").select("id", { count: "exact", head: true }).eq("paper_id", run.paper_id).not("r2_key", "is", null);
        pagesStored = (count ?? 0) > 0;
      }
    }
    await failRun(
      sb,
      runId,
      pagesStored
        ? "We couldn't finish checking this paper's marks just now — your pages are kept, and you can try again."
        : "We could not find the pages for this paper. Try scanning it again."
    );
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, ReconcileMessage>;
