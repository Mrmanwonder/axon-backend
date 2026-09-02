import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { reconcile, type QuestionMarks } from "@mastery/shared/reconcile.js";
import { assess, numberingSoundness, downgradeRecognition, type Recognition } from "@mastery/shared/confidence.js";
import type { Env } from "@mastery/shared/env.js";

interface ReconcileMessage {
  run_id: string;
  _retries?: number;
}

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
      .select("id, order_index, question_label, marks_awarded, marks_available, confidence_tier, confidence_signals, extract_status, page_spans")
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

    // Which specific pages used the non-red-ink / student-wrote-red fallback —
    // not just whether the paper has any (AXON_FIX_BRIEF.md §6.3). A region
    // whose own pages never touched that fallback has no reason to be
    // downgraded for a page elsewhere in the booklet.
    const { data: fallbackPages } = await sb
      .from("paper_page")
      .select("page_number")
      .eq("paper_id", run.paper_id)
      .not("layer_fallback", "is", null);
    const fallbackPageNumbers = new Set((fallbackPages ?? []).map((p: any) => p.page_number));

    // One RPC for every region on the run, not one UPDATE per region — a
    // ~35+ question paper would otherwise hit the same subrequest ceiling
    // that took down mastery-content before batch_size was capped at 1 (see
    // AXON_FIX_BRIEF.md §3.3, §9.1). apply_region_confidence() does the same
    // update in a single statement, tested against a synthetic 60-question
    // batch before this shipped.
    const confidenceRows = regions.map((region: any, i: number) => {
      const spans: Array<{ page: number }> = region.page_spans ?? [];
      const touchesFallbackPage = spans.some((s) => fallbackPageNumbers.has(s.page));
      const recognition: Recognition = touchesFallbackPage
        ? downgradeRecognition(marks[i].recognition)
        : marks[i].recognition;

      // The paper's totals not adding up is not this region's problem unless
      // this region is *why*. reconcile() already ranks which region(s) the
      // discrepancy is most likely attributable to; mastery-adjudicate is what
      // actually confirms and applies that (setting confidence_tier to
      // 'unsure' on the specific regions its corrections name — see
      // workers/adjudicate/src/index.ts) once it runs, which the run's status
      // machine guarantees happens *before* the student ever reaches the
      // review screen on an unreconciled paper (the run sits in
      // 'adjudicating' until then). So this pass never fails the arithmetic
      // signal itself — a clean question is not punished for a bad total
      // reconcile can't yet attribute to it. See AXON_FIX_BRIEF.md §6.3.
      const { tier, signals } = assess({
        recognition,
        numberingSound: sound[i] ?? false,
        arithmeticOk: true,
        awarded: marks[i].awarded,
        available: marks[i].available,
        unreadable: region.confidence_tier === "unreadable" || region.extract_status === "failed",
      });
      return { id: region.id, tier, signals, needs_review: true };
    });
    const { error: confidenceError } = await sb.rpc("apply_region_confidence", { p_rows: confidenceRows });
    if (confidenceError) throw new Error("apply_region_confidence failed: " + confidenceError.message);

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
