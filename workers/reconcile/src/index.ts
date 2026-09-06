import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { reconcile, type QuestionMarks } from "@mastery/shared/reconcile.js";
import { assess, numberingSoundness, downgradeRecognition, type Recognition } from "@mastery/shared/confidence.js";
import { checkAnswer } from "@mastery/shared/arithmetic.js";
import { checkLabels } from "@mastery/shared/labels.js";
import { readAnswerBlock, checkableText } from "@mastery/shared/answer_block.js";
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
      .select("id, order_index, question_label, marks_awarded, marks_available, confidence_tier, confidence_signals, extract_status, page_spans, student_answer, answer_block")
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

    // The label set, checked structurally rather than trusted. Two regions
    // claiming the same part means the marks on at least one of them are
    // attached to the wrong question — decidable without a model, and a
    // condition the paper must not be committed under. `2a` and `2. a)` are the
    // same part, which is why this compares canonical forms: production holds
    // both spellings for one question, so a string comparison sees no clash.
    const labelCheck = checkLabels(regions.map((r: any) => r.question_label));
    if (!labelCheck.ok) {
      console.info("duplicate question labels on this run", runId,
        labelCheck.problems.filter((p) => p.kind === "duplicate").map((p) => p.label).join(","));
    }

    // Arithmetic, evaluated. This used to be `arithmeticOk: true`, hardcoded —
    // the signal never looked at a single character of the student's working,
    // while other runs wrote `false` onto byte-identical text. Now each chain is
    // parsed and evaluated over exact rationals, and the verdict is the same
    // every time because it is a computation rather than an opinion.
    const arithmetic = regions.map((r: any) => {
      const block = readAnswerBlock(r.answer_block, r.student_answer);
      const verdict = checkAnswer(checkableText(block, r.student_answer));
      if (verdict.kind === "consistent") return true as const;
      if (verdict.kind === "inconsistent") return false as const;
      return "unknown" as const;
    });

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
      // this region is *why*. reconcile() ranks which region(s) the discrepancy
      // is attributable to, and mastery-adjudicate confirms and applies that
      // before the student reaches the review screen. So a clean question is
      // still not punished for a bad total elsewhere on the paper.
      //
      // What HAS changed is that `arithmetic` now means this region's own
      // working, evaluated — not a hardcoded true. An inconsistent chain makes
      // the region unsure and routes it back to its crop; it never touches a
      // mark, because which of the student and the transcription is wrong is
      // not knowable from the text. Production holds handwritten `8/2` stored
      // as `8+1`, which turns a correct step into a false one.
      const { tier, signals } = assess({
        recognition,
        // A duplicated label is a structural failure of the whole set, so every
        // region on the run carries it: the marks may be on the wrong question
        // and there is no way to tell which one from here.
        numberingSound: (sound[i] ?? false) && labelCheck.ok,
        arithmeticOk: arithmetic[i],
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
