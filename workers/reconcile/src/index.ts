import { pipelineWrite } from "@mastery/shared/pipeline_write.js";
import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { QUEUE_TUNING } from "@mastery/shared/queue_tuning.js";
import { mustOk, mustData, mustMaybe, mustRpc } from "@mastery/shared/db.js";
import { failureCodeFor } from "@mastery/shared/failure_codes.js";
import { adjudicationTriggers, reconcile, type QuestionMarks } from "@mastery/shared/reconcile.js";
import { paperShownUnmarked } from "@mastery/shared/confidence.js";
import { checkAnswer } from "@mastery/shared/arithmetic.js";
import { checkLabels } from "@mastery/shared/labels.js";
import { byReadingOrder, placementInput } from "@mastery/shared/placement.js";
import { judgeRegions, recognitionFor } from "@mastery/shared/review_rule.js";
import { readAnswerBlock, checkableText } from "@mastery/shared/answer_block.js";
import type { Env } from "@mastery/shared/env.js";

export { recognitionFor };

interface ReconcileMessage {
  run_id: string;
  _retries?: number;
}

const handler = consumeQueue<ReconcileMessage>(
  async ({ env, sb, msg }) => {
    const runId = msg.run_id;
    const run = await mustMaybe<any>(sb.from("extraction_run").select("id, paper_id, student_id, status, tier_routing").eq("id", runId).maybeSingle(), "reconcile run read");
    if (!run) return { detail: { skipped: "no such run" } };
    if (["failed", "rejected", "committed", "needs_review", "ready"].includes(run.status)) {
      return { detail: { skipped: run.status } };
    }

    if (run.status === "adjudicating") {
      if (!env.ADJUDICATE_QUEUE) throw new Error("Adjudication queue is not configured");
      await env.ADJUDICATE_QUEUE.send({ run_id: runId });
      return { detail: { resumed: "adjudication dispatch" } };
    }
    if (!await pipelineWrite(sb, runId, "reconcile_start")) return { detail: { skipped: "stale reconciliation" } };

    const paper = await mustMaybe<any>(sb.from("paper").select("reported_total, stated_maximum").eq("id", run.paper_id).maybeSingle(), "reconcile paper read");
    const regions = await mustData<any[]>(sb
      .from("question_region")
      .select("id, order_index, question_label, marks_awarded, marks_available, confidence_tier, confidence_signals, extract_status, page_spans, student_answer, question_text, answer_block")
      .eq("run_id", runId)
      .order("order_index"), "reconcile regions read");
    if (!regions?.length) {
      await failRun(sb, runId, "We could not find any questions on this paper. Try scanning it again in better light.", "reconcile_no_questions");
      return { detail: { failed: "no questions" } };
    }

    const marks: QuestionMarks[] = regions.map((r: any) => ({
      order_index: r.order_index,
      label: r.question_label,
      awarded: r.marks_awarded === null ? null : Number(r.marks_awarded),
      available: r.marks_available === null ? null : Number(r.marks_available),
      recognition: r.confidence_tier === "unreadable" ? "low" : recognitionFor(r.confidence_signals?.recognition_confidence),
    }));

    const result = reconcile(
      marks,
      paper?.reported_total === null || paper?.reported_total === undefined ? null : Number(paper.reported_total),
      paper?.stated_maximum === null || paper?.stated_maximum === undefined ? null : Number(paper.stated_maximum)
    );

    // The label set, checked structurally rather than trusted. Two regions
    // claiming the same part means the marks on at least one of them are
    // attached to the wrong question — decidable without a model. `2a` and
    // `2. a)` are the same part, which is why this compares canonical forms.
    // Read in page order (council D1): stored order was the order pages
    // finished structure, which says nothing about the paper. This feeds only
    // the adjudication trigger; the per-region verdict is the placement walk's.
    const inReadingOrder = regions
      .map((r: any) => ({ ...placementInput(r), label: r.question_label as string | null }))
      .sort(byReadingOrder);
    const labelCheck = checkLabels(inReadingOrder.map((r) => r.label));
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
    const fallbackPages = await mustData<any[]>(sb
      .from("paper_page")
      .select("page_number")
      .eq("paper_id", run.paper_id)
      .not("layer_fallback", "is", null), "reconcile fallback pages read");
    const fallbackPageNumbers = new Set((fallbackPages ?? []).map((p: any) => p.page_number));

    // One RPC for every region on the run, not one UPDATE per region — a
    // ~35+ question paper would otherwise hit the same subrequest ceiling
    // that took down mastery-content before batch_size was capped at 1 (see
    // AXON_FIX_BRIEF.md §3.3, §9.1). apply_region_confidence() does the same
    // update in a single statement, tested against a synthetic 60-question
    // batch before this shipped.
    //
    // The paper's totals not adding up is not a region's problem unless that
    // region is *why*: reconcile() ranks the suspects and mastery-adjudicate
    // confirms them. `arithmetic` is the region's own working, evaluated; an
    // inconsistent chain makes the region unsure, never touches a mark, and —
    // council D1 — does not on its own ask the student.
    //
    // needs_review used to be true for every region on every paper, so review
    // was a whole screen that protected nothing. It is now the D1 ask-rule:
    // unreadable or low/null recognition, a missing or impossible teacher mark
    // on a marked paper, or a part the placement walk cannot place. The reasons
    // travel in confidence_signals.ask so the inline card can say why.
    const verdicts = judgeRegions(regions, {
      paperUnmarked: paperShownUnmarked(run.tier_routing),
      fallbackPages: fallbackPageNumbers,
      arithmetic,
    });
    const confidenceRows = regions.map((region: any, i: number) => ({
      id: region.id,
      tier: verdicts[i].tier,
      signals: { ...region.confidence_signals, ...verdicts[i].signals },
      needs_review: verdicts[i].needs_review,
    }));

    // A failed write here used to be ignored: the run moved on with a stale or missing
    // reconciliation. Throwing lets the queue retry the stage and the terminal handler classify it.
    //
    // A paper with no printed total is "unchecked" (reconciled = null) with a machine reason, never
    // "false": there is no mismatch, only nothing to compare against. Its total is the sum of the
    // teacher marks Axon read, labelled as such, and partial when a mark could not be read.
    const unchecked = result.reconciled === null;
    const triggers = adjudicationTriggers(result, marks, !labelCheck.ok);
    if (!await pipelineWrite(sb, runId, "reconcile_result", {
      confidence: confidenceRows,
      run_result: { reconciled: result.reconciled, reconcile_delta: result.delta,
        status_reason_code: result.added_up ? "no_printed_total" : null },
      paper_result: { total_awarded: result.sum_awarded, total_available: result.sum_available || null,
        reconciled: result.reconciled, total_basis: result.added_up ? "added_up" : "printed",
        total_partial: result.partial },
      to: triggers.length > 0 ? "adjudicating" : "needs_review", reason: result.message,
    })) return { detail: { skipped: "stale reconciliation result" } };

    if (triggers.length > 0) {
      if (!env.ADJUDICATE_QUEUE) throw new Error("Adjudication queue is not configured");
      await env.ADJUDICATE_QUEUE.send({ run_id: runId });
      return { detail: { reconciled: result.reconciled, delta: result.delta, adjudicate: triggers } };
    }

    return { detail: { reconciled: result.reconciled, unchecked, questions: regions.length } };
  },
  async ({ sb, msg }, error) => {
    const runId = msg.run_id;
    let pagesStored = false;
    if (runId) {
      const run = await mustMaybe<any>(sb.from("extraction_run").select("paper_id").eq("id", runId).maybeSingle(), "reconcile terminal run read");
      if (run?.paper_id) {
        const pages = await mustData<any[]>(sb.from("paper_page").select("id").eq("paper_id", run.paper_id).not("r2_key", "is", null).limit(1), "reconcile terminal pages read");
        pagesStored = pages.length > 0;
      }
    }
    await failRun(
      sb,
      runId,
      pagesStored
        ? "We couldn't finish checking this paper's marks just now — your pages are kept, and you can try again."
        : "We could not find the pages for this paper. Try scanning it again.",
      pagesStored ? failureCodeFor("reconcile", error) : "reconcile_pages_missing",
    );
  },
  { concurrency: QUEUE_TUNING.reconcile.concurrency },
);

export default { queue: handler } satisfies ExportedHandler<Env, ReconcileMessage>;
