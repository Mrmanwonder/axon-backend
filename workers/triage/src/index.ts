import { mustData, mustMaybe } from "@mastery/shared/db.js";
import { pipelineWrite } from "@mastery/shared/pipeline_write.js";
import { callModel } from "@mastery/shared/model-client.js";
import { failureCodeFor } from "@mastery/shared/failure_codes.js";
import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { imageRef } from "@mastery/shared/r2.js";
import { SYSTEM, instruction, SCHEMA, validate, REJECTION_REASON, qualityFailureMessage, type QualitySignals } from "@mastery/shared/prompts/triage.v1.js";
import { CAPTURE } from "@mastery/shared/contract.js";
import { resolveAssessmentIdentity } from "@mastery/shared/assessment.js";
import type { Env } from "@mastery/shared/env.js";
import { chunkedSendBatch } from "@mastery/shared/chunked_send.js";

const PAGES_TO_LOOK_AT = 6;
/** Verdicts that refuse a paper for having no marking (or no answers) on it. */
const UNMARKED = new Set(["ungraded_paper", "blank_paper"]);

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

/** Keep the model budget fixed while covering the whole booklet, including the last page. */
function samplePages(pages: Page[], limit = PAGES_TO_LOOK_AT): Page[] {
  if (pages.length <= limit) return pages;
  const sampled: Page[] = [];
  for (let i = 0; i < limit; i++) {
    const index = Math.round((i * (pages.length - 1)) / (limit - 1));
    sampled.push(pages[index]);
  }
  return sampled;
}

export async function enqueueStructure(env: Env, sb: any, runId: string, paperId: string): Promise<number> {
  const pages = await mustData<any[]>(sb.from("paper_page").select("id").eq("paper_id", paperId)
    .in("structure_status", ["pending","running"]).not("r2_key","is",null), "triage structure dispatch pages");
  if (pages.length) {
    if (!env.STRUCTURE_QUEUE) throw new Error("Structure queue is not configured");
    await chunkedSendBatch(env.STRUCTURE_QUEUE, pages, page => ({ body: { run_id: runId, page_id: page.id } }));
  }
  return pages.length;
}

const handler = consumeQueue<TriageMessage>(
  async ({ env, sb, msg, attempt, beat }) => {
    const runId = msg.run_id;
    const run = await mustMaybe<any>(sb.from("extraction_run")
      .select("id, paper_id, student_id, status, route_override").eq("id", runId).maybeSingle(), "triage run read");
    if (!run) return { detail: { skipped: "no such run" } };
    // Accepts both "queued" and "triaging": a retryable error used to advance
    // the run to "triaging" before the model call, so a retry after that
    // point permanently found the run in the wrong status and was skipped
    // forever. See AXON_FIX_BRIEF.md §3.1.
    if (run.status === "structure") {
      await enqueueStructure(env, sb, runId, run.paper_id);
      return { detail: { resumed: "structure dispatch" } };
    }
    if (!["queued", "triaging"].includes(run.status)) return { detail: { skipped: run.status } };
    const override = run.route_override;

    const pages = await mustData<any[]>(sb
      .from("paper_page")
      .select("page_number, r2_bucket, r2_key, thumb_key, quality_verdict, quality_signals")
      .eq("paper_id", run.paper_id)
      .not("r2_key", "is", null)
      .order("page_number")
      .limit(CAPTURE.MAX_PAGES), "triage pages read");
    if (!pages?.length) {
      await failRun(sb, runId, "We could not find the pages for this paper. Try scanning it again.", "triage_pages_missing");
      return { detail: { failed: "no pages" } };
    }
    const sampledPages = samplePages(pages as Page[]);
    if (sampledPages.every((p: Page) => p.quality_verdict === "fail")) {
      const message = qualityFailureMessage(sampledPages) ?? "These pages did not come out clearly enough to read. Please retake them and try again.";
      await pipelineWrite(sb, runId, "triage_reject", { reason: message });
      return { detail: { rejected: "quality", pages: pages.length } };
    }

    if (!await pipelineWrite(sb, runId, "triage_start")) return { detail: { skipped: "stale triage work" } };
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
      sampledPages.map((p: Page) => (p.thumb_key
        ? imageRef(env, "derived", p.thumb_key, "low")
        : imageRef(env, (p.r2_bucket as any) ?? "derived", p.r2_key, "low")))
    );
    const onThumbs = sampledPages.filter((p: Page) => !!p.thumb_key).length;

    const ask = (imgs: typeof images) => callModel({
      env,
      sb,
      stage: "triage",
      system: SYSTEM,
      instruction: instruction(sampledPages.length),
      images: imgs,
      schema: SCHEMA,
      validate,
      runId,
      paperId: run.paper_id,
      studentId: run.student_id,
      attempt,
      routeOverride: override,
    });

    let { parsed } = await ask(images);
    let secondLook = false;

    // "No marking" is the one verdict that throws a marked paper away, and the
    // first look is at 512 px thumbnails, where a small tick or a pencil mark
    // is a few pixels (owner, 5 Oct 2026: a marked 12-page past paper refused
    // as unmarked). Before refusing for want of marking, look again at the
    // pages themselves, at full detail. Same prompt, same question: only the
    // resolution changes, and only the second answer stands.
    if (UNMARKED.has(parsed.classification) && onThumbs > 0) {
      const full = await Promise.all(
        sampledPages.map((p: Page) => imageRef(env, (p.r2_bucket as any) ?? "derived", p.r2_key, "high"))
      );
      ({ parsed } = await ask(full));
      secondLook = true;
    }

    // The student told us this paper is marked, after it was refused as
    // unmarked. They are the authority on their own paper; the later stages
    // read only the marking that is actually there, and a question with no
    // readable mark is shown as unmarked, never guessed.
    const studentSaysMarked = (override as any)?.student_says_marked === true;
    if (UNMARKED.has(parsed.classification) && studentSaysMarked) {
      parsed = { ...parsed, classification: "graded_exam" };
    }

    if (parsed.classification !== "graded_exam") {
      const isUncertainReject = parsed.classification === "not_schoolwork" && parsed.confidence === "low";
      const reason = (isUncertainReject && qualityFailureMessage(sampledPages)) || REJECTION_REASON[parsed.classification];
      await pipelineWrite(sb, runId, "triage_reject", { reason });
      return { detail: { rejected: parsed.classification, second_look: secondLook } };
    }

    let resolvedAssessment = null;
    if (parsed.assessment_identity) {
      try {
        resolvedAssessment = await resolveAssessmentIdentity(sb, {
          studentId: run.student_id,
          paperId: run.paper_id,
          candidate: parsed.assessment_identity,
          persist: false,
        });
      } catch (error) {
        // Identity lookup is enrichment, never a reason to lose a scan. A
        // transient DB failure is recorded and the paper proceeds unbound;
        // exact scheme retrieval remains disabled until a later retry resolves it.
        console.warn("assessment identity resolution failed", run.paper_id, String(error));
      }
    }

    if (!await pipelineWrite(sb, runId, "triage", {
      fallback: parsed.ink_colour !== "red" ? "non_red_marking" : null,
      assessment_identity_id: resolvedAssessment?.id ?? null,
      tier_routing: { triage: parsed, assessment_identity_id: resolvedAssessment?.id ?? null,
        assessment_identity_status: resolvedAssessment ? "exact" : "unresolved" },
    })) return { detail: { skipped: "stale triage result" } };
    const pagesDispatched = await enqueueStructure(env, sb, runId, run.paper_id);

    // Recorded so §7.7's "triage latency drops to single-digit seconds" can be
    // read straight off the data: `model_call.latency_ms` alone cannot say
    // whether a given call was on thumbnails or on full pages.
    return {
      detail: {
        classification: parsed.classification,
        pages: pagesDispatched,
        looked_at: sampledPages.length,
        on_thumbnails: onThumbs,
        second_look: secondLook,
        student_says_marked: studentSaysMarked,
        assessment_identity: resolvedAssessment?.id ?? null,
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
    await failRun(sb, runId, "We ran into a problem reading this paper. Nothing was lost — please try again.", failureCodeFor("triage", error));
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, TriageMessage>;
