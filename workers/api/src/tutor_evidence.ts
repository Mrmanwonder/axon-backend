/**
 * Server-side paper evidence for the Tutor (AXO-36).
 *
 * The intelligence orchestrator refuses paper feedback unless it holds
 * `paper` or `teacher` evidence, and the gateway used to forward only a
 * verified `paperId`, so every paper question ended in "I can't determine
 * that". This module loads that evidence from the database, never from the
 * caller:
 *
 *   - through the caller's own RLS-scoped client (question_region is readable
 *     only under the live Student Mode scope), AND an explicit student_id +
 *     paper_id join on every query, so a foreign id fails twice over;
 *   - only the fields a walkthrough needs: label, question text, the student's
 *     answer, the teacher's marks and remark. No names, no image keys, no boxes;
 *   - from the paper's latest run that reached review, so a half-finished run
 *     is never presented as the paper;
 *   - bounded: one question when asked about one, else at most MAX_REGIONS,
 *     with each text field truncated.
 *
 * Teacher marks go in as `source: "teacher"`, authority `primary`. They are
 * the fact the tutor explains and must never contradict (hard rule 1). An
 * `unsure` read is forwarded as `unverified`, never as verified, and an
 * unreadable region is omitted rather than guessed at (hard rule 4).
 */

export const MAX_REGIONS = 12;
const MAX_TEXT = 1500;
const MAX_REMARK = 500;

/** Run states whose regions a student has been shown. */
export const REVIEWABLE_RUN_STATES = ["needs_review", "explaining", "ready", "committed"] as const;

export interface RegionRow {
  id: string;
  question_label: string | null;
  question_text: string | null;
  student_answer: string | null;
  marks_awarded: number | string | null;
  marks_available: number | string | null;
  teacher_remark: string | null;
  confidence_tier: "confident" | "unsure" | "unreadable" | null;
  student_confirmed_at: string | null;
}

export interface TutorEvidence {
  id: string;
  informationClass: "OBSERVED";
  source: "paper" | "teacher";
  authority: "primary";
  value: Record<string, unknown>;
  provenance: { paperId: string };
  verification: "verified" | "probable" | "unverified";
}

function clip(text: string | null, max: number): string | null {
  if (typeof text !== "string") return null;
  const t = text.trim();
  if (!t) return null;
  return t.length > max ? t.slice(0, max) + "…" : t;
}

function num(value: number | string | null): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/** A student's own confirmation outranks the reader's confidence. */
function verificationOf(row: RegionRow): TutorEvidence["verification"] {
  if (row.student_confirmed_at) return "verified";
  return row.confidence_tier === "confident" ? "probable" : "unverified";
}

export function regionsToEvidence(rows: RegionRow[], paperId: string): TutorEvidence[] {
  const out: TutorEvidence[] = [];
  for (const row of rows.slice(0, MAX_REGIONS)) {
    if (row.confidence_tier === "unreadable") continue;
    const verification = verificationOf(row);
    const label = clip(row.question_label, 40);
    const question = clip(row.question_text, MAX_TEXT);
    const answer = clip(row.student_answer, MAX_TEXT);
    if (question || answer) {
      out.push({
        id: `paper:${row.id}`,
        informationClass: "OBSERVED",
        source: "paper",
        authority: "primary",
        value: { label, questionText: question, studentAnswer: answer },
        provenance: { paperId },
        verification,
      });
    }
    const awarded = num(row.marks_awarded);
    const available = num(row.marks_available);
    const remark = clip(row.teacher_remark, MAX_REMARK);
    if (awarded !== null || remark) {
      out.push({
        id: `teacher:${row.id}`,
        informationClass: "OBSERVED",
        source: "teacher",
        authority: "primary",
        value: { label, marksAwarded: awarded, marksAvailable: available, teacherRemark: remark },
        provenance: { paperId },
        verification,
      });
    }
  }
  return out;
}

type Client = { from: (table: string) => any };

export type EvidenceResult =
  | { ok: true; evidence: TutorEvidence[]; regions: RegionRow[] }
  | { ok: false; status: number; message: string };

export async function loadPaperEvidence(
  user: Client,
  args: { studentId: string; paperId: string; questionId?: string },
): Promise<EvidenceResult> {
  const run = await user
    .from("extraction_run")
    .select("id")
    .eq("paper_id", args.paperId)
    .eq("student_id", args.studentId)
    .in("status", REVIEWABLE_RUN_STATES as unknown as string[])
    .order("started_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (run.error) return { ok: false, status: 503, message: "We could not load that paper just now." };
  if (!run.data) {
    // A paper with no reviewable run has no evidence yet. Say so through the
    // orchestrator's own insufficient-evidence path rather than an error.
    return args.questionId
      ? { ok: false, status: 403, message: "That question is not available for this student." }
      : { ok: true, evidence: [], regions: [] };
  }

  let query = user
    .from("question_region")
    .select("id,question_label,question_text,student_answer,marks_awarded,marks_available,teacher_remark,confidence_tier,student_confirmed_at")
    .eq("run_id", run.data.id)
    .eq("paper_id", args.paperId)
    .eq("student_id", args.studentId);
  if (args.questionId) query = query.eq("id", args.questionId);
  const regions = await query.order("order_index", { ascending: true }).limit(MAX_REGIONS);
  if (regions.error) return { ok: false, status: 503, message: "We could not load that paper just now." };
  const rows = (regions.data ?? []) as RegionRow[];
  if (args.questionId && !rows.length) {
    return { ok: false, status: 403, message: "That question is not available for this student." };
  }
  // The rows go back too: grounding (tutor_grounding.ts) keys scheme and
  // syllabus evidence to exactly these regions and no others.
  const kept = rows.slice(0, MAX_REGIONS).filter((row) => row.confidence_tier !== "unreadable");
  return { ok: true, evidence: regionsToEvidence(rows, args.paperId), regions: kept };
}
