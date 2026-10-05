/**
 * Verified grounding for the Tutor (AXO-36, AXO-12).
 *
 * tutor_evidence.ts gives the Tutor what is on the paper: the question, the
 * student's answer, the teacher's marks. This module adds the two kinds of
 * verified reference material Axon holds, and nothing else:
 *
 *   1. Official marking-scheme evidence — only through the same fail-closed
 *      resolver the explain stage uses (`resolveSchemeEvidence`): the paper must
 *      already be bound to one exact assessment identity, the canonical question
 *      must belong to that assessment, the document must be ready, unrevoked,
 *      unsuperseded, from an active reproduction-permitted policy on the stored
 *      host. Cambridge and IB are metadata-only by policy, so for them this
 *      returns nothing — by construction, not by a check here.
 *
 *   2. Syllabus learning objectives — verbatim objective text from a syllabus a
 *      person has verified, for topics the question was tagged with and the
 *      student has not rejected. Draft syllabi never appear: RLS hides them and
 *      the status is checked again here.
 *
 * Both are reference material, not judgement. They never carry or imply a mark.
 * Generated prose (region_explanation, past Tutor answers) is never loaded here:
 * a model's earlier words are not evidence for its next ones.
 *
 * Service-role reads: scheme_document and scheme_source_policy are not
 * readable by end users, so the resolver runs on the service client. Every
 * service-role path starts with an explicit ownership join (paper.id AND
 * paper.student_id), in addition to the session-scoped check the gateway has
 * already made.
 */

import { resolveSchemeEvidence, type SchemeEvidence } from "@mastery/shared/assessment.js";
import type { RegionRow } from "./tutor_evidence.js";

export const MAX_SCHEME_ITEMS = 12;
export const MAX_SYLLABUS_ITEMS = 12;
const MAX_SCHEME_TEXT = 4000;
const MAX_OBJECTIVE_TEXT = 1500;

export interface GroundingEvidence {
  id: string;
  informationClass: "VERIFIED_EXTERNAL";
  source: "axon_db";
  authority: "primary";
  value: Record<string, unknown>;
  provenance: { paperId: string; url?: string; artifactHash?: string };
  verification: "verified";
}

type Client = { from: (table: string) => any };

function clip(text: unknown, max: number): string | null {
  if (typeof text !== "string") return null;
  const t = text.trim();
  if (!t) return null;
  return t.length > max ? t.slice(0, max) + "…" : t;
}

function httpsUrl(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.toString() : undefined;
  } catch {
    return undefined;
  }
}

function num(value: number | string | null): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function schemeToEvidence(
  evidence: SchemeEvidence,
  region: Pick<RegionRow, "question_label">,
  paperId: string,
): GroundingEvidence | null {
  const markingScheme = clip(evidence.markingScheme, MAX_SCHEME_TEXT);
  const url = httpsUrl(evidence.sourceUrl);
  if (!markingScheme || !url) return null;
  return {
    id: `scheme:${evidence.canonicalQuestionId}`,
    informationClass: "VERIFIED_EXTERNAL",
    source: "axon_db",
    authority: "primary",
    value: {
      kind: "official_marking_scheme",
      questionLabel: clip(region.question_label, 40),
      schemeQuestionLabel: evidence.questionLabel,
      markingScheme,
      schemeSource: evidence.source,
      schemeVersion: evidence.version,
      retrievalMode: evidence.retrievalMode,
      note: "Official marking scheme for this exact assessment. It describes how marks are earned; the teacher's recorded mark stays authoritative.",
    },
    provenance: { paperId, url, artifactHash: `scheme_document:${evidence.schemeDocumentId}` },
    verification: "verified",
  };
}

/**
 * Scheme evidence for the given regions, or [] whenever anything is uncertain.
 * Never throws: a grounding failure must cost the student only the grounding,
 * not the answer.
 */
export async function loadSchemeGrounding(
  admin: Client | null,
  args: { studentId: string; paperId: string; regions: RegionRow[] },
  resolve: typeof resolveSchemeEvidence = resolveSchemeEvidence,
): Promise<GroundingEvidence[]> {
  if (!admin || !args.regions.length) return [];
  try {
    // Ownership join first. A paper that is not this student's, or that is not
    // bound to an exact assessment, ends retrieval here.
    const owned = await admin.from("paper")
      .select("id,assessment_identity_id")
      .eq("id", args.paperId)
      .eq("student_id", args.studentId)
      .maybeSingle();
    if (owned.error || !owned.data?.assessment_identity_id) return [];

    const out: GroundingEvidence[] = [];
    const seen = new Set<string>();
    for (const region of args.regions.slice(0, MAX_SCHEME_ITEMS)) {
      const found = await resolve(admin, {
        paperId: args.paperId,
        questionLabel: region.question_label,
        questionText: region.question_text,
        marksAvailable: num(region.marks_available),
      });
      // Defence in depth: the resolver already scopes to the paper's bound
      // assessment; refuse anything that disagrees.
      if (!found || found.assessmentIdentityId !== owned.data.assessment_identity_id) continue;
      const item = schemeToEvidence(found, region, args.paperId);
      if (!item || seen.has(item.id)) continue;
      seen.add(item.id);
      out.push(item);
    }
    return out;
  } catch (cause) {
    console.warn("tutor scheme grounding withheld", cause instanceof Error ? cause.message : "error");
    return [];
  }
}

interface TopicRow { id: string; document_id: string; code: string; kind: string; title: string; objective_text: string | null }
interface DocumentRow { id: string; provider_key: string; syllabus_code: string; version_label: string; source_url: string; status: string }

/**
 * Verbatim learning objectives for the topics these regions were tagged with,
 * from verified syllabi only. Read through the session's own RLS-scoped client
 * with an explicit student join. Never throws.
 */
export async function loadSyllabusGrounding(
  user: Client,
  args: { studentId: string; paperId: string; regions: RegionRow[] },
): Promise<GroundingEvidence[]> {
  const regionIds = args.regions.map((r) => r.id);
  if (!regionIds.length) return [];
  try {
    const tags = await user.from("region_topic")
      .select("region_id,topic_id,is_primary")
      .eq("student_id", args.studentId)
      .in("region_id", regionIds)
      .is("student_rejected_at", null)
      .neq("confidence", "unsure");
    if (tags.error || !tags.data?.length) return [];

    const labelsByTopic = new Map<string, Set<string>>();
    const labelOf = new Map(args.regions.map((r) => [r.id, clip(r.question_label, 40)] as const));
    const primaryFirst = [...(tags.data as Array<{ region_id: string; topic_id: string; is_primary: boolean }>)]
      .sort((a, b) => Number(b.is_primary) - Number(a.is_primary));
    for (const tag of primaryFirst) {
      if (!regionIds.includes(tag.region_id)) continue;
      const labels = labelsByTopic.get(tag.topic_id) ?? new Set<string>();
      const label = labelOf.get(tag.region_id);
      if (label) labels.add(label);
      labelsByTopic.set(tag.topic_id, labels);
    }
    const topicIds = [...labelsByTopic.keys()].slice(0, MAX_SYLLABUS_ITEMS);
    if (!topicIds.length) return [];

    const topics = await user.from("syllabus_topic")
      .select("id,document_id,code,kind,title,objective_text")
      .in("id", topicIds);
    if (topics.error || !topics.data?.length) return [];
    const topicRows = topics.data as TopicRow[];

    const docIds = [...new Set(topicRows.map((t) => t.document_id))];
    const docs = await user.from("syllabus_document")
      .select("id,provider_key,syllabus_code,version_label,source_url,status")
      .in("id", docIds)
      .eq("status", "verified");
    if (docs.error || !docs.data?.length) return [];
    const docById = new Map((docs.data as DocumentRow[]).filter((d) => d.status === "verified").map((d) => [d.id, d]));

    const out: GroundingEvidence[] = [];
    for (const topicId of topicIds) {
      const topic = topicRows.find((t) => t.id === topicId);
      const doc = topic ? docById.get(topic.document_id) : undefined;
      if (!topic || !doc) continue;
      const url = httpsUrl(doc.source_url);
      out.push({
        id: `syllabus:${topic.id}`,
        informationClass: "VERIFIED_EXTERNAL",
        source: "axon_db",
        authority: "primary",
        value: {
          kind: "syllabus_objective",
          questionLabels: [...(labelsByTopic.get(topicId) ?? [])],
          syllabus: `${doc.syllabus_code} (${doc.version_label})`,
          board: doc.provider_key,
          code: topic.code,
          title: clip(topic.title, 200),
          objectiveText: clip(topic.objective_text, MAX_OBJECTIVE_TEXT),
          note: "Verbatim from the published, person-verified syllabus. It states what is assessed, not how marks are awarded.",
        },
        provenance: { paperId: args.paperId, ...(url ? { url } : {}), artifactHash: `syllabus_document:${doc.id}` },
        verification: "verified",
      });
    }
    return out;
  } catch (cause) {
    console.warn("tutor syllabus grounding withheld", cause instanceof Error ? cause.message : "error");
    return [];
  }
}
