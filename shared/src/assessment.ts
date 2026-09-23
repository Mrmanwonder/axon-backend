/**
 * Deterministic official-assessment resolution.
 *
 * Vision/model output is allowed to transcribe header metadata. It is never
 * allowed to choose the matching paper or marking scheme. These helpers bind
 * only when the student's own curriculum/subject selection and the transcribed
 * metadata resolve to exactly one stored official identity.
 */

export type AssessmentCandidate = {
  subject_code: string | null;
  level: "SL" | "HL" | null;
  exam_year: number | null;
  session: string | null;
  paper_code: string | null;
  component_code: string | null;
  variant: string | null;
  zone: string | null;
  assessment_route: string | null;
  confidence: "high" | "low";
};

export type ResolvedAssessment = {
  id: string;
  programme_id: string;
  subject_offering_id: string | null;
  level: string | null;
  exam_year: number | null;
  session: string | null;
  paper_code: string | null;
  component_code: string | null;
  variant: string | null;
  zone: string | null;
  assessment_route: string | null;
  title: string;
  official_source_url: string | null;
};

export type SchemeEvidence = {
  canonicalQuestionId: string;
  markingScheme: string;
  source: string;
  version: string;
  sourceUrl: string;
  schemeDocumentId: string;
  assessmentIdentityId: string;
};

function compact(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const v = value.trim().replace(/\s+/g, " ");
  return v || null;
}

function token(value: unknown): string | null {
  const v = compact(value);
  return v ? v.toUpperCase().replace(/[^A-Z0-9]+/g, "") : null;
}

export function normaliseQuestionLabel(value: unknown): string | null {
  const v = compact(value);
  if (!v) return null;
  return v.toUpperCase().replace(/\s+/g, "").replace(/[.]+$/g, "");
}

export function candidateIsResolvable(candidate: AssessmentCandidate): boolean {
  if (candidate.confidence !== "high") return false;
  if (!token(candidate.subject_code)) return false;
  if (!Number.isInteger(candidate.exam_year) || Number(candidate.exam_year) < 2000 || Number(candidate.exam_year) > 2100) return false;
  // Subject + year alone is never a paper identity. Require at least one
  // official paper/component discriminator visible in the header.
  return !!(token(candidate.paper_code) || token(candidate.component_code));
}

function sameNullable(a: unknown, b: unknown): boolean {
  return token(a) === token(b);
}

export function identityMatchesCandidate(
  identity: Omit<ResolvedAssessment, "id" | "programme_id" | "subject_offering_id" | "title" | "official_source_url">,
  candidate: AssessmentCandidate,
  selectedLevel: string | null,
): boolean {
  if (Number(identity.exam_year) !== Number(candidate.exam_year)) return false;
  const effectiveLevel = candidate.level ?? (selectedLevel as "SL" | "HL" | null);
  return sameNullable(identity.level, effectiveLevel)
    && sameNullable(identity.session, candidate.session)
    && sameNullable(identity.paper_code, candidate.paper_code)
    && sameNullable(identity.component_code, candidate.component_code)
    && sameNullable(identity.variant, candidate.variant)
    && sameNullable(identity.zone, candidate.zone)
    && sameNullable(identity.assessment_route, candidate.assessment_route);
}

export async function resolveAssessmentIdentity(
  sb: any,
  args: { studentId: string; paperId: string; candidate: AssessmentCandidate },
): Promise<ResolvedAssessment | null> {
  const { candidate } = args;
  if (!candidateIsResolvable(candidate)) return null;

  const { data: student, error: studentError } = await sb.from("student")
    .select("programme_id")
    .eq("id", args.studentId)
    .maybeSingle();
  if (studentError) throw studentError;
  if (!student?.programme_id) return null;

  const { data: selections, error: selectionsError } = await sb.from("student_subject")
    .select("subject_offering_id,selected_level")
    .eq("student_id", args.studentId)
    .not("subject_offering_id", "is", null);
  if (selectionsError) throw selectionsError;
  const offeringIds = (selections ?? []).map((row: any) => row.subject_offering_id).filter(Boolean);
  if (!offeringIds.length) return null;

  const { data: offerings, error: offeringsError } = await sb.from("subject_offering")
    .select("id,programme_id,external_code")
    .eq("programme_id", student.programme_id)
    .in("id", offeringIds);
  if (offeringsError) throw offeringsError;

  const subjectCode = token(candidate.subject_code);
  const matchingOfferings = (offerings ?? []).filter((row: any) => token(row.external_code) === subjectCode);
  if (matchingOfferings.length !== 1) return null;

  const offering = matchingOfferings[0];
  const selection = (selections ?? []).find((row: any) => row.subject_offering_id === offering.id);
  const selectedLevel = selection?.selected_level ?? null;

  const { data: identities, error: identitiesError } = await sb.from("assessment_identity")
    .select("id,programme_id,subject_offering_id,level,exam_year,session,paper_code,component_code,variant,zone,assessment_route,title,official_source_url")
    .eq("programme_id", student.programme_id)
    .eq("subject_offering_id", offering.id)
    .eq("exam_year", candidate.exam_year);
  if (identitiesError) throw identitiesError;

  const exact = (identities ?? []).filter((row: any) =>
    identityMatchesCandidate(row, candidate, selectedLevel)
  ) as ResolvedAssessment[];
  if (exact.length !== 1) return null;

  const resolved = exact[0];
  const { error: paperError } = await sb.from("paper")
    .update({ assessment_identity_id: resolved.id })
    .eq("id", args.paperId)
    .eq("student_id", args.studentId);
  if (paperError) throw paperError;
  return resolved;
}

export async function resolveSchemeEvidence(
  sb: any,
  args: { paperId: string; questionLabel: string | null },
): Promise<SchemeEvidence | null> {
  const label = normaliseQuestionLabel(args.questionLabel);
  if (!label) return null;

  const { data: paper, error: paperError } = await sb.from("paper")
    .select("assessment_identity_id")
    .eq("id", args.paperId)
    .maybeSingle();
  if (paperError) throw paperError;
  if (!paper?.assessment_identity_id) return null;

  const { data: rows, error: questionError } = await sb.from("canonical_question")
    .select("id,assessment_identity_id,question_label,marking_scheme,scheme_source,scheme_version,scheme_document_id")
    .eq("assessment_identity_id", paper.assessment_identity_id);
  if (questionError) throw questionError;

  const exact = (rows ?? []).filter((row: any) => normaliseQuestionLabel(row.question_label) === label);
  if (exact.length !== 1) return null;
  const question = exact[0];
  if (!question.marking_scheme || !question.scheme_source || !question.scheme_version || !question.scheme_document_id) return null;

  const { data: document, error: documentError } = await sb.from("scheme_document")
    .select("id,source_url,copyright_access_class,extraction_status")
    .eq("id", question.scheme_document_id)
    .eq("assessment_identity_id", paper.assessment_identity_id)
    .maybeSingle();
  if (documentError) throw documentError;
  if (!document) return null;
  if (!["public_official", "licensed_official"].includes(document.copyright_access_class)) return null;
  if (!["ready", "complete", "extracted"].includes(document.extraction_status)) return null;
  if (!document.source_url) return null;

  return {
    canonicalQuestionId: question.id,
    markingScheme: question.marking_scheme,
    source: question.scheme_source,
    version: question.scheme_version,
    sourceUrl: document.source_url,
    schemeDocumentId: document.id,
    assessmentIdentityId: paper.assessment_identity_id,
  };
}
