export type LearningConsentState = "GRANTED" | "DENIED" | "UNVERIFIED";

export interface LearningConsentDecision {
  state: LearningConsentState;
  granted: boolean;
  reason: "CURRENT_GRANT" | "CURRENT_DENIAL" | "NO_DECISION" | "INVALID_STUDENT_ID" | "MISSING_ADMIN_KEY" | "STUDENT_NOT_FOUND" | "LOOKUP_FAILED";
  seq?: number;
  noticeVersion?: string;
}

interface ConsentEnvironment {
  SUPABASE_URL: string;
  SUPABASE_SECRET_KEY?: string;
  SUPABASE_SERVICE_ROLE_KEY?: string;
}

interface ConsentRow {
  student_id: string | null;
  granted: boolean;
  notice_version: string;
  seq: number;
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function validStudentId(value: string | null): value is string {
  return value !== null && UUID.test(value);
}

const unverified = (reason: LearningConsentDecision["reason"]): LearningConsentDecision => ({ state: "UNVERIFIED", granted: false, reason });

async function rows<T>(response: Response): Promise<T[] | undefined> {
  if (!response.ok) return undefined;
  const value: unknown = await response.json();
  return Array.isArray(value) ? value as T[] : undefined;
}

export async function resolveLearningConsent(
  env: ConsentEnvironment,
  studentId: string,
  fetcher: typeof fetch = fetch
): Promise<LearningConsentDecision> {
  if (!validStudentId(studentId)) return unverified("INVALID_STUDENT_ID");
  const adminKey = env.SUPABASE_SECRET_KEY ?? env.SUPABASE_SERVICE_ROLE_KEY;
  if (!adminKey) return unverified("MISSING_ADMIN_KEY");
  const headers = { accept: "application/json", apikey: adminKey };
  const requestOptions = { headers, signal: AbortSignal.timeout(3_000) };
  try {
    const studentUrl = new URL("/rest/v1/student", env.SUPABASE_URL);
    studentUrl.searchParams.set("select", "guardian_id");
    studentUrl.searchParams.set("id", `eq.${studentId}`);
    studentUrl.searchParams.set("limit", "1");
    const studentRows = await rows<{ guardian_id?: unknown }>(await fetcher(studentUrl, requestOptions));
    if (!studentRows) return unverified("LOOKUP_FAILED");
    const guardianId = studentRows[0]?.guardian_id;
    if (typeof guardianId !== "string" || !validStudentId(guardianId)) return unverified("STUDENT_NOT_FOUND");

    const consentUrl = new URL("/rest/v1/consent_current", env.SUPABASE_URL);
    consentUrl.searchParams.set("select", "student_id,granted,notice_version,seq");
    consentUrl.searchParams.set("guardian_id", `eq.${guardianId}`);
    consentUrl.searchParams.set("purpose", "eq.improve_extraction");
    consentUrl.searchParams.set("or", `(student_id.eq.${studentId},student_id.is.null)`);
    consentUrl.searchParams.set("limit", "2");
    const consentRows = await rows<ConsentRow>(await fetcher(consentUrl, requestOptions));
    if (!consentRows) return unverified("LOOKUP_FAILED");
    const current = consentRows.find((row) => row.student_id === studentId) ?? consentRows.find((row) => row.student_id === null);
    if (!current) return { state: "DENIED", granted: false, reason: "NO_DECISION" };
    if (typeof current.granted !== "boolean" || !Number.isSafeInteger(current.seq) || typeof current.notice_version !== "string") return unverified("LOOKUP_FAILED");
    const noticeVersion = current.notice_version.trim();
    if (!noticeVersion || noticeVersion.length > 128) return unverified("LOOKUP_FAILED");
    return {
      state: current.granted ? "GRANTED" : "DENIED",
      granted: current.granted,
      reason: current.granted ? "CURRENT_GRANT" : "CURRENT_DENIAL",
      seq: current.seq,
      noticeVersion
    };
  } catch {
    return unverified("LOOKUP_FAILED");
  }
}
