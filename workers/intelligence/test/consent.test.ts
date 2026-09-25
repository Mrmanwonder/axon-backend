import { describe, expect, it } from "vitest";
import { resolveLearningConsent } from "../src/intelligence/corrections/consent";
import { TEST_GUARDIAN_ID, TEST_STUDENT_ID } from "./supabase-consent.mock";

const environment = { SUPABASE_URL: "https://project.supabase.co", SUPABASE_SECRET_KEY: "sb_secret_test" };

function consentFetcher(consents: unknown[], status = 200): { fetcher: typeof fetch; requests: Request[] } {
  const requests: Request[] = [];
  const fetcher = ((input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    requests.push(request);
    if (request.headers.get("apikey") !== "sb_secret_test" || request.headers.has("authorization")) {
      return Promise.resolve(Response.json({ message: "bad key transport" }, { status: 401 }));
    }
    if (status !== 200) return Promise.resolve(Response.json({ message: "unavailable" }, { status }));
    return Promise.resolve(request.url.includes("/student?")
      ? Response.json([{ guardian_id: TEST_GUARDIAN_ID }])
      : Response.json(consents));
  }) as typeof fetch;
  return { fetcher, requests };
}

describe("authoritative correction-learning consent", () => {
  it("uses the student-specific decision before guardian scope and sends a modern secret only as apikey", async () => {
    const { fetcher, requests } = consentFetcher([
      { student_id: null, granted: true, notice_version: "privacy.v2", seq: 99 },
      { student_id: TEST_STUDENT_ID, granted: false, notice_version: "privacy.v1", seq: 12 }
    ]);
    await expect(resolveLearningConsent(environment, TEST_STUDENT_ID, fetcher)).resolves.toEqual({
      state: "DENIED", granted: false, reason: "CURRENT_DENIAL", seq: 12, noticeVersion: "privacy.v1"
    });
    expect(requests).toHaveLength(2);
    expect(requests.every((request) => request.headers.get("apikey") === "sb_secret_test" && !request.headers.has("authorization"))).toBe(true);
  });

  it("accepts a current student grant with auditable sequence and notice provenance", async () => {
    const { fetcher } = consentFetcher([{ student_id: TEST_STUDENT_ID, granted: true, notice_version: "privacy.v3", seq: 123 }]);
    await expect(resolveLearningConsent(environment, TEST_STUDENT_ID, fetcher)).resolves.toEqual({
      state: "GRANTED", granted: true, reason: "CURRENT_GRANT", seq: 123, noticeVersion: "privacy.v3"
    });
  });

  it("treats no optional decision as denied and malformed provenance as unverifiable", async () => {
    const noDecision = consentFetcher([]);
    await expect(resolveLearningConsent(environment, TEST_STUDENT_ID, noDecision.fetcher)).resolves.toEqual({ state: "DENIED", granted: false, reason: "NO_DECISION" });
    const malformed = consentFetcher([{ student_id: TEST_STUDENT_ID, granted: true, notice_version: "", seq: 1 }]);
    await expect(resolveLearningConsent(environment, TEST_STUDENT_ID, malformed.fetcher)).resolves.toMatchObject({ state: "UNVERIFIED", granted: false, reason: "LOOKUP_FAILED" });
  });

  it("fails closed when the student, admin key, or Supabase lookup cannot be verified", async () => {
    const unavailable = consentFetcher([], 503);
    await expect(resolveLearningConsent(environment, TEST_STUDENT_ID, unavailable.fetcher)).resolves.toMatchObject({ state: "UNVERIFIED", granted: false, reason: "LOOKUP_FAILED" });
    await expect(resolveLearningConsent({ SUPABASE_URL: environment.SUPABASE_URL }, TEST_STUDENT_ID, unavailable.fetcher)).resolves.toMatchObject({ state: "UNVERIFIED", granted: false, reason: "MISSING_ADMIN_KEY" });
    await expect(resolveLearningConsent(environment, "not-a-student", unavailable.fetcher)).resolves.toMatchObject({ state: "UNVERIFIED", granted: false, reason: "INVALID_STUDENT_ID" });
  });
});
