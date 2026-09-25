import { vi } from "vitest";

export const TEST_STUDENT_ID = "aaaaaaaa-0000-4000-8000-000000000002";
export const TEST_GUARDIAN_ID = "aaaaaaaa-0000-4000-8000-000000000001";

export function mockSupabaseConsent(granted = true): ReturnType<typeof vi.spyOn> {
  const originalFetch = globalThis.fetch;
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    const url = new URL(request.url);
    if (url.origin !== "https://dlgcqieyevoebefhcggi.supabase.co") return originalFetch(input, init);
    if (request.headers.get("apikey") !== "test-only" || request.headers.has("authorization")) {
      return Response.json({ message: "invalid admin-key transport" }, { status: 401 });
    }
    if (url.pathname === "/rest/v1/student") return Response.json([{ guardian_id: TEST_GUARDIAN_ID }]);
    if (url.pathname === "/rest/v1/consent_current") {
      return Response.json([{ student_id: TEST_STUDENT_ID, granted, notice_version: "privacy.v1", seq: 42 }]);
    }
    return Response.json({ message: "unexpected Supabase path" }, { status: 404 });
  });
}
