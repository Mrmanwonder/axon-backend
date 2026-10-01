import { describe, expect, test } from "vitest";
import { jwtSubject, tutorRollout } from "../src/tutor_rollout.js";

const b64 = (o: unknown) => btoa(JSON.stringify(o)).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
const bearer = (sub: string) => `Bearer ${b64({ alg: "HS256" })}.${b64({ sub, role: "authenticated" })}.sig`;

describe("AXO-126 tutor rollout switch", () => {
  test("unset means off", () => {
    expect(tutorRollout({}, "u1")).toEqual({ allowed: false, reason: "off" });
  });
  test("an unknown stage value is treated as off, never as open", () => {
    expect(tutorRollout({ TUTOR_ROLLOUT: "beta" }, "u1").allowed).toBe(false);
    expect(tutorRollout({ TUTOR_ROLLOUT: "on" }, "u1").allowed).toBe(false);
  });
  test("internal admits only listed users", () => {
    const env = { TUTOR_ROLLOUT: "internal", TUTOR_INTERNAL_USERS: " u1 , u2 " };
    expect(tutorRollout(env, "u1").allowed).toBe(true);
    expect(tutorRollout(env, "u3")).toEqual({ allowed: false, reason: "not_in_stage" });
    expect(tutorRollout(env, null).allowed).toBe(false);
    expect(tutorRollout({ TUTOR_ROLLOUT: "internal" }, "u1").allowed).toBe(false);
  });
  test("ga admits everyone who reached the gate", () => {
    expect(tutorRollout({ TUTOR_ROLLOUT: "GA" }, null).allowed).toBe(true);
  });
  test("reads sub from a bearer token; malformed tokens give null", () => {
    expect(jwtSubject(bearer("11111111-2222"))).toBe("11111111-2222");
    expect(jwtSubject("Bearer not-a-jwt")).toBeNull();
    expect(jwtSubject(null)).toBeNull();
  });
});
