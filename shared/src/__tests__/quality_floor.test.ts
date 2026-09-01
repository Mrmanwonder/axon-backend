import { test } from "node:test";
import assert from "node:assert/strict";
import { clearsTheFloor } from "../quality_floor.js";

test("clearsTheFloor: rejects null and empty advice", () => {
  assert.equal(clearsTheFloor(null), false);
  assert.equal(clearsTheFloor(undefined), false);
  assert.equal(clearsTheFloor(""), false);
});

test("clearsTheFloor: rejects advice that's too short even if specific-sounding", () => {
  assert.equal(clearsTheFloor("Show your working."), false);
});

test("clearsTheFloor: rejects the known-generic patterns from CLAUDE.md's failing examples", () => {
  assert.equal(clearsTheFloor("Revise Newton's laws before the next test."), false);
  assert.equal(clearsTheFloor("Practice more numericals every week."), false);
  assert.equal(clearsTheFloor("Be more careful with your units next time."), false);
});

test("clearsTheFloor: accepts specific, performable advice — CLAUDE.md's passing example", () => {
  assert.equal(clearsTheFloor("Write the formula on its own line before you substitute."), true);
});
