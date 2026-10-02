import { test } from "node:test";
import assert from "node:assert/strict";
import { providerKeyForBoard, providerKeyForProgramme } from "../provider_identity.js";

test("only explicit legacy Cambridge boards resolve to cambridge", () => {
  for (const board of ["CAIE", "IGCSE", "AS_A_LEVEL"]) assert.equal(providerKeyForBoard(board), "cambridge");
  assert.equal(providerKeyForBoard("CBSE"), "cbse");
  assert.equal(providerKeyForBoard("IBDP"), "ib");
});

test("an unrecognised or missing board is unknown, never Cambridge", () => {
  for (const board of [undefined, null, "", "ICSE", "STATE", "caie", "IB", "SOMETHING_NEW", 7, {}]) {
    assert.equal(providerKeyForBoard(board), null, JSON.stringify(board));
  }
});

test("programme keys resolve by prefix and unknown keys stay unknown", () => {
  assert.equal(providerKeyForProgramme("cambridge_igcse"), "cambridge");
  assert.equal(providerKeyForProgramme("cbse_secondary"), "cbse");
  assert.equal(providerKeyForProgramme("ibdp"), "ib");
  for (const key of [undefined, null, "", "icse_x", "ib", "cambridge"]) assert.equal(providerKeyForProgramme(key), null, String(key));
});
