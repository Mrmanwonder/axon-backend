// Shape checks for the synthetic content.v2 / structure.v2 golden sets
// (eval/golden/*-v2.json). They are specifications for a live run, so the
// checks here are about the labels being self-consistent, not about a model.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";

const load = (name: string) =>
  JSON.parse(readFileSync(path.join(import.meta.dirname, "../../../eval/golden", name), "utf8"));

const content = load("content-v2.json");
const structure = load("structure-v2.json");

test("golden sets are draft-labelled, synthetic, and have unique ids", () => {
  for (const set of [content, structure]) {
    assert.match(set.labels, /^DRAFT/);
    const ids = set.cases.map((c: any) => c.id);
    assert.equal(new Set(ids).size, ids.length);
    for (const c of set.cases) {
      assert.equal(c.needs_human_label, true);
      assert.equal(c.requires_image, true);
    }
  }
});

test("content.v2 cases: a diagram with no written text expects null; a table expects one LaTeX array", () => {
  for (const c of content.cases) {
    const lines = c.expected.student_answer_lines;
    if (c.expected.region_type === "table") {
      assert.equal(lines.length, 1, c.key);
      assert.match(lines[0], /^\\begin\{array\}\{(\|c)+\|\} \\hline .* \\end\{array\}$/, c.key);
      assert.equal(c.expected.is_latex_array, true);
    }
    if (lines === null) assert.equal(c.expected.region_type, "diagram", c.key);
    assert.equal(c.expected.every_value_has_box, true);
  }
  assert.ok(content.cases.some((c: any) => c.expected.region_type === "diagram" && c.expected.student_answer_lines?.length));
});

test("structure.v2 cases: no accepted label is a forbidden one", () => {
  for (const c of structure.cases) {
    for (const r of c.expected.regions) {
      for (const label of r.accept) {
        if (label === null) continue;
        for (const bad of c.expected.forbidden_labels) assert.notEqual(label, bad, c.key);
      }
    }
  }
});
