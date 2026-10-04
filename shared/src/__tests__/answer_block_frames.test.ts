import { test } from "node:test";
import assert from "node:assert/strict";
import { mapAnswerBlockToPages } from "../answer_block_frames.js";
import { readAnswerBlock } from "../answer_block.js";

const block = (page_index: number) => ({ raw_text: "8/2", lines: [{ role: "working", segments: [{ type: "math", latex: "8/2", bbox: { x: 100, y: 200, w: 300, h: 100, page_index } }] }] });
test("a segment on a continuation page uses that page's dimensions and number", () => {
  const result = mapAnswerBlockToPages(block(1), null, [
    { kind: "page", pageNumber: 2, pageWidth: 1000, pageHeight: 2000 },
    { kind: "page", pageNumber: 5, pageWidth: 2000, pageHeight: 3000 },
  ])!;
  assert.equal(result.source_space, "page_pixels_v1");
  assert.deepEqual(result.lines[0].segments[0].bbox, { page: 5, x: 200, y: 600, w: 600, h: 300 });
  assert.deepEqual(readAnswerBlock(result, null), result);
});
test("a cropped model image translates segment coordinates back into page pixels", () => {
  const result = mapAnswerBlockToPages(block(0), null, [{ kind: "crop", pageNumber: 3, pageWidth: 2000, pageHeight: 3000, band: { x: 50, y: 700, w: 1000, h: 500 } }])!;
  assert.deepEqual(result.lines[0].segments[0].bbox, { page: 3, x: 150, y: 800, w: 300, h: 50 });
});
test("an invalid model image index cannot become a highlight on the first page", () => {
  const result = mapAnswerBlockToPages(block(9), null, [{ kind: "page", pageNumber: 1, pageWidth: 1000, pageHeight: 1000 }])!;
  assert.equal(result.lines[0].segments[0].bbox, null);
});
