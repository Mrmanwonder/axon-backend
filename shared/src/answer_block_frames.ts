import { readAnswerBlock, type AnswerBlock } from "./answer_block.js";
import { frameForIndex, mapModelBoxToPage, type ModelFrame } from "./frames.js";

/** Persist provenance in page pixels, never in an undisclosed model-image grid. */
export function mapAnswerBlockToPages(raw: unknown, fallback: string | null, frames: ModelFrame[]): AnswerBlock | null {
  const block = readAnswerBlock(raw, fallback);
  if (!block) return null;
  return {
    ...block,
    source_space: "page_pixels_v1",
    lines: block.lines.map(line => ({
      ...line,
      segments: line.segments.map(segment => {
        const frame = frameForIndex(frames, segment.bbox?.page_index);
        const bbox = segment.bbox && frame ? mapModelBoxToPage(frame, segment.bbox) : null;
        return { ...segment, bbox };
      }),
    })),
  };
}
