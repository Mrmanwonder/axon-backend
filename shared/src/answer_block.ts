/**
 * The answer is not a string.
 *
 * `question_region.student_answer` is `text`, and that single fact is why the
 * screen shows what a student described as "numbers, alphabets and signs paired
 * together". There is nowhere to put structure, so structure is destroyed at
 * write time and no frontend can recover it. From the live rows:
 *
 *   handwritten `8/2`        stored as `8+1`     — a correct step becomes false
 *   handwritten `3/16 × 2⁵`  stored as `3/16 | 2^5` — an invented character
 *   `32` struck, `2` beneath  stored as nothing   — cancellation working gone
 *   final `6` in a drawn box  stored as `6`       — the student's own emphasis
 *   `4.5₁₀`, `100.1₂`         stored as `_10`, `_2`, alongside `^` for exponent
 *
 * Two of those are not cosmetic. `8/2` → `8+1` turns a correct student into an
 * incorrect one, and any grader reasoning over the transcript will penalise a
 * step they got right. And under CAIE marking crossed-out work still earns
 * marks when nothing replaces it, so discarding strikethroughs is not tidying,
 * it is discarding mark-bearing evidence.
 *
 * `raw_text` is kept alongside, always. It is what search runs on, what the
 * fallback renders, and what makes this additive rather than a migration of
 * meaning.
 */

export type SegmentType = "math" | "prose" | "numeral" | "binary" | "label";

export type Annotation =
  | "struck_through" | "boxed" | "circled" | "underlined" | "inserted" | "overwritten";

export type LineRole = "working" | "final_answer" | "restatement" | "crossed_out";

/** A box on the page image, in the same 0–1000 grid the rest of the pipeline uses. */
export interface Bbox { x: number; y: number; w: number; h: number; page_index?: number }

export interface Segment {
  type: SegmentType;
  /** Math only. Untrusted: model-generated, rendered through a hardened KaTeX. */
  latex?: string | null;
  /** Prose only. */
  text?: string | null;
  annotations: Annotation[];
  /**
   * Where this segment was read from. The highest-value field here: it turns
   * "the app misread me" from an argument into a two-second check, and it lets
   * a re-read target one segment instead of re-running the page.
   */
  bbox?: Bbox | null;
  confidence?: number | null;
}

export interface Line {
  segments: Segment[];
  role: LineRole;
}

export interface AnswerBlock {
  lines: Line[];
  /** Declares the `^` / `_` / radix conventions so each call stops inventing one. */
  notation_profile: string;
  raw_text: string;
}

const SEGMENT_TYPES = new Set<string>(["math", "prose", "numeral", "binary", "label"]);
const ANNOTATIONS = new Set<string>([
  "struck_through", "boxed", "circled", "underlined", "inserted", "overwritten",
]);
const ROLES = new Set<string>(["working", "final_answer", "restatement", "crossed_out"]);

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function readBbox(v: unknown): Bbox | null {
  if (!v || typeof v !== "object") return null;
  const b = v as Record<string, unknown>;
  const x = num(b.x), y = num(b.y), w = num(b.w), h = num(b.h);
  if (x === null || y === null || w === null || h === null) return null;
  const page = num(b.page_index);
  return page === null ? { x, y, w, h } : { x, y, w, h, page_index: page };
}

/**
 * Validate a model-produced answer block.
 *
 * Everything here arrives from a model and is treated as untrusted: unknown
 * segment types, unknown annotations and unknown roles are dropped rather than
 * stored, so the closed vocabularies stay closed and the renderer never meets a
 * value it has no case for. A block with nothing left after that is null, and
 * the caller falls back to `raw_text` — which is why raw_text is required.
 */
export function readAnswerBlock(raw: unknown, rawTextFallback: string | null): AnswerBlock | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const rawText = typeof o.raw_text === "string" && o.raw_text
    ? o.raw_text
    : (rawTextFallback ?? "");

  const linesIn = Array.isArray(o.lines) ? o.lines : [];
  const lines: Line[] = [];

  for (const l of linesIn) {
    if (!l || typeof l !== "object") continue;
    const li = l as Record<string, unknown>;
    const segsIn = Array.isArray(li.segments) ? li.segments : [];
    const segments: Segment[] = [];

    for (const sg of segsIn) {
      if (!sg || typeof sg !== "object") continue;
      const s = sg as Record<string, unknown>;
      const type = typeof s.type === "string" && SEGMENT_TYPES.has(s.type)
        ? (s.type as SegmentType) : null;
      if (!type) continue;

      const latex = typeof s.latex === "string" && s.latex.trim() ? s.latex : null;
      const text = typeof s.text === "string" && s.text.trim() ? s.text : null;
      // A segment carrying neither is empty, and an empty segment rendered is a
      // silent hole in the student's answer.
      if (!latex && !text) continue;

      const annotations = (Array.isArray(s.annotations) ? s.annotations : [])
        .filter((a): a is Annotation => typeof a === "string" && ANNOTATIONS.has(a));

      const confidence = num(s.confidence);
      segments.push({
        type,
        latex: type === "math" || type === "binary" || type === "numeral" ? latex : null,
        text,
        annotations,
        bbox: readBbox(s.bbox),
        confidence: confidence === null ? null : Math.min(1, Math.max(0, confidence)),
      });
    }

    if (!segments.length) continue;
    const role = typeof li.role === "string" && ROLES.has(li.role)
      ? (li.role as LineRole) : "working";
    lines.push({ segments, role });
  }

  if (!lines.length) return null;
  return {
    lines,
    notation_profile: typeof o.notation_profile === "string" && o.notation_profile
      ? o.notation_profile : "caie_default",
    raw_text: rawText,
  };
}

/**
 * The text an arithmetic check should run over.
 *
 * Crossed-out lines are excluded: the student withdrew them, and checking work
 * they abandoned would raise re-read triggers on writing they already replaced.
 * They stay in the block — CAIE still marks them when nothing replaces them —
 * they are simply not evidence of what the student's final method was.
 */
export function checkableText(block: AnswerBlock | null, rawText: string | null): string {
  if (!block) return rawText ?? "";
  const lines: string[] = [];
  for (const line of block.lines) {
    if (line.role === "crossed_out") continue;
    const parts = line.segments
      .filter((s) => !s.annotations.includes("struck_through"))
      .map((s) => s.latex ?? s.text ?? "")
      .filter(Boolean);
    if (parts.length) lines.push(parts.join(" "));
  }
  return lines.length ? lines.join("\n") : (rawText ?? "");
}
