/**
 * Per-region confidence and the council D1 ask-rule, decided without I/O.
 *
 * mastery-reconcile calls this with what it read; the offline gate
 * (scripts/ask-rule-gate.mjs) calls it with production rows read-only. One
 * function, so the gate measures exactly what the worker will do.
 */

import { assess, askReasons, downgradeRecognition, type AskReason, type AssessSignals, type ConfidenceTier, type Recognition } from "./confidence.js";
import { placementInput, placementVerdicts } from "./placement.js";

export function recognitionFor(value: unknown): Recognition {
  return value === "high" || value === "medium" || value === "low" ? value : null;
}

export interface RegionRow {
  id: string;
  order_index: number;
  question_label: string | null;
  marks_awarded: number | string | null;
  marks_available: number | string | null;
  confidence_tier: string | null;
  confidence_signals: Record<string, unknown> | null;
  extract_status: string | null;
  page_spans: unknown;
  student_answer?: string | null;
  question_text?: string | null;
}

export interface JudgeOptions {
  /** Shown unmarked (see `paperShownUnmarked`). */
  paperUnmarked: boolean;
  /** Page numbers whose marking was read through the layer fallback. */
  fallbackPages: Set<number>;
  /** Each region's own working, evaluated (shared/src/arithmetic.ts); input order. */
  arithmetic: Array<boolean | "unknown">;
}

export interface RegionVerdict {
  id: string;
  tier: ConfidenceTier;
  signals: AssessSignals & { ask: AskReason[]; placed_label: string | null };
  needs_review: boolean;
  ask: AskReason[];
  placedLabel: string | null;
}

const num = (v: unknown): number | null => (v === null || v === undefined ? null : Number(v));

export function judgeRegions(rows: RegionRow[], opts: JudgeOptions): RegionVerdict[] {
  const placement = placementVerdicts(rows.map(placementInput));
  return rows.map((region, i) => {
    const spans: Array<{ page: number }> = Array.isArray(region.page_spans) ? region.page_spans as Array<{ page: number }> : [];
    const touchesFallbackPage = spans.some((s) => opts.fallbackPages.has(s.page));
    const read: Recognition = region.confidence_tier === "unreadable"
      ? "low"
      : recognitionFor(region.confidence_signals?.recognition_confidence);
    const recognition = touchesFallbackPage ? downgradeRecognition(read) : read;
    const unreadable = region.confidence_tier === "unreadable" || region.extract_status === "failed";
    const place = placement[i];

    const { tier, signals } = assess({
      recognition,
      numberingSound: place.structural,
      arithmeticOk: opts.arithmetic[i] ?? "unknown",
      awarded: num(region.marks_awarded),
      available: num(region.marks_available),
      unreadable,
      paperUnmarked: opts.paperUnmarked,
    });
    const ask = askReasons({
      unreadable,
      recognition,
      paperMarked: !opts.paperUnmarked,
      plausibility: signals.plausibility,
      unplaceable: place.unplaceable,
      counted: place.counted,
    });
    return {
      id: region.id,
      tier,
      signals: { ...signals, ask, placed_label: place.placedLabel },
      needs_review: ask.length > 0,
      ask,
      placedLabel: place.placedLabel,
    };
  });
}
