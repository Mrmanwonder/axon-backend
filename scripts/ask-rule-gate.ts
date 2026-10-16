// Council D1 offline gate (7 Oct 2026, AXO-216).
//
// Re-applies the new tier rule and ask-rule to production regions, read-only,
// and prints aggregate counts. Nothing is written anywhere. Before shipping
// option B, every student-corrected region must come out asked; if one does
// not, the council's fallback is option A for the beta.
//
// Input: a JSON file produced by the read-only query below (run it with the
// Supabase SQL editor or MCP `execute_sql`, save the rows as {"runs": [...]}).
// It carries no student text: answers and question text are reduced to one
// boolean (has_text_or_answer), which is all the placement walk needs.
//
//   select r.id run, <owner-paper flag> owner,
//     r.tier_routing->'triage'->>'classification' cls, r.tier_routing->'triage'->>'confidence' conf,
//     (select jsonb_agg(pp.page_number order by pp.page_number) from paper_page pp
//        where pp.paper_id=r.paper_id and pp.layer_fallback is not null) fb,
//     (select jsonb_agg(jsonb_build_array(q.order_index, q.question_label, q.marks_awarded, q.marks_available,
//        q.confidence_tier, q.extract_status, q.needs_review,
//        (q.page_spans->0->>'page')::int, (q.page_spans->0->'box'->>'y')::numeric,
//        coalesce(q.student_answer,'')<>'' or coalesce(q.question_text,'')<>'',
//        q.confidence_signals->'recognition_confidence', q.confidence_signals->'arithmetic',
//        q.confidence_signals->'plausibility', q.student_corrected,
//        exists(select 1 from learning.signal s where s.region_id=q.id and s.field='marks_awarded'),
//        (select jsonb_agg(s->'page') from jsonb_array_elements(q.page_spans) s)) order by q.order_index)
//      from question_region q where q.run_id=r.id) rows
//   from extraction_run r where r.id in (<runs holding a corrected region>, <owner's runs>);
//
// Two reconstructions, both stated in the output:
//   - A corrected teacher mark: the row now holds the student's corrected
//     value, not the value Axon read. The stored `plausibility` signal was
//     written at reconcile time from the value as read, so where it is false the
//     as-read mark was missing or impossible; the gate restores "missing" for
//     that row (on a marked paper the ask is the same either way).
//   - Arithmetic: the stored `arithmetic` signal (a deterministic evaluation
//     of the same working) stands in for re-running it, so no student text is
//     needed here.
//
// Usage: npx tsx scripts/ask-rule-gate.ts <data.json>

import { readFileSync } from "node:fs";
import { judgeRegions, type RegionRow } from "../shared/src/review_rule.js";
import { paperShownUnmarked } from "../shared/src/confidence.js";

type Row = [number, string | null, number | null, number | null, string, string, boolean, number | null, number | null,
  boolean, string | null, boolean | "unknown" | null, boolean | "unknown" | null, boolean, boolean, number[] | null];
interface Run { run: string; owner: boolean; cls: string | null; conf: string | null; fb: number[] | null; rows: Row[] }

const path = process.argv[2];
if (!path) throw new Error("usage: tsx scripts/ask-rule-gate.ts <data.json>");
const { runs } = JSON.parse(readFileSync(path, "utf8")) as { runs: Run[] };

function toRegion(row: Row, i: number, opts: { recognitionOverride?: string }): RegionRow & { _corrected: boolean } {
  const [order_index, label, awarded, available, tier, extract_status, , page, y, hasText, rec, , storedPl, corrected, correctedMarks, spanPages] = row;
  const asReadAwarded = corrected && correctedMarks && storedPl === false ? null : awarded;
  const pages = spanPages?.length ? spanPages : page === null ? [] : [page];
  return {
    id: `r${i}`,
    order_index,
    question_label: label,
    marks_awarded: asReadAwarded,
    marks_available: available,
    confidence_tier: tier,
    confidence_signals: { recognition_confidence: opts.recognitionOverride ?? rec },
    extract_status,
    page_spans: pages.map((p, k) => ({ page: p, box: { y: k === 0 ? y : null } })),
    student_answer: hasText ? "x" : null,
    question_text: null,
    _corrected: corrected,
  };
}

function evaluate(opts: { recognitionNullAs?: string } = {}) {
  const out = {
    corrected: { total: 0, asked: 0, reasons: {} as Record<string, number> },
    owner: { total: 0, askedBefore: 0, askedAfter: 0, reasons: {} as Record<string, number>,
      tierBefore: {} as Record<string, number>, tierAfter: {} as Record<string, number> },
  };
  const bump = (m: Record<string, number>, k: string) => { m[k] = (m[k] ?? 0) + 1; };
  for (const run of runs) {
    const regions = run.rows.map((row, i) => toRegion(row, i, {
      recognitionOverride: row[10] === null && opts.recognitionNullAs ? opts.recognitionNullAs : undefined,
    }));
    const verdicts = judgeRegions(regions, {
      paperUnmarked: paperShownUnmarked({ triage: { classification: run.cls, confidence: run.conf } }),
      fallbackPages: new Set(run.fb ?? []),
      arithmetic: run.rows.map((row) => (row[11] === true || row[11] === false ? row[11] : "unknown")),
    });
    verdicts.forEach((v, i) => {
      const row = run.rows[i];
      if (regions[i]._corrected) {
        out.corrected.total += 1;
        if (v.needs_review) out.corrected.asked += 1;
        for (const r of v.ask) bump(out.corrected.reasons, r);
        if (!v.ask.length) bump(out.corrected.reasons, "not asked");
      }
      if (run.owner) {
        out.owner.total += 1;
        if (row[6]) out.owner.askedBefore += 1;
        if (v.needs_review) out.owner.askedAfter += 1;
        for (const r of v.ask) bump(out.owner.reasons, r);
        bump(out.owner.tierBefore, row[4]);
        bump(out.owner.tierAfter, v.tier);
      }
    });
  }
  return out;
}

const primary = evaluate();
const altered = evaluate({ recognitionNullAs: "high" });
console.log(JSON.stringify({
  rule: "council D1 ask-rule (a) unreadable/low/null recognition, (b) marked paper + missing/impossible mark, (c) unplaceable",
  corrected_regions: primary.corrected,
  owner_regions: primary.owner,
  sensitivity_recognition_null_treated_as_high: { corrected_regions: altered.corrected },
  gate: primary.corrected.asked === primary.corrected.total ? "PASS: every corrected region is asked" : "FAIL: a corrected region is not asked (fall back to option A)",
}, null, 2));
