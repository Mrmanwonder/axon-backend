# AXO-223: prompt evidence recovery and validation

Read-only audit on 2026-10-09. Backend base: afd24420bcf576a799ce789c9e45fc9ab4af7d21.

The prompt manifest had two topic-tag keys. JSON.parse selected the later bare declared-pass record (e45f9a23-4562-4321-9876-123456789abc), masking the earlier measured 50-case run. No row for that placeholder or the all-zero ID from rejected PR #171 exists in public.eval_run. The manifest now references the measured run once; the placeholder artifact is explicitly invalidated and cannot gate a prompt.

| Stage | Actual run | Completed / distinct / schema-valid cases | Ordered result SHA-256 |
| --- | --- | --- | --- |
| topic_tag | 7ca49ba7-6f11-4d11-b212-416d7d1987d1 | 50 / 50 / 50 | 5f25b720046fd1322f65ccd2f1cb11cdd8df53ac9e1bce0c3156ba310b504e07 |
| scheme_check | c7a66b34-3c94-4609-9fe4-a8917fb8b2c5 | 30 / 30 / 30 | 50d0904d5f1070d500356f1a7cfb4c9c87293f3aee425a9b1ca2d0b8ce04c1df |

Production records show both completed runs passed their stored stage thresholds on gemini-3.8-flash. All 50 topic-tag and 30 scheme-check cases have owner-confirmed labels dated 2026-10-06. The cases are synthetic questions, not a frozen real-paper extraction corpus. Existing historical notes about labels at run time are retained; later confirmation is appended. No prompt, route, model or production database row changes.

The gate rejects duplicate target JSON keys, zero/mismatched IDs, bare passed declarations, missing model/corpus/date/case/call/threshold metadata and missing exact-run per-case provenance. It validates measured artifacts even when a prompt hash is unchanged, so a run-reference-only edit cannot bypass checks. Existing unchanged dated baselines remain explicitly unevaluated, not certified. Draft labels still cannot gate extraction release.

## Limits and reproduction

Structural validation and a digest do not prove the truth of a deliberately fabricated record. A reviewer must independently verify the actual run, outputs, labels and accepted thresholds. The supplied read-only SQL reproduces the database counts/digests without returning question or student text. The ordered digest covers case ID, candidate, model, prompt version, status, schema validity and output. It does not turn these stage runs into cross-curriculum academic release certification. Real-paper evaluation and release sign-off remain AXO-13/44.
