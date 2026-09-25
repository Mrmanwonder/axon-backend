# Release evidence contract

`npm run release:preflight` does not trust self-reported benchmark counts. It derives coverage and quality metrics from a private, human-reviewed evidence bundle and verifies every artifact against the SHA-256 digests in `certification/release.json`.

Keep the bundle outside Git and point `AXON_RELEASE_EVIDENCE_DIR` at its local directory. Store the immutable copy at the `evidenceUri` recorded in the certification. The checked-in `.gitignore` excludes `certification/evidence/` as an additional safeguard against accidentally committing student data or provider contracts.

## Required artifacts

The `artifacts` object in `release.json` maps these logical names to paths beneath the evidence directory:

- `scannerCases`: JSON Lines, one independently reviewed question per line.
- `tutorCases`: JSON Lines, one independently reviewed tutor response per line.
- `retrievalCases`: JSON Lines, one independently reviewed retrieval decision/result per line.
- `reviewManifest`: approval and digest manifest.
- `geminiZdrEvidence` and `visionZdrEvidence`: contractual zero-data-retention attestations.
- `capabilityProbe`: live Gemini, Tavily, and vision capability-probe result for the certified deployment and config revision.
- `modelComparison`: human-reviewed baseline-versus-candidate results on the certified tutor dataset.
- `rollbackEvidence`: executed rollback-drill result.
- `canaryEvidence`: observed rollout-stage signals.

Artifact paths may not be absolute or escape the evidence directory. Each artifact is limited to 64 MiB during local verification.

## Scanner case fields

Every line requires `paperId`, `questionId`, `classLevel`, `subject`, `teacherId`, `writingStyleId`, `deviceQuality`, `conditions`, `humanReviewed`, `reviewer`, and `reviewedAt`, plus these evaluated booleans:

```text
questionDiscovered
questionNumberCorrect
markAttributionApplicable / markAttributedCorrectly
numericMarkApplicable / numericMarkCorrect
reportedTotalApplicable / reportedTotalCorrect
catastrophicWrongBinding
```

The verifier derives unique paper/question counts, diversity coverage, all required difficult conditions, and the scanner release metrics. It enforces the specification's initial thresholds: at least 100 papers and 1,500 questions; at least 99% question discovery and mark attribution; at least 99.5% numeric-mark, question-number, and reported-total accuracy; and catastrophic wrong binding below 0.05%.

## Tutor case fields

Every line requires `caseId`, `category`, the human-review fields, `factualClaims`, `unsupportedFactualClaims`, `violations`, `abstentionExpected`, `sufficientEvidence`, `abstained`, and this rubric:

```text
answersActualQuestion
factuallyCorrect
appropriateLevel
relevantConceptIdentified
reasoningUnderstandable
unnecessaryInformationAvoided
misconceptionIdentifiedWhenSupported
actionableAdviceWhenAppropriate
```

Use `null` only when a rubric field is genuinely inapplicable. The verifier requires all specified tutor categories, 500 unique hand-reviewed cases, at least 95% on every rubric dimension, at least 95% correct abstention, false abstention below 5%, unsupported-claim escape below 0.1%, and zero P0 hallucination violations.

## Retrieval case fields

Every line requires `caseId`, human-review fields, `retrievalRequired`, `retrievalUsed`, `latencyMs`, `queryQualityScore`, and `fabricatedCitation`. Retrieval-required cases also require `authoritativeSource`, `freshSource`, and `citationCorrect`. Contradiction cases set `contradictionApplicable` and `contradictionHandled`.

The set must contain retrieval-required, retrieval-not-required, and contradictory-source cases. Required retrieval, authority, freshness, citations, and contradiction handling must all pass; unnecessary retrieval and latency remain separately reported.

## Operational attestations

JSON artifacts use these version identifiers:

```text
axon-review-manifest.v1
axon-capability-probe.v1
axon-model-comparison.v1
axon-rollback-evidence.v1
axon-canary-evidence.v1
```

The capability probe must prove the served model is `gemini-3.5-flash-lite` and that Gemini, Tavily, and vision passed for the exact certified deployment SHA and config revision. The model comparison must run the same 500-or-more-case dataset against baseline and candidate, report correctness, unsupported claims, latency, token usage, cost, tool-call rate, false uncertainty, and pedagogy, and prove that the candidate meets the specification's tutoring and latency targets without regressing the protected quality metrics. Rollback evidence must prove health, tutor, document, and trace-provenance checks after moving to a distinct prior revision.

Generate `capability-probe.json` by making an authenticated `POST` request to `/v1/admin/capabilities/probe` on the deployed intelligence Worker. Its response is already the `axon-capability-probe.v1` artifact expected by this verifier. It exercises Gemini structured output and thinking, Tavily search plus extraction against an authoritative source, and the private document-vision service's staged layout-to-targeted-crop path. The synthetic printed page contains no student data. A passing vision result requires positive region/read-group counts and at least two distinct reader identities. Preserve the response exactly; do not hand-edit passing flags. The nested `details` object is diagnostic and contains no prompt, source text, image bytes, or secret.

Set `AXON_RELEASE_TARGET_STAGE` to the stage being promoted. It defaults to `FULL`. Canary evidence must contain every completed predecessor stage in this order:

```text
BENCHMARK → INTERNAL → ONE_PERCENT → FIVE_PERCENT → TWENTY_FIVE_PERCENT → FIFTY_PERCENT → FULL
```

Each stage is rejected if correction rate, mark attribution, unsupported-claim escapes, p95 latency, provider failures, or cost crosses the thresholds implemented by the rollout controller.

The preflight prints the derived scanner, tutor, and retrieval metrics only after all artifact hashes and semantic checks pass.
