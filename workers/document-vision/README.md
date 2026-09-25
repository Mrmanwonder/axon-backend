# AXON document vision

Private, service-binding-only document analysis for the AXON Intelligence Worker.

The Worker accepts JPEG, PNG, and WebP `axon-document-vision.v1` page requests from `axon-intelligence`, bounds and verifies encoded image bytes before decoding, and computes deterministic pixel-quality signals. Document reading is staged: Workers AI Moondream 3.1 receives the page once for layout and OCR-first printed-text discovery, then Gemini 3.5 Flash-Lite receives only padded WebP crops for readable regions, with the proposed layer, page-space box, nearby OCR context, and quality metadata. The targeted reader never receives the complete page. Calls are bounded to 120 regions with concurrency four; unprocessed readable regions remain without a recognition group and are therefore routed to review downstream.

Printed regions begin with Moondream OCR and use the targeted reader as adjudication before authoritative ingestion. Handwriting, teacher annotations, and marks use the same crop-only second reader as an independent semantic/HTR check. A region is only emitted at high confidence when both readers agree on class, layer, and normalized text. Disagreements remain as separate reads for deterministic reconciliation and review; the system never invents a consensus. PDF and HEIC originals remain immutable but are routed to review until a separately benchmarked, non-generative normalization path is certified.

Inference is disabled unless both `GEMINI_PRIVACY_MODE` and `WORKERS_AI_PRIVACY_MODE` equal `zdr`. Those flags represent reviewed contractual evidence; they must not be changed merely because a provider publishes a general data-use statement. The release preflight separately requires the signed attestations.

The Worker has no public `workers.dev` endpoint, preview URL, or route. It must be deployed before `axon-intelligence`, because the latter binds it as `DOCUMENT_VISION`. Its only secret is `GOOGLE_API_KEY`; Workers AI is an account binding. The private capability probe uses a non-student synthetic printed page and passes only when it observes at least one recognition group containing both distinct reader identities.
