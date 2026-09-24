# AXON document vision

Private, service-binding-only document analysis for the AXON Intelligence Worker.

The Worker accepts JPEG, PNG, and WebP `axon-document-vision.v1` page requests from `axon-intelligence`, bounds and verifies encoded image bytes before decoding, computes deterministic pixel-quality signals, and reconciles two distinct readers (Gemini 3.5 Flash-Lite for handwriting semantics and Workers AI Moondream 3.1 for layout/printed OCR). A region is only emitted at high confidence when class and geometry agree; content is only copied onto the region when both readers return the same normalized text. Conflicts remain separate reads for the Intelligence Worker to route to review. PDF and HEIC originals remain immutable but are routed to review until a separately benchmarked, non-generative normalization path is certified.

Inference is disabled unless both `GEMINI_PRIVACY_MODE` and `WORKERS_AI_PRIVACY_MODE` equal `zdr`. Those flags represent reviewed contractual evidence; they must not be changed merely because a provider publishes a general data-use statement. The release preflight separately requires the signed attestations.

The Worker has no public `workers.dev` endpoint, preview URL, or route. It must be deployed before `axon-intelligence`, because the latter binds it as `DOCUMENT_VISION`. Its only secret is `GOOGLE_API_KEY`; Workers AI is an account binding.
