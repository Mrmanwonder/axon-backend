# AXON document vision contract

The Worker calls the configured `AXON_VISION_API_BASE` only when `AXON_VISION_PRIVACY_MODE=zdr` and `AXON_VISION_TOKEN` is present. The service must not retain request bodies, use them for training, or log raw images. It receives JSON with `contractVersion: "axon-document-vision.v1"`, `pageId`, `mimeType`, and `dataBase64`.

The response is schema-validated and bounded to 32 MB. It must include normalized quality metrics, orientation, layout regions, multi-signal ink features, and one or more recognition reads per textual region. Two genuinely independent reader IDs are required for automatic trust. Conditioning may return `conditionedImageBase64`; AXON stores it as a separate immutable R2 artifact linked to the original hash. Generative enhancement is forbidden.

The provider does not decide truth, marks, question ownership, or insight eligibility. Those decisions remain in the Worker: quality policy, question graph construction, ink classification, global mark matching, read reconciliation, confidence calculation, and trusted-field state are deterministic owners.

If the service is missing, privacy is unattested, output fails schema validation, or evidence is ambiguous, the page is moved to `REVIEW_REQUIRED`. The Worker never substitutes the tutor model or a less private endpoint.
