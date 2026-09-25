# Correction learning contract

`POST /v1/corrections` preserves the prediction, correction, and accepted value in the immutable `student_correction` record. It also creates an `active_learning_queue` item and routes the event into one or more private learning targets:

- `BENCHMARK_EXPANSION`
- `CONFIDENCE_RECALIBRATION`
- `LAYOUT_TRAINING`
- `HTR_DATASET`
- `PROMPT_REGRESSION`
- `ERROR_CLUSTERING`

Targets are metadata pointers to the correction. They do not duplicate raw student values. Error-cluster rows contain a one-way signature of bounded categorical metadata, an allowlisted field category, counts, and timestamps only; an unknown caller-supplied field name becomes `other`. Calibration rows contain the submitted system confidence, its decile, and whether the prediction matched the accepted value; they contain no answer text or image data.

Human review remains mandatory. Target states follow the parent item from `QUEUED` to `LABELLED`; a correction becomes `READY` only after the reviewer supplies an evidence URI and a passing evaluation-run ID. Rejection marks every target `REJECTED`. `READY` means eligible for a controlled private export—it does not train or modify any production model automatically.

The admin-only read endpoints are:

- `GET /v1/admin/learning/targets`
- `GET /v1/admin/learning/error-clusters`
- `GET /v1/admin/learning/calibration`

The first endpoint accepts bounded `status`, `target`, and `limit` filters. The other endpoints expose aggregate triage/calibration data. None returns predicted, corrected, or accepted values.
