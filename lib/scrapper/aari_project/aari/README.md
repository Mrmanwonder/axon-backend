# AARI — Axon Autonomous Resource Intelligence

> Multi-stage, asynchronous educational content crawler for IB, IGCSE, and A-Levels (CAIE / Edexcel).  
> Designed to feed the **Axon ResourceCrawlerService** via idempotent Firestore batch uploads.

---

## Architecture Overview

```
subjects.yaml
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Discovery                                    │
│  ┌──────────────────┐   ┌──────────────────────────┐   │
│  │ OfficialScraper  │   │  CommunityRepoScraper     │   │
│  │ (CAIE/IBO/Pearson│   │  (PapaCambridge/GCEGuide) │   │
│  │  Deep-Tree DFS)  │   │  + Self-Healing Resolver  │   │
│  └────────┬─────────┘   └─────────────┬────────────┘   │
│           └──────────────┬────────────┘                 │
│                          ▼                              │
│               DeduplicationEngine                       │
│               (URL-level + SHA-256)                     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — Download & Validate                          │
│  ResourceDownloader  (bounded async concurrency)        │
│  └─► PDFValidator    (magic-byte + structural check)    │
│  └─► QualityScorer   (0–100 composite score)            │
│  └─► DeduplicationEngine.check_and_register_content()  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 — Export                                       │
│  ExportEngine                                           │
│  ├── output/batch_upload.json   (Firestore payload)     │
│  └── output/manifest.lock       (idempotent sync)       │
└─────────────────────────────────────────────────────────┘
```

---

## Project Layout

```
aari/
├── config.py               — Sources, crawl tuning, field names
├── __main__.py             — CLI entrypoint (yaml + inline modes)
├── main.py                 — AARIPipeline orchestrator
├── subjects.yaml           — Subject/board/qualification config
├── demo.py                 — Generates sample output without live HTTP
│
├── models/
│   └── resource.py         — AARIResource, CrawlSession, ManifestEntry (Pydantic v2)
│
├── core/
│   ├── dedup.py            — SHA-256 content deduplication engine
│   ├── polite.py           — robots.txt gate, per-domain rate limiter, retry
│   ├── healer.py           — Self-healing URL resolver (fuzzy DOM matching)
│   └── downloader.py       — Async concurrent file downloader
│
├── scrapers/
│   ├── official.py         — Deep-Tree DFS over CAIE / IBO / Edexcel
│   └── community.py        — PapaCambridge / GCE Guide crawler
│
├── validators/
│   ├── pdf_validator.py    — PDF magic-byte + pypdf structural check
│   └── quality.py          — 0–100 quality_score compositor
│
├── exporters/
│   └── export.py           — batch_upload.json + manifest.lock writer
│
└── tests/
    └── test_core.py        — 18 unit tests (all passing)
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: YAML config support
pip install pyyaml
```

**requirements.txt**
```
httpx[http2]>=0.27.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
pydantic>=2.7.0
pypdf>=4.2.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

---

## Usage

### YAML config (recommended for production runs)

```bash
python -m aari --config subjects.yaml
```

Edit `subjects.yaml` to control which boards, subjects, and years are crawled.

### Inline subjects

```bash
python -m aari \
  --subjects "0580:Mathematics:IGCSE:CAIE" "9709:Mathematics:A-Level:CAIE" \
  --years 2023 2024 2025 2026
```

### Dry-run (discovery only, no file downloads)

```bash
python -m aari --config subjects.yaml --dry-run
```

### Demo (no network required)

```bash
python demo.py
```

Generates `output/batch_upload.json` and `output/manifest.lock` from synthetic
data so you can inspect the exact Firestore schema before running a live crawl.

---

## Output Files

### `output/batch_upload.json`

Firestore-ready batch payload for `ResourceCrawlerService`:

```json
{
  "meta": {
    "session_id": "uuid4",
    "generated_at": "2026-04-02T04:45:23Z",
    "total": 106,
    "new": 100,
    "dupes": 4,
    "errors": 0
  },
  "resources": [
    {
      "uid": "acf7952c-...",
      "board": "CAIE",
      "qualification": "IGCSE",
      "subject_code": "0580",
      "subject_name": "Mathematics",
      "resource_type": "Past Paper",
      "year": 2024,
      "session": "May-Jun",
      "paper_number": "1",
      "variant": "2",
      "source_type": "Official",
      "source_url": "https://cambridgeinternational.org/...",
      "file_url": "https://cambridgeinternational.org/.../0580_24_may_jun_qp_12.pdf",
      "sha256": "d340bfd6b80a7527...",
      "file_size_bytes": 1482240,
      "verified_status": "verified",
      "quality_score": 87,
      "crawled_at": "2026-04-02T04:45:23.665913+00:00"
    }
  ]
}
```

### `output/manifest.lock`

Idempotent re-sync manifest. `ResourceCrawlerService` skips any entry where
`synced: true` and the SHA-256 has not changed since last write:

```json
{
  "lock_version": "1.0",
  "session_id": "uuid4",
  "locked_at": "2026-04-02T04:45:23Z",
  "total_entries": 106,
  "entries": {
    "acf7952c-...": {
      "uid": "acf7952c-...",
      "sha256": "d340bfd6b80a7527...",
      "file_url": "https://...",
      "resource_type": "Past Paper",
      "crawled_at": "2026-04-02T04:45:23Z",
      "synced": false
    }
  }
}
```

After `ResourceCrawlerService` uploads a batch, it should PATCH `synced: true`
on each confirmed document to prevent re-upload on the next run.

---

## Firestore Schema

Each resource maps to `/resources/{uid}`:

| Field            | Type      | Description                                      |
|------------------|-----------|--------------------------------------------------|
| `uid`            | `string`  | UUID4, server-authoritative                      |
| `board`          | `string`  | `CAIE` \| `IBO` \| `Edexcel`                    |
| `qualification`  | `string`  | `IGCSE` \| `A-Level` \| `IB-DP` \| …            |
| `subject_code`   | `string`  | Board subject code (e.g. `0580`, `9709`)         |
| `subject_name`   | `string`  | Human-readable subject name                      |
| `resource_type`  | `string`  | `Past Paper` \| `Mark Scheme` \| `Examiner Report` \| `Syllabus` \| `Datesheet` |
| `year`           | `integer` | Exam year                                        |
| `session`        | `string?` | `Feb-Mar` \| `May-Jun` \| `Oct-Nov`              |
| `paper_number`   | `string?` | `"1"` – `"6"`                                    |
| `variant`        | `string?` | `"1"` – `"3"`                                    |
| `source_type`    | `string`  | `Official` \| `Community`                        |
| `source_url`     | `string`  | Page where resource was discovered               |
| `file_url`       | `string`  | Direct PDF link                                  |
| `sha256`         | `string`  | 64-char hex digest for integrity checks          |
| `file_size_bytes`| `integer` | Raw file size                                    |
| `verified_status`| `string`  | `verified` \| `unverified` \| `flagged`          |
| `quality_score`  | `integer` | 0–100 composite score (see below)                |
| `crawled_at`     | `string`  | ISO-8601 UTC timestamp                           |

### Quality Score Breakdown (max 100)

| Component              | Weight | Condition                                |
|------------------------|--------|------------------------------------------|
| Official source        | 40     | `source_type == Official`                |
| Paired Mark Scheme     | 20     | MS for same subject/year/session exists  |
| Paired Examiner Report | 15     | ER for same subject/year exists          |
| Recency                | 15     | Normalised over 4-year window            |
| PDF structural quality | 10     | pypdf page count + metadata presence     |

---

## Core Design Principles

### Ethics & Compliance
- Every domain is checked against `robots.txt` before any request is dispatched
- The User-Agent is declared honestly: `AARI-EducationalBot/1.0`
- Per-domain delays are randomised between 1.5–4 seconds (courtesy, not evasion)
- No WAF bypass, no credential stuffing, no authenticated-zone access
- PDF files are validated structurally — content is never altered

### Self-Healing Resolver
When a known URL returns 404, `core/healer.py` walks up the parent directory
tree (up to 3 levels), parses all anchor links from the parent page, and uses
`difflib.SequenceMatcher` to find the best candidate by fuzzy-matching the
filename hint against anchor text + href strings. This handles routine
directory restructuring on official boards without manual intervention.

### Deduplication
Two-level dedup prevents redundant downloads:
1. **URL-level** — exact URL seen this session or in a previous cached run
2. **Content-level** — SHA-256 digest matches a previously downloaded file
   (catches mirrors and re-uploads of identical content)

The seen-sets are persisted to `.cache/dedup_cache.json` between runs,
enabling true incremental crawls.

---

## Running Tests

```bash
pytest aari/tests/ -v
# Expected: 18 passed in < 1s
```

---

## Extending AARI

### Add a new official source

1. Add an entry to `OFFICIAL_SOURCES` in `config.py`
2. Map its `board` value to a `Board` enum member in `main.py`
3. (Optional) Override `_infer_resource_type()` in `scrapers/official.py` if
   the new board uses non-standard filename conventions

### Add a new community source

1. Add to `COMMUNITY_SOURCES` in `config.py` with a `board_map`
2. The `CommunityRepoScraper` will pick it up automatically on next run

### Custom quality weights

Edit `QUALITY_WEIGHTS` in `config.py`. The scorer normalises automatically so
individual weights can be tuned without breaking the 0–100 ceiling.
