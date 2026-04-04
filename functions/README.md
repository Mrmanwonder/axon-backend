# Axon Backend - FastAPI AI Service

This is the FastAPI backend for YOLO layout detection, authenticated PDF jobs, handwritten grading, and public leaderboard mirroring.

## Quick Deploy to Railway

1. **Install Railway CLI:**
```bash
npm install -g @railway/cli
```

2. **Login:**
```bash
railway login
```

3. **Initialize Project:**
```bash
cd functions
railway init
```

4. **Add Model File:**
Upload `assets/models/exam_layout.pt` to Railway Files:
```bash
railway variables set MODEL_PATH=exam_layout.pt
```
Then upload the model file via Railway dashboard.

5. **Deploy:**
```bash
railway up
```

## Alternative: Deploy to Render

1. Create account at [render.com](https://render.com)
2. Connect GitHub repository
3. Create new Web Service
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -k uvicorn.workers.UvicornWorker app:app --workers 2 --timeout 120`
5. Add environment variable `MODEL_PATH=exam_layout.pt`
6. Upload model file via Render dashboard

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_PATH | Path to YOLO model | exam_layout.pt |
| PORT | Server port | 8080 |
| FIRESTORE_DATABASE_ID | Firestore database ID used by the app | axon |
| GOOGLE_APPLICATION_CREDENTIALS | Path to Firebase Admin service account JSON | required for auth / Firestore jobs |

## API Endpoints

- `POST /detectLayout` - Detect layout in one image
- `POST /detectLayoutBatch` - Detect layout across multiple images with per-item status/error reporting
- `POST /extractText` - OCR text extraction through Google Vision
- `POST /analyzePdf` - Queue an authenticated async PDF job (`pdf_url` or `pdf_base64`)
- `GET /jobs/{job_id}` - Read async PDF job status/result
- `POST /syncLeaderboardProfile` - Mirror safe fields into `users_public`
- `POST /generate/grade-handwriting` - Grade a handwritten response against a marking scheme
- `POST /generate-daily-plan` - Build and persist today's study blocks into `users_private/{uid}/daily_plan`
- `POST /analyze-study-pulse` - Compute readiness, exam risk, and advisor insights into `users_private/{uid}/analytics/current`
- `POST /sync-exam-dates` - Scrape official board datesheets and persist matching deadlines into `users_private/{uid}/deadlines`
- `GET /health` - Health check

Swagger docs are available at `/docs` once deployed.

## Batch Layout Detection

`POST /detectLayoutBatch` accepts up to 24 images per request and returns a result envelope for each item.

Example payload:
```json
{
  "items": [
    {
      "item_id": "page-1",
      "image_url": "https://example.com/page-1.jpg"
    },
    {
      "item_id": "page-2",
      "image_data": "data:image/jpeg;base64,/9j/4AAQSk..."
    }
  ]
}
```

## Official Exam Date Sync

`POST /sync-exam-dates` uses official board source pages and datesheet PDFs to populate `users_private/{uid}/deadlines`.

Example payload:
```json
{
  "board": "IGCSE",
  "subjects": ["Physics", "Mathematics"],
  "year": 2026,
  "series": "may",
  "administrative_zone": "zone4",
  "persist": true
}
```

Notes:
- `administrative_zone` matters most for Cambridge-style boards. If you omit it, the scraper may ingest dates from multiple official timetable variants.
- The planner already reads from `users_private/{uid}/deadlines`, so this route becomes the source of truth for exam anchors.

Example response shape:
```json
{
  "owner_uid": "abc123",
  "source_type": "MODEL_INFERENCE_BATCH",
  "item_count": 2,
  "results": [
    {
      "index": 0,
      "item_id": "page-1",
      "status": "completed",
      "detections": []
    },
    {
      "index": 1,
      "item_id": "page-2",
      "status": "failed",
      "detections": [],
      "error": "..."
    }
  ]
}
```

## Firebase Admin Setup

The backend now verifies Firebase bearer tokens and writes job status into Firestore. Before deployment:

1. Create a Firebase service account with Firestore access.
2. Download the JSON key.
3. Set `GOOGLE_APPLICATION_CREDENTIALS` to the absolute path of that JSON file.
4. Set `FIRESTORE_DATABASE_ID=axon` if your project uses the same custom Firestore database as the Flutter app.

Example local run:
```bash
cd functions
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:FIRESTORE_DATABASE_ID="axon"
$env:GEMINI_API_KEY="your-gemini-key"
uvicorn app:app --reload --port 8080
```

## Legacy User Migration

To move legacy `/users/{uid}` data into `users_private/{uid}` and seed `users_public/{uid}`:

Dry run:
```bash
cd functions
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:FIRESTORE_DATABASE_ID="axon"
python migrate_legacy_users.py --dry-run
```

Execute:
```bash
cd functions
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:FIRESTORE_DATABASE_ID="axon"
python migrate_legacy_users.py
```

## Seed `syllabus_maps`

The handwriting grader and mastery engine need real objective metadata in Firestore. Seed the initial graph before relying on grading output:

Dry run:
```bash
cd functions
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:FIRESTORE_DATABASE_ID="axon"
python seed_syllabus_maps.py --dry-run
```

Execute:
```bash
cd functions
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:FIRESTORE_DATABASE_ID="axon"
python seed_syllabus_maps.py
```

## After Deployment

Update the backend URL in Flutter:
```dart
// In lib/services/layout_detection_service.dart
void setBackendUrl(String url) {
  _backendUrl = 'https://your-app.railway.app';
}
```

Initialize in main.dart:
```dart
void main() {
  LayoutDetectionService().setBackendUrl('https://your-app.railway.app');
  runApp(const MyApp());
}
```

## Migration Note

The legacy Flask/Functions Framework path has been retired. `functions/main.py` now re-exports the FastAPI app so there is one backend runtime, not two divergent implementations.
