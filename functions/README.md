# Axon Backend - ML Processing Server

This is the backend for YOLO layout detection and PDF processing.

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
   - Start Command: `gunicorn app:app`
5. Add environment variable `MODEL_PATH=exam_layout.pt`
6. Upload model file via Render dashboard

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_PATH | Path to YOLO model | exam_layout.pt |
| PORT | Server port | 8080 |

## API Endpoints

- `POST /detectLayout` - Detect layout in image
- `POST /detectLayoutBatch` - Batch detection
- `POST /analyzePdf` - Extract text from PDF
- `POST /extractText` - OCR text extraction
- `GET /health` - Health check

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
