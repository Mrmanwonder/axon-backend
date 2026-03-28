import os
import json
import base64
import tempfile
from pathlib import Path

import functions_framework
from firebase_admin import initialize_app, storage, firestore
from google.cloud import vision
from google.cloud.vision_v1 import types
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO

initialize_app()
db = firestore.client()
bucket = storage.bucket()

MODEL_PATH = os.environ.get('MODEL_PATH', '/tmp/exam_layout.pt')
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            blob = bucket.blob('models/exam_layout.pt')
            if blob.exists():
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    blob.download_to_filename(tmp.name)
                    model = YOLO(tmp.name)
            else:
                raise Exception("Model not found")
    return model

@functions_framework.http
def detect_layout(request):
    request_json = request.get_json(silent=True)
    if not request_json:
        return {"error": "No JSON provided"}, 400
    
    image_data = request_json.get('image_data') or request_json.get('image_url')
    if not image_data:
        return {"error": "No image data or URL provided"}, 400
    
    try:
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        elif image_data.startswith('http'):
            import requests
            image_bytes = requests.get(image_data).content
        else:
            with open(image_data, 'rb') as f:
                image_bytes = f.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        yolo_model = get_model()
        results = yolo_model(tmp_path, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detections.append({
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'x': float(box.xywh[0][0]),
                    'y': float(box.xywh[0][1]),
                    'width': float(box.xywh[0][2]),
                    'height': float(box.xywh[0][3]),
                })
        
        os.unlink(tmp_path)
        return {"detections": detections}, 200
    
    except Exception as e:
        return {"error": str(e)}, 500

@functions_framework.http
def detect_layout_batch(request):
    request_json = request.get_json(silent=True)
    if not request_json:
        return {"error": "No JSON provided"}, 400
    
    image_urls = request_json.get('image_urls', [])
    if not image_urls:
        return {"error": "No image URLs provided"}, 400
    
    results = {}
    try:
        yolo_model = get_model()
        
        for idx, url in enumerate(image_urls):
            try:
                import requests
                image_bytes = requests.get(url).content
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name
                
                detections = yolo_model(tmp_path, verbose=False)
                results[idx] = _parse_detections(detections)
                os.unlink(tmp_path)
            except Exception as e:
                results[idx] = {"error": str(e)}
        
        return {"results": results}, 200
    
    except Exception as e:
        return {"error": str(e)}, 500

def _parse_detections(results):
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detections.append({
                'class': int(box.cls[0]),
                'confidence': float(box.conf[0]),
                'x': float(box.xywh[0][0]),
                'y': float(box.xywh[0][1]),
                'width': float(box.xywh[0][2]),
                'height': float(box.xywh[0][3]),
            })
    return detections

@functions_framework.http
def extract_text(request):
    request_json = request.get_json(silent=True)
    if not request_json:
        return {"error": "No JSON provided"}, 400
    
    image_data = request_json.get('image_data') or request_json.get('image_url')
    if not image_data:
        return {"error": "No image data or URL provided"}, 400
    
    try:
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        elif image_data.startswith('http'):
            import requests
            image_bytes = requests.get(image_data).content
        else:
            with open(image_data, 'rb') as f:
                image_bytes = f.read()
        
        vision_client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        
        texts = [{'text': t.description, 'bounds': t.bounding_poly} 
                 for t in response.text_annotations]
        
        return {"texts": texts}, 200
    
    except Exception as e:
        return {"error": str(e)}, 500

@functions_framework.http
def analyze_pdf(request):
    request_json = request.get_json(silent=True)
    if not request_json:
        return {"error": "No JSON provided"}, 400
    
    pdf_url = request_json.get('pdf_url')
    if not pdf_url:
        return {"error": "No PDF URL provided"}, 400
    
    try:
        import requests
        from PyPDF2 import PdfReader
        
        response = requests.get(pdf_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        reader = PdfReader(tmp_path)
        pages_text = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages_text.append({'page': i + 1, 'text': text})
        
        os.unlink(tmp_path)
        return {"pages": pages_text}, 200
    
    except Exception as e:
        return {"error": str(e)}, 500

@functions_framework.http
def health(request):
    return {"status": "healthy", "torch_available": True}, 200
