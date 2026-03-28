import os
import json
import base64
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'exam_layout.pt')
model = None

def get_model():
    global model
    if model is None:
        try:
            from ultralytics import YOLO
            if os.path.exists(MODEL_PATH):
                model = YOLO(MODEL_PATH)
            else:
                print(f"Model not found at {MODEL_PATH}, using fallback")
                model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    return model

@app.route('/detectLayout', methods=['POST'])
def detect_layout():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON provided"}), 400
    
    image_data = data.get('image_data') or data.get('image_url')
    if not image_data:
        return jsonify({"error": "No image data or URL provided"}), 400
    
    try:
        yolo_model = get_model()
        
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
        
        if yolo_model is None:
            return jsonify({"detections": [], "warning": "Model not loaded"}), 200
        
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
        return jsonify({"detections": detections}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detectLayoutBatch', methods=['POST'])
def detect_layout_batch():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON provided"}), 400
    
    image_urls = data.get('image_urls', [])
    if not image_urls:
        return jsonify({"error": "No image URLs provided"}), 400
    
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
                
                if yolo_model is None:
                    results[idx] = []
                    continue
                
                detections = yolo_model(tmp_path, verbose=False)
                results[idx] = _parse_detections(detections)
                os.unlink(tmp_path)
            except Exception as e:
                results[idx] = {"error": str(e)}
        
        return jsonify({"results": results}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.route('/extractText', methods=['POST'])
def extract_text():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON provided"}), 400
    
    image_data = data.get('image_data') or data.get('image_url')
    if not image_data:
        return jsonify({"error": "No image data or URL provided"}), 400
    
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
        
        from google.cloud import vision
        vision_client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        
        texts = [{'text': t.description, 'bounds': str(t.bounding_poly)} 
                 for t in response.text_annotations]
        
        return jsonify({"texts": texts}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyzePdf', methods=['POST'])
def analyze_pdf():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON provided"}), 400
    
    pdf_url = data.get('pdf_url')
    if not pdf_url:
        return jsonify({"error": "No PDF URL provided"}), 400
    
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
        return jsonify({"pages": pages_text}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
