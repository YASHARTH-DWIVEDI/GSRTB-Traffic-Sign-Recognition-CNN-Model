# 🚀 Deployment Guide

How to deploy the GTSRB CNN model in production.

**Author:** [Yasharth Dwivedi](https://github.com/YASHARTH-DWIVEDI)

---

## Overview

The GTSRB CNN model can be deployed in multiple ways:

| Deployment | Latency | Scalability | Cost |
|-----------|---------|-------------|------|
| **Web API** | ~50ms | High | Medium |
| **Mobile** | ~10ms | Mobile-only | Low |
| **Edge Device** | ~15ms | Low | Low |
| **Cloud** | ~100ms | Very High | Medium-High |
| **Embedded System** | ~20ms | Low | Very Low |

---

## Option 1: Flask Web API

### Create `app.py`
```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('gtsrb_best_model.keras')
CLASS_NAMES = [f"Class {i}" for i in range(43)]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Model is running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess
        image = image.resize((48, 48))
        image_array = np.array(image, dtype='float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        prediction = model.predict(image_array)
        class_id = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100
        
        return jsonify({
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'confidence': f"{confidence:.2f}%",
            'all_probabilities': {
                CLASS_NAMES[i]: float(prediction[0][i]) 
                for i in range(43)
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Handle multiple images
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            image = Image.open(io.BytesIO(file.read()))
            image = image.resize((48, 48))
            image_array = np.array(image, dtype='float32') / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            prediction = model.predict(image_array, verbose=0)
            class_id = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0])) * 100
            
            results.append({
                'filename': file.filename,
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'confidence': f"{confidence:.2f}%"
            })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Run Flask App
```bash
pip install flask pillow
python app.py
```

### Test with curl
```bash
# Single prediction
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict

# Batch prediction
curl -X POST -F "images=@img1.jpg" -F "images=@img2.jpg" \
     http://localhost:5000/batch_predict
```

---

## Option 2: FastAPI (Modern Alternative)

### Create `main.py`
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="GTSRB Traffic Sign Classifier")

# Load model
model = tf.keras.models.load_model('gtsrb_best_model.keras')
CLASS_NAMES = [f"Class {i}" for i in range(43)]

@app.get("/")
async def root():
    return {"message": "GTSRB Traffic Sign Classifier API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((48, 48))
    image_array = np.array(image, dtype='float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array)
    class_id = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0])) * 100
    
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run FastAPI
```bash
pip install fastapi uvicorn
python -m uvicorn main:app --reload
```

### Access Documentation
```
http://localhost:8000/docs
```

---

## Option 3: Docker Deployment

### Create `Dockerfile`
```dockerfile
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app
COPY gtsrb_best_model.keras .
COPY app.py .

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
```

### Create `.dockerignore`
```
*.csv
*.zip
Train/
Test/
__pycache__
.git
.gitignore
```

### Build Docker Image
```bash
docker build -t gtsrb-cnn:latest .
```

### Run Docker Container
```bash
docker run -p 5000:5000 gtsrb-cnn:latest
```

### Test Container
```bash
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
```

---

## Option 4: Google Cloud Deployment

### 1. Create `main.py` for Cloud Functions
```python
import functions_framework
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from werkzeug.datastructures import FileStorage

model = tf.keras.models.load_model('gtsrb_best_model.keras')
CLASS_NAMES = [f"Class {i}" for i in range(43)]

@functions_framework.http
def predict_traffic_sign(request):
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((48, 48))
        image_array = np.array(image, dtype='float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = model.predict(image_array)
        class_id = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100
        
        return {
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'confidence': f"{confidence:.2f}%"
        }
    except Exception as e:
        return {'error': str(e)}, 400
```

### 2. Deploy to Cloud Functions
```bash
gcloud functions deploy predict-traffic-sign \
  --runtime python39 \
  --trigger-http \
  --allow-unauthenticated
```

---

## Option 5: AWS Lambda

### Create Lambda Handler
```python
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import base64

model = tf.keras.models.load_model('gtsrb_best_model.keras')
CLASS_NAMES = [f"Class {i}" for i in range(43)]

def lambda_handler(event, context):
    try:
        # Get image from base64
        image_data = base64.b64decode(event['image'])
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((48, 48))
        image_array = np.array(image, dtype='float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        prediction = model.predict(image_array)
        class_id = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'confidence': f"{confidence:.2f}%"
            })
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
```

---

## Option 6: Mobile Deployment (TFLite)

### Convert Model
```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('gtsrb_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Android Implementation
```kotlin
// Load TFLite model
val tfliteModel = FileUtil.loadMappedFile(context, "gtsrb_model.tflite")
val options = Interpreter.Options()
val interpreter = Interpreter(tfliteModel, options)

// Prepare input
val inputBitmap = // ... prepare 48x48 image
val pixelValues = IntArray(48 * 48)
for (i in 0 until 48 * 48) {
    pixelValues[i] = (inputBitmap.getPixel(i % 48, i / 48) shr 16) and 0xFF
}
val input = Array(1) { FloatArray(48 * 48 * 3) }
for (i in 0 until 48 * 48 * 3) {
    input[0][i] = pixelValues[i / 3].toFloat() / 255.0f
}

// Run inference
val output = Array(1) { FloatArray(43) }
interpreter.run(input, output)

// Get result
val classId = output[0].withIndex().maxByOrNull { it.value }?.index ?: 0
```

### iOS Implementation
```swift
import CoreML
import Vision

// Load model
let model = try gtsrb_best_modelV1(configuration: .init())

// Prepare input
let pixelBuffer = // ... prepare CVPixelBuffer (48x48)
let request = VNCoreMLRequest(model: try VNCoreMLModel(for: model.model)) { 
    request, error in
    guard let results = request.results as? [VNClassificationObservation] else { return }
    
    let topResult = results.first!
    print("Class: \(topResult.identifier), Confidence: \(topResult.confidence)")
}

let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
try handler.perform([request])
```

---

## Option 7: Edge Deployment (Raspberry Pi)

### Setup Raspberry Pi
```bash
# Install TensorFlow Lite
pip install --index-url https://google-coral.github.io/py-repo/ tflite-runtime

# Or full TensorFlow (slower)
pip install tensorflow
```

### Inference Script
```python
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# Load TFLite model
interpreter = tflite.Interpreter(model_path="gtsrb_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read image
image = cv2.imread('traffic_sign.jpg')
image = cv2.resize(image, (48, 48))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype('float32') / 255.0

# Reshape to match input
input_data = np.expand_dims(image, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get prediction
class_id = np.argmax(output_data[0])
confidence = np.max(output_data[0]) * 100
print(f"Class: {class_id}, Confidence: {confidence:.2f}%")
```

---

## Performance Optimization

### Model Quantization
```python
# Integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized = converter.convert()

# Save quantized model
with open('gtsrb_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized)
```

### Model Pruning
```python
# Prune non-important weights
import tensorflow_model_optimization as tfmot

pruning_schedule = tfmot.sparsity.keras.PruningSchedule.PolynomialDecay(
    initial_sparsity=0.36,
    final_sparsity=0.80,
    begin_step=0,
    end_step=end_step
)

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    pruning_schedule=pruning_schedule
)
```

---

## Monitoring in Production

### Health Check
```python
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'gtsrb_cnn',
        'accuracy': '98.52%',
        'version': '1.0'
    }), 200
```

### Logging Predictions
```python
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_prediction(class_id, confidence, timestamp):
    logger.info(json.dumps({
        'class_id': class_id,
        'confidence': confidence,
        'timestamp': timestamp
    }))
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram

predictions_counter = Counter(
    'traffic_sign_predictions_total',
    'Total predictions made'
)

prediction_latency = Histogram(
    'traffic_sign_prediction_latency_seconds',
    'Prediction latency'
)

@prediction_latency.time()
def make_prediction(image):
    predictions_counter.inc()
    # ... prediction logic
```

---

## Security Considerations

### Input Validation
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = ['jpg', 'jpeg', 'png']

def validate_image(file):
    if file.size > MAX_FILE_SIZE:
        raise ValueError('File too large')
    
    ext = file.filename.split('.')[-1].lower()
    if ext not in ALLOWED_FORMATS:
        raise ValueError('Invalid file format')
    
    return True
```

### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... prediction logic
```

### Authentication
```python
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != SECRET_API_KEY:
            return {'error': 'Unauthorized'}, 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... prediction logic
```

---

## Scaling Strategies

### Horizontal Scaling
- Deploy multiple instances
- Use load balancer
- Container orchestration (Kubernetes)

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def get_prediction(image_hash):
    # Cache frequently requested predictions
    pass
```

### Batch Processing
```python
# Process multiple images at once
def batch_predict(images):
    # Stack images
    batch = np.vstack(images)
    
    # Single inference pass
    predictions = model.predict(batch)
    
    # Much faster than individual predictions
    return predictions
```

---

## Next Steps

1. Choose deployment option based on requirements
2. Follow setup instructions
3. Test with sample images
4. Monitor performance
5. Scale as needed

---

**Author:** [Yasharth Dwivedi](https://github.com/YASHARTH-DWIVEDI)
**GitHub:** https://github.com/YASHARTH-DWIVEDI/GTSRB-Traffic-Sign-CNN
