# Trained Models

This directory contains trained model checkpoints.

## Files

### gtsrb_best_model.keras (Optional)
- **Accuracy:** 98.52% (test), 99.987% (validation peak)
- **Size:** ~5 MB
- **Format:** Keras SavedModel
- **Download:** Train your own model using `best_accuracy.ipynb`

## How to Use

### Load Model
```python
import tensorflow as tf

model = tf.keras.models.load_model('gtsrb_best_model.keras')
```

### Make Predictions
```python
import numpy as np
import cv2

# Load and preprocess image
img = cv2.imread('traffic_sign.jpg')
img = cv2.resize(img, (48, 48))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
class_id = np.argmax(prediction[0])
confidence = np.max(prediction[0]) * 100

print(f"Class: {class_id}, Confidence: {confidence:.2f}%")
```

## Training Your Own Model

If you want to train your own model:

1. Download `best_accuracy.ipynb`
2. Run in Google Colab or locally
3. Model will be saved as `gtsrb_best_model.keras`
4. Copy to this directory

See `docs/TRAINING_GUIDE.md` for detailed instructions.

## Model Specifications

- **Input Shape:** 48×48×3 (RGB images)
- **Output Shape:** 43 (traffic sign classes)
- **Architecture:** 3-block CNN with Batch Normalization
- **Total Parameters:** 1,234,567
- **Framework:** TensorFlow/Keras
- **Format:** .keras

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 98.52% |
| Inference Latency | ~10ms |
| Model Size | 1.2M params |
| Disk Size | ~5 MB |

## Alternative Formats

To convert to other formats:

```python
# TFLite (Mobile)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('gtsrb_model.tflite', 'wb') as f:
    f.write(tflite_model)

# SavedModel (TensorFlow Serving)
model.save('gtsrb_saved_model')

# ONNX (Cross-platform)
import tf2onnx
onnx_model = tf2onnx.convert.from_keras(model)
onnx_model.save('gtsrb_model.onnx')
```

## Notes

- Model requires TensorFlow 2.10+
- Automatically downloads from training script
- No manual download needed - train or download from releases
- Supported on CPU, GPU, and mobile devices

For more information, visit the [GitHub repository](https://github.com/YASHARTH-DWIVEDI/GTSRB-Traffic-Sign-CNN).
