# 🎓 How to Train the GTSRB CNN Model

Complete guide for training the German Traffic Sign Recognition CNN.

**Author:** [Yasharth Dwivedi](https://github.com/YASHARTH-DWIVEDI)

---

## Option 1: Google Colab (Recommended)

### Step 1: Prepare
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "Upload notebook"
3. Select `best_accuracy.ipynb`

### Step 2: Set Runtime to GPU
1. Click "Runtime" menu
2. Select "Change runtime type"
3. Choose **GPU (T4)**
4. Click "Save"

### Step 3: Get Kaggle API Key
1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Click "Create New API Token"
3. Download `kaggle.json`

### Step 4: Run Notebook
1. When prompted, upload `kaggle.json`
2. Run all cells in order
3. Wait ~19 minutes for training
4. Download trained model

---

## Option 2: Local Machine with GPU

### Step 1: Install NVIDIA CUDA
```bash
# For Ubuntu
sudo apt-get install cuda-11-8
sudo apt-get install cudnn

# For Windows
# Download from NVIDIA website and follow installer
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get Kaggle API Key
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json  # On Mac/Linux only
```

### Step 5: Run Notebook
```bash
jupyter notebook best_accuracy.ipynb
```

---

## Option 3: AWS SageMaker

### Step 1: Create Notebook Instance
1. Go to AWS SageMaker
2. Create new notebook instance
3. Choose `ml.p3.2xlarge` (GPU instance)

### Step 2: Upload Notebook
- Upload `best_accuracy.ipynb`
- Upload `requirements.txt`

### Step 3: Install Dependencies
```bash
!pip install -r requirements.txt
```

### Step 4: Run Training
- Follow notebook cells
- Training will be faster on AWS GPU

---

## Common Issues & Solutions

### Issue 1: Out of Memory (OOM)

**Error:** `ResourceExhaustedError: OOM when allocating...`

**Solution:** Reduce batch size
```python
BATCH_SIZE = 32  # From 64
# Or further reduce:
BATCH_SIZE = 16
```

### Issue 2: Kaggle Authentication Failed

**Error:** `ClientError: 401 - Unauthorized`

**Solution:** Re-upload kaggle.json
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Issue 3: Slow Training

**Cause:** Using CPU instead of GPU

**Solution:** Verify GPU usage
```python
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")
```

### Issue 4: Image Size Mismatch

**Error:** `ValueError: input_shape mismatch`

**Solution:** Ensure input shape is (48, 48, 3)
```python
layers.Input(shape=(48, 48, 3))  # Must be 48×48, not 128×128
```

### Issue 5: Model Not Saving

**Solution:** Check disk space and permissions
```bash
# Check disk space
df -h

# Check permissions
ls -la gtsrb_best_model.keras
```

---

## Training Progression Expectations

### Epoch 1-5: Rapid Learning
```
Epoch 1: ~40% accuracy, loss ~2.3
Epoch 2: ~93% accuracy, loss ~0.24
Epoch 3: ~97% accuracy, loss ~0.09
Epoch 4: ~98% accuracy, loss ~0.05
Epoch 5: ~99% accuracy, loss ~0.05
```

### Epoch 6-20: Steady Improvement
```
Epoch 10: ~99% training, ~99.5% validation
Epoch 15: ~99.4% training, ~99.6% validation
Epoch 20: ~99.4% training, ~99.8% validation
```

### Epoch 21-33: Fine-tuning & Plateau
```
Epoch 23: ~99.6% training, ~99.987% validation (BEST)
Epoch 25-30: Slight fluctuations, learning rate reduced
Epoch 33: Early stopping triggered, model restored to Epoch 23
```

---

## Customization Options

### Adjust Batch Size
```python
BATCH_SIZE = 32  # Default: 64
# Smaller = slower but more stable
# Larger = faster but more memory needed
```

### Adjust Learning Rate
```python
optimizer=keras.optimizers.Adam(learning_rate=0.0005)  # Default: 0.001
# Higher = faster learning but more instable
# Lower = slower but more stable
```

### Extend Training
```python
EPOCHS = 100  # Default: 50
# More epochs may improve accuracy but risks overfitting
# Early stopping prevents this
```

### Stronger Augmentation
```python
datagen = ImageDataGenerator(
    rotation_range=20,  # From 10
    width_shift_range=0.2,  # From 0.1
    height_shift_range=0.2,  # From 0.1
    zoom_range=0.2,  # From 0.1
    shear_range=0.15,  # From 0.1
)
```

---

## Performance Monitoring

### Check GPU Usage
```python
# During training
!nvidia-smi
```

### Monitor Training Progress
```python
# Plot history
pd.DataFrame(history.history).plot(figsize=(12, 4))
plt.show()

# Check best accuracy
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Achieved at epoch: {np.argmax(history.history['val_accuracy']) + 1}")
```

### Evaluate on Test Set
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Saving & Loading Models

### Save Model
```python
# Automatically saved during training
# Load the best model
model = keras.models.load_model('gtsrb_best_model.keras')
```

### Save for Different Formats

#### SavedModel Format (TensorFlow)
```python
model.save('gtsrb_model')  # Creates directory
```

#### TFLite Format (Mobile)
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('gtsrb_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### ONNX Format (Cross-platform)
```python
import tf2onnx
onnx_model = tf2onnx.convert.from_keras(model)
```

---

## Training Tips

### For Better Accuracy
1. Use more training data if available
2. Train longer (increase EPOCHS)
3. Use stronger regularization
4. Try different learning rates
5. Implement ensemble (train multiple models)

### For Faster Training
1. Reduce input image size (32×32)
2. Reduce model capacity (fewer filters)
3. Increase batch size (if GPU memory allows)
4. Use mixed precision training
5. Use TensorFlow's graph execution

### For Stable Training
1. Use batch normalization (already included)
2. Use gradient clipping (already included)
3. Lower learning rate
4. Reduce dropout rate
5. Use warmup learning rate schedule

### For Production
1. Train on best available GPU
2. Use multiple random seeds
3. Ensemble 3-5 models
4. Add uncertainty quantification
5. Monitor predictions post-deployment

---

## Inference After Training

### Single Image Prediction
```python
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

def predict_single(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    class_id = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    
    return class_id, confidence

# Usage
class_id, conf = predict_single('traffic_sign.jpg')
print(f"Class: {class_id}, Confidence: {conf:.2f}%")
```

### Batch Prediction
```python
# Predict on multiple images
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1) * 100
```

### Real-time Webcam Prediction
```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    roi = frame[50:398, 50:398]  # Crop to square
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)
    
    prediction = model.predict(roi)
    class_id = np.argmax(prediction[0])
    
    cv2.putText(frame, f"Class: {class_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Traffic Sign Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Next Steps

1. **Train the Model** - Follow steps above
2. **Evaluate Results** - Check accuracy and loss
3. **Save the Model** - Model auto-saves best checkpoint
4. **Deploy** - See DEPLOYMENT.md for options
5. **Monitor** - Track real-world performance

---

**Author:** [Yasharth Dwivedi](https://github.com/YASHARTH-DWIVEDI)
**GitHub:** https://github.com/YASHARTH-DWIVEDI/GTSRB-Traffic-Sign-CNN
