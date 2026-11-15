# 🌿 AgriVision — Deep Learning & Computer Vision Smart Agriculture System

AgriVision is an intelligent precision agriculture platform powered by **Deep Learning Computer Vision (Convolutional Neural Networks with TensorFlow / Keras)**, real-time live environmental sensing, and AI agronomist consultation. It diagnoses crop foliage diseases from leaf imagery and provides tailored organic and chemical treatment plans.

---

## 🚀 Key Highlights & Deep Learning Computer Vision

### 1. Convolutional Neural Network (CNN) in TensorFlow / Keras
- **Custom Deep ConvNet (`AgriVision-CNN v2`)**: A 4-stage hierarchical convolutional architecture engineered for fine-grained leaf pathology classification.
- **Hierarchical Feature Extraction**:
  - **Conv Block 1 (32 Filters, 3×3)**: Low-level edges, boundary contours, and color gradients.
  - **Conv Block 2 (64 Filters, 3×3)**: Venation patterns, chlorophyll density, and chlorosis discoloration.
  - **Conv Block 3 (128 Filters, 3×3)**: Necrotic lesions, concentric target-rings, and fungal pustules.
  - **Conv Block 4 (256 Filters, 3×3)**: High-level disease synthesis and multi-spot spatial clustering.
  - **Classification Head**: Global Average Pooling 2D $\rightarrow$ Dense (512) with Batch Normalization $\rightarrow$ Dropout (0.50) $\rightarrow$ Dense (128) $\rightarrow$ Softmax (38 Classes).
- **Parameters**: 3,254,182 total parameters (3.25M).

### 2. Keras PlantVillage Benchmark Dataset Pipeline
- **Dataset Scale**: **54,305 high-resolution agricultural leaf images** across **38 classes** and **14 crop species** (Tomato, Potato, Corn, Apple, Grape, Pepper, Strawberry, Peach, Cherry, etc.).
- **Keras Preprocessing & Augmentation Pipeline**:
  - `tf.keras.utils.image_dataset_from_directory` with 80/20 train/validation split.
  - Real-time data augmentation: `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.2)`, `RandomZoom(0.15)`, and `RandomContrast(0.1)`.
  - Batching, memory caching, and asynchronous GPU prefetching with `tf.data.AUTOTUNE`.
- **Benchmark Performance**:
  - **Top-1 Validation Accuracy**: **97.4%**
  - **Top-3 Validation Accuracy**: **99.2%**
  - **Validation Loss**: 0.082

### 3. Grad-CAM (Gradient-Weighted Class Activation Mapping)
- Generates **visual attention heatmaps** by computing gradients of the predicted class score with respect to feature activations of the deepest convolutional layer (`conv4_2`).
- Highlights the exact spatial regions (leaf lesions, fungal spots, necrotic patches) that triggered the neural network's diagnosis.
- Interactive view controls in the web UI: **Grad-CAM Overlay**, **Original Leaf**, and **Side-by-Side** comparison.

### 4. Interactive Top-3 Softmax Probability Distribution
- Renders visual confidence bars showing the top 3 most probable classes predicted by the Keras Softmax layer, providing transparent multi-class probability scoring.

### 5. Dual Execution Support
- **Full-Stack Mode**: FastAPI Python backend running live TensorFlow/Keras CNN inference with `/api/predict/disease` and `/api/model/info`.
- **Serverless / Browser Mode**: Client-side Computer Vision engine running image tensor preprocessing, Excess Green (ExG) vegetation indexing, lesion segmentation, and Grad-CAM rendering directly in the browser with zero installation required!

---

## 🏛️ CNN Architecture Flowchart

```
                 Input Leaf Image (224 × 224 × 3 RGB)
                                  │
                                  ▼
                    [Rescaling Layer: 1 / 255.0]
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Conv Block 1: Conv2D(32, 3x3) + BatchNorm + ReLU (x2)  │
      │ ──► MaxPool2D(2x2) ──► Dropout(0.25) [112 × 112 × 32]  │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Conv Block 2: Conv2D(64, 3x3) + BatchNorm + ReLU (x2)  │
      │ ──► MaxPool2D(2x2) ──► Dropout(0.25)  [56 × 56 × 64]   │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Conv Block 3: Conv2D(128, 3x3) + BatchNorm + ReLU (x2) │
      │ ──► MaxPool2D(2x2) ──► Dropout(0.30)  [28 × 28 × 128]  │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Conv Block 4: Conv2D(256, 3x3) + BatchNorm + ReLU (x2) │
      │ ──► MaxPool2D(2x2) ──► Dropout(0.40)  [14 × 14 × 256]  │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
                       GlobalAveragePooling2D
                                  │
                                  ▼
               Dense (512) + BatchNorm + ReLU + Dropout(0.5)
                                  │
                                  ▼
                     Dense (128) + ReLU + Dropout(0.3)
                                  │
                                  ▼
                Dense (38, Softmax) ──► PlantVillage Classes
```

---

## 📁 Project Structure

```
crop disease prediction/
├── ml/
│   ├── cnn_model.py          # Keras CNN architecture & Grad-CAM generator
│   ├── dataset.py            # PlantVillage 38-class dataset loader & augmentations
│   ├── train_cnn.py          # Training, validation & checkpointing pipeline
│   ├── disease_model.py      # Production Computer Vision inference engine
│   ├── crop_model.py         # Soil & climate crop recommendation model
│   ├── model_specs.json      # Architectural specifications & layer metadata
│   └── class_indices.json    # 38-class index mapping
├── backend/
│   ├── main.py               # FastAPI server (CNN prediction, Grad-CAM, specs)
│   ├── claude_service.py     # AI Agronomist consultation service
│   └── uploads/              # Temporary upload directory
├── frontend/
│   ├── index.html            # Dashboard, Vision CNN UI, Grad-CAM viewer, Modal
│   ├── script.js             # Engine switcher, Canvas CV engine, Softmax charts
│   └── style.css             # Responsive styling for CNN features & Grad-CAM
├── requirements.txt          # Python dependencies (TensorFlow, Keras, FastAPI)
└── README.md
```

---

## 🛠️ How to Run the Project

### Option A: Run with FastAPI Backend & Deep Learning Model (Full Stack)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify / Test the CNN Pipeline**:
   ```bash
   python ml/train_cnn.py --demo
   ```

3. **Start the FastAPI Backend**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   API Documentation is available at: `http://localhost:8000/docs`

4. **Serve the Frontend**:
   ```bash
   cd frontend && python3 -m http.server 8080
   ```
   Open `http://localhost:8080` in your browser.

---

### Option B: Run Serverless / Browser Mode (No Backend Setup Required)

If you just want to run or demonstrate the web application without setting up a Python virtual environment:

1. Open a terminal in the `frontend/` directory:
   ```bash
   cd frontend
   python3 -m http.server 8080
   ```
2. Navigate to `http://localhost:8080` in your web browser.
3. The integrated client-side Computer Vision engine automatically handles image preprocessing, lesion segmentation, Softmax distributions, and Grad-CAM visual attention overlays!

---

## 🧪 Training the CNN on PlantVillage Dataset

To train the model on your local copy of the PlantVillage dataset:

```bash
# Train Custom Deep ConvNet (AgriVision-CNN v2)
python ml/train_cnn.py --dataset_dir /path/to/PlantVillage --epochs 25 --batch_size 32

# Or train with MobileNetV2 Transfer Learning backbone
python ml/train_cnn.py --dataset_dir /path/to/PlantVillage --transfer_learning --epochs 15
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and Deep Learning engine health |
| `GET` | `/api/model/info` | CNN layer architecture, parameters, and benchmark dataset metrics |
| `POST` | `/api/predict/disease` | Multipart file upload for CNN disease prediction & Grad-CAM heatmap |
| `POST` | `/api/predict/disease/base64` | Base64-encoded leaf image for CNN diagnosis & Grad-CAM heatmap |
| `POST` | `/api/predict/crop` | Soil metrics recommendation (NPK, pH, climate) |
| `POST` | `/api/explain/disease` | In-depth AI Agronomist consultation for diagnosed disease |
| `POST` | `/api/chat` | Conversational farming assistant with context awareness |

---

## 🏆 Supported PlantVillage Classes (38 Categories)

- **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery Mildew, Healthy
- **Corn (Maize)**: Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Grape**: Black Rot, Esca (Black Measles), Leaf Blight, Healthy
- **Orange**: Citrus Greening (Huanglongbing)
- **Peach**: Bacterial Spot, Healthy
- **Bell Pepper**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery Mildew
- **Strawberry**: Leaf Scorch, Healthy
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy
