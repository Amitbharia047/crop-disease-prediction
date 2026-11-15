"""
main.py — AgriVision FastAPI Backend
─────────────────────────────────────────────────────────────────
Endpoints:
  GET  /                              Health check
  POST /api/predict/disease           ML disease detection
  POST /api/predict/crop              ML crop recommendation
  POST /api/explain/disease           Claude explanation for disease result
  POST /api/explain/crop              Claude explanation for crop result
  POST /api/chat                      Claude chatbot (multi-turn)
─────────────────────────────────────────────────────────────────
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os, shutil

# Add the project root to sys.path so we can import ml + backend modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from ml.disease_model import predict_disease
from ml.cnn_model import get_model_summary_dict
from ml.crop_model import predict_crop
from backend.claude_service import (
    explain_disease_prediction,
    explain_crop_recommendation,
    chat_with_agribot,
    quick_explain,
)

# ──────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgriVision API",
    description="ML + Claude LLM powered Smart Agriculture API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS  (Pydantic schemas)
# ──────────────────────────────────────────────────────────────

class SoilMetrics(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class Base64ImageRequest(BaseModel):
    image: str   # Base64 string or data:image/...;base64,... string
    filename: str = "leaf_upload.png"


class DiseaseExplainRequest(BaseModel):
    """Send the ML prediction result here to get a Claude explanation."""
    disease: str
    confidence: float          # 0.0 – 1.0
    plant: str
    severity: str              # None / Mild / Moderate / Severe
    treatment: str


class CropExplainRequest(BaseModel):
    """Send both the recommendation + original soil metrics."""
    recommended_crop: str
    soil_health_score: float
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: dict | None = None   # Optional prediction context


class QuickExplainRequest(BaseModel):
    """Send a simple topic + context string for a quick one-shot explanation."""
    topic: str
    context: str = ""


# ──────────────────────────────────────────────────────────────
# ROUTE 1 — HEALTH CHECK
# ──────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "AgriVision API v2.0 — Deep Learning CNN + Keras + Computer Vision ready",
        "deep_learning_engine": "TensorFlow/Keras CNN (PlantVillage 38 Classes)",
    }


# ──────────────────────────────────────────────────────────────
# ROUTE 1.5 — MODEL & DATASET SPECIFICATIONS
# ──────────────────────────────────────────────────────────────

@app.get("/api/model/info")
def model_info_endpoint():
    """Returns deep learning model architecture specs and PlantVillage dataset benchmarks."""
    return {
        "status": "success",
        "model": get_model_summary_dict()
    }


# ──────────────────────────────────────────────────────────────
# ROUTE 2 — ML DISEASE PREDICTION (Keras Deep Learning CNN)
# ──────────────────────────────────────────────────────────────

@app.post("/api/predict/disease")
async def predict_disease_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        result = predict_disease(file_path)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/api/predict/disease/base64")
async def predict_disease_base64_endpoint(req: Base64ImageRequest):
    """Allows sending base64-encoded leaf images directly from web frontends."""
    import base64
    raw_data = req.image
    if "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]

    file_path = os.path.join(UPLOAD_DIR, req.filename)
    try:
        img_bytes = base64.b64decode(raw_data)
        with open(file_path, "wb") as buf:
            buf.write(img_bytes)
        result = predict_disease(file_path)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ──────────────────────────────────────────────────────────────
# ROUTE 3 — ML CROP PREDICTION (unchanged)
# ──────────────────────────────────────────────────────────────

@app.post("/api/predict/crop")
def predict_crop_endpoint(metrics: SoilMetrics):
    try:
        result = predict_crop(
            n=metrics.N, p=metrics.P, k=metrics.K,
            temperature=metrics.temperature,
            humidity=metrics.humidity,
            ph=metrics.ph,
            rainfall=metrics.rainfall,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# ROUTE 4 — CLAUDE: EXPLAIN DISEASE RESULT  ← NEW
# ──────────────────────────────────────────────────────────────

@app.post("/api/explain/disease")
def explain_disease_endpoint(req: DiseaseExplainRequest):
    """
    Takes ML disease prediction and returns a rich Claude explanation.
    Call this AFTER getting ML results to add depth to the UI.
    """
    try:
        explanation = explain_disease_prediction(
            disease=req.disease,
            confidence=req.confidence,
            plant=req.plant,
            severity=req.severity,
            treatment=req.treatment,
        )
        return {"status": "success", "data": explanation}
    except ValueError as e:
        # API key not configured
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# ROUTE 5 — CLAUDE: EXPLAIN CROP RESULT  ← NEW
# ──────────────────────────────────────────────────────────────

@app.post("/api/explain/crop")
def explain_crop_endpoint(req: CropExplainRequest):
    """
    Takes ML crop recommendation + soil data and returns a Claude explanation.
    """
    try:
        explanation = explain_crop_recommendation(
            recommended_crop=req.recommended_crop,
            soil_health_score=req.soil_health_score,
            n=req.N, p=req.P, k=req.K,
            temperature=req.temperature,
            humidity=req.humidity,
            ph=req.ph,
            rainfall=req.rainfall,
        )
        return {"status": "success", "data": explanation}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# ROUTE 6 — CLAUDE: CHATBOT  ← NEW
# ──────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chatbot_endpoint(req: ChatRequest):
    """
    Multi-turn conversational chatbot powered by Claude.
    Maintains conversation history for context-aware replies.
    """
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        reply = chat_with_agribot(messages=messages, context=req.context)
        return {"status": "success", "reply": reply}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# ROUTE 7 — CLAUDE: QUICK ONE-SHOT EXPLAIN  ← NEW
# ──────────────────────────────────────────────────────────────

@app.post("/api/explain/quick")
def quick_explain_endpoint(req: QuickExplainRequest):
    """
    Lightweight one-shot explainer — great for inline "What does this mean?" calls.
    """
    try:
        explanation = quick_explain(topic=req.topic, context=req.context)
        return {"status": "success", "explanation": explanation}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
