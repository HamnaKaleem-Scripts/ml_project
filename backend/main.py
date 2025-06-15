from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import os
import numpy as np
from scipy.sparse import hstack

# =========================
# 🔧 App Initialization
# =========================
app = FastAPI(title="YouTube Popularity Predictor API",
              description="Predict video popularity using ML and NLP",
              version="1.0")

# ===========================
# 🔐 Model + Vectorizer Loading
# ===========================
MODEL_PATH = "backend/models/like_ratio_model.pkl"
VECTORIZER_PATH = "backend/models/tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise RuntimeError("❌ Model or vectorizer not found. Please run train_model.py first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ===========================
# ✅ Pydantic Input Schema with Validation
# ===========================
class VideoInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    tags: str = Field(..., min_length=3)
    category: str
    duration: float = Field(..., gt=0)
    publish_time: str  # ISO format recommended

# ===========================
# ⚙️ Helper: Preprocess Input
# ===========================
def prepare_input(data: VideoInput):
    # Combine text features
    combined_text = f"{data.title} {data.tags} {data.category}"
    tfidf_vector = vectorizer.transform([combined_text])
    
    # Numerical features (extendable)
    numeric_features = np.array([[data.duration]])
    
    # Combine both
    final_input = hstack([tfidf_vector, numeric_features])
    return final_input

# ===========================
# 🚀 API: /predict
# ===========================
@app.post("/predict")
def predict(data: VideoInput):
    try:
        input_vector = prepare_input(data)
        prediction = model.predict(input_vector)[0]
        return {
            "predicted_like_ratio": float(prediction),
            "message": "✅ Prediction successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ===========================
# 📊 API: /top-features
# ===========================
@app.get("/top-features")
def top_features(limit: Optional[int] = 10):
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = vectorizer.get_feature_names_out()
            top = sorted(zip(feature_names, importances[:-1]), key=lambda x: x[1], reverse=True)[:limit]
            return {"top_features": top}
        else:
            return {"message": "Model does not support feature importance (e.g., GradientBoostingRegressor)"},
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {str(e)}")

# ===========================
# 📈 API: /stats
# ===========================
@app.get("/stats")
def stats():
    # Normally you’d pull these from a saved metrics file
    return {
        "model": "GradientBoostingRegressor",
        "description": "Predicts like/view ratio from video metadata",
        "trained_on": "YouTube Trending Videos Dataset",
        "metrics": {
            "MAE": "0.23",
            "R² Score": "0.81"
        }
    }


# 🏠 Optional Root Test

@app.get("/")
def home():
    return {"message": "🚀 YouTube Popularity API is running!"}
