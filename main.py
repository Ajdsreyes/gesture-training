from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from model_training import (
    train_model,
    evaluate_model
)

app = FastAPI()


# =========================
# REQUEST MODELS
# =========================

class TrainRequest(BaseModel):
    X_data: List[List[float]]
    y_data: List[str]


class EvaluateRequest(BaseModel):
    features: List[float]


# =========================
# ROOT
# =========================

@app.get("/")
def home():
    return {
        "message": "Gesture Authentication API Running"
    }


# =========================
# BUILD / TRAIN MODEL
# =========================

@app.post("/build-model")
def build_model(data: TrainRequest):

    result = train_model(
        data.X_data,
        data.y_data
    )

    return {
        "message": "Model trained successfully",
        "result": result
    }


# =========================
# EVALUATE GESTURE
# =========================

@app.post("/evaluate")
def evaluate(data: EvaluateRequest):

    result = evaluate_model(
        data.features
    )

    return result