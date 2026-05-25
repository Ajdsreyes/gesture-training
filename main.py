from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_training import (
    train_model,
    evaluate_model
)

app = FastAPI()

# ALLOW FRONTEND TO CONNECT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluate-raw")
async def evaluate_raw(data: dict):

    gesture = data.get("gesture", [])

    if len(gesture) < 1:
        return {
            "error": "No gesture points received"
        }

    start_time = gesture[0]["time"]
    end_time   = gesture[-1]["time"]

    duration_ms = round(end_time - start_time, 2)

    return {
        "duration_ms": duration_ms,
        "points": len(gesture)
    }


# REQUEST MODELS

class TrainRequest(BaseModel):
    X_data: List[List[float]]
    y_data: List[str]


class EvaluateRequest(BaseModel):
    features: List[float]


# ROOT

@app.get("/")
def home():
    return {
        "message": "Gesture Authentication API Running"
    }


# BUILD / TRAIN MODEL

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


# EVALUATE GESTURE

@app.post("/evaluate")
def evaluate(data: EvaluateRequest):

    result = evaluate_model(
        data.features
    )

    return result