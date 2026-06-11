"""
FastAPI backend for SVM Gesture Authentication
Endpoints cover dataset generation, model training, authentication, and evaluation.
"""

from __future__ import annotations

import uuid
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gesture_data import (
    generate_synthetic_dataset,
    GestureSequence,
    Gesture,
    TouchEvent,
)
from feature_extractor import FeatureExtractor, ZScoreNormaliser, FEATURE_NAMES
from svm_trainer import train_user_model, UserSVM, compute_eer_threshold
from evaluator import (
    AuthEvaluationPipeline,
    UserResult,
    AggregateMetrics,
    compute_dprime,
    compute_eer_from_scores,
    compute_far,
    compute_frr,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SVM Gesture Authentication API",
    description=(
        "REST API for training One-Class SVM models on touch-gesture "
        "biometric data and evaluating authentication performance."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory stores  (replace with a DB for production)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.datasets:     Dict[str, List[GestureSequence]] = {}
        self.models:       Dict[str, UserSVM]               = {}
        self.normalisers:  Dict[str, ZScoreNormaliser]      = {}
        self.eval_results: Dict[str, dict]                  = {}
        self.jobs:         Dict[str, dict]                  = {}
        self.extractor = FeatureExtractor()

state = AppState()

# ---------------------------------------------------------------------------
# In-memory training buffer for frontend submissions (keyed by participant_id)
# ---------------------------------------------------------------------------

training_buffer: Dict[str, List[GestureSequence]] = defaultdict(list)
MIN_TRAIN_SEQS = 5


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class GenerateDatasetRequest(BaseModel):
    n_participants: int   = Field(10,  ge=2,  le=100)
    n_sessions:     int   = Field(3,   ge=1,  le=10)
    n_repetitions:  int   = Field(10,  ge=3,  le=50)
    seed:           int   = Field(42)


class TrainModelRequest(BaseModel):
    dataset_id:     str
    participant_id: str
    train_session:  int          = Field(1, ge=1)
    val_session:    int          = Field(2, ge=1)
    candidate_nus:  List[float]  = Field([0.05, 0.1, 0.2, 0.3, 0.5])
    kernel:         str          = Field("rbf")


class BatchTrainRequest(BaseModel):
    dataset_id:    str
    train_session: int          = Field(1, ge=1)
    val_session:   int          = Field(2, ge=1)
    candidate_nus: List[float]  = Field([0.05, 0.1, 0.2, 0.3, 0.5])
    kernel:        str          = Field("rbf")


class TouchEventSchema(BaseModel):
    timestamp:  float
    x:          float
    y:          float
    pressure:   float = Field(ge=0.0, le=1.0)
    finger_id:  int   = Field(0, ge=0)


class GestureSchema(BaseModel):
    gesture_type:   str
    orientation:    str
    events:         List[TouchEventSchema]
    session_id:     int = 1
    participant_id: str = "unknown"
    repetition:     int = 0


class GestureSequenceSchema(BaseModel):
    gestures:       List[GestureSchema] = Field(..., min_length=3, max_length=3)
    participant_id: str = "unknown"
    session_id:     int = 1


# CHANGED: replaces old AuthenticateRequest to match frontend payload shape
class AuthenticateRequest(BaseModel):
    participant_id: str
    session_id:     int
    model_id:       str
    sequence:       "SequencePayload"


class EvaluateRequest(BaseModel):
    dataset_id:    str
    candidate_nus: List[float] = Field([0.05, 0.1, 0.2, 0.3, 0.5])
    kernel:        str         = Field("rbf")
    test_session:  int         = Field(3, ge=1)


class DatasetInfo(BaseModel):
    dataset_id:     str
    n_participants: int
    n_sequences:    int
    participants:   List[str]
    sessions:       List[int]


class ModelInfo(BaseModel):
    model_id:       str
    participant_id: str
    nu:             float
    threshold:      float
    is_fitted:      bool


class AuthenticateResponse(BaseModel):
    accepted:       bool
    score:          float
    threshold:      float
    model_id:       str
    participant_id: str


class ScoreDistributionResponse(BaseModel):
    model_id:        str
    participant_id:  str
    genuine_scores:  List[float]
    impostor_scores: List[float]
    far:             float
    frr:             float
    eer:             float
    dprime:          float
    accuracy:        float
    threshold:       float


class AggregateMetricsResponse(BaseModel):
    session_id:  int
    n_users:     int
    mean_far:    float
    std_far:     float
    mean_frr:    float
    std_frr:     float
    mean_eer:    float
    std_eer:     float
    mean_dprime: float
    std_dprime:  float
    mean_acc:    float
    std_acc:     float


class UserMetricsResponse(BaseModel):
    participant_id: str
    session_id:     int
    far:            float
    frr:            float
    eer:            float
    dprime:         float
    accuracy:       float
    threshold:      float


class EvalJobResponse(BaseModel):
    job_id:  str
    status:  str
    message: str


class JobStatusResponse(BaseModel):
    job_id:  str
    status:  str
    message: str
    result:  Optional[dict] = None


# ---------------------------------------------------------------------------
# NEW: Pydantic schemas for frontend submit_gestures / authenticate
# ---------------------------------------------------------------------------

class TouchEventPayload(BaseModel):
    timestamp:  float
    x:          float
    y:          float
    pressure:   float = 0.5
    finger_id:  int   = 0


class GesturePayload(BaseModel):
    gesture_type: str
    orientation:  str
    events:       List[TouchEventPayload]


class SequencePayload(BaseModel):
    gestures: List[GesturePayload]


class SubmitGesturesRequest(BaseModel):
    participant_id: str
    session_id:     int
    sequences:      List[SequencePayload]
    mode:           str = "train"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema_to_gesture_sequence(schema: GestureSequenceSchema) -> GestureSequence:
    gestures = []
    for g in schema.gestures:
        events = [
            TouchEvent(
                timestamp=e.timestamp, x=e.x, y=e.y,
                pressure=e.pressure, finger_id=e.finger_id,
            )
            for e in g.events
        ]
        gestures.append(Gesture(
            gesture_type=g.gesture_type, orientation=g.orientation,
            events=events, session_id=g.session_id,
            participant_id=g.participant_id, repetition=g.repetition,
        ))
    return GestureSequence(
        gestures=gestures,
        participant_id=schema.participant_id,
        session_id=schema.session_id,
    )


# NEW: converts flat frontend SequencePayload to GestureSequence
def _payload_to_gesture_sequence(
    seq: SequencePayload,
    participant_id: str,
    session_id: int,
) -> GestureSequence:
    gestures = []
    for g in seq.gestures:
        events = [
            TouchEvent(
                timestamp  = e.timestamp,
                x          = e.x,
                y          = e.y,
                pressure   = e.pressure,
                finger_id  = e.finger_id,
            )
            for e in g.events
        ]
        gestures.append(Gesture(
            gesture_type   = g.gesture_type,
            orientation    = g.orientation,
            events         = events,
            session_id     = session_id,
            participant_id = participant_id,
        ))
    return GestureSequence(
        gestures       = gestures,
        participant_id = participant_id,
        session_id     = session_id,
    )


def _sequences_by_participant(
    sequences: List[GestureSequence],
) -> Dict[str, Dict[int, List[GestureSequence]]]:
    grouped: Dict[str, Dict[int, List[GestureSequence]]] = defaultdict(lambda: defaultdict(list))
    for s in sequences:
        grouped[s.participant_id][s.session_id].append(s)
    return grouped


def _score_sequences(
    model:      UserSVM,
    normaliser: ZScoreNormaliser,
    extractor:  FeatureExtractor,
    sequences:  List[GestureSequence],
) -> np.ndarray:
    scores = []
    for s in sequences:
        mat = extractor.sequence_to_matrix(s)
        mat = normaliser.transform(mat)
        scores.append(model.score(mat))
    return np.array(scores, dtype=float)


# ---------------------------------------------------------------------------
# Routes – Dataset
# ---------------------------------------------------------------------------

@app.post("/dataset/generate", response_model=DatasetInfo, tags=["Dataset"])
def generate_dataset(req: GenerateDatasetRequest):
    dataset_id = str(uuid.uuid4())
    seqs = generate_synthetic_dataset(
        n_participants=req.n_participants,
        n_sessions=req.n_sessions,
        n_repetitions=req.n_repetitions,
        seed=req.seed,
    )
    state.datasets[dataset_id] = seqs
    participants = sorted({s.participant_id for s in seqs})
    sessions     = sorted({s.session_id     for s in seqs})
    logger.info(f"Dataset {dataset_id}: {len(seqs)} sequences, {len(participants)} participants")
    return DatasetInfo(
        dataset_id=dataset_id, n_participants=len(participants),
        n_sequences=len(seqs), participants=participants, sessions=sessions,
    )


@app.get("/dataset/{dataset_id}", response_model=DatasetInfo, tags=["Dataset"])
def get_dataset_info(dataset_id: str):
    if dataset_id not in state.datasets:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    seqs         = state.datasets[dataset_id]
    participants = sorted({s.participant_id for s in seqs})
    sessions     = sorted({s.session_id     for s in seqs})
    return DatasetInfo(
        dataset_id=dataset_id, n_participants=len(participants),
        n_sequences=len(seqs), participants=participants, sessions=sessions,
    )


@app.get("/dataset/{dataset_id}/participants", response_model=List[str], tags=["Dataset"])
def list_participants(dataset_id: str):
    if dataset_id not in state.datasets:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    return sorted({s.participant_id for s in state.datasets[dataset_id]})


@app.delete("/dataset/{dataset_id}", tags=["Dataset"])
def delete_dataset(dataset_id: str):
    if dataset_id not in state.datasets:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    del state.datasets[dataset_id]
    return {"message": f"Dataset '{dataset_id}' deleted."}


# ---------------------------------------------------------------------------
# Routes – Training
# ---------------------------------------------------------------------------

@app.post("/model/train", response_model=ModelInfo, tags=["Training"])
def train_single_user_model(req: TrainModelRequest):
    """Train a One-Class SVM for a single participant."""
    if req.dataset_id not in state.datasets:
        raise HTTPException(404, f"Dataset '{req.dataset_id}' not found.")

    seqs    = state.datasets[req.dataset_id]
    grouped = _sequences_by_participant(seqs)

    if req.participant_id not in grouped:
        raise HTTPException(404, f"Participant '{req.participant_id}' not found in dataset.")

    pid_data   = grouped[req.participant_id]
    train_seqs = pid_data.get(req.train_session, [])
    val_seqs   = pid_data.get(req.val_session,   [])

    if len(train_seqs) < 3:
        raise HTTPException(
            422,
            f"Participant '{req.participant_id}' has only {len(train_seqs)} training sequences "
            f"(minimum 3 required)."
        )

    imp_seqs = [
        s for pid, sessions in grouped.items()
        for s in sessions.get(req.val_session, [])
        if pid != req.participant_id
    ]

    X_tr, _ = state.extractor.sequences_to_arrays(train_seqs)
    norm = ZScoreNormaliser()
    norm.fit(X_tr)

    try:
        model = train_user_model(
            participant_id    = req.participant_id,
            train_sequences   = train_seqs,
            val_genuine_seqs  = val_seqs if val_seqs else train_seqs[:2],
            val_impostor_seqs = imp_seqs[:20],
            extractor         = state.extractor,
            normaliser        = norm,
            candidate_nus     = req.candidate_nus,
            kernel            = req.kernel,
            verbose           = False,
        )
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(500, f"Training failed: {exc}")

    model_id = str(uuid.uuid4())
    state.models[model_id]      = model
    state.normalisers[model_id] = norm

    logger.info(f"Trained SVM model {model_id} for participant {req.participant_id}")
    return ModelInfo(
        model_id       = model_id,
        participant_id = model.participant_id,
        nu             = model.svm.nu,
        threshold      = model.threshold,
        is_fitted      = model.svm.is_fitted,
    )


@app.post("/model/train/batch", response_model=EvalJobResponse, tags=["Training"])
def batch_train(req: BatchTrainRequest, background_tasks: BackgroundTasks):
    """Train SVM models for all participants in a dataset asynchronously."""
    if req.dataset_id not in state.datasets:
        raise HTTPException(404, f"Dataset '{req.dataset_id}' not found.")

    job_id = str(uuid.uuid4())
    state.jobs[job_id] = {"status": "pending", "message": "Queued", "result": None}

    def _run():
        state.jobs[job_id].update({"status": "running", "message": "Training in progress ..."})
        seqs    = state.datasets[req.dataset_id]
        grouped = _sequences_by_participant(seqs)
        trained_models = []

        for pid, pid_sessions in grouped.items():
            train_seqs = pid_sessions.get(req.train_session, [])
            val_seqs   = pid_sessions.get(req.val_session,   [])
            if len(train_seqs) < 3:
                continue
            imp_seqs = [
                s for other_pid, sessions in grouped.items()
                for s in sessions.get(req.val_session, [])
                if other_pid != pid
            ]
            X_tr, _ = state.extractor.sequences_to_arrays(train_seqs)
            norm = ZScoreNormaliser()
            norm.fit(X_tr)
            try:
                model = train_user_model(
                    participant_id    = pid,
                    train_sequences   = train_seqs,
                    val_genuine_seqs  = val_seqs if val_seqs else train_seqs[:2],
                    val_impostor_seqs = imp_seqs[:20],
                    extractor         = state.extractor,
                    normaliser        = norm,
                    candidate_nus     = req.candidate_nus,
                    kernel            = req.kernel,
                    verbose           = False,
                )
                mid = str(uuid.uuid4())
                state.models[mid]      = model
                state.normalisers[mid] = norm
                trained_models.append({
                    "model_id": mid, "participant_id": pid,
                    "nu": model.svm.nu, "threshold": model.threshold,
                })
            except Exception as exc:
                logger.warning(f"Batch train failed for {pid}: {exc}")

        state.jobs[job_id].update({
            "status":  "done",
            "message": f"Trained {len(trained_models)} models.",
            "result":  {"trained_models": trained_models},
        })

    background_tasks.add_task(_run)
    return EvalJobResponse(job_id=job_id, status="pending", message="Batch training queued.")


@app.get("/model/{model_id}", response_model=ModelInfo, tags=["Training"])
def get_model_info(model_id: str):
    if model_id not in state.models:
        raise HTTPException(404, f"Model '{model_id}' not found.")
    m = state.models[model_id]
    return ModelInfo(
        model_id=model_id, participant_id=m.participant_id,
        nu=m.svm.nu, threshold=m.threshold, is_fitted=m.svm.is_fitted,
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Training"])
def list_models():
    return [
        ModelInfo(
            model_id=mid, participant_id=m.participant_id,
            nu=m.svm.nu, threshold=m.threshold, is_fitted=m.svm.is_fitted,
        )
        for mid, m in state.models.items()
    ]


@app.delete("/model/{model_id}", tags=["Training"])
def delete_model(model_id: str):
    if model_id not in state.models:
        raise HTTPException(404, f"Model '{model_id}' not found.")
    del state.models[model_id]
    state.normalisers.pop(model_id, None)
    return {"message": f"Model '{model_id}' deleted."}


# ---------------------------------------------------------------------------
# Routes – Frontend endpoints (submit_gestures + authenticate)
# ---------------------------------------------------------------------------

@app.post("/submit_gestures", tags=["Frontend"])
def submit_gestures(req: SubmitGesturesRequest):
    """
    Called by training.ts after collecting gesture sequences.
    Buffers sequences per participant and trains the SVM once
    MIN_TRAIN_SEQS sequences are collected.
    """
    pid        = req.participant_id
    session_id = req.session_id

    try:
        parsed = [
            _payload_to_gesture_sequence(s, pid, session_id)
            for s in req.sequences
        ]
    except Exception as e:
        raise HTTPException(422, f"Parse error: {e}")

    training_buffer[pid].extend(parsed)
    all_seqs = training_buffer[pid]

    if len(all_seqs) < MIN_TRAIN_SEQS:
        return {
            "status":  "pending",
            "message": f"Collected {len(all_seqs)}/{MIN_TRAIN_SEQS} sequences. Keep training!",
        }

    X_all, _ = state.extractor.sequences_to_arrays(all_seqs)
    norm = ZScoreNormaliser()
    norm.fit(X_all)

    split     = max(3, int(len(all_seqs) * 0.8))
    train_seq = all_seqs[:split]
    val_seq   = all_seqs[split:] or all_seqs[:2]

    imp_seqs = [
        s for other_pid, seqs in training_buffer.items()
        if other_pid != pid
        for s in seqs[:5]
    ]

    try:
        model = train_user_model(
            participant_id    = pid,
            train_sequences   = train_seq,
            val_genuine_seqs  = val_seq,
            val_impostor_seqs = imp_seqs if imp_seqs else val_seq,
            extractor         = state.extractor,
            normaliser        = norm,
            verbose           = True,
        )
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(500, f"Training failed: {e}")

    # Store by participant_id so eval.ts can find it using the same id
    state.models[pid]      = model
    state.normalisers[pid] = norm

    logger.info(f"Trained SVM model for participant {pid}")
    return {
        "status":    "trained",
        "model_id":  pid,
        "threshold": round(float(model.threshold), 4),
        "n_seqs":    len(all_seqs),
    }


# CHANGED: replaces the original /authenticate to match frontend payload shape
@app.post("/authenticate", response_model=AuthenticateResponse, tags=["Authentication"])
def authenticate(req: AuthenticateRequest):
    """
    Called by eval.ts. Accepts participant_id, session_id, model_id,
    and a flat sequence payload — matches the shape sent by api.ts.
    """
    model_id = req.model_id

    if model_id not in state.models:
        raise HTTPException(404, f"No trained model for '{model_id}'")

    model = state.models[model_id]
    norm  = state.normalisers[model_id]

    try:
        seq = _payload_to_gesture_sequence(req.sequence, req.participant_id, req.session_id)
    except Exception as e:
        raise HTTPException(422, f"Invalid gesture sequence: {e}")

    mat          = state.extractor.sequence_to_matrix(seq)
    mat          = norm.transform(mat)
    accepted, sc = model.authenticate(mat)

    return AuthenticateResponse(
        accepted       = bool(accepted),
        score          = round(float(sc), 4),
        threshold      = round(float(model.threshold), 4),
        model_id       = model_id,
        participant_id = req.participant_id,
    )


# ---------------------------------------------------------------------------
# Routes – Evaluation pipeline
# ---------------------------------------------------------------------------

@app.post("/evaluate", response_model=EvalJobResponse, tags=["Evaluation"])
def run_evaluation(req: EvaluateRequest, background_tasks: BackgroundTasks):
    """Run the full SVM AuthEvaluationPipeline over a dataset asynchronously."""
    if req.dataset_id not in state.datasets:
        raise HTTPException(404, f"Dataset '{req.dataset_id}' not found.")

    job_id = str(uuid.uuid4())
    state.jobs[job_id] = {"status": "pending", "message": "Queued", "result": None}

    def _run():
        state.jobs[job_id].update({"status": "running", "message": "Pipeline running ..."})
        try:
            seqs     = state.datasets[req.dataset_id]
            pipeline = AuthEvaluationPipeline(state.extractor, verbose=False)
            pipeline.run(seqs, candidate_nus=req.candidate_nus, kernel=req.kernel)

            agg       = pipeline.aggregate_by_session()
            stability = pipeline.temporal_stability()

            agg_out: Dict[int, dict] = {}
            for sess, m in agg.items():
                agg_out[sess] = {
                    "session_id":  sess,        "n_users":     m.n_users,
                    "mean_far":    round(m.mean_far,    4), "std_far":     round(m.std_far,     4),
                    "mean_frr":    round(m.mean_frr,    4), "std_frr":     round(m.std_frr,     4),
                    "mean_eer":    round(m.mean_eer,    4), "std_eer":     round(m.std_eer,     4),
                    "mean_dprime": round(m.mean_dprime, 4), "std_dprime":  round(m.std_dprime,  4),
                    "mean_acc":    round(m.mean_acc,    4), "std_acc":     round(m.std_acc,     4),
                }

            user_results_out = [
                {
                    "participant_id": r.participant_id, "session_id": r.session_id,
                    "far":     round(r.far,      4), "frr":      round(r.frr,      4),
                    "eer":     round(r.eer,       4), "dprime":   round(r.dprime,   4),
                    "accuracy":round(r.accuracy,  4), "threshold":round(r.threshold,4),
                }
                for r in pipeline.user_results
            ]

            sig_test = None
            s2_eers  = [r.eer for r in pipeline.user_results if r.session_id == 2]
            s3_eers  = [r.eer for r in pipeline.user_results if r.session_id == req.test_session]
            if len(s2_eers) > 2 and len(s3_eers) > 2:
                n        = min(len(s2_eers), len(s3_eers))
                sig_test = pipeline.significance_test(s2_eers[:n], s3_eers[:n])

            result = {
                "aggregate_by_session": agg_out,
                "user_results":         user_results_out,
                "temporal_stability": {
                    pid: {str(sess): round(dp, 4) for sess, dp in smap.items()}
                    for pid, smap in stability.items()
                },
                "significance_test_s2_vs_s3": sig_test,
            }

            for pid, model in pipeline.user_models.items():
                mid = f"eval_{job_id}_{pid}"
                state.models[mid]      = model
                state.normalisers[mid] = pipeline.normalisers[pid]

            state.jobs[job_id].update({"status": "done", "message": "Evaluation complete.", "result": result})

        except Exception as exc:
            logger.exception("Evaluation pipeline failed")
            state.jobs[job_id].update({"status": "error", "message": str(exc)})

    background_tasks.add_task(_run)
    return EvalJobResponse(job_id=job_id, status="pending", message="Evaluation job queued.")


@app.get("/evaluate/{job_id}/metrics", tags=["Evaluation"])
def get_aggregate_metrics(job_id: str):
    job = _require_done_job(job_id)
    return job["result"]["aggregate_by_session"]


@app.get("/evaluate/{job_id}/user-results", response_model=List[UserMetricsResponse], tags=["Evaluation"])
def get_user_results(job_id: str):
    job = _require_done_job(job_id)
    return [UserMetricsResponse(**r) for r in job["result"]["user_results"]]


@app.get("/evaluate/{job_id}/stability", tags=["Evaluation"])
def get_temporal_stability(job_id: str):
    job = _require_done_job(job_id)
    return job["result"]["temporal_stability"]


@app.get("/evaluate/{job_id}/significance", tags=["Evaluation"])
def get_significance_test(job_id: str):
    job    = _require_done_job(job_id)
    result = job["result"].get("significance_test_s2_vs_s3")
    if result is None:
        raise HTTPException(422, "Not enough sessions to run significance test.")
    return result


# ---------------------------------------------------------------------------
# Routes – Jobs
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
def get_job_status(job_id: str):
    if job_id not in state.jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    j = state.jobs[job_id]
    return JobStatusResponse(
        job_id=job_id, status=j["status"],
        message=j["message"], result=j.get("result"),
    )


@app.get("/jobs", tags=["Jobs"])
def list_jobs():
    return [
        {"job_id": jid, "status": j["status"], "message": j["message"]}
        for jid, j in state.jobs.items()
    ]


# ---------------------------------------------------------------------------
# Routes – Utilities
# ---------------------------------------------------------------------------

@app.get("/features", tags=["Utilities"])
def list_feature_names():
    return {"n_features": len(FEATURE_NAMES), "features": FEATURE_NAMES}


@app.get("/health", tags=["Utilities"])
def health_check():
    return {
        "status":   "ok",
        "datasets": len(state.datasets),
        "models":   len(state.models),
        "jobs":     len(state.jobs),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_done_job(job_id: str) -> dict:
    if job_id not in state.jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    j = state.jobs[job_id]
    if j["status"] == "error":
        raise HTTPException(500, f"Job failed: {j['message']}")
    if j["status"] != "done":
        raise HTTPException(202, f"Job not yet complete (status: {j['status']}).")
    return j