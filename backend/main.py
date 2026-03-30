from pathlib import Path
from typing import List, Dict, Any
import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import InferenceConfig, PredictiveMaintenanceService

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# ---------------------------
# Request Models
# ---------------------------

class TelemetryRecord(BaseModel):
    engine_id: int = Field(..., description="Unique engine identifier")
    cycle: int = Field(..., description="Cycle index for this measurement")

    setting_1: float
    setting_2: float
    setting_3: float

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class TelemetryBatch(BaseModel):
    records: List[TelemetryRecord]


# ---------------------------
# Lifespan (Startup Loader)
# ---------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load trained models and preprocessing artifacts once when the API starts.
    """
    global service
    cfg = InferenceConfig()
    service = PredictiveMaintenanceService(cfg)

    print("✅ Predictive Maintenance models loaded")

    yield

    print("🛑 API shutting down")


# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI(
    title="Aircraft Predictive Maintenance API",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------------------
# Enable CORS (for frontend)
# ---------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Helper Function
# ---------------------------

def _batch_to_dataframe(batch: TelemetryBatch) -> pd.DataFrame:
    """
    Convert telemetry batch into pandas DataFrame.
    """
    records: List[Dict[str, Any]] = [r.dict() for r in batch.records]
    return pd.DataFrame.from_records(records)


# ---------------------------
# API Endpoints
# ---------------------------

@app.post("/predict_rul")
def predict_rul(batch: TelemetryBatch) -> Dict[str, Any]:
    """
    Predict Remaining Useful Life (RUL).
    """
    df = _batch_to_dataframe(batch)

    preds = service.predict_rul(df)

    rul_rf = preds["rul_rf"]
    rul_lstm = preds["rul_lstm"]

    return {
        "rul_rf_latest": float(rul_rf[-1]) if len(rul_rf) > 0 else None,
        "rul_lstm_latest": float(rul_lstm[-1]) if len(rul_lstm) > 0 else None,
    }


@app.post("/predict_failure_probability")
def predict_failure_probability(
    batch: TelemetryBatch,
    horizon: float = 30.0
) -> Dict[str, Any]:
    """
    Predict probability of failure within given horizon.
    """
    df = _batch_to_dataframe(batch)

    probs = service.predict_failure_probability(df, horizon=horizon)

    prob_rf = probs["failure_prob_rf"]
    prob_lstm = probs["failure_prob_lstm"]

    # ✅ Get latest LSTM probability (most important)
    latest_lstm = float(prob_lstm[-1]) if len(prob_lstm) > 0 else 0.0

    # ✅ Status logic
    if latest_lstm < 0.3:
        status = "GOOD ✅"
    elif latest_lstm < 0.7:
        status = "WARNING ⚠️"
    else:
        status = "CRITICAL 🔴"

    return {
        "failure_prob_rf": prob_rf.tolist(),
        "failure_prob_lstm": prob_lstm.tolist(),
        "failure_prob_lstm_latest": latest_lstm,
        "status": status
    }

@app.post("/anomaly_score")
def anomaly_score(batch: TelemetryBatch) -> Dict[str, Any]:
    """
    Compute anomaly score of engine telemetry.
    """
    df = _batch_to_dataframe(batch)

    scores = service.anomaly_scores(df)

    # ✅ Convert to list
    scores_list = scores.tolist()

    # ✅ Get latest score
    latest_score = float(scores_list[-1]) if len(scores_list) > 0 else 0.0

    mean = np.mean(scores_list)
    std = np.std(scores_list)

    if latest_score < mean:
        status = "NORMAL ✅"
    elif latest_score < mean + std:
        status = "WARNING ⚠️"
    else:
        status = "ANOMALY 🔴"

    return {
        "anomaly_scores": scores_list,
        "anomaly_score_latest": latest_score,
        "status": status
    }


@app.post("/health_score")
def health_score(
    batch: TelemetryBatch,
    horizon: float = 30.0
) -> Dict[str, Any]:
    """
    Compute overall engine health score (final interpreted output).
    """
    df = _batch_to_dataframe(batch)

    # 🔹 Get all model outputs
    rul_preds = service.predict_rul(df)
    probs = service.predict_failure_probability(df, horizon=horizon)
    anomaly_scores = service.anomaly_scores(df)

    # 🔹 Extract latest values
    rul_lstm = rul_preds["rul_lstm"]
    prob_lstm = probs["failure_prob_lstm"]
    anomaly_list = anomaly_scores.tolist()

    latest_rul = float(rul_lstm[-1]) if len(rul_lstm) > 0 else 0.0
    latest_failure = float(prob_lstm[-1]) if len(prob_lstm) > 0 else 0.0
    latest_anomaly = float(anomaly_list[-1]) if len(anomaly_list) > 0 else 0.0

    # ---------------------------
    # 🔹 NORMALIZATION
    # ---------------------------
    rul_score = min(latest_rul / 150.0, 1.0)
    failure_score = 1.0 - latest_failure
    anomaly_score = 1.0 - latest_anomaly

    # ---------------------------
    # 🔹 FINAL HEALTH SCORE
    # ---------------------------
    health_score = (
        0.5 * rul_score +
        0.3 * failure_score +
        0.2 * anomaly_score
    )

    # ---------------------------
    # 🔹 STATUS LOGIC (FINAL TUNED)
    # ---------------------------
    if latest_failure > 0.7 or latest_anomaly > 0.75:
        status = "CRITICAL 🔴"
        reason = "High failure risk or strong anomaly detected"
    elif latest_failure > 0.3 or latest_anomaly > 0.55:
        status = "WARNING ⚠️"
        reason = "Moderate anomaly or increasing failure risk"
    else:
        status = "GOOD ✅"
        reason = "Engine operating within normal conditions"

    return {
        "rul": round(latest_rul, 2),
        "failure_probability": round(latest_failure, 4),
        "anomaly_score": round(latest_anomaly, 4),
        "health_score": round(health_score, 3),
        "status": status,
        "reason": reason
    }


__all__ = ["app"]