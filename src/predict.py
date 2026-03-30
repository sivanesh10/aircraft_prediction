from dataclasses import dataclass
from typing import Dict, Any, Optional
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .preprocessing import (
    PreprocessingConfig,
    load_scaler,
    get_feature_columns,
    transform_features,
    create_sequences_lstm,
    create_features_rf,
    create_rul_labels_train,
)
from models.random_forest import RandomForestRULModel, RandomForestConfig
from models.lstm_model import LSTMRULTrainer, LSTMConfig
from models.anomaly_model import AnomalyDetector, AnomalyConfig


@dataclass
class InferenceConfig:
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    rf: RandomForestConfig = RandomForestConfig()
    lstm: LSTMConfig = LSTMConfig()
    anomaly: AnomalyConfig = AnomalyConfig()


class PredictiveMaintenanceService:
    """
    High-level facade for running RUL predictions, failure probabilities,
    anomaly scores, and health scores on top of trained models.
    """

    def __init__(self, cfg: Optional[InferenceConfig] = None) -> None:
        self.cfg = cfg or InferenceConfig()

        self.scaler = load_scaler(self.cfg.preprocessing)
        self.rf_model = RandomForestRULModel.load(config=self.cfg.rf)
        self.lstm_trainer = LSTMRULTrainer.load(config=self.cfg.lstm)
        self.anomaly_model = AnomalyDetector.load(config=self.cfg.anomaly)

    def _prepare_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Normalize a raw engine DataFrame and create features for RF and LSTM.
        """
        feature_cols = get_feature_columns(self.cfg.preprocessing, df)
        df_scaled = transform_features(df, feature_cols, self.scaler)

        # For inference, treat each engine's history as-is and construct windows.
        dummy_rul = create_rul_labels_train(df_scaled)

        X_lstm, y_lstm = create_sequences_lstm(df_scaled, dummy_rul, feature_cols, self.cfg.preprocessing)
        X_rf, y_rf = create_features_rf(df_scaled, dummy_rul, feature_cols, self.cfg.preprocessing)

        # For anomaly detector, we use average over windows to form per-sample summary.
        X_anom = None
        if X_lstm.size > 0:
            X_anom = np.mean(X_lstm, axis=1)

        return {
            "X_lstm": X_lstm,
            "y_lstm": y_lstm,
            "X_rf": X_rf,
            "y_rf": y_rf,
            "X_anom": X_anom,
        }

    def predict_rul(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict RUL using both RF and LSTM and return window-level predictions.
        """
        feats = self._prepare_features(df)

        X_lstm = feats["X_lstm"]
        X_rf = feats["X_rf"]

        if X_rf.size == 0 and X_lstm.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Not enough telemetry records. Send at least ~30 cycles."
            )

        rul_rf = self.rf_model.predict(X_rf) if X_rf.size > 0 else np.array([])
        rul_lstm = self.lstm_trainer.predict(X_lstm) if X_lstm.size > 0 else np.array([])

        return {
            "rul_rf": rul_rf,
            "rul_lstm": rul_lstm,
        }

    @staticmethod
    def _rul_to_failure_probability(rul: np.ndarray, horizon: float = 30.0) -> np.ndarray:
        """
        Map RUL values to failure probabilities within a given horizon.

        Here we use a simple logistic-like transform to keep things lightweight.
        """
        # Avoid division by zero
        horizon = max(horizon, 1.0)
        scaled = np.exp(-rul / horizon)
        return scaled

    def predict_failure_probability(self, df: pd.DataFrame, horizon: float = 30.0) -> Dict[str, Any]:
        """
        Compute failure probabilities derived from RUL predictions.
        """
        rul_preds = self.predict_rul(df)
        prob_rf = self._rul_to_failure_probability(rul_preds["rul_rf"], horizon)
        prob_lstm = self._rul_to_failure_probability(rul_preds["rul_lstm"], horizon)

        return {
            "failure_prob_rf": prob_rf,
            "failure_prob_lstm": prob_lstm,
        }

    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores per window (aggregated from sequences).
        """
        feats = self._prepare_features(df)
        X_anom = feats["X_anom"]
        if X_anom is None or X_anom.size == 0:
            return np.array([])
        return self.anomaly_model.anomaly_score(X_anom)

    def health_score(self, df: pd.DataFrame, horizon: float = 30.0) -> Dict[str, float]:
        """
        Combine RUL, failure probability, and anomaly scores into a simple
        normalized health score.
        """
        rul_preds = self.predict_rul(df)
        probs = self.predict_failure_probability(df, horizon)
        anomalies = self.anomaly_scores(df)

        # For a high-level health score we take the mean across windows.
        def _safe_mean(arr: np.ndarray) -> float:
            return float(np.mean(arr)) if arr.size > 0 else float("nan")

        avg_rul_lstm = _safe_mean(rul_preds["rul_lstm"])
        avg_failure_prob = _safe_mean(probs["failure_prob_lstm"])
        avg_anomaly = _safe_mean(anomalies)

        # Simple heuristic: start from perfect health = 1.0 and subtract risk terms.
        risk = 0.5 * (avg_failure_prob if not np.isnan(avg_failure_prob) else 0.0)
        if not np.isnan(avg_anomaly):
            risk += 0.5 * (avg_anomaly - np.min(anomalies)) / (np.ptp(anomalies) + 1e-8)

        health = float(np.clip(1.0 - risk, 0.0, 1.0))

        return {
            "avg_rul_lstm": avg_rul_lstm,
            "avg_failure_probability": avg_failure_prob,
            "avg_anomaly_score": avg_anomaly,
            "health_score": health,
        }


__all__ = ["InferenceConfig", "PredictiveMaintenanceService"]

