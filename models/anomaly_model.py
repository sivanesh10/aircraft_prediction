from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyConfig:
    """
    Configuration for the Isolation Forest anomaly detector.
    """

    n_estimators: int = 200
    max_samples: str = "auto"
    contamination: float = 0.05
    random_state: int = 42

    model_dir: Path = Path("models")
    model_filename: str = "isolation_forest_fd001.joblib"

    def ensure_model_dir(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename


class AnomalyDetector:
    """
    Isolation Forest based anomaly detector for engine sensor data.
    """

    def __init__(self, config: Optional[AnomalyConfig] = None) -> None:
        self.config = config or AnomalyConfig()
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the Isolation Forest on historical sensor data assumed to be mostly normal.
        """
        self.model.fit(X)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for each sample.

        IsolationForest.score_samples returns higher scores for inliers,
        so we invert it to represent anomaly intensity.
        """
        raw_scores = self.model.score_samples(X)
        return -raw_scores

    def predict_flags(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Convert anomaly scores into binary flags.

        If no threshold is provided, use the model's built-in predict method,
        which classifies -1 as anomaly.
        """
        if threshold is None:
            preds = self.model.predict(X)
            return (preds == -1).astype(int)

        scores = self.anomaly_score(X)
        return (scores > threshold).astype(int)

    def fit_from_dataframe(self, df: pd.DataFrame, feature_cols: Optional[list] = None) -> None:
        """
        Convenience wrapper to fit on a DataFrame directly.
        """
        cols = feature_cols or [c for c in df.columns if c not in {"engine_id", "cycle"}]
        X = df[cols].values.astype(np.float32)
        self.fit(X)

    def scores_from_dataframe(self, df: pd.DataFrame, feature_cols: Optional[list] = None) -> pd.Series:
        """
        Compute anomaly scores for each row in a DataFrame and return as a Series
        aligned with the DataFrame index.
        """
        cols = feature_cols or [c for c in df.columns if c not in {"engine_id", "cycle"}]
        X = df[cols].values.astype(np.float32)
        scores = self.anomaly_score(X)
        return pd.Series(scores, index=df.index, name="anomaly_score")

    def save(self, path: Optional[Path] = None) -> None:
        """
        Persist the trained anomaly detector to disk.
        """
        target_path = path or self.config.model_path
        self.config.ensure_model_dir()
        joblib.dump(
            {
                "config": self.config,
                "model": self.model,
            },
            target_path,
        )

    @classmethod
    def load(cls, path: Optional[Path] = None, config: Optional[AnomalyConfig] = None) -> "AnomalyDetector":
        """
        Load a previously saved anomaly detector.
        """
        cfg = config or AnomalyConfig()
        target_path = path or cfg.model_path
        if not target_path.exists():
            raise FileNotFoundError(f"Anomaly model not found at {target_path}")

        payload: Dict[str, Any] = joblib.load(target_path)
        loaded_config: AnomalyConfig = payload.get("config", cfg)
        loaded_model: IsolationForest = payload["model"]

        instance = cls(loaded_config)
        instance.model = loaded_model
        return instance


__all__ = ["AnomalyConfig", "AnomalyDetector"]

