from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class RandomForestConfig:
    """
    Configuration for the Random Forest baseline model.

    Defaults are chosen to be CPU-friendly while still reasonably expressive.
    """

    n_estimators: int = 150
    max_depth: Optional[int] = None
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    n_jobs: int = -1
    random_state: int = 42

    model_dir: Path = Path("models")
    model_filename: str = "random_forest_fd001.joblib"

    def ensure_model_dir(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename


class RandomForestRULModel:
    """
    Wrapper around sklearn's RandomForestRegressor for RUL prediction.
    """

    def __init__(self, config: Optional[RandomForestConfig] = None) -> None:
        self.config = config or RandomForestConfig()
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Compute RMSE and MAE for the provided dataset.
        """
        y_pred = self.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Optional[Path] = None) -> None:
        """
        Persist the trained model to disk.
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
    def load(cls, path: Optional[Path] = None, config: Optional[RandomForestConfig] = None) -> "RandomForestRULModel":
        """
        Load a previously saved model.
        """
        cfg = config or RandomForestConfig()
        target_path = path or cfg.model_path
        if not target_path.exists():
            raise FileNotFoundError(f"Random Forest model not found at {target_path}")

        payload: Dict[str, Any] = joblib.load(target_path)
        loaded_config: RandomForestConfig = payload.get("config", cfg)
        loaded_model: RandomForestRegressor = payload["model"]

        instance = cls(loaded_config)
        instance.model = loaded_model
        return instance


__all__ = ["RandomForestConfig", "RandomForestRULModel"]

