from dataclasses import dataclass
from typing import Optional, Dict, Any
import sys
from pathlib import Path

import numpy as np

# Add project root to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .preprocessing import PreprocessingConfig, preprocess_fd001_for_models
from .evaluate import metrics_dict
from models.random_forest import RandomForestConfig, RandomForestRULModel
from models.lstm_model import LSTMConfig, LSTMRULTrainer
from models.anomaly_model import AnomalyConfig, AnomalyDetector


@dataclass
class TrainingConfig:
    """
    High-level training configuration, tying together preprocessing and models.
    """

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    rf: RandomForestConfig = RandomForestConfig()
    lstm: LSTMConfig = LSTMConfig()
    anomaly: AnomalyConfig = AnomalyConfig()


def train_random_forest(cfg: TrainingConfig) -> Dict[str, Any]:
    """
    Train the Random Forest baseline on FD001 and return metrics.
    """
    data = preprocess_fd001_for_models(cfg.preprocessing)
    X_train, y_train = data["train_rf"]
    X_test, y_test = data["test_rf"]

    rf_model = RandomForestRULModel(cfg.rf)
    rf_model.fit(X_train, y_train)
    rf_model.save()

    train_metrics = rf_model.evaluate(X_train, y_train)
    test_metrics = rf_model.evaluate(X_test, y_test)

    return {
        "train": train_metrics,
        "test": test_metrics,
    }


def train_lstm(cfg: TrainingConfig) -> Dict[str, Any]:
    """
    Train the LSTM RUL model on FD001 and return metrics.
    """
    data = preprocess_fd001_for_models(cfg.preprocessing)
    X_train, y_train = data["train_lstm"]
    X_test, y_test = data["test_lstm"]

    # Simple split of training set into train/val
    num_train = X_train.shape[0]
    split = int(num_train * 0.8)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    lstm_trainer = LSTMRULTrainer(cfg.lstm)
    history = lstm_trainer.fit(X_tr, y_tr, X_val, y_val)
    lstm_trainer.save()

    # Evaluate on test set
    y_test_pred = lstm_trainer.predict(X_test)
    test_metrics = metrics_dict(y_test, y_test_pred)

    return {
        "history": history,
        "test": test_metrics,
    }


def train_anomaly_detector(cfg: TrainingConfig) -> None:
    """
    Train the Isolation Forest anomaly model on FD001 sensor data.

    We reuse the normalized training features produced by the preprocessing step.
    """
    data = preprocess_fd001_for_models(cfg.preprocessing)
    X_train_lstm, _ = data["train_lstm"]

    # Collapse sequences into per-cycle samples by averaging across time.
    X_cycle = np.mean(X_train_lstm, axis=1)

    anomaly_model = AnomalyDetector(cfg.anomaly)
    anomaly_model.fit(X_cycle)
    anomaly_model.save()


__all__ = ["TrainingConfig", "train_random_forest", "train_lstm", "train_anomaly_detector"]

