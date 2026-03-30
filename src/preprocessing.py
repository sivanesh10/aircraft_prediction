import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


FD001_DEFAULT_COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


@dataclass
class PreprocessingConfig:
    """
    Configuration for FD001 preprocessing.

    This keeps all paths and hyperparameters in one place so that
    training, evaluation, and the backend can share consistent settings.
    """

from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class PreprocessingConfig:
    # 🔥 This line fixes everything
    base_dir: Path = Path(__file__).resolve().parent.parent

    data_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)

    train_filename: str = "train_FD001.txt"
    test_filename: str = "test_FD001.txt"
    rul_filename: str = "RUL_FD001.txt"

    sequence_length: int = 30
    sequence_stride: int = 1

    feature_columns: Optional[List[str]] = None
    scaler_filename: str = "scaler_fd001.joblib"

    def __post_init__(self):
        self.data_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"

    def ensure_processed_dir(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def scaler_path(self) -> Path:
        return self.processed_dir / self.scaler_filename


    def ensure_processed_dir(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def scaler_path(self) -> Path:
        return self.processed_dir / self.scaler_filename


def _read_fd001_file(path: Path) -> pd.DataFrame:
    """
    Read a single FD001 txt file into a DataFrame.

    The NASA C-MAPSS FD001 files are space-separated with variable whitespace
    and may end with extra empty columns, so we use `delim_whitespace=True`
    and then select the expected columns.
    """
    df = pd.read_csv(path, delim_whitespace=True, header=None)

    # Drop any all-NaN trailing columns that sometimes appear.
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < len(FD001_DEFAULT_COLUMNS):
        raise ValueError(
            f"Unexpected column count in {path}. "
            f"Got {df.shape[1]}, expected at least {len(FD001_DEFAULT_COLUMNS)}."
        )

    df = df.iloc[:, : len(FD001_DEFAULT_COLUMNS)]
    df.columns = FD001_DEFAULT_COLUMNS
    return df


def load_fd001_data(config: PreprocessingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load FD001 train, test, and RUL label files into memory.

    This function assumes all three files live under `config.data_dir`, but
    the actual filenames are configurable so you can point to custom names.
    """
    train_path = config.data_dir / config.train_filename
    test_path = config.data_dir / config.test_filename
    rul_path = config.data_dir / config.rul_filename

    if not train_path.exists() or not test_path.exists() or not rul_path.exists():
        missing = [str(p) for p in [train_path, test_path, rul_path] if not p.exists()]
        raise FileNotFoundError(f"One or more FD001 files are missing: {missing}")

    train_df = _read_fd001_file(train_path)
    test_df = _read_fd001_file(test_path)

    rul_df = pd.read_csv(rul_path, header=None, names=["RUL"])
    rul_series = rul_df["RUL"]

    return train_df, test_df, rul_series


def create_rul_labels_train(train_df: pd.DataFrame) -> pd.Series:
    """
    Create RUL labels for the FD001 training set.

    Standard C-MAPSS convention:
    RUL = (max cycle for that engine) - current_cycle
    """
    max_cycles = train_df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    merged = train_df.merge(max_cycles, on="engine_id", how="left")
    rul = merged["max_cycle"] - merged["cycle"]
    return rul.astype(float)


def create_rul_labels_test(test_df: pd.DataFrame, rul_series: pd.Series) -> pd.Series:
    """
    Create RUL labels for the FD001 test set using the provided RUL file.

    The RUL_FD001.txt file provides the RUL for each engine at its last observed cycle.
    We propagate that back to each cycle via:

    RUL(cycle) = RUL_last_engine + (max_cycle_engine - cycle)
    """
    max_cycles = test_df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    merged = test_df.merge(max_cycles, on="engine_id", how="left")

    # RUL file is ordered by engine_id starting from 1.
    rul_last_by_engine: Dict[int, float] = {
        engine_id: float(rul_series.iloc[engine_id - 1])
        for engine_id in merged["engine_id"].unique()
    }

    rul_last = merged["engine_id"].map(rul_last_by_engine)
    rul = rul_last + (merged["max_cycle"] - merged["cycle"])
    return rul.astype(float)


def get_feature_columns(config: PreprocessingConfig, df: pd.DataFrame) -> List[str]:
    """
    Determine which columns to use as model features.

    By default we take all settings and sensor columns (everything except
    engine_id, cycle, and any explicit label columns if present).
    """
    if config.feature_columns is not None:
        return config.feature_columns

    exclude = {"engine_id", "cycle"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str], config: PreprocessingConfig) -> StandardScaler:
    """
    Fit a StandardScaler on the training features and persist it under `processed_dir`.
    """
    config.ensure_processed_dir()
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values.astype(np.float32))
    joblib.dump(scaler, config.scaler_path)
    return scaler


def load_scaler(config: PreprocessingConfig) -> StandardScaler:
    """
    Load a previously fitted scaler. Raises if it does not exist.
    """
    if not config.scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {config.scaler_path}. "
            f"Make sure to run preprocessing/training first."
        )
    return joblib.load(config.scaler_path)


def transform_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply feature scaling to the specified columns, preserving original index
    and non-feature columns.
    """
    scaled_values = scaler.transform(df[feature_cols].values.astype(np.float32))
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaled_values
    return scaled_df


def _sliding_windows_indices(length: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    """
    Compute (start, end) indices for sliding windows over a sequence of given length.
    """
    indices: List[Tuple[int, int]] = []
    if length < window_size:
        return indices

    for start in range(0, length - window_size + 1, stride):
        end = start + window_size
        indices.append((start, end))
    return indices


def create_sequences_lstm(
    df: pd.DataFrame,
    rul: pd.Series,
    feature_cols: List[str],
    config: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D sequences for LSTM models.

    Returns:
        X: shape (num_samples, seq_len, num_features)
        y: shape (num_samples,)
    """
    seq_len = config.sequence_length
    stride = config.sequence_stride

    sequences: List[np.ndarray] = []
    labels: List[float] = []

    # Work per-engine to respect temporal ordering.
    grouped = df.sort_values(["engine_id", "cycle"]).groupby("engine_id")

    for engine_id, engine_df in grouped:
        engine_rul = rul.loc[engine_df.index]
        feature_values = engine_df[feature_cols].values.astype(np.float32)
        window_indices = _sliding_windows_indices(len(engine_df), seq_len, stride)

        for start, end in window_indices:
            window_features = feature_values[start:end]
            # Use RUL at the last cycle in the window as label.
            window_label = float(engine_rul.iloc[end - 1])
            sequences.append(window_features)
            labels.append(window_label)

    if not sequences:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.stack(sequences, axis=0)
    y = np.array(labels, dtype=np.float32)
    return X, y


def create_features_rf(
    df: pd.DataFrame,
    rul: pd.Series,
    feature_cols: List[str],
    config: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fixed-length feature vectors for tree-based models from sliding windows.

    For simplicity and CPU efficiency we flatten each window:
        [f1_t1, f2_t1, ..., fN_t1, f1_t2, ..., fN_tL]

    Returns:
        X_rf: shape (num_samples, seq_len * num_features)
        y_rf: shape (num_samples,)
    """
    seq_len = config.sequence_length
    stride = config.sequence_stride

    vectors: List[np.ndarray] = []
    labels: List[float] = []

    grouped = df.sort_values(["engine_id", "cycle"]).groupby("engine_id")

    for engine_id, engine_df in grouped:
        engine_rul = rul.loc[engine_df.index]
        feature_values = engine_df[feature_cols].values.astype(np.float32)
        window_indices = _sliding_windows_indices(len(engine_df), seq_len, stride)

        for start, end in window_indices:
            window_features = feature_values[start:end].reshape(-1)
            window_label = float(engine_rul.iloc[end - 1])
            vectors.append(window_features)
            labels.append(window_label)

    if not vectors:
        return np.empty((0, seq_len * len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_rf = np.stack(vectors, axis=0)
    y_rf = np.array(labels, dtype=np.float32)
    return X_rf, y_rf


def preprocess_fd001_for_models(
    config: Optional[PreprocessingConfig] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    High-level convenience function to run the full FD001 preprocessing pipeline.

    It returns a dictionary with arrays ready for training:
        {
            "train_lstm": (X_train_lstm, y_train_lstm),
            "test_lstm": (X_test_lstm, y_test_lstm),
            "train_rf": (X_train_rf, y_train_rf),
            "test_rf": (X_test_rf, y_test_rf),
        }

    This function assumes the standard FD001 train/test split and does not create
    an additional validation set. Validation can be created later by splitting
    the returned training arrays.
    """
    if config is None:
        config = PreprocessingConfig()

    train_df, test_df, rul_series_test_last = load_fd001_data(config)

    # RUL labels
    rul_train = create_rul_labels_train(train_df)
    rul_test = create_rul_labels_test(test_df, rul_series_test_last)

    # Feature selection and scaling
    feature_cols = get_feature_columns(config, train_df)
    scaler = fit_scaler(train_df, feature_cols, config)

    train_scaled = transform_features(train_df, feature_cols, scaler)
    test_scaled = transform_features(test_df, feature_cols, scaler)

    # Sequences for LSTM
    X_train_lstm, y_train_lstm = create_sequences_lstm(train_scaled, rul_train, feature_cols, config)
    X_test_lstm, y_test_lstm = create_sequences_lstm(test_scaled, rul_test, feature_cols, config)

    # Features for Random Forest
    X_train_rf, y_train_rf = create_features_rf(train_scaled, rul_train, feature_cols, config)
    X_test_rf, y_test_rf = create_features_rf(test_scaled, rul_test, feature_cols, config)

    return {
        "train_lstm": (X_train_lstm, y_train_lstm),
        "test_lstm": (X_test_lstm, y_test_lstm),
        "train_rf": (X_train_rf, y_train_rf),
        "test_rf": (X_test_rf, y_test_rf),
    }


__all__ = [
    "PreprocessingConfig",
    "load_fd001_data",
    "create_rul_labels_train",
    "create_rul_labels_test",
    "get_feature_columns",
    "fit_scaler",
    "load_scaler",
    "transform_features",
    "create_sequences_lstm",
    "create_features_rf",
    "preprocess_fd001_for_models",
]

