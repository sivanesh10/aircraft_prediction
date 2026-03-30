from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowAggregationConfig:
    """
    Configuration for window-level feature aggregation.

    These aggregations are used for tree-based models (Random Forest)
    and for anomaly detection, where fixed-length vectors are preferred.
    """

    aggregations: Tuple[str, ...] = ("mean", "std", "min", "max", "last")


def aggregate_window_features(
    window: np.ndarray,
    feature_names: List[str],
    config: WindowAggregationConfig,
) -> Dict[str, float]:
    """
    Aggregate a single sliding window (seq_len, num_features) into
    a flat dictionary of statistics per feature.

    The resulting keys follow the pattern: `<feature>_<agg>`.
    """
    agg_result: Dict[str, float] = {}

    for j, feat_name in enumerate(feature_names):
        series = window[:, j]

        if "mean" in config.aggregations:
            agg_result[f"{feat_name}_mean"] = float(np.mean(series))
        if "std" in config.aggregations:
            # Small epsilon to avoid NaNs when window length is 1
            agg_result[f"{feat_name}_std"] = float(np.std(series) + 1e-8)
        if "min" in config.aggregations:
            agg_result[f"{feat_name}_min"] = float(np.min(series))
        if "max" in config.aggregations:
            agg_result[f"{feat_name}_max"] = float(np.max(series))
        if "last" in config.aggregations:
            agg_result[f"{feat_name}_last"] = float(series[-1])

    return agg_result


def build_aggregated_feature_matrix(
    windows: np.ndarray,
    feature_names: List[str],
    config: WindowAggregationConfig,
) -> pd.DataFrame:
    """
    Convert a 3D array of windows into a 2D aggregated feature matrix.

    Args:
        windows: Array with shape (num_samples, seq_len, num_features).
        feature_names: Names of the base features corresponding to the
            last dimension of `windows`.
        config: Aggregation configuration.

    Returns:
        DataFrame of shape (num_samples, num_aggregated_features).
    """
    if windows.ndim != 3:
        raise ValueError(
            f"`windows` must be 3D (num_samples, seq_len, num_features), got shape {windows.shape}."
        )

    records = []
    for i in range(windows.shape[0]):
        window = windows[i]
        record = aggregate_window_features(window, feature_names, config)
        records.append(record)

    return pd.DataFrame.from_records(records)


__all__ = [
    "WindowAggregationConfig",
    "aggregate_window_features",
    "build_aggregated_feature_matrix",
]

