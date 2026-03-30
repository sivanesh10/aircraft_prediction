from typing import Tuple, Dict

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute RMSE and MAE between true and predicted RUL values.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Convenience wrapper returning a metrics dictionary.
    """
    rmse, mae = compute_rmse_mae(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}


__all__ = ["compute_rmse_mae", "metrics_dict"]

