## Aircraft Predictive Maintenance System (NASA C-MAPSS FD001)

This project implements a production-oriented, CPU-friendly predictive maintenance pipeline around the NASA C-MAPSS FD001 dataset, including:

- LSTM model for Remaining Useful Life (RUL) prediction
- Random Forest baseline model
- Isolation Forest anomaly detection
- FastAPI backend
- Streamlit aerospace-style dashboard with SHAP explainability and PDF reporting

### 1. Project structure

- `data/raw/` – place the raw `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` files here.
- `data/processed/` – scalers and processed artifacts.
- `models/` – saved Random Forest, LSTM, and anomaly models.
- `src/` – preprocessing, feature engineering, training, evaluation, prediction.
- `backend/` – FastAPI app (`main.py`).
- `dashboard/` – Streamlit UI (`dashboard.py`).
- `reports/` – generated reports (optional; PDF downloads stream from memory).

### 2. Environment setup

```bash
cd aircraft_predictive_maintenance
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Data placement

1. Download the NASA C-MAPSS FD001 subset.
2. Copy the following files into `data/raw/`:
   - `train_FD001.txt`
   - `test_FD001.txt`
   - `RUL_FD001.txt`

The preprocessing code expects this layout by default.

### 4. Training the models

The high-level training entry points live in `src/train.py`.

In a Python shell or script:

```python
from src.train import TrainingConfig, train_random_forest, train_lstm, train_anomaly_detector

cfg = TrainingConfig()

rf_metrics = train_random_forest(cfg)
print("Random Forest metrics:", rf_metrics)

lstm_metrics = train_lstm(cfg)
print("LSTM metrics:", lstm_metrics)

train_anomaly_detector(cfg)
print("Anomaly detector trained.")
```

This will:

- Run preprocessing on FD001 (labeling, normalization, sliding windows).
- Train and save:
  - Random Forest (`models/random_forest_fd001.joblib`)
  - LSTM (`models/lstm_fd001.pt`)
  - Isolation Forest (`models/isolation_forest_fd001.joblib`)

### 5. Running the FastAPI backend

Start the API (after models are trained and environment is active):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Available endpoints (JSON):

- `POST /predict_rul` – RUL predictions (RF + LSTM).
- `POST /predict_failure_probability` – failure probabilities within a configurable horizon.
- `POST /anomaly_score` – anomaly scores per window.
- `POST /health_score` – aggregate health score per aircraft.

All endpoints expect a JSON body of the form:

```json
{
  "records": [
    {
      "engine_id": 1,
      "cycle": 1,
      "setting_1": 0.0,
      "setting_2": 0.0,
      "setting_3": 0.0,
      "sensor_1": 0.0,
      "...": 0.0,
      "sensor_21": 0.0
    }
  ]
}
```

### 6. Running the Streamlit dashboard

With the backend running:

```bash
streamlit run dashboard/dashboard.py
```

The dashboard provides:

- Dark aerospace control-room theme.
- Auto-refresh cadence (configurable; default 5 seconds).
- Fleet comparison dropdown (select engine).
- RUL trend line chart (LSTM).
- Failure probability gauge.
- SHAP feature importance bar chart for Random Forest.
- Sensor anomaly heatmap across a small fleet.
- Maintenance recommendation panel.
- Downloadable PDF maintenance report per engine.

### 7. SHAP explainability

The dashboard loads the trained Random Forest model and uses SHAP TreeExplainer on a small sample of recent telemetry to compute:

- Mean absolute SHAP values per feature.
- A bar chart for the top contributing sensors/settings.

This keeps runtime overhead low while still providing interpretable importance rankings.

### 8. Notes on CPU optimization

- LSTM model uses modest hidden size and layers, with configurable batch size and epochs, and runs on CPU.
- Random Forest and Isolation Forest use reasonable defaults and `n_jobs=-1` to utilize available cores.
- Preprocessing is vectorized with `pandas`/`numpy`, and sequence length is configurable (default 30).

### 9. Next steps

- Extend configuration (e.g., via a `config.py` or YAML) for more flexible hyperparameter tuning.
- Integrate real-time telemetry ingestion and persistence (e.g., message queue + database).
- Harden security and deployment (Docker, reverse proxy, authentication) for production environments.

