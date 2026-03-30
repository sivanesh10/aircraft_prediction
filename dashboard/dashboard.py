import io
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


API_BASE_URL = "http://localhost:8000"


CUSTOM_CSS = """
<style>
/* Global dark aerospace theme */
body, .stApp {
    background-color: #050814;
    color: #f5f7ff;
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050814 0%, #060b1a 40%, #02030a 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: radial-gradient(circle at top left, #152a63, #050814);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 0 30px rgba(0, 200, 255, 0.15);
}

/* Headers */
h1, h2, h3 {
    color: #e5ecff;
    letter-spacing: 0.04em;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #00bcd4, #2962ff);
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
}
.stButton>button:hover {
    box-shadow: 0 0 20px rgba(0, 188, 212, 0.6);
}

/* Cards / containers */
.aero-panel {
    background: radial-gradient(circle at top left, rgba(41, 98, 255, 0.25), rgba(5, 8, 20, 0.95));
    border-radius: 16px;
    padding: 1rem 1.4rem;
    border: 1px solid rgba(0, 172, 193, 0.4);
    box-shadow: 0 0 25px rgba(0, 172, 193, 0.25);
}

/* Hide Streamlit default header & footer */
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def call_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{endpoint}"
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_payload_from_demo(engine_id: int) -> Dict[str, Any]:

    cycles = np.arange(1, 51)
    rng = np.random.default_rng(seed=engine_id)

    sensor_base = [
        518.7, 641.8, 1589.7, 1400.6, 14.6, 21.6, 554.3,
        2388.0, 9046.2, 1.3, 47.4, 521.6, 2388.0,
        8138.6, 8.41, 0.03, 392, 2388, 100, 39.0, 23.4
    ]

    data = []

    for c in cycles:
        record = {
            "engine_id": engine_id,
            "cycle": int(c),
            "setting_1": float(rng.normal(0.0, 1.0)),
            "setting_2": float(rng.normal(0.0, 1.0)),
            "setting_3": float(rng.normal(0.0, 1.0)),
        }

        for s in range(1, 22):
            record[f"sensor_{s}"] = float(sensor_base[s-1] + rng.normal(0, 0.2))

        data.append(record)

    return {"records": data}


def plot_rul_trend(cycles: np.ndarray, rul_lstm: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=rul_lstm[: len(cycles)],
            mode="lines+markers",
            line=dict(color="#00bcd4", width=3),
            marker=dict(size=6),
            name="Predicted RUL (LSTM)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,20,0.9)",
        xaxis_title="Cycle",
        yaxis_title="RUL (cycles)",
    )
    return fig


def plot_failure_gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100.0,
            title={"text": "Failure Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff6e40"},
                "steps": [
                    {"range": [0, 40], "color": "#1b5e20"},
                    {"range": [40, 70], "color": "#f9a825"},
                    {"range": [70, 100], "color": "#b71c1c"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_anomaly_heatmap(engine_ids: List[int], anomaly_matrix: np.ndarray) -> go.Figure:
    fig = px.imshow(
        anomaly_matrix,
        labels=dict(x="Cycle Window", y="Engine", color="Anomaly Score"),
        x=[f"W{i+1}" for i in range(anomaly_matrix.shape[1])],
        y=[f"E{eid}" for eid in engine_ids],
        color_continuous_scale="Turbo",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=40, b=40),
    )
    return fig


def generate_pdf_report(
    aircraft_id: int,
    predicted_rul: float,
    failure_prob: float,
    top_sensors: List[str],
    recommendation: str,
) -> bytes:
    """
    Generate a PDF maintenance report into a bytes buffer.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFillColorRGB(0.0, 0.8, 1.0)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(20 * mm, height - 30 * mm, "Aircraft Predictive Maintenance Report")

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica", 11)
    y = height - 45 * mm

    lines = [
        f"Date & Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Aircraft ID: {aircraft_id}",
        f"Predicted RUL: {predicted_rul:.1f} cycles",
        f"Failure Probability (next 30 cycles): {failure_prob * 100.0:.1f}%",
        "Top Contributing Sensors: " + ", ".join(top_sensors),
        "",
        "Maintenance Recommendation:",
        recommendation,
    ]

    for line in lines:
        c.drawString(20 * mm, y, line)
        y -= 7 * mm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


def main() -> None:
    st.set_page_config(
        page_title="Aerospace Predictive Maintenance Control Room",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🛩️",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Top hero header
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.markdown("### 🛩️")
    with col_title:
        st.markdown("## Aerospace Predictive Maintenance Control Room")
        st.caption("NASA C-MAPSS FD001 · LSTM + Random Forest · Isolation Forest Anomaly Detection")

    # Sidebar: fleet selection and controls
    with st.sidebar:
        st.markdown("### ✈ Fleet Selection")
        fleet_ids = list(range(1, 11))
        selected_engine = st.selectbox("Select Aircraft Engine", fleet_ids, index=0)

        st.markdown("---")
        st.markdown("### ⏱ Refresh & Horizon")
        refresh_seconds = st.slider("Refresh interval hint (seconds)", 5, 60, 5, step=5)
        horizon = st.slider("Failure Horizon (cycles)", 10, 60, 30, step=5)
        st.markdown(
            f"<small>Use this interval as a guideline for how often to manually refresh.</small>",
            unsafe_allow_html=True,
        )

    # Note: auto-refresh is disabled; users can refresh the page manually when needed.

    # Layout: three main rows
    top_row = st.columns([2, 1, 1])
    mid_row = st.columns([2, 2])
    bottom_row = st.columns([2, 2])

    # Fetch data from backend for the selected engine
    payload = build_payload_from_demo(selected_engine)

    with st.spinner("Contacting prediction engine and updating fleet telemetry..."):
        try:
            rul_resp = call_api("/predict_rul", payload)
            prob_resp = call_api(f"/predict_failure_probability?horizon={horizon}", payload)
            anom_resp = call_api("/anomaly_score", payload)
            health_resp = call_api(f"/health_score?horizon={horizon}", payload)
        except Exception as e:
            st.error(f"Backend error: {e}")
            return

    # ----- TOP ROW: Metrics and gauge -----
    with top_row[0]:
        st.markdown('<div class="aero-panel">', unsafe_allow_html=True)
        st.markdown("#### 🛰 Fleet Snapshot")
        cols = st.columns(3)

        avg_rul = health_resp.get("avg_rul_lstm", float("nan"))
        avg_failure_prob = health_resp.get("avg_failure_probability", float("nan"))
        health = health_resp.get("health_score", float("nan"))

        delta_health = (health * 100.0) - 100.0 if not np.isnan(health) else 0.0

        cols[0].metric("Avg RUL (LSTM)", f"{avg_rul:.1f} cycles")
        cols[1].metric("Failure Prob (30c)", f"{avg_failure_prob * 100.0:.1f}%")
        cols[2].metric(
            "Health Score",
            f"{health * 100.0:.1f}%",
            delta=f"{delta_health:+.1f} pts" if not np.isnan(health) else None,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with top_row[1]:
        st.markdown('<div class="aero-panel">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Failure Probability Gauge")
        last_prob = float(prob_resp["failure_prob_lstm"][-1]) if prob_resp["failure_prob_lstm"] else 0.0
        st.plotly_chart(plot_failure_gauge(last_prob), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_row[2]:
        st.markdown('<div class="aero-panel">', unsafe_allow_html=True)
        st.markdown("#### 🛠 Maintenance Recommendation")
        if last_prob > 0.7:
            rec = "IMMEDIATE INSPECTION REQUIRED: Schedule unscheduled maintenance within the next 5 cycles."
        elif last_prob > 0.4:
            rec = "Plan maintenance window in the next 10–20 cycles and increase monitoring frequency."
        else:
            rec = "System operating nominally. Maintain standard inspection interval."
        st.write(rec)
        st.markdown("</div>", unsafe_allow_html=True)

    # ----- MIDDLE ROW: RUL trend & SHAP -----
    cycles = np.arange(1, len(rul_resp["rul_lstm"]) + 1)

    with mid_row[0]:
        st.markdown('<div class="aero-panel">', unsafe_allow_html=True)
        st.markdown("#### 📉 RUL Trend (LSTM)")
        st.plotly_chart(plot_rul_trend(cycles, np.array(rul_resp["rul_lstm"])), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with mid_row[1]:
        st.markdown('<div class="aero-panel">', unsafe_allow_html=True)
        st.markdown("#### 🔍 SHAP Feature Importance (Random Forest)")

        # For demo purposes, build a small sample matrix from payload and call SHAP on the RF model.
        demo_df = pd.DataFrame(payload["records"])
        feature_cols = [c for c in demo_df.columns if c not in {"engine_id", "cycle"}]
        X_sample = demo_df[feature_cols].values.astype(np.float32)[:100]

        try:
            from models.random_forest import RandomForestRULModel, RandomForestConfig

            rf_model = RandomForestRULModel.load(RandomForestConfig().model_path)
            explainer = shap.TreeExplainer(rf_model.model)
            shap_values = explainer.shap_values(X_sample)
            mean_abs = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-10:][::-1]
            top_features = [feature_cols[i] for i in top_idx]
            top_importance = mean_abs[top_idx]

            fig = px.bar(
                x=top_importance,
                y=top_features,
                orientation="h",
                labels={"x": "Mean |SHAP value|", "y": "Feature"},
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=80, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ----- BOTTOM ROW: Anomaly heatmap & PDF download -----
    with bottom_row[0]:
        st.markdown('<div class="aero-panel">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Sensor Anomaly Heatmap (Fleet)")

        # Build a simple fleet anomaly matrix by querying anomalies for all engines in the fleet.
        fleet_ids = list(range(1, 6))
        anomaly_matrix = []

        for eid in fleet_ids:
            p = build_payload_from_demo(eid)
            try:
                resp = call_api("/anomaly_score", p)
                scores = np.array(resp["anomaly_scores"])
            except Exception:
                scores = np.zeros(10)

            if scores.size == 0:
                scores = np.zeros(10)

            # Downsample/reshape into fixed-length windows for display
            window_len = min(10, scores.size)
            idx = np.linspace(0, scores.size - 1, window_len).astype(int)
            anomaly_matrix.append(scores[idx])

        anomaly_matrix_arr = np.vstack(anomaly_matrix)
        st.plotly_chart(
            plot_anomaly_heatmap(fleet_ids, anomaly_matrix_arr),
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_row[1]:
        st.markdown('<div class="aero-panel">', unsafe_allow_html=True)
        st.markdown("#### 📄 Download Maintenance PDF Report")

        top_sensors = ["sensor_2", "sensor_11", "sensor_15"]
        recommendation = rec
        pdf_bytes = generate_pdf_report(
            aircraft_id=selected_engine,
            predicted_rul=avg_rul,
            failure_prob=avg_failure_prob,
            top_sensors=top_sensors,
            recommendation=recommendation,
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"maintenance_report_engine_{selected_engine}.pdf",
            mime="application/pdf",
        )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

