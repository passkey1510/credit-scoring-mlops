"""Streamlit monitoring dashboard for the credit scoring API."""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests as http_requests
import streamlit as st

API_URL = "http://localhost:8000"
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "reference_data.parquet"
THRESHOLD = 0.11

st.set_page_config(page_title="Credit Scoring Monitor", layout="wide")
st.title("Credit Scoring — Monitoring Dashboard")


def load_logs() -> pd.DataFrame:
    """Load prediction logs from JSONL file."""
    if not LOG_PATH.exists():
        return pd.DataFrame()
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def send_test_predictions(n: int = 50):
    """Send test predictions using real data from the reference dataset."""
    ref = pd.read_parquet(DATA_PATH)
    sample = ref.sample(n=n, random_state=random.randint(0, 9999))
    success = 0
    for _, row in sample.iterrows():
        features = {}
        for col in row.index:
            val = row[col]
            if isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
                features[col] = float(val)
        try:
            resp = http_requests.post(
                f"{API_URL}/predict", json={"features": features}, timeout=5
            )
            if resp.status_code == 200:
                success += 1
        except Exception:
            pass
        time.sleep(0.1)
    return success


# --- Sidebar: send test predictions ---
with st.sidebar:
    st.header("Test Predictions")
    n_preds = st.slider("Number of predictions", 10, 100, 50)
    if st.button("Send test predictions"):
        with st.spinner(f"Sending {n_preds} predictions..."):
            ok = send_test_predictions(n_preds)
        st.success(f"Sent {ok}/{n_preds} predictions. Refresh the page to see results.")

df = load_logs()

if df.empty:
    st.info("No predictions logged yet. Use the sidebar to send test predictions.")
    st.stop()

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", len(df))
col2.metric("Default Rate", f"{df['prediction'].mean():.1%}")
col3.metric("Avg Latency", f"{df['latency_ms'].mean():.1f} ms")
col4.metric("Avg Probability", f"{df['probability'].mean():.4f}")

st.divider()

# --- Score distribution ---
left, right = st.columns(2)

with left:
    st.subheader("Score Distribution")
    counts, edges = np.histogram(df["probability"].dropna(), bins=20)
    bin_labels = [f"{edges[i]:.3f}" for i in range(len(counts))]
    hist_df = pd.DataFrame({"Score Range": bin_labels, "Count": counts})
    st.bar_chart(hist_df, x="Score Range", y="Count", use_container_width=True)

with right:
    st.subheader("Latency Over Time")
    latency_series = df.set_index("timestamp")["latency_ms"]
    st.line_chart(latency_series, use_container_width=True)

st.divider()

# --- Prediction outcome breakdown ---
left2, right2 = st.columns(2)

with left2:
    st.subheader("Prediction Outcomes")
    outcome_counts = df["prediction"].value_counts().rename({0: "Approved", 1: "Default"})
    fig = px.pie(
        names=outcome_counts.index,
        values=outcome_counts.values,
        color=outcome_counts.index,
        color_discrete_map={"Approved": "#2ecc71", "Default": "#e74c3c"},
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

with right2:
    st.subheader("Probability Over Time")
    prob_series = df.set_index("timestamp")["probability"]
    fig2 = px.line(prob_series, markers=True)
    fig2.add_hline(y=THRESHOLD, line_dash="dash", line_color="#e74c3c",
                   annotation_text=f"Threshold ({THRESHOLD})")
    fig2.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis_title="",
        yaxis_title="Probability",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Recent predictions table ---
st.subheader("Recent Predictions")
st.dataframe(
    df.sort_values("timestamp", ascending=False).head(50)[
        ["timestamp", "probability", "prediction", "latency_ms", "n_features_provided", "n_nan_features"]
    ],
    use_container_width=True,
)
