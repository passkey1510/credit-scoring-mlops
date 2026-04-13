"""Streamlit monitoring dashboard for the credit scoring API."""

import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import requests as http_requests
import streamlit as st

API_URL = os.environ.get("API_URL", "https://passkey1510-credit-scoring-mlops.hf.space")
THRESHOLD = 0.11

st.set_page_config(page_title="Credit Scoring Monitor", layout="wide")
st.title("Credit Scoring — Monitoring Dashboard")


def send_predictions(n: int):
    """Send test predictions to the API using random feature values."""
    results = []
    for _ in range(n):
        features = {
            "AMT_INCOME_TOTAL": random.uniform(50000, 500000),
            "AMT_CREDIT": random.uniform(100000, 2000000),
            "AMT_ANNUITY": random.uniform(5000, 80000),
            "AMT_GOODS_PRICE": random.uniform(50000, 1500000),
            "DAYS_BIRTH": random.randint(-25000, -7000),
            "DAYS_EMPLOYED": random.randint(-15000, 0),
            "EXT_SOURCE_1": random.uniform(0, 1),
            "EXT_SOURCE_2": random.uniform(0, 1),
            "EXT_SOURCE_3": random.uniform(0, 1),
        }
        try:
            t0 = time.time()
            resp = http_requests.post(
                f"{API_URL}/predict", json={"features": features}, timeout=10
            )
            latency_ms = round((time.time() - t0) * 1000, 2)
            if resp.status_code == 200:
                data = resp.json()
                data["latency_ms"] = latency_ms
                data["timestamp"] = datetime.now(timezone.utc).isoformat()
                results.append(data)
        except Exception:
            pass
        time.sleep(0.05)
    return results


# --- Sidebar ---
with st.sidebar:
    st.header("API Connection")
    st.code(API_URL, language=None)

    # Health check
    try:
        health = http_requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API healthy — {health['n_features']} features, threshold {health['threshold']}")
    except Exception:
        st.error("API unreachable")

    st.divider()
    st.header("Test Predictions")
    n_preds = st.slider("Number of predictions", 10, 100, 30)
    if st.button("Send test predictions"):
        with st.spinner(f"Sending {n_preds} predictions..."):
            preds = send_predictions(n_preds)
        if preds:
            st.success(f"{len(preds)} predictions sent!")
            st.session_state["predictions"] = st.session_state.get("predictions", []) + preds
        else:
            st.error("No predictions succeeded. Check API connection.")

    st.divider()
    st.header("Single Prediction")
    income = st.number_input("Income", value=200000.0, step=10000.0)
    credit = st.number_input("Credit amount", value=500000.0, step=50000.0)
    ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5)
    if st.button("Predict"):
        try:
            t0 = time.time()
            resp = http_requests.post(
                f"{API_URL}/predict",
                json={"features": {"AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "EXT_SOURCE_2": ext2}},
                timeout=10,
            )
            latency_ms = round((time.time() - t0) * 1000, 2)
            r = resp.json()
            label = "Default" if r["prediction"] == 1 else "Approved"
            color = "red" if r["prediction"] == 1 else "green"
            st.markdown(f"**Probability**: {r['probability']:.4f}")
            st.markdown(f"**Decision**: :{color}[{label}]")
            st.markdown(f"**Latency**: {latency_ms} ms")
        except Exception as e:
            st.error(f"Error: {e}")


# --- Main content ---
predictions = st.session_state.get("predictions", [])

if not predictions:
    st.info("No predictions yet. Use the sidebar to send test predictions or make a single prediction.")
    st.stop()

df = pd.DataFrame(predictions)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", len(df))
col2.metric("Default Rate", f"{df['prediction'].mean():.1%}")
col3.metric("Avg Latency", f"{df['latency_ms'].mean():.1f} ms")
col4.metric("Avg Probability", f"{df['probability'].mean():.4f}")

st.divider()

# --- Charts ---
left, right = st.columns(2)

with left:
    st.subheader("Latency Over Time")
    latency_series = df.set_index("timestamp")["latency_ms"]
    st.line_chart(latency_series, use_container_width=True)

with right:
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

st.divider()

# --- Recent predictions table ---
st.subheader("Recent Predictions")
st.dataframe(
    df[["timestamp", "probability", "prediction", "latency_ms"]].sort_values("timestamp", ascending=False).head(50),
    use_container_width=True,
)
