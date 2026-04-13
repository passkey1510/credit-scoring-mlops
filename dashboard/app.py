"""Streamlit monitoring dashboard for the credit scoring API."""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests as http_requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
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
            resp = http_requests.post(
                f"{API_URL}/predict", json={"features": features}, timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                data["features"] = features
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
            resp = http_requests.post(
                f"{API_URL}/predict",
                json={"features": {"AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "EXT_SOURCE_2": ext2}},
                timeout=10,
            )
            r = resp.json()
            label = "Default" if r["prediction"] == 1 else "Approved"
            color = "red" if r["prediction"] == 1 else "green"
            st.markdown(f"**Probability**: {r['probability']:.4f}")
            st.markdown(f"**Decision**: :{color}[{label}]")
            st.markdown(f"**Latency**: {r.get('latency_ms', 'N/A')} ms")
        except Exception as e:
            st.error(f"Error: {e}")


# --- Main content ---
predictions = st.session_state.get("predictions", [])

if not predictions:
    st.info("No predictions yet. Use the sidebar to send test predictions or make a single prediction.")
    st.stop()

df = pd.DataFrame(predictions)

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", len(df))
col2.metric("Default Rate", f"{df['prediction'].mean():.1%}")
col3.metric("Avg Probability", f"{df['probability'].mean():.4f}")
col4.metric("Threshold", f"{THRESHOLD}")

st.divider()

# --- Charts ---
left, right = st.columns(2)

with left:
    st.subheader("Score Distribution")
    counts, edges = np.histogram(df["probability"].dropna(), bins=20)
    bin_labels = [f"{edges[i]:.3f}" for i in range(len(counts))]
    hist_df = pd.DataFrame({"Score Range": bin_labels, "Count": counts})
    st.bar_chart(hist_df, x="Score Range", y="Count", use_container_width=True)

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
    df[["probability", "prediction", "threshold"]].tail(50),
    use_container_width=True,
)
