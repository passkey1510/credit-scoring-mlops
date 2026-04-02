#!/bin/bash
# Start FastAPI (background) and Streamlit (foreground on port 7860 for HF Spaces)

uvicorn app.main:app --host 0.0.0.0 --port 8000 &

streamlit run dashboard/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
