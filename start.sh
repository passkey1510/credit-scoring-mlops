#!/bin/bash
# Start FastAPI on port 7860 (HF Spaces exposed port)

uvicorn app.main:app --host 0.0.0.0 --port 7860
