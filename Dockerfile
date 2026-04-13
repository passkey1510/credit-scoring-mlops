FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgomp required by LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ app/
COPY dashboard/ dashboard/
COPY model/ model/
COPY data/ data/
COPY start.sh .

RUN chmod +x start.sh
RUN mkdir -p logs

EXPOSE 7860

CMD ["./start.sh"]
