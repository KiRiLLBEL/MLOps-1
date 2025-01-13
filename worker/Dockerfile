FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/.cache/huggingface

COPY main.py .
COPY .env .

ENV RABBITMQ_HOST="localhost"
ENV RABBITMQ_PORT="5672"
ENV RABBITMQ_USER="guest"
ENV RABBITMQ_PASSWORD="guest"
ENV REQUEST_QUEUE="incoming_tasks"
ENV RESPONSE_QUEUE="ready_tasks"
ENV LORA_WEIGHTS_PATH="/app/lora_adapters/default_adapter"
ENV HF_TOKEN=""
ENV LORA_ADAPTER_URL=""


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

