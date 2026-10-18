# TimeXer AI — Dự đoán độ tuổi từ giọng nói (FastAPI + ONNX Runtime)
# Dùng được cho Hugging Face Spaces (Docker SDK, port 7860), Render, Fly.io, Railway, VPS.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/tmp/hf APP_DATA_DIR=/tmp/app_data PRIVATE_RAW_DIR=/tmp/private_raw

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY app/requirements-deploy.txt /code/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /code/requirements.txt

COPY src /code/src
COPY configs /code/configs
COPY app /code/app

# HF Spaces chạy user id 1000
RUN useradd -m -u 1000 user && chown -R user /code && mkdir -p /tmp/app_data /tmp/private_raw && chown -R user /tmp/app_data /tmp/private_raw
USER user

EXPOSE 7860
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
