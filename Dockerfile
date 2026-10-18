# Render / Fly / local — context = thư mục gốc repo. Image nhẹ (ONNX, không PyTorch).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    APP_DATA_DIR=/tmp/app_data PRIVATE_RAW_DIR=/tmp/private_raw

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY app/requirements-render.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY src /code/src
COPY configs /code/configs
COPY app /code/app

RUN mkdir -p /tmp/app_data /tmp/private_raw \
    && test -f /code/app/model/timexer_dinov3.onnx

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
