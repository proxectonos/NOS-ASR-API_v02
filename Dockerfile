# Dockerfile for the NÓS Galician ASR API.
# Supports Wav2Vec2, Wav2Vec2+LM, W2V-BERT and Whisper backends.

FROM python:3.10-slim-bookworm

ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ cmake \
        ffmpeg libsndfile1 \
        libboost-system-dev libboost-thread-dev libboost-program-options-dev \
        libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev \
        git curl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --quiet --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip install --quiet -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /app

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models/.hf_cache

CMD ["gunicorn", "server:app", "-b", ":8000", "--workers", "1", "--threads", "1", "--worker-class", "gthread", "--timeout", "3600"]
