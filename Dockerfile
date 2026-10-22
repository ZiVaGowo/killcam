# Kill Cam: один образ для API и воркера.
#   docker build -t killcam .                                         # CPU (явный фолбэк)
#   docker build -t killcam:gpu --build-arg EXTRA_REQS=requirements-gpu.txt .   # + faster-whisper/torch
FROM python:3.12-slim

ARG EXTRA_REQS=""
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements*.txt ./
RUN pip install -r requirements.txt && if [ -n "$EXTRA_REQS" ]; then pip install -r "$EXTRA_REQS"; fi

COPY app ./app
COPY evals ./evals
COPY scripts ./scripts
COPY prompts ./prompts
COPY registry ./registry
COPY data/reference ./data/reference
COPY data/data.lock.json ./data/data.lock.json
COPY params.yaml ./

RUN useradd -m -u 1000 killcam && mkdir -p /app/data/runtime && chown -R killcam /app/data /app/registry
USER killcam

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
