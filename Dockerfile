FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system raguser \
    && useradd --system --gid raguser --create-home raguser

COPY pyproject.toml README.md ./
COPY rag_hybrid/ ./rag_hybrid/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[pinecone]" \
    && mkdir -p /app/data \
    && chown -R raguser:raguser /app/data

USER raguser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)" || exit 1

CMD ["uvicorn", "rag_hybrid.src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
