# Dockerfile for rag_hybrid project
# Use a slim Python image and install the editable package with all dependencies.
# The same image is reused for both the FastAPI and Streamlit services.

FROM python:3.11-slim AS base

# System deps (git for version info, curl for health checks, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non‑root user
ARG USERNAME=raguser
ARG UID=1000
RUN useradd -m -u $UID $USERNAME
WORKDIR /app

# Copy only the pyproject and source files (Docker layer cache friendly)
COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY rag_hybrid/ ./rag_hybrid/

# Install project in editable mode
RUN pip install --upgrade pip && \
    pip install -e .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command (overridden in docker‑compose) – keep it alive.
CMD ["sleep", "infinity"]
