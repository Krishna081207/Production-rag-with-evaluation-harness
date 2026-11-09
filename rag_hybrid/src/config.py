"""Runtime configuration helpers shared by the API and command-line tools."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def data_dir() -> Path:
    """Return the writable directory used for local corpus and vector data."""
    configured = os.getenv("RAG_DATA_DIR")
    return Path(configured).expanduser() if configured else PACKAGE_DATA_DIR


def corpus_path() -> Path:
    return data_dir() / "corpus.jsonl"


def max_upload_bytes() -> int:
    """Return a positive upload limit, defaulting to 10 MiB."""
    raw_value = os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("MAX_UPLOAD_BYTES must be a positive integer") from exc
    if value <= 0:
        raise ValueError("MAX_UPLOAD_BYTES must be a positive integer")
    return value
