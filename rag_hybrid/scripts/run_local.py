"""Start the local FastAPI and Streamlit services from one terminal."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]


def _api_is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=1) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def _wait_for_api(url: str, process: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "FastAPI exited during startup. Review the API output above; "
                'if dependencies are missing, run `python -m pip install -e ".[dev]"`.'
            )
        if _api_is_ready(url):
            return
        time.sleep(0.25)
    raise TimeoutError(f"FastAPI did not become healthy within {timeout:.0f} seconds: {url}")


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start FastAPI and Streamlit together for local testing."
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8501)
    args = parser.parse_args()

    api_base = f"http://{args.api_host}:{args.api_port}"
    readiness_url = f"{api_base}/ready"
    environment = os.environ.copy()
    environment["RAG_API_URL"] = api_base

    api_process = None
    ui_process = None
    try:
        if _api_is_ready(readiness_url):
            print(f"Using existing FastAPI service at {api_base}")
        else:
            api_command = [
                sys.executable,
                "-m",
                "uvicorn",
                "rag_hybrid.src.api.app:app",
                "--host",
                args.api_host,
                "--port",
                str(args.api_port),
            ]
            print(
                f"Starting FastAPI at {api_base}. First startup may take a little longer "
                "while Python loads the retrieval dependencies.",
                flush=True,
            )
            api_process = subprocess.Popen(api_command, cwd=REPO_ROOT, env=environment)
            _wait_for_api(readiness_url, api_process)

        ui_command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(REPO_ROOT / "rag_hybrid" / "scripts" / "app_streamlit.py"),
            "--server.address",
            args.ui_host,
            "--server.port",
            str(args.ui_port),
        ]
        ui_url = f"http://{args.ui_host}:{args.ui_port}"
        print(f"Starting Streamlit at {ui_url}. Press Ctrl+C to stop.")
        ui_process = subprocess.Popen(ui_command, cwd=REPO_ROOT, env=environment)

        while ui_process.poll() is None:
            if api_process is not None and api_process.poll() is not None:
                raise RuntimeError("FastAPI stopped while Streamlit was running")
            time.sleep(0.25)
        return ui_process.returncode or 0
    except KeyboardInterrupt:
        print("Stopping local RAG services.")
        return 0
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"Could not start the local RAG app: {exc}", file=sys.stderr)
        return 1
    finally:
        _stop_process(ui_process)
        _stop_process(api_process)


if __name__ == "__main__":
    raise SystemExit(main())
