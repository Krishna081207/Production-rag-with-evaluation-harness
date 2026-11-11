"""Create Streamlit's OIDC secrets file from runtime environment variables."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def write_oidc_configuration() -> bool:
    if os.getenv("AUTH_REQUIRED", "false").strip().lower() != "true":
        os.environ["RAG_OIDC_CONFIGURED"] = "false"
        return False

    allowed_emails = [
        email.strip()
        for email in os.getenv("AUTHORIZED_EMAILS", "").split(",")
        if email.strip()
    ]
    client_id = os.getenv("OIDC_CLIENT_ID", "").strip()
    client_secret = os.getenv("OIDC_CLIENT_SECRET", "").strip()
    metadata_url = os.getenv(
        "OIDC_SERVER_METADATA_URL",
        "https://accounts.google.com/.well-known/openid-configuration",
    ).strip()
    cookie_secret = os.getenv("OIDC_COOKIE_SECRET", "").strip()
    redirect_uri = os.getenv("OIDC_REDIRECT_URI", "").strip()
    if not redirect_uri:
        public_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
        if public_url:
            redirect_uri = f"{public_url}/oauth2callback"

    required_values = (allowed_emails, client_id, client_secret, metadata_url, cookie_secret, redirect_uri)
    if not all(required_values):
        os.environ["RAG_OIDC_CONFIGURED"] = "false"
        return False

    secrets_path = Path.home() / ".streamlit" / "secrets.toml"
    secrets_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    secrets_path.parent.chmod(0o700)
    secrets_path.write_text(
        "[auth]\n"
        f"redirect_uri = {json.dumps(redirect_uri)}\n"
        f"cookie_secret = {json.dumps(cookie_secret)}\n"
        f"client_id = {json.dumps(client_id)}\n"
        f"client_secret = {json.dumps(client_secret)}\n"
        f"server_metadata_url = {json.dumps(metadata_url)}\n",
        encoding="utf-8",
    )
    secrets_path.chmod(0o600)
    os.environ["RAG_OIDC_CONFIGURED"] = "true"
    return True


def main() -> None:
    write_oidc_configuration()
    api_hostport = os.getenv("RAG_API_HOSTPORT", "").strip()
    if api_hostport and not os.getenv("RAG_API_URL"):
        os.environ["RAG_API_URL"] = f"http://{api_hostport}"
    app_path = "/app/rag_hybrid/scripts/app_streamlit.py"
    port = os.getenv("PORT", "10000")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path,
            "--server.address",
            "0.0.0.0",
            "--server.port",
            port,
        ],
    )


if __name__ == "__main__":
    main()
