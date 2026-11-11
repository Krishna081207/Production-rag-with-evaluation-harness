"""Prepare Render's persistent disk, then run the API without root privileges."""

from __future__ import annotations

import os
import pwd
import sys
from pathlib import Path


def prepare_data_directory(path: Path, uid: int, gid: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.stat().st_uid == uid and path.stat().st_gid == gid:
        return

    for root, directories, files in os.walk(path, followlinks=False):
        for name in directories + files:
            target = Path(root) / name
            if not target.is_symlink():
                os.chown(target, uid, gid)
    os.chown(path, uid, gid)


def main() -> None:
    command = sys.argv[1:] or [
        "uvicorn",
        "rag_hybrid.src.api.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        os.getenv("PORT", "10000"),
    ]

    if os.geteuid() == 0:
        service_user = pwd.getpwnam("raguser")
        data_path = Path(os.getenv("RAG_DATA_DIR", "/app/data"))
        prepare_data_directory(data_path, service_user.pw_uid, service_user.pw_gid)
        os.setgroups([])
        os.setgid(service_user.pw_gid)
        os.setuid(service_user.pw_uid)

    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
