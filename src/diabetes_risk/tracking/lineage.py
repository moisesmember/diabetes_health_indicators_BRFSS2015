"""Metadados de linhagem anexados a cada run (versão do código)."""

from __future__ import annotations

import os
import subprocess


def git_commit() -> str:
    """Commit atual do código.

    Ordem de busca: variável ``GIT_COMMIT`` (definida no Docker, onde não há
    ``.git``), depois ``git rev-parse``; ``"unknown"`` se nenhuma funcionar.
    """
    if commit := os.environ.get("GIT_COMMIT"):
        return commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
