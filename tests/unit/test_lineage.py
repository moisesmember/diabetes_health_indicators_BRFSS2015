import subprocess

import pytest

from diabetes_risk.tracking import lineage


def test_env_var_has_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    assert lineage.git_commit() == "abc1234"


def test_reads_commit_from_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="def5678\n")
    monkeypatch.setattr(lineage.subprocess, "run", lambda *a, **k: completed)
    assert lineage.git_commit() == "def5678"


def test_falls_back_to_unknown_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIT_COMMIT", raising=False)

    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(lineage.subprocess, "run", missing_git)
    assert lineage.git_commit() == "unknown"
