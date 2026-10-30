"""Shared fixtures: a throw-away workspace so tests never write into the real repo folders."""
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A copy of config/, champion/ and program.md in a temp dir, selected through WV_ROOT."""
    for name in ("config",):
        shutil.copytree(REPO / name, tmp_path / name)
    shutil.copytree(REPO / "champion", tmp_path / "champion", ignore=shutil.ignore_patterns("history"))
    (tmp_path / "champion" / "history").mkdir()
    shutil.copy(REPO / "program.md", tmp_path / "program.md")
    monkeypatch.setenv("WV_ROOT", str(tmp_path))
    for key in ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "OANDA_TOKEN", "OANDA_ACCOUNT_ID", "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
                "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


@pytest.fixture
def cfg():
    from weathervane import config
    return config.load_engine()
