from __future__ import annotations

import sys
from pathlib import Path

import scripts.generate_candidates as script


def test_generate_candidates_script_bootstrap(monkeypatch):
    """The CLI helper should always ensure the repo root is importable."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)

    # Simulate an environment where the repository root is missing from sys.path.
    new_path = [entry for entry in sys.path if entry != repo_str]
    monkeypatch.setattr(sys, "path", new_path, raising=False)
    assert repo_str not in sys.path

    script._ensure_project_root()

    assert repo_str in sys.path
