from __future__ import annotations

import sys
from pathlib import Path

from app._entrypoint_utils import ensure_repo_root_on_path


def test_ensure_repo_root_on_path_inserts_repo_root(monkeypatch) -> None:
    module_file = Path(__file__)
    expected_root = module_file.resolve().parents[2]

    # Start from an empty sys.path to validate insertion order.
    monkeypatch.setattr(sys, "path", [])

    root = ensure_repo_root_on_path(module_file)

    assert root == expected_root
    assert sys.path[0] == str(expected_root)
    # Idempotent on subsequent calls.
    ensure_repo_root_on_path(module_file)
    assert sys.path.count(str(expected_root)) == 1
