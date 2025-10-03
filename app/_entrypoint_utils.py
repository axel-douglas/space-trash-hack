"""Utilities to prepare Streamlit entrypoints before importing ``app`` modules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def _candidate_roots(start: Path) -> Iterable[Path]:
    """Yield potential repository roots starting from ``start`` upward."""

    resolved = start.resolve()
    yield resolved
    yield from resolved.parents


def ensure_repo_root_on_path(module_file: str | Path) -> Path:
    """Ensure the repository root is on ``sys.path`` for Streamlit scripts."""

    module_path = Path(module_file).resolve()
    root: Path | None = None
    for candidate in _candidate_roots(module_path):
        if not candidate.is_dir():
            continue
        app_dir = candidate / "app"
        if (app_dir / "__init__.py").is_file():
            root = candidate
            break
    if root is None:
        root = module_path.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


__all__ = ["ensure_repo_root_on_path"]
