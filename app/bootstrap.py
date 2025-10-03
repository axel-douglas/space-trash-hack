"""Bootstrap helpers for Streamlit entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def ensure_streamlit_entrypoint(module_file: str | Path) -> Path:
    """Ensure the Streamlit entrypoint can import ``app`` modules."""

    root = _find_project_root(Path(module_file))
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _candidate_roots(start: Path) -> Iterable[Path]:
    """Yield candidate roots from ``start`` up to the filesystem root."""

    resolved = start.resolve()
    yield resolved
    yield from resolved.parents


def _find_project_root(start: Path) -> Path:
    """Locate the repository root given a starting path within the project."""

    for candidate in _candidate_roots(start):
        app_dir = candidate / "app"
        if app_dir.is_dir() and (app_dir / "__init__.py").is_file():
            return candidate
    return start.resolve().parents[-1]


def ensure_project_root(start: str | Path | None = None) -> Path:
    """Ensure the repository root is present on ``sys.path``.

    Parameters
    ----------
    start:
        Optional path used to determine where the search should begin. When
        ``None`` (the default) it falls back to the location of this module.
    """

    base = Path(start) if start is not None else Path(__file__)
    root = _find_project_root(base)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


__all__ = [
    "ensure_streamlit_entrypoint",
    "ensure_project_root",
]
