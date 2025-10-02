"""Bootstrap helpers for Streamlit entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    """Ensure the repository root is present on ``sys.path``.

    Streamlit executes scripts as modules, and when running standalone files
    the repository root might be missing from ``sys.path``. This helper makes
    sure the absolute path two levels above this file (the project root) is
    inserted so imports like ``app.modules`` succeed reliably.
    """

    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


__all__ = ["ensure_project_root"]
