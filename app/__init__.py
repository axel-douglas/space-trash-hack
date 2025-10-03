"""Streamlit app package."""

from __future__ import annotations

from .bootstrap import ensure_project_root

ROOT = ensure_project_root()

__all__ = ["ROOT", "ensure_project_root"]
