"""Streamlit app package."""

from __future__ import annotations

from .bootstrap import ensure_project_root, ensure_streamlit_path

ROOT = ensure_streamlit_path()

__all__ = ["ROOT", "ensure_project_root", "ensure_streamlit_path"]
