"""Smoke test to ensure the Streamlit home entrypoint can be imported."""

from __future__ import annotations

import importlib


def test_home_module_importable() -> None:
    """Import ``app.Home`` ensuring bootstrap wiring prevents path issues."""

    module = importlib.import_module("app.Home")
    assert module is not None
