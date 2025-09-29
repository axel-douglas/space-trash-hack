"""Test configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _reset_model_registry_cache():
    """Ensure the cached model registry does not leak state across tests."""

    try:
        from app.modules.ml_models import get_model_registry
    except Exception:
        yield
        return

    get_model_registry.clear()
    try:
        yield
    finally:
        get_model_registry.clear()
