"""Test configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT_CANDIDATE = Path(__file__).resolve().parents[1]

try:
    from app.bootstrap import ensure_project_root
except ModuleNotFoundError:  # pragma: no cover - fallback when PYTHONPATH lacks repo
    if str(PROJECT_ROOT_CANDIDATE) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_CANDIDATE))
    from app.bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root(PROJECT_ROOT_CANDIDATE)


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
