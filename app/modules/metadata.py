"""Utilities to expose training metadata and validation logs to the UI."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_META_PATH = DATA_DIR / "training_metadata.json"
_VALIDATION_PATH = DATA_DIR / "model_validation.csv"

_training_cache: Dict[str, Any] | None = None
_validation_cache: pd.DataFrame | None = None


def load_training_metadata() -> Dict[str, Any]:
    """Return training metadata if available, otherwise an empty dict."""
    global _training_cache
    if _training_cache is None:
        if _META_PATH.exists():
            try:
                _training_cache = json.loads(_META_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                _training_cache = {}
        else:
            _training_cache = {}
    return deepcopy(_training_cache)


def load_validation_results() -> pd.DataFrame:
    """Return a DataFrame with validation experiments (may be empty)."""
    global _validation_cache
    if _validation_cache is None:
        if _VALIDATION_PATH.exists():
            _validation_cache = pd.read_csv(_VALIDATION_PATH)
        else:
            _validation_cache = pd.DataFrame(
                columns=["process_id", "metric", "batch", "mean", "lo", "hi", "observed"]
            )
    return _validation_cache.copy()


def metric_interval(metric: str, default: float = 0.05) -> float:
    """Return an empirical error band for a metric from metadata metrics."""
    meta = load_training_metadata()
    metrics = meta.get("metrics", {})
    key = f"{metric}_rmse"
    if key in metrics and isinstance(metrics[key], (int, float)):
        return float(metrics[key])
    # fallback to MAE/MAPE when RMSE is not logged
    alt_key = {
        "energy": "energy_mape",
        "water": "water_mape",
        "crew": "crew_mae",
    }.get(metric)
    if alt_key and isinstance(metrics.get(alt_key), (int, float)):
        return float(metrics[alt_key])
    return default
