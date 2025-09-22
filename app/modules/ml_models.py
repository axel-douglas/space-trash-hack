"""Model loading utilities for Rex-AI predictions.

This module centralises the logic to load machine learning artefacts that
replace the heuristic estimations in :mod:`app.modules.generator`.

Models are trained with :mod:`app.modules.model_training` and persisted inside
``data/models``. The registry automatically discovers the packaged pipeline and
provides a small inference helper that consumes the engineered feature dict
attached to each candidate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
MODEL_DIR = DATA_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


@dataclass(slots=True)
class PredictionResult:
    """Structured payload returned by :class:`ModelRegistry`."""

    rigidez: float
    estanqueidad: float
    energy_kwh: float
    water_l: float
    crew_min: float
    source: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rigidez": self.rigidez,
            "estanqueidad": self.estanqueidad,
            "energy_kwh": self.energy_kwh,
            "water_l": self.water_l,
            "crew_min": self.crew_min,
            "source": self.source,
            "metadata": self.metadata,
        }


class ModelRegistry:
    """Simple registry that exposes trained regression pipelines."""

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.pipeline = None
        self.metadata: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not PIPELINE_PATH.exists():
            LOGGER.info("Rex-AI models not found at %s", PIPELINE_PATH)
            return

        try:
            self.pipeline = joblib.load(PIPELINE_PATH)
        except Exception as exc:  # pragma: no cover - defensive logging.
            LOGGER.warning("Unable to load Rex-AI pipeline %s: %s", PIPELINE_PATH, exc)
            self.pipeline = None

        if METADATA_PATH.exists():
            try:
                self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive logging.
                LOGGER.warning("Failed to parse metadata %s: %s", METADATA_PATH, exc)
                self.metadata = {}

    # ------------------------------------------------------------------
    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        if not self.ready:
            return {}

        try:
            frame = pd.DataFrame([features])
            predictions = self.pipeline.predict(frame)
        except Exception as exc:  # pragma: no cover - defensive logging.
            LOGGER.warning("Prediction failed, falling back to heuristics: %s", exc)
            return {}

        if predictions is None:
            return {}

        preds = np.asarray(predictions, dtype=float).reshape(-1)
        if preds.size < 5:
            LOGGER.warning("Invalid prediction size %s", preds.size)
            return {}

        result = PredictionResult(
            rigidez=float(np.clip(preds[0], 0.0, 1.0)),
            estanqueidad=float(np.clip(preds[1], 0.0, 1.0)),
            energy_kwh=float(max(preds[2], 0.0)),
            water_l=float(max(preds[3], 0.0)),
            crew_min=float(max(preds[4], 0.0)),
            source=str(self.metadata.get("model_name", "rexai-ml")),
            metadata={
                "trained_at": self.metadata.get("trained_at"),
                "n_samples": self.metadata.get("n_samples"),
                "features": self.metadata.get("feature_columns"),
                "targets": self.metadata.get("targets"),
            },
        )
        return result.as_dict()


MODEL_REGISTRY = ModelRegistry()

__all__ = ["MODEL_REGISTRY", "ModelRegistry", "PredictionResult"]
