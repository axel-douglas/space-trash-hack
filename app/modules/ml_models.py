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

# --- Optional torch/autoencoder support (safe if torch is not installed) ---
try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore
    nn = None     # type: ignore
    _HAS_TORCH = False

LOGGER = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
MODEL_DIR = DATA_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
AUTOENCODER_PATH = MODEL_DIR / "rexai_autoencoder.pt"
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
    latent_vector: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "rigidez": self.rigidez,
            "estanqueidad": self.estanqueidad,
            "energy_kwh": self.energy_kwh,
            "water_l": self.water_l,
            "crew_min": self.crew_min,
            "source": self.source,
            "metadata": self.metadata,
            "latent_vector": list(self.latent_vector),
        }


# --- Optional AE model definition ---
if _HAS_TORCH:
    class _Autoencoder(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )

        def forward(self, x):  # pragma: no cover - unused forward
            return self.encoder(x)

        def encode(self, x):
            return self.encoder(x)
else:
    _Autoencoder = None  # type: ignore[assignment]


class ModelRegistry:
    """Simple registry that exposes trained regression pipelines."""

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.pipeline = None
        self.metadata: dict[str, Any] = {}
        self.preprocessor = None
        self.autoencoder: _Autoencoder | None = None  # type: ignore[valid-type]
        self.latent_dim: int | None = None
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
            if hasattr(self.pipeline, "named_steps"):
                # for sklearn Pipeline([...('preprocess', ...), ('regressor', ...)])
                self.preprocessor = getattr(self.pipeline, "named_steps", {}).get("preprocess")
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Unable to load Rex-AI pipeline %s: %s", PIPELINE_PATH, exc)
            self.pipeline = None
            self.preprocessor = None

        if METADATA_PATH.exists():
            try:
                self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to parse metadata %s: %s", METADATA_PATH, exc)
                self.metadata = {}

        # Optional autoencoder
        if _HAS_TORCH and AUTOENCODER_PATH.exists():
            try:
                checkpoint = torch.load(AUTOENCODER_PATH, map_location="cpu")  # type: ignore[arg-type]
                input_dim = int(checkpoint.get("input_dim"))
                latent_dim = int(checkpoint.get("latent_dim"))
                model = _Autoencoder(input_dim, latent_dim)  # type: ignore[operator]
                model.load_state_dict(checkpoint.get("state_dict", {}))
                model.eval()
                self.autoencoder = model
                self.latent_dim = latent_dim
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Unable to load autoencoder %s: %s", AUTOENCODER_PATH, exc)
                self.autoencoder = None
                self.latent_dim = None

    # ------------------------------------------------------------------
    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        if not self.ready:
            return {}

        try:
            frame = pd.DataFrame([features])
            predictions = self.pipeline.predict(frame)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Prediction failed, falling back to heuristics: %s", exc)
            return {}

        if predictions is None:
            return {}

        preds = np.asarray(predictions, dtype=float).reshape(-1)
        if preds.size < 5:
            LOGGER.warning("Invalid prediction size %s", preds.size)
            return {}

        latent_vector = self._encode(frame)

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
            latent_vector=tuple(latent_vector),
        )
        return result.as_dict()

    # ------------------------------------------------------------------
    def _encode(self, frame: pd.DataFrame) -> list[float]:
        """Return latent vector if AE & preprocess are available; [] otherwise."""
        if not (_HAS_TORCH and self.preprocessor is not None and self.autoencoder is not None):
            return []

        try:
            matrix = self.preprocessor.transform(frame)
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float32))  # type: ignore[attr-defined]
            with torch.no_grad():  # type: ignore[attr-defined]
                latent = self.autoencoder.encode(tensor).numpy()  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to compute latent vector: %s", exc)
            return []

        if getattr(latent, "size", 0) == 0:
            return []
        return np.asarray(latent, dtype=np.float32).reshape(-1).tolist()

    def embed(self, features: Mapping[str, Any]) -> list[float]:
        if not self.ready:
            return []
        frame = pd.DataFrame([features])
        return self._encode(frame)


MODEL_REGISTRY = ModelRegistry()

__all__ = ["MODEL_REGISTRY", "ModelRegistry", "PredictionResult"]
