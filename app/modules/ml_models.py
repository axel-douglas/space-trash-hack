"""Model loading utilities for Rex-AI predictions.

This module centralises the logic to load machine learning artefacts that
replace the heuristic estimations in :mod:`app.modules.generator`.

Models are trained with :mod:`app.modules.model_training` and persisted inside
``data/models``. The registry automatically discovers the packaged pipeline and
provides inference helpers that also expose explainability artefacts such as
feature importance, confidence intervals and alternative model suggestions.
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

try:  # Optional dependency for lightweight deployments
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - fallback when torch is absent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
MODEL_DIR = DATA_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
AUTOENCODER_PATH = MODEL_DIR / "rexai_autoencoder.pt"
XGBOOST_PATH = MODEL_DIR / "rexai_xgboost.joblib"
TABTRANSFORMER_PATH = MODEL_DIR / "rexai_tabtransformer.pt"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET_COLUMNS = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]


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
    uncertainty: dict[str, float]
    confidence_interval: dict[str, tuple[float, float]]
    feature_importance: list[tuple[str, float]]
    comparisons: dict[str, dict[str, float]]
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
            "uncertainty": self.uncertainty,
            "confidence_interval": self.confidence_interval,
            "feature_importance": self.feature_importance,
            "comparisons": self.comparisons,
            "latent_vector": list(self.latent_vector),
        }


class _Autoencoder(nn.Module if torch is not None and nn is not None else object):
    if torch is None or nn is None:  # pragma: no cover - executed only without torch
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PyTorch is required to use the Rex-AI autoencoder")

        def encode(self, *_args: object, **_kwargs: object) -> Any:  # type: ignore[override]
            raise RuntimeError("PyTorch is required to use the Rex-AI autoencoder")

    else:
        def __init__(self, input_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 192),
                nn.ReLU(),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Linear(96, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 192),
                nn.ReLU(),
                nn.Linear(192, input_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - unused forward
            latent = self.encoder(x)
            return self.decoder(latent)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)


class SimpleTabTransformer(nn.Module if torch is not None and nn is not None else object):
    if torch is None or nn is None:  # pragma: no cover - executed only without torch
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PyTorch is required to use the Rex-AI TabTransformer")

    else:
        def __init__(
            self,
            input_dim: int,
            n_tokens: int,
            d_model: int,
            out_dim: int,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.n_tokens = n_tokens
            self.d_model = d_model
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.token_projection = nn.Linear(input_dim, n_tokens * d_model)
            self.positional = nn.Parameter(torch.randn(1, n_tokens, d_model))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_tokens * d_model, 128),
                nn.GELU(),
                nn.Linear(128, out_dim),
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            tokens = self.token_projection(x).view(-1, self.n_tokens, self.d_model)
            tokens = tokens + self.positional
            encoded = self.encoder(tokens)
            encoded = self.norm(encoded)
            return self.head(encoded)


class ModelRegistry:
    """Simple registry that exposes trained regression pipelines."""

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.pipeline = None
        self.preprocessor = None
        self.metadata: dict[str, Any] = {}
        self.autoencoder: _Autoencoder | None = None
        self.latent_dim: int | None = None
        self.feature_names: list[str] = []
        self.feature_means: dict[str, float] = {}
        self.feature_stds: dict[str, float] = {}
        self.residual_std: np.ndarray | None = None
        self.feature_importance_avg: list[tuple[str, float]] = []
        self.xgb_models: dict[str, Any] = {}
        self.tab_model: SimpleTabTransformer | None = None
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
                self.preprocessor = self.pipeline.named_steps.get("preprocess")
        except Exception as exc:  # pragma: no cover - defensive logging.
            LOGGER.warning("Unable to load Rex-AI pipeline %s: %s", PIPELINE_PATH, exc)
            self.pipeline = None
            self.preprocessor = None

        if METADATA_PATH.exists():
            try:
                self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive logging.
                LOGGER.warning("Failed to parse metadata %s: %s", METADATA_PATH, exc)
                self.metadata = {}

        self.feature_names = list(self.metadata.get("post_transform_features", []))
        self.feature_means = {
            key: float(value) for key, value in self.metadata.get("feature_means", {}).items()
        }
        self.feature_stds = {
            key: float(value) for key, value in self.metadata.get("feature_stds", {}).items()
        }
        residual_std = self.metadata.get("residual_std", {})
        self.residual_std = np.array([float(residual_std.get(t, 0.0)) for t in TARGET_COLUMNS])
        importance_section = (
            self.metadata.get("random_forest", {}).get("feature_importance", {}).get("average", [])
        )
        self.feature_importance_avg = [
            (str(name), float(weight)) for name, weight in importance_section
        ]

        if torch is not None and AUTOENCODER_PATH.exists():
            try:
                checkpoint = torch.load(AUTOENCODER_PATH, map_location="cpu")
                input_dim = int(checkpoint.get("input_dim"))
                latent_dim = int(checkpoint.get("latent_dim"))
                model = _Autoencoder(input_dim, latent_dim)
                model.load_state_dict(checkpoint.get("state_dict", {}))
                model.eval()
                self.autoencoder = model
                self.latent_dim = latent_dim
            except Exception as exc:  # pragma: no cover - defensive logging.
                LOGGER.warning("Unable to load autoencoder %s: %s", AUTOENCODER_PATH, exc)
                self.autoencoder = None
                self.latent_dim = None

        if XGBOOST_PATH.exists():
            try:
                payload = joblib.load(XGBOOST_PATH)
                self.xgb_models = payload.get("models", {})
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Unable to load XGBoost ensemble: %s", exc)
                self.xgb_models = {}

        if torch is not None and TABTRANSFORMER_PATH.exists():
            try:
                checkpoint = torch.load(TABTRANSFORMER_PATH, map_location="cpu")
                model = SimpleTabTransformer(
                    checkpoint.get("input_dim"),
                    checkpoint.get("n_tokens", 8),
                    checkpoint.get("d_model", 64),
                    len(TARGET_COLUMNS),
                )
                model.load_state_dict(checkpoint.get("state_dict", {}))
                model.eval()
                self.tab_model = model
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Unable to load TabTransformer weights: %s", exc)
                self.tab_model = None

    # ------------------------------------------------------------------
    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        if not self.ready:
            return {}

        try:
            frame, matrix = self._prepare_frame(features)
            rf_predictions = self.pipeline.predict(frame)
        except Exception as exc:  # pragma: no cover - defensive logging.
            LOGGER.warning("Prediction failed, falling back to heuristics: %s", exc)
            return {}

        if rf_predictions is None:
            return {}

        preds = np.asarray(rf_predictions, dtype=float).reshape(-1)
        if preds.size < len(TARGET_COLUMNS):
            LOGGER.warning("Invalid prediction size %s", preds.size)
            return {}

        uncertainty = self._rf_uncertainty(matrix)
        combined_std = self._combine_uncertainty(uncertainty)
        ci = self._confidence_interval(preds, combined_std)
        latent_vector = self._encode(frame)
        importance = self._feature_contributions(matrix[0])
        variants = self._predict_variants(matrix)

        result = PredictionResult(
            rigidez=float(np.clip(preds[0], 0.0, 1.0)),
            estanqueidad=float(np.clip(preds[1], 0.0, 1.0)),
            energy_kwh=float(max(preds[2], 0.0)),
            water_l=float(max(preds[3], 0.0)),
            crew_min=float(max(preds[4], 0.0)),
            source=str(self.metadata.get("model_name", "rexai-rf-ensemble")),
            metadata={
                "trained_at": self.metadata.get("trained_at"),
                "n_samples": self.metadata.get("n_samples"),
                "features": self.feature_names,
                "targets": TARGET_COLUMNS,
            },
            uncertainty={t: float(combined_std[idx]) for idx, t in enumerate(TARGET_COLUMNS)},
            confidence_interval=ci,
            feature_importance=importance,
            comparisons=variants,
            latent_vector=tuple(latent_vector),
        )
        return result.as_dict()

    # ------------------------------------------------------------------
    def _prepare_frame(self, features: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
        frame = pd.DataFrame([features])
        if self.preprocessor is None:
            return frame, np.zeros((1, len(self.feature_names)), dtype=float)
        matrix = self.preprocessor.transform(frame)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return frame, np.asarray(matrix, dtype=float)

    # ------------------------------------------------------------------
    def _rf_uncertainty(self, matrix: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            return np.zeros((matrix.shape[0], len(TARGET_COLUMNS)))
        regressor = getattr(self.pipeline, "named_steps", {}).get("regressor")
        if regressor is None:
            return np.zeros((matrix.shape[0], len(TARGET_COLUMNS)))

        stds = np.zeros((matrix.shape[0], len(TARGET_COLUMNS)))
        for target_idx, estimator in enumerate(getattr(regressor, "estimators_", [])):
            try:
                tree_preds = np.stack([tree.predict(matrix) for tree in estimator.estimators_], axis=0)
                stds[:, target_idx] = tree_preds.std(axis=0)
            except Exception:  # pragma: no cover - defensive
                stds[:, target_idx] = 0.0
        return stds

    # ------------------------------------------------------------------
    def _combine_uncertainty(self, tree_std: np.ndarray) -> np.ndarray:
        residual = self.residual_std if self.residual_std is not None else np.zeros(len(TARGET_COLUMNS))
        residual = residual.reshape(1, -1)
        while residual.shape[1] < tree_std.shape[1]:
            residual = np.pad(residual, ((0, 0), (0, 1)), constant_values=0.0)
        return np.sqrt(tree_std ** 2 + residual ** 2)[0]

    # ------------------------------------------------------------------
    def _confidence_interval(self, preds: np.ndarray, std: np.ndarray) -> dict[str, tuple[float, float]]:
        ci = {}
        for idx, target in enumerate(TARGET_COLUMNS):
            sigma = float(std[idx])
            lower = preds[idx] - 1.96 * sigma
            upper = preds[idx] + 1.96 * sigma
            if target in {"rigidez", "estanqueidad"}:
                lower = float(np.clip(lower, 0.0, 1.0))
                upper = float(np.clip(upper, 0.0, 1.0))
            else:
                lower = float(max(lower, 0.0))
                upper = float(max(upper, 0.0))
            ci[target] = (lower, upper)
        return ci

    # ------------------------------------------------------------------
    def _feature_contributions(self, vector: np.ndarray) -> list[tuple[str, float]]:
        if not self.feature_importance_avg or not self.feature_names:
            return []
        contributions: list[tuple[str, float]] = []
        index_map = {name: idx for idx, name in enumerate(self.feature_names)}
        for feature, importance in self.feature_importance_avg[:8]:
            idx = index_map.get(feature)
            if idx is None:
                continue
            value = float(vector[idx])
            mean = self.feature_means.get(feature, 0.0)
            contrib = (value - mean) * importance
            contributions.append((feature, float(contrib)))
        return contributions

    # ------------------------------------------------------------------
    def _predict_variants(self, matrix: np.ndarray) -> dict[str, dict[str, float]]:
        comparisons: dict[str, dict[str, float]] = {}

        if self.xgb_models:
            preds = []
            for target in TARGET_COLUMNS:
                model = self.xgb_models.get(target)
                if model is None:
                    preds.append(np.zeros(matrix.shape[0]))
                else:
                    preds.append(np.asarray(model.predict(matrix), dtype=float))
            stacked = np.stack(preds, axis=1)[0]
            comparisons["xgboost"] = {
                target: (
                    float(np.clip(stacked[idx], 0.0, 1.0))
                    if target in {"rigidez", "estanqueidad"}
                    else float(max(0.0, stacked[idx]))
                )
                for idx, target in enumerate(TARGET_COLUMNS)
            }

        if torch is not None and self.tab_model is not None:
            with torch.no_grad():
                tensor = torch.from_numpy(matrix.astype(np.float32))
                preds = self.tab_model(tensor).cpu().numpy()[0]
            comparisons["tabtransformer"] = {
                target: float(max(0.0, preds[idx])) if target not in {"rigidez", "estanqueidad"} else float(np.clip(preds[idx], 0.0, 1.0))
                for idx, target in enumerate(TARGET_COLUMNS)
            }

        return comparisons

    # ------------------------------------------------------------------
    def _encode(self, frame: pd.DataFrame) -> list[float]:
        if torch is None or self.preprocessor is None or self.autoencoder is None:
            return []

        try:
            matrix = self.preprocessor.transform(frame)
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
            with torch.no_grad():
                latent = self.autoencoder.encode(tensor).numpy()
        except Exception as exc:  # pragma: no cover - defensive logging.
            LOGGER.debug("Failed to compute latent vector: %s", exc)
            return []

        if latent.size == 0:
            return []
        return latent.reshape(-1).tolist()

    # ------------------------------------------------------------------
    def embed(self, features: Mapping[str, Any]) -> list[float]:
        if not self.ready:
            return []
        frame, _ = self._prepare_frame(features)
        return self._encode(frame)


MODEL_REGISTRY = ModelRegistry()

if torch is None:  # pragma: no cover - log once when optional deps missing
    LOGGER.warning(
        "PyTorch is not installed; latent embeddings and transformer comparisons are disabled."
    )
