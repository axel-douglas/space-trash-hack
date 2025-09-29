"""Latent-space utilities for optional Sprint 4 explorations.

This module provides helpers to:

* Detect near-duplicate recipes in the autoencoder latent space.
* Sample new candidates around a seed recipe using gaussian moves in latent
  space and score them with a user-defined objective.

The functionality gracefully degrades when the optional autoencoder artifact
is not available, so it can be imported safely even in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from app.modules.ml_models import TARGET_COLUMNS, ModelRegistry, get_model_registry


ScoreFunction = Callable[[Mapping[str, float]], float]


@dataclass(slots=True)
class LatentCandidate:
    """Container for latent optimisation results."""

    features: Dict[str, Any]
    prediction: Dict[str, float]
    latent: Sequence[float]
    score: float

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "features": self.features,
            "prediction": self.prediction,
            "score": float(self.score),
            "latent": [float(x) for x in self.latent],
        }
        return payload


class LatentSpaceExplorer:
    """High-level API to interact with the Rex-AI autoencoder."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        if registry is None:
            try:
                registry = get_model_registry()
            except Exception:  # pragma: no cover - cache unavailable in minimal envs
                registry = None
        self.registry = registry

    # ------------------------------------------------------------------
    # Availability & preprocessing helpers
    # ------------------------------------------------------------------
    def available(self) -> bool:
        return bool(self.registry and getattr(self.registry, "ready", False) and self.registry.has_autoencoder())

    def _sanitise_features(
        self,
        base: Mapping[str, Any],
        decoded: Mapping[str, Any],
    ) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = dict(base)
        feature_columns: Iterable[str]
        feature_columns = self.registry.metadata.get("feature_columns") or decoded.keys()
        for column in feature_columns:
            if column == "process_id":
                candidate = decoded.get(column)
                if candidate:
                    cleaned[column] = str(candidate).strip().upper()
                continue

            candidate = decoded.get(column)
            if candidate is None:
                continue

            try:
                numeric = float(candidate)
            except (TypeError, ValueError):
                continue

            if column.endswith("_frac"):
                numeric = float(np.clip(numeric, 0.0, 1.0))
            elif column.endswith("_pct"):
                numeric = float(np.clip(numeric, 0.0, 100.0))
            else:
                numeric = float(max(numeric, 0.0))

            cleaned[column] = numeric

        return cleaned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_duplicates(self, frame: pd.DataFrame, threshold: float = 0.12) -> List[Dict[str, Any]]:
        """Return pairs of indices that are closer than *threshold* in latent space."""

        if not self.available():
            return []

        if frame.empty:
            return []

        matrix = self.registry.transform_features(frame)
        if matrix.size == 0:
            return []

        latent = self.registry.encode_matrix(matrix)
        if latent.size == 0:
            return []

        if len(frame) < 2:
            return []

        latent_array = np.asarray(latent, dtype=float)
        norms = np.sum(np.square(latent_array), axis=1, keepdims=True)
        squared = norms + norms.T - 2.0 * latent_array @ latent_array.T
        squared = np.maximum(squared, 0.0)
        distances_matrix = np.sqrt(squared, dtype=float)

        mask = distances_matrix <= float(threshold)
        candidate_pairs = np.argwhere(np.triu(mask, k=1))

        distances: List[Dict[str, Any]] = []
        for left, right in candidate_pairs:
            distances.append(
                {
                    "left_index": str(frame.index[left]),
                    "right_index": str(frame.index[right]),
                    "distance": float(distances_matrix[left, right]),
                }
            )

        return distances

    def propose_candidates(
        self,
        seed_features: Mapping[str, Any],
        objective: ScoreFunction,
        *,
        radius: float = 0.35,
        samples: int = 64,
        top_k: int = 10,
        random_state: int | None = None,
    ) -> List[LatentCandidate]:
        """Sample new candidates around *seed_features* and return the best ones."""

        if samples <= 0 or top_k <= 0:
            return []

        if not self.available():
            return []

        frame = pd.DataFrame([seed_features])
        matrix = self.registry.transform_features(frame)
        if matrix.size == 0:
            return []

        latent = self.registry.encode_matrix(matrix)
        if latent.size == 0:
            return []

        seed_latent = latent.reshape(-1)
        rng = np.random.default_rng(random_state)
        proposals = rng.normal(loc=seed_latent, scale=radius, size=(samples, seed_latent.shape[0]))

        candidates: List[LatentCandidate] = []
        seen_keys: set[str] = set()
        for vector in proposals:
            decoded = self.registry.decode_latent(vector)
            if not decoded:
                continue

            candidate_features = self._sanitise_features(seed_features, decoded)
            prediction = self.registry.predict(candidate_features)
            if not prediction:
                continue

            score = float(objective(prediction))
            fingerprint = _fingerprint(candidate_features)
            if fingerprint in seen_keys:
                continue
            seen_keys.add(fingerprint)

            cast_prediction = {str(k): float(v) for k, v in prediction.items() if k in TARGET_COLUMNS}
            candidates.append(
                LatentCandidate(
                    features=candidate_features,
                    prediction=cast_prediction,
                    latent=tuple(float(x) for x in vector.tolist()),
                    score=score,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]


def make_objective(weights: Mapping[str, float], *, maximise: bool = True) -> ScoreFunction:
    """Build a deterministic scoring function from *weights* over the targets."""

    filtered: Dict[str, float] = {str(k): float(v) for k, v in weights.items() if float(v) != 0.0}
    if not filtered:
        filtered = {"rigidez": 1.0}

    multiplier = 1.0 if maximise else -1.0

    def _objective(prediction: Mapping[str, float]) -> float:
        score = 0.0
        for key, weight in filtered.items():
            value = float(prediction.get(key, 0.0))
            score += weight * value
        return multiplier * score

    return _objective


def _fingerprint(features: Mapping[str, Any]) -> str:
    items = []
    for key in sorted(features):
        value = features[key]
        if isinstance(value, (int, float, np.floating)):
            items.append(f"{key}:{value:.6f}")
        else:
            items.append(f"{key}:{value}")
    return "|".join(items)


__all__ = [
    "LatentCandidate",
    "LatentSpaceExplorer",
    "make_objective",
]
