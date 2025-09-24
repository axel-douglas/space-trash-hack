from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
import pytest

from app.modules import latent_optimizer


class DummyRegistry:
    def __init__(self) -> None:
        self.metadata = {"feature_columns": ["a", "b", "process_id"]}
        self._ready = True

    @property
    def ready(self) -> bool:  # pragma: no cover - simple property
        return self._ready

    def has_autoencoder(self) -> bool:
        return True

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[["a", "b"]].to_numpy(dtype=float)

    def encode_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return matrix * 0.5

    def decode_latent(self, latent: Mapping[str, float] | np.ndarray) -> Dict[str, float]:
        arr = np.asarray(latent, dtype=float)
        return {"a": float(arr[0]), "b": float(arr[1]), "process_id": "P02"}

    def predict(self, features: Mapping[str, float]) -> Dict[str, float]:
        return {
            "rigidez": float(features.get("a", 0.0)),
            "crew_min": float(features.get("b", 0.0)),
            "energy_kwh": float(features.get("a", 0.0) * 2),
        }


def test_available_without_autoencoder(monkeypatch: pytest.MonkeyPatch) -> None:
    explorer = latent_optimizer.LatentSpaceExplorer(registry=None)
    # Force the global registry to None to simulate missing artefact
    monkeypatch.setattr(latent_optimizer, "MODEL_REGISTRY", None)
    explorer.registry = None
    assert explorer.available() is False


def test_detect_duplicates() -> None:
    registry = DummyRegistry()
    explorer = latent_optimizer.LatentSpaceExplorer(registry)
    frame = pd.DataFrame([{"a": 0.2, "b": 0.4, "process_id": "P02"}, {"a": 0.25, "b": 0.42, "process_id": "P03"}])
    dupes = explorer.detect_duplicates(frame, threshold=0.1)
    assert dupes  # the rows are close in latent space
    assert dupes[0]["left_index"] == "0"
    assert dupes[0]["right_index"] == "1"


def test_propose_candidates_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = DummyRegistry()
    explorer = latent_optimizer.LatentSpaceExplorer(registry)
    objective = latent_optimizer.make_objective({"rigidez": 1.0, "crew_min": -0.5})
    results = explorer.propose_candidates(
        {"a": 0.4, "b": 0.3, "process_id": "P02"},
        objective,
        radius=0.05,
        samples=32,
        top_k=5,
        random_state=7,
    )
    assert results
    scores = [candidate.score for candidate in results]
    assert scores == sorted(scores, reverse=True)
    latents = {tuple(candidate.latent) for candidate in results}
    assert len(latents) == len(results)
