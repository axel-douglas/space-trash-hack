from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from app.modules import generator


class DummyRegistry:
    ready = True
    metadata = {"model_hash": "dummy-hash"}

    def predict(self, features):
        return {
            "rigidez": 0.8,
            "estanqueidad": 0.7,
            "energy_kwh": 3.5,
            "water_l": 1.2,
            "crew_min": 15.0,
            "uncertainty": {"rigidez": 0.05},
        }

    def embed(self, features):
        return [0.1, 0.2, 0.3]


def test_generate_candidates_appends_inference_log(monkeypatch):
    monkeypatch.setattr(generator, "MODEL_REGISTRY", DummyRegistry())

    log_dir = generator.LOGS_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"inference_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    if log_path.exists():
        log_path.unlink()

    waste_df = pd.DataFrame(
        {
            "id": ["W1", "W2", "W3"],
            "category": ["packaging", "eva", "metal"],
            "material": ["plastic", "foam", "aluminum"],
            "kg": [1.0, 2.0, 0.5],
            "volume_l": [10.0, 5.0, 2.5],
            "flags": ["", "ctb", ""],
        }
    )
    proc_df = pd.DataFrame(
        {
            "process_id": ["P01"],
            "name": ["Demo"],
            "energy_kwh_per_kg": [1.0],
            "water_l_per_kg": [0.5],
            "crew_min_per_batch": [30.0],
        }
    )

    candidates, history = generator.generate_candidates(waste_df, proc_df, target={}, n=1)

    assert candidates, "Expected at least one candidate to be generated"
    assert history.empty

    assert log_path.exists(), "Inference log parquet file was not created"

    log_df = pd.read_parquet(log_path)
    for column in ["timestamp", "input_features", "prediction", "uncertainty", "model_hash"]:
        assert column in log_df.columns

    last_event = log_df.iloc[-1]
    assert last_event["model_hash"] == "dummy-hash"

    prediction_payload = json.loads(last_event["prediction"])
    assert prediction_payload["rigidez"] == 0.8

    uncertainty_payload = json.loads(last_event["uncertainty"])
    assert "rigidez" in uncertainty_payload

    cand = candidates[0]
    breakdown = cand.get("score_breakdown")
    assert isinstance(breakdown, dict)
    assert "contributions" in breakdown
    assert "auxiliary" in breakdown
    assert pytest.approx(breakdown.get("total", cand["score"]), rel=1e-2) == cand["score"]
    auxiliary = cand.get("auxiliary")
    assert isinstance(auxiliary, dict)
    assert "passes_seal" in auxiliary

    log_path.unlink(missing_ok=True)


def test_generate_candidates_heuristic_mode_skips_ml(monkeypatch):
    calls: list[str] = []

    class NoCallRegistry:
        ready = True
        metadata = {"model_hash": "noop"}

        def predict(self, features):
            calls.append("predict")
            return {}

        def embed(self, features):
            return []

    monkeypatch.setattr(generator, "MODEL_REGISTRY", NoCallRegistry())

    log_dir = generator.LOGS_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"inference_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    log_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)

def test_generate_candidates_heuristic_mode_skips_ml(monkeypatch):
    calls: list[str] = []

    class NoCallRegistry:
        ready = True
        metadata = {"model_hash": "noop"}

        def predict(self, features):
            calls.append("predict")
            return {}

        def embed(self, features):
            return []

    monkeypatch.setattr(generator, "MODEL_REGISTRY", NoCallRegistry())

    log_dir = generator.LOGS_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"inference_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    log_path.unlink(missing_ok=True)
    waste_df = pd.DataFrame(
        {
            "id": ["W1", "W2"],
            "category": ["packaging", "eva"],
            "material": ["plastic", "foam"],
            "kg": [1.0, 2.0],
            "volume_l": [10.0, 5.0],
            "flags": ["", "ctb"],
        }
    )
    proc_df = pd.DataFrame(
        {
            "process_id": ["P01"],
            "name": ["Demo"],
            "energy_kwh_per_kg": [1.0],
            "water_l_per_kg": [0.5],
            "crew_min_per_batch": [30.0],
        }
    )

    candidates, history = generator.generate_candidates(
        waste_df, proc_df, target={}, n=1, use_ml=False
    )

    assert candidates, "Expected heuristic candidate even when ML disabled"
    assert history.empty
    assert not calls, "ML predict should not be invoked in heuristic mode"
    assert not log_path.exists(), "Inference log should not be created in heuristic mode"

    cand = candidates[0]
    assert "score_breakdown" in cand
    assert "auxiliary" in cand
