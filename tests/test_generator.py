from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

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

    log_path.unlink(missing_ok=True)
