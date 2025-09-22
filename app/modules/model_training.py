"""Training utilities for Rex-AI regression models.

The training workflow generates a synthetic dataset that emulates the
manufacturing behaviour encoded in the heuristic generator. The purpose is to
package a reproducible baseline model that can later be replaced with real NASA
logs without changing the application code.

Running the module as a script will:

1. Sample synthetic runs using the current waste inventory and process catalog.
2. Train a multi-output regression pipeline (RandomForest + preprocessing).
3. Persist the fitted artefact and accompanying metadata under ``data/models``.
4. Export the training dataset to ``data/processed`` for traceability.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from app.modules import generator

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_ROOT / "processed" / "ml"
MODEL_DIR = DATA_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
DATASET_PATH = PROCESSED_DIR / "synthetic_runs.parquet"

TARGET_COLUMNS = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]
FEATURE_COLUMNS = [
    "process_id",
    "mass_input_kg",
    "num_items",
    "problematic_mass_frac",
    "problematic_item_frac",
    "regolith_pct",
    "aluminum_frac",
    "foam_frac",
    "eva_frac",
    "textile_frac",
    "multilayer_frac",
    "glove_frac",
]


@dataclass(slots=True)
class SyntheticRun:
    features: dict
    target: dict

    def as_row(self) -> dict:
        payload = {**self.features}
        payload.update(self.target)
        return payload


def _load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    waste = pd.read_csv(DATA_ROOT / "waste_inventory_sample.csv")
    processes = pd.read_csv(DATA_ROOT / "process_catalog.csv")
    return waste, processes


def _default_target() -> dict:
    return {
        "rigidity": 0.8,
        "tightness": 0.75,
        "max_energy_kwh": 9.0,
        "max_water_l": 6.0,
        "max_crew_min": 60.0,
    }


def _sample_candidate(waste: pd.DataFrame, processes: pd.DataFrame, rng: random.Random) -> dict | None:
    target = _default_target()
    n = rng.randint(1, 3)
    cands, _ = generator.generate_candidates(
        waste,
        processes,
        target,
        n=n,
        crew_time_low=rng.random() < 0.4,
        optimizer_evals=0,
    )
    if not cands:
        return None
    return rng.choice(cands)


def _create_synthetic_runs(n_samples: int, seed: int | None = None) -> List[SyntheticRun]:
    previous = os.environ.get("REXAI_FORCE_HEURISTIC")
    os.environ["REXAI_FORCE_HEURISTIC"] = "1"
    rng = random.Random(seed or 42)
    waste, processes = _load_sources()
    runs: List[SyntheticRun] = []

    try:
        while len(runs) < n_samples:
            candidate = _sample_candidate(waste, processes, rng)
            if not candidate:
                continue
            props = candidate["props"]
            features = {key: candidate["features"].get(key) for key in FEATURE_COLUMNS}
            # Heuristic props are deterministic; add mild gaussian noise to simulate measurement.
            noise = rng.random()
            target = {
                "rigidez": float(np.clip(props.rigidity + rng.gauss(0, 0.03), 0.0, 1.0)),
                "estanqueidad": float(np.clip(props.tightness + rng.gauss(0, 0.04), 0.0, 1.0)),
                "energy_kwh": float(max(props.energy_kwh * (0.95 + 0.1 * noise), 0.0)),
                "water_l": float(max(props.water_l * (0.95 + 0.12 * noise), 0.0)),
                "crew_min": float(max(props.crew_min * (0.9 + 0.2 * noise), 0.0)),
            }
            runs.append(SyntheticRun(features=features, target=target))
    finally:
        if previous is None:
            os.environ.pop("REXAI_FORCE_HEURISTIC", None)
        else:
            os.environ["REXAI_FORCE_HEURISTIC"] = previous
    return runs


def build_training_dataframe(n_samples: int = 600, seed: int | None = 21) -> pd.DataFrame:
    runs = _create_synthetic_runs(n_samples, seed)
    df = pd.DataFrame([run.as_row() for run in runs])
    return df


def train_pipeline(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    categorical = ["process_id"]
    numeric = [col for col in FEATURE_COLUMNS if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
        ]
    )

    base_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=7,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", MultiOutputRegressor(base_model)),
    ])

    pipeline.fit(X, y)
    return pipeline


def persist_artifacts(df: pd.DataFrame, pipeline: Pipeline) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, PIPELINE_PATH)
    df.to_parquet(DATASET_PATH, index=False)

    metadata = {
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "n_samples": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "model_name": "rexai-rf-v1",
        "dataset_path": str(DATASET_PATH.relative_to(DATA_ROOT)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_and_save(n_samples: int = 600, seed: int | None = 21) -> None:
    df = build_training_dataframe(n_samples=n_samples, seed=seed)
    pipeline = train_pipeline(df)
    persist_artifacts(df, pipeline)


if __name__ == "__main__":
    train_and_save()
