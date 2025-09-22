# app/modules/model_training.py
"""Training utilities for Rex-AI regression models.

El flujo de entrenamiento genera un dataset sintético que emula el
comportamiento del generador heurístico. Sirve como baseline reproducible
que puede luego reemplazarse por logs reales sin cambiar el código de la app.

Al ejecutar este módulo como script se hará:
1) Muestreo de corridas sintéticas usando el inventario y catálogo de procesos.
2) Entrenamiento de un pipeline de regresión multi-salida (RandomForest + preprocess).
3) Persistencia del artefacto y metadatos en ``data/models``.
4) Export del dataset de entrenamiento a ``data/processed/ml`` para trazabilidad.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.modules import generator

# ------------------------------ Paths ------------------------------
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_ROOT / "processed" / "ml"
MODEL_DIR = DATA_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
DATASET_PATH = PROCESSED_DIR / "synthetic_runs.parquet"

# --------------------------- Schema -------------------------------
TARGET_COLUMNS = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]

# Conjunto amplio para alinearnos con generator.py (incluye métricas ricas y simples)
FEATURE_COLUMNS = [
    "process_id",
    # métricas ricas
    "total_mass_kg",
    "density_kg_m3",
    "num_items",
    "moisture_frac",
    "difficulty_index",
    "problematic_mass_frac",
    "problematic_item_frac",
    "packaging_frac",
    "gas_recovery_index",
    "logistics_reuse_index",
    # versiones simples para compatibilidad hacia atrás
    "mass_input_kg",
    "problematic_mass_frac_simple",
    "problematic_item_frac_simple",
    # regolito y fracciones por palabra clave
    "regolith_pct",
    "aluminum_frac",
    "foam_frac",
    "eva_frac",
    "textile_frac",
    "multilayer_frac",
    "glove_frac",
    "polyethylene_frac",
    "carbon_fiber_frac",
    "hydrogen_rich_frac",
    # óxidos escalados por participación de regolito
    "oxide_sio2",
    "oxide_feot",
    "oxide_mgo",
    "oxide_cao",
    "oxide_so3",
    "oxide_h2o",
]

# --------------------------- Data helpers --------------------------
def _load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga muestras mínimas de inventario y catálogo de procesos desde data/."""
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


@dataclass(slots=True)
class SyntheticRun:
    features: dict
    target: dict

    def as_row(self) -> dict:
        payload = {**self.features}
        payload.update(self.target)
        return payload


def _sample_candidate(waste: pd.DataFrame, processes: pd.DataFrame, rng: random.Random) -> dict | None:
    target = _default_target()
    n = rng.randint(2, 3)
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
    """Fuerza modo heurístico para no depender de modelos previos."""
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
            # Seleccionamos solo columnas esperadas; si falta alguna, se completa luego con NaN.
            features = {key: candidate["features"].get(key) for key in FEATURE_COLUMNS}
            # Ruido suave para simular mediciones reales
            noise = rng.random()
            target = {
                "rigidez": float(np.clip(props.rigidity + rng.gauss(0, 0.03), 0.0, 1.0)),
                "estanqueidad": float(np.clip(props.tightness + rng.gauss(0, 0.04), 0.0, 1.0)),
                "energy_kwh": float(max(props.energy_kwh * (0.95 + 0.10 * noise), 0.0)),
                "water_l": float(max(props.water_l * (0.95 + 0.12 * noise), 0.0)),
                "crew_min": float(max(props.crew_min * (0.90 + 0.20 * noise), 0.0)),
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

# --------------------------- Training ---------------------------------
def _build_preprocessor(available_features: list[str]) -> ColumnTransformer:
    categorical = [col for col in ["process_id"] if col in available_features]
    numeric = [col for col in available_features if col not in categorical]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
        ]
    )


def train_pipeline(df: pd.DataFrame) -> Pipeline:
    # Usamos solo columnas realmente presentes (por si alguna feature faltó).
    present_feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[present_feats]
    y = df[TARGET_COLUMNS]

    preprocessor = _build_preprocessor(present_feats)
    base_model = RandomForestRegressor(
        n_estimators=220,
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

# --------------------------- Persistence ------------------------------
def persist_artifacts(df: pd.DataFrame, pipeline: Pipeline) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, PIPELINE_PATH)
    df.to_parquet(DATASET_PATH, index=False)

    metadata = {
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "n_samples": int(len(df)),
        "feature_columns": [c for c in FEATURE_COLUMNS if c in df.columns],
        "targets": TARGET_COLUMNS,
        "model_name": "rexai-rf-v1",
        "dataset_path": str(DATASET_PATH.relative_to(DATA_ROOT)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# ----------------------------- Entry ----------------------------------
def train_and_save(n_samples: int = 600, seed: int | None = 21) -> None:
    df = build_training_dataframe(n_samples=n_samples, seed=seed)
    pipeline = train_pipeline(df)
    persist_artifacts(df, pipeline)


if __name__ == "__main__":
    train_and_save()
