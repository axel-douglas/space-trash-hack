"""Advanced training workflow for the Rex-AI demo.

This module composes the full machine-learning stack requested in the
Space Trash Hack roadmap:

* RandomForest multi-output regressor (fase Demo+) with uncertainty
  estimates from tree ensembles.
* Gradient boosted trees (XGBoost) and a lightweight TabTransformer to
  deliver the "wow" effect for hackathon judging.
* A small autoencoder that compresses engineered features into latent
  vectors used for similarity search and exploratory UX components.

The pipeline consumes the curated NASA datasets stored under
``datasets/raw`` and produces reproducible artefacts in
``data/models`` plus a processed dataset in ``datasets/processed``.
The resulting metadata captures training provenance, feature
importances and calibration statistics so the Streamlit experience can
surface explainability without additional heavy computation.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from typing import Dict, Iterable, List, Sequence, Tuple
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
RAW_DIR = DATASETS_ROOT / "raw"
PROCESSED_DIR = DATASETS_ROOT / "processed"
MODEL_DIR = DATA_ROOT / "models"

PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
AUTOENCODER_PATH = MODEL_DIR / "rexai_autoencoder.pt"
XGBOOST_PATH = MODEL_DIR / "rexai_xgboost.joblib"
TABTRANSFORMER_PATH = MODEL_DIR / "rexai_tabtransformer.pt"
METADATA_PATH = MODEL_DIR / "metadata.json"
DATASET_PATH = PROCESSED_DIR / "rexai_training_dataset.parquet"

TARGET_COLUMNS = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]
FEATURE_COLUMNS = [
    "process_id",
    "regolith_pct",
    "total_mass_kg",
    "density_kg_m3",
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
    "polyethylene_frac",
    "carbon_fiber_frac",
    "hydrogen_rich_frac",
    "packaging_frac",
    "gas_recovery_index",
    "logistics_reuse_index",
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
LATENT_DIM = 12
TABTRANSFORMER_TOKENS = 8
TABTRANSFORMER_DIM = 64


# ---------------------------------------------------------------------------
# Utility dataclasses and helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SampledCombination:
    """Represent a simulated manufacturing run used for training."""

    features: Dict[str, float | str]
    targets: Dict[str, float]

    def as_row(self) -> Dict[str, float | str]:
        payload = {**self.features}
        payload.update(self.targets)
        return payload


def _set_seed(seed: int | None) -> None:
    seed = seed or 21
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Raw data helpers
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return pd.read_csv(path)


def _load_inventory() -> DataFrame:
    df = _load_csv(RAW_DIR / "nasa_waste_inventory.csv")
    df["density_kg_m3"] = df["mass_kg"] / df["volume_m3"].clip(lower=1e-6)
    df["moisture_pct"] = df.get("moisture_pct", 0).fillna(0.0)
    df["difficulty_factor"] = df.get("difficulty_factor", 1).fillna(1).astype(float)
    df["tokens"] = (
        df["key_materials"].fillna("").astype(str).str.lower()
        + " "
        + df["flags"].fillna("").astype(str).str.lower()
        + " "
        + df["category"].fillna("").astype(str).str.lower()
        + " "
        + df["item"].fillna("").astype(str).str.lower()
    )
    return df


def _load_process_catalog() -> DataFrame:
    return _load_csv(DATA_ROOT / "process_catalog.csv")


def _load_regolith_vectors() -> Dict[str, float]:
    oxides = _load_csv(RAW_DIR / "mgs1_oxides.csv")
    vector = {f"oxide_{row.oxide.lower()}": float(row.wt_percent) / 100.0 for row in oxides.itertuples()}
    return vector


def _load_regolith_properties() -> Dict[str, float]:
    props = _load_csv(RAW_DIR / "mgs1_properties.csv")
    pivot: Dict[str, float] = {}
    for row in props.itertuples():
        key = str(row.property).lower()
        try:
            pivot[key] = float(row.value)
        except ValueError:
            continue
    return pivot


def _load_gas_metrics() -> Dict[str, float]:
    gas_df = _load_csv(RAW_DIR / "nasa_trash_to_gas.csv")
    gas_df["yield_ratio"] = gas_df["o2_ch4_yield_kg"] / gas_df["water_makeup_kg"].clip(lower=1e-6)
    gas_df["delta_v"] = gas_df["delta_v_ms"].astype(float)
    return {
        "mean_yield_ratio": float(gas_df["yield_ratio"].mean()),
        "max_delta_v": float(gas_df["delta_v"].max()),
    }


def _load_logistics_metrics() -> Dict[str, float]:
    logistics = _load_csv(RAW_DIR / "logistics_to_living.csv")
    logistics["reuse_efficiency"] = (
        (logistics["outfitting_replaced_kg"] - logistics["residual_waste_kg"]) / logistics["packaging_kg"].clip(lower=1e-6)
    ).clip(lower=0)
    logistics["ctb_density"] = logistics["packaging_kg"] / logistics["ctb_count"].clip(lower=1e-6)
    return {
        "mean_reuse_efficiency": float(logistics["reuse_efficiency"].mean()),
        "ctb_density": float(logistics["ctb_density"].mean()),
    }


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------


def _sample_weights(n: int, rng: random.Random) -> np.ndarray:
    raw = np.array([rng.gammavariate(1.0, 1.0) for _ in range(n)], dtype=float)
    raw = np.clip(raw, 1e-6, None)
    return raw / raw.sum()


def _keyword_fraction(tokens: Sequence[str], weights: Sequence[float], keywords: Sequence[str]) -> float:
    score = 0.0
    for token, weight in zip(tokens, weights):
        if any(keyword in token for keyword in keywords):
            score += weight
    return float(np.clip(score, 0.0, 1.0))


def _category_fraction(categories: Sequence[str], weights: Sequence[float], targets: Sequence[str]) -> float:
    score = 0.0
    for category, weight in zip(categories, weights):
        if any(target in category for target in targets):
            score += weight
    return float(np.clip(score, 0.0, 1.0))


def _compute_features(
    picks: DataFrame,
    weights: np.ndarray,
    process: pd.Series,
    regolith_pct: float,
    regolith_vector: Dict[str, float],
    gas_metrics: Dict[str, float],
    logistics_metrics: Dict[str, float],
) -> Dict[str, float | str]:
    total_mass = float(np.dot(weights, picks["mass_kg"]))
    densities = picks["density_kg_m3"].to_numpy(dtype=float)
    density = float(np.dot(weights, densities))
    moisture = float(np.dot(weights, picks["moisture_pct"].to_numpy(dtype=float)) / 100.0)
    difficulty = float(np.dot(weights, picks["difficulty_factor"].to_numpy(dtype=float)) / 3.0)

    tokens = picks["tokens"].tolist()
    categories = picks["category"].str.lower().tolist()

    keyword_map = {
        "aluminum_frac": ("aluminum", " alloy"),
        "foam_frac": ("foam", "closed_cell", "plastazote"),
        "eva_frac": ("eva", "ctb", "nomex"),
        "textile_frac": ("textile", "cotton", "garment", "wipe"),
        "multilayer_frac": ("multilayer", "pouch", "foil"),
        "polyethylene_frac": ("polyethylene", "pvdf", "ldpe"),
        "carbon_fiber_frac": ("carbon fiber", "composite"),
        "hydrogen_rich_frac": ("polyethylene", "cotton", "pvdf"),
    }

    features: Dict[str, float | str] = {
        "process_id": str(process["process_id"]),
        "regolith_pct": float(np.clip(regolith_pct, 0.0, 0.6)),
        "total_mass_kg": total_mass,
        "density_kg_m3": density,
        "moisture_frac": float(np.clip(moisture, 0.0, 1.0)),
        "difficulty_index": float(np.clip(difficulty, 0.0, 1.0)),
        "problematic_mass_frac": float(
            np.clip(np.dot(weights, picks["pct_mass"].to_numpy(dtype=float)) / 100.0, 0.0, 1.0)
        ),
        "problematic_item_frac": float(
            np.clip(np.dot(weights, picks["pct_volume"].to_numpy(dtype=float)) / 100.0, 0.0, 1.0)
        ),
        "packaging_frac": _category_fraction(categories, weights, ["packaging", "food packaging"]),
    }

    for name, keywords in keyword_map.items():
        features[name] = _keyword_fraction(tokens, weights, keywords)

    gas_index = gas_metrics["mean_yield_ratio"] * (
        0.7 * float(features["polyethylene_frac"])
        + 0.4 * float(features["foam_frac"])
        + 0.5 * float(features["eva_frac"])
        + 0.2 * float(features["textile_frac"])
    )
    features["gas_recovery_index"] = float(np.clip(gas_index / 10.0, 0.0, 1.0))

    logistics_index = logistics_metrics["mean_reuse_efficiency"] * (
        float(features["packaging_frac"]) + 0.5 * float(features["eva_frac"])
    )
    features["logistics_reuse_index"] = float(np.clip(logistics_index, 0.0, 2.0))

    for oxide, value in regolith_vector.items():
        features[oxide] = float(value * regolith_pct)

    return features


def _compute_targets(
    features: Dict[str, float | str],
    weights: np.ndarray,
    picks: DataFrame,
    process: pd.Series,
    regolith_props: Dict[str, float],
) -> Dict[str, float]:
    structural_boost = 0.55 * float(features.get("aluminum_frac", 0.0)) + 0.35 * float(
        features.get("carbon_fiber_frac", 0.0)
    )
    foam_penalty = 0.18 * float(features.get("foam_frac", 0.0))
    regolith_bonus = 0.3 * float(regolith_props.get("crystalline_fraction", 0.0)) * float(
        features.get("regolith_pct", 0.0)
    )

    rigidity = 0.25 + structural_boost + regolith_bonus - foam_penalty
    rigidity = float(np.clip(rigidity, 0.05, 1.0))

    tightness = 0.3
    tightness += 0.45 * float(features.get("multilayer_frac", 0.0))
    tightness += 0.2 * float(features.get("textile_frac", 0.0))
    tightness += 0.15 * float(features.get("polyethylene_frac", 0.0))
    tightness -= 0.12 * float(features.get("regolith_pct", 0.0))
    tightness = float(np.clip(tightness, 0.05, 1.0))

    process_energy = float(process["energy_kwh_per_kg"])
    process_water = float(process["water_l_per_kg"])
    process_crew = float(process["crew_min_per_batch"])

    difficulty = float(features.get("difficulty_index", 0.0))
    moisture = float(features.get("moisture_frac", 0.0))
    regolith_pct = float(features.get("regolith_pct", 0.0))
    gas_index = float(features.get("gas_recovery_index", 0.0))
    reuse_index = float(features.get("logistics_reuse_index", 0.0))

    total_mass = float(features.get("total_mass_kg", 0.0))

    energy_kwh = total_mass * (
        process_energy
        + 0.25 * difficulty
        + 0.12 * moisture
        + 0.18 * regolith_pct
    )
    energy_kwh *= (1.0 - 0.15 * gas_index)

    water_l = total_mass * (
        process_water
        + 0.35 * moisture
        + 0.08 * regolith_pct
    )
    water_l *= (1.0 - 0.1 * gas_index)

    crew_min = process_crew + 18.0 * difficulty + 10.0 * regolith_pct
    crew_min *= (1.0 - 0.08 * np.clip(reuse_index, 0.0, 1.5))
    crew_min += 3.0 * len(weights)

    return {
        "rigidez": float(np.clip(rigidity, 0.0, 1.0)),
        "estanqueidad": float(np.clip(tightness, 0.0, 1.0)),
        "energy_kwh": float(max(0.0, energy_kwh)),
        "water_l": float(max(0.0, water_l)),
        "crew_min": float(max(1.0, crew_min)),
    }


def _generate_samples(
    inventory: DataFrame,
    processes: DataFrame,
    n_samples: int,
    seed: int | None,
) -> List[SampledCombination]:
    rng = random.Random(seed or 0)
    regolith_vector = _load_regolith_vectors()
    regolith_props = _load_regolith_properties()
    gas_metrics = _load_gas_metrics()
    logistics_metrics = _load_logistics_metrics()

    samples: List[SampledCombination] = []

    while len(samples) < n_samples:
        picks = inventory.sample(
            n=rng.choice([2, 3]),
            replace=False,
            weights=inventory["mass_kg"],
            random_state=rng.randint(0, 10_000),
        )
        weights = _sample_weights(len(picks), rng)
        process = processes.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]

        regolith_pct = 0.0
        if str(process["process_id"]).upper() == "P03":
            regolith_pct = rng.uniform(0.15, 0.35)

        features = _compute_features(
            picks,
            weights,
            process,
            regolith_pct,
            regolith_vector,
            gas_metrics,
            logistics_metrics,
        )

        targets = _compute_targets(features, weights, picks, process, regolith_props)
        samples.append(SampledCombination(features=features, targets=targets))

    return samples


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_training_dataframe(n_samples: int = 1600, seed: int | None = 21) -> DataFrame:
    inventory = _load_inventory()
    processes = _load_process_catalog()
    samples = _generate_samples(inventory, processes, n_samples, seed)
    df = pd.DataFrame([sample.as_row() for sample in samples])
    return df


def _build_preprocessor() -> ColumnTransformer:
    categorical = ["process_id"]
    numeric = [col for col in FEATURE_COLUMNS if col not in categorical]
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

def _build_rf_pipeline(random_state: int = 42) -> Pipeline:
    rf = RandomForestRegressor(
        n_estimators=240,
        max_depth=8,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=-1,
        oob_score=False,
    )
    model = MultiOutputRegressor(rf, n_jobs=None)
    return Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            ("regressor", model),
        ]
    )


# ---------------------------------------------------------------------------
# Metrics & explainability helpers
# ---------------------------------------------------------------------------


def _collect_scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def _collect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    overall = _collect_scalar_metrics(y_true, y_pred)
    per_target = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        per_target[target] = _collect_scalar_metrics(y_true[:, idx], y_pred[:, idx])
    return {"overall": overall, "per_target": per_target}


def _aggregate_importances(
    model: MultiOutputRegressor, feature_names: Sequence[str]
) -> Dict[str, Dict[str, float] | List[Tuple[str, float]]]:
    importances = np.zeros(len(feature_names))
    per_target: Dict[str, Dict[str, float]] = {}
    for target, estimator in zip(TARGET_COLUMNS, model.estimators_):
        tree_imp = getattr(estimator, "feature_importances_", np.zeros(len(feature_names)))
        per_target[target] = {
            feature: float(weight)
            for feature, weight in zip(feature_names, tree_imp.tolist())
        }
        importances += tree_imp
    importances = importances / max(len(model.estimators_), 1)
    ordered = sorted(
        [(feature, float(weight)) for feature, weight in zip(feature_names, importances.tolist())],
        key=lambda item: item[1],
        reverse=True,
    )
    return {"average": ordered, "per_target": per_target}


# ---------------------------------------------------------------------------
# Autoencoder & TabTransformer definitions
# ---------------------------------------------------------------------------


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SimpleTabTransformer(nn.Module):
    """Compact transformer-style regressor for tabular data."""

    def __init__(
        self,
        num_features: int,
        n_tokens: int = TABTRANSFORMER_TOKENS,
        d_model: int = TABTRANSFORMER_DIM,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        out_dim: int = len(TARGET_COLUMNS),
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.n_tokens = n_tokens
        self.d_model = d_model

        self.token_projection = nn.Linear(num_features, n_tokens * d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_tokens * d_model, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )
        self.positional = nn.Parameter(torch.randn(1, n_tokens, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tokens = self.token_projection(x).view(-1, self.n_tokens, self.d_model)
        tokens = tokens + self.positional
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        return self.head(encoded)


# ---------------------------------------------------------------------------
# Training routines for each component
# ---------------------------------------------------------------------------


def train_autoencoder(feature_matrix: np.ndarray, latent_dim: int = LATENT_DIM) -> _Autoencoder:
    tensor = torch.from_numpy(feature_matrix.astype(np.float32))
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = _Autoencoder(feature_matrix.shape[1], latent_dim=latent_dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(100):
        for (batch,) in loader:
            optimiser.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimiser.step()

    model.eval()
    return model


def train_tabtransformer(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    epochs: int = 120,
) -> Tuple[SimpleTabTransformer, Dict[str, Dict[str, float]]]:
    model = SimpleTabTransformer(feature_matrix.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    X_train = torch.from_numpy(feature_matrix[train_idx]).float()
    y_train = torch.from_numpy(targets[train_idx]).float()
    X_valid = torch.from_numpy(feature_matrix[valid_idx]).float()
    y_valid = torch.from_numpy(targets[valid_idx]).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=128, shuffle=False)

    best_state = None
    best_loss = float("inf")
    patience = 12
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            losses = []
            preds_valid: List[np.ndarray] = []
            for xb, yb in valid_loader:
                preds = model(xb)
                losses.append(loss_fn(preds, yb).item())
                preds_valid.append(preds.cpu().numpy())
        val_loss = float(np.mean(losses))
        if val_loss + 1e-4 < best_loss:
            best_loss = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        preds_valid = model(X_valid).cpu().numpy()
    metrics = _collect_metrics(y_valid.numpy(), preds_valid)
    return model, metrics


def train_xgboost_models(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> Tuple[Dict[str, xgb.XGBRegressor], Dict[str, Dict[str, float]]]:
    models: Dict[str, xgb.XGBRegressor] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    X_train = feature_matrix[train_idx]
    X_valid = feature_matrix[valid_idx]
    y_train = targets[train_idx]
    y_valid = targets[valid_idx]

    preds_valid = np.zeros_like(y_valid)

    for col_idx, target in enumerate(TARGET_COLUMNS):
        booster = xgb.XGBRegressor(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.9,
            min_child_weight=1.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )
        booster.fit(X_train, y_train[:, col_idx], eval_set=[(X_valid, y_valid[:, col_idx])], verbose=False)
        preds = booster.predict(X_valid)
        preds_valid[:, col_idx] = preds
        models[target] = booster
        metrics[target] = _collect_scalar_metrics(y_valid[:, col_idx], preds)

    metrics["overall"] = _collect_scalar_metrics(y_valid, preds_valid)
    return models, metrics


# ---------------------------------------------------------------------------
# Persistence & orchestration
# ---------------------------------------------------------------------------


def persist_artifacts(
    df: DataFrame,
    pipeline: Pipeline,
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
    residual_std: Sequence[float],
    importances: Dict[str, Dict[str, float] | List[Tuple[str, float]]],
    rf_metrics: Dict[str, Dict[str, float]],
    xgb_models: Dict[str, xgb.XGBRegressor],
    xgb_metrics: Dict[str, Dict[str, float]],
    tab_model: SimpleTabTransformer,
    tab_metrics: Dict[str, Dict[str, float]],
    autoencoder: _Autoencoder,
) -> None:

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
    joblib.dump({"models": xgb_models, "feature_names": list(feature_names)}, XGBOOST_PATH)

    torch.save(
        {
            "state_dict": autoencoder.state_dict(),
            "input_dim": feature_matrix.shape[1],
            "latent_dim": LATENT_DIM,
        },
        AUTOENCODER_PATH,
    )

    torch.save(
        {
            "state_dict": tab_model.state_dict(),
            "input_dim": feature_matrix.shape[1],
            "n_tokens": TABTRANSFORMER_TOKENS,
            "d_model": TABTRANSFORMER_DIM,
            "targets": TARGET_COLUMNS,
        },
        TABTRANSFORMER_PATH,
    )

    dataset_hash = hashlib.sha1(df.to_csv(index=False).encode("utf-8")).hexdigest()

    metadata = {
        "model_name": "rexai-rf-ensemble",
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "n_samples": int(len(df)),
        "dataset": {
            "path": str(DATASET_PATH.relative_to(DATASETS_ROOT)),
            "hash": dataset_hash,
        },
        "feature_columns": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "post_transform_features": list(feature_names),
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "residual_std": {t: float(std) for t, std in zip(TARGET_COLUMNS, residual_std)},
        "random_forest": {
            "metrics": rf_metrics,
            "feature_importance": importances,
            "n_estimators": 240,
        },
        "xgboost": {
            "metrics": xgb_metrics,
            "path": str(XGBOOST_PATH.relative_to(MODEL_DIR)),
        },
        "tabtransformer": {
            "metrics": tab_metrics,
            "path": str(TABTRANSFORMER_PATH.relative_to(MODEL_DIR)),
            "tokens": TABTRANSFORMER_TOKENS,
            "d_model": TABTRANSFORMER_DIM,
        },
        "autoencoder": {
            "latent_dim": LATENT_DIM,
            "path": str(AUTOENCODER_PATH.relative_to(MODEL_DIR)),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def train_and_save(n_samples: int = 1600, seed: int | None = 21) -> None:
    _set_seed(seed)
    df = build_training_dataframe(n_samples=n_samples, seed=seed)

    indices = np.arange(len(df))
    train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=seed or 21)

    X_train = df.iloc[train_idx][FEATURE_COLUMNS]
    y_train = df.iloc[train_idx][TARGET_COLUMNS]
    X_valid = df.iloc[valid_idx][FEATURE_COLUMNS]
    y_valid = df.iloc[valid_idx][TARGET_COLUMNS]

    rf_pipeline_eval = _build_rf_pipeline()
    rf_pipeline_eval.fit(X_train, y_train)
    preds_valid = rf_pipeline_eval.predict(X_valid)
    rf_metrics = _collect_metrics(y_valid.to_numpy(), preds_valid)

    residual_std = (y_valid.to_numpy() - preds_valid).std(axis=0)

    rf_pipeline = _build_rf_pipeline()
    rf_pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMNS])
    preprocessor: ColumnTransformer = rf_pipeline.named_steps["preprocess"]  # type: ignore[assignment]
    regressor: MultiOutputRegressor = rf_pipeline.named_steps["regressor"]  # type: ignore[assignment]

    feature_matrix = preprocessor.transform(df[FEATURE_COLUMNS])
    if hasattr(feature_matrix, "toarray"):
        feature_matrix = feature_matrix.toarray()
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

    feature_names = preprocessor.get_feature_names_out()
    feature_means = {
        feature: float(value)
        for feature, value in zip(feature_names, feature_matrix.mean(axis=0))
    }
    feature_stds = {
        feature: float(value)
        for feature, value in zip(feature_names, feature_matrix.std(axis=0))
    }

    importances = _aggregate_importances(regressor, feature_names)

    xgb_models, xgb_metrics = train_xgboost_models(feature_matrix, df[TARGET_COLUMNS].to_numpy(), train_idx, valid_idx)
    tab_model, tab_metrics = train_tabtransformer(feature_matrix, df[TARGET_COLUMNS].to_numpy(), train_idx, valid_idx)

    autoencoder = train_autoencoder(feature_matrix)

    persist_artifacts(
        df,
        rf_pipeline,
        feature_matrix,
        feature_names,
        feature_means,
        feature_stds,
        residual_std,
        importances,
        rf_metrics,
        xgb_models,
        xgb_metrics,
        tab_model,
        tab_metrics,
        autoencoder,
    )


if __name__ == "__main__":  # pragma: no cover - script entry point

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
