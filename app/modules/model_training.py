"""Training utilities for Rex-AI using NASA-aligned datasets.

This module ingests the curated raw tables placed under ``datasets/raw`` and
combines them with the existing process catalogue to generate a reproducible
training dataset. The workflow follows these steps:

1. Build a tabular dataset that mixes non-metabolic waste (NASA taxonomy) with
   MGS-1 regolith properties and process constraints.
2. Engineer physicochemical features (density, moisture, composition) and
   mission-level factors (gas recovery potential, logistics reuse efficiency).
3. Train a neural network regressor that predicts multi-output targets used by
   the Streamlit demo (rigidez, estanqueidad, energía, agua, tiempo de crew).
4. Fit a lightweight autoencoder to obtain latent embeddings for candidate
   recipes, enabling similarity search and generative sampling in the UI.
5. Persist artefacts (regression pipeline, autoencoder weights, dataset and
   metadata) in ``data/models`` and ``datasets/processed``.

Running ``python -m app.modules.model_training`` regenerates the dataset and all
artefacts so the demo remains explainable and reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
RAW_DIR = DATASETS_ROOT / "raw"
PROCESSED_DIR = DATASETS_ROOT / "processed"
MODEL_DIR = DATA_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
AUTOENCODER_PATH = MODEL_DIR / "rexai_autoencoder.pt"
METADATA_PATH = MODEL_DIR / "metadata.json"
DATASET_PATH = PROCESSED_DIR / "rexai_training_dataset.parquet"

TARGET_COLUMNS = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]
FEATURE_COLUMNS = [
    "process_id",
    "regolith_pct",
    "total_mass_kg",
    "density_kg_m3",
    "moisture_frac",
    "difficulty_index",
    "problematic_mass_frac",
    "problematic_item_frac",
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
    "oxide_sio2",
    "oxide_feot",
    "oxide_mgo",
    "oxide_cao",
    "oxide_so3",
    "oxide_h2o",
]

LATENT_DIM = 8


@dataclass(slots=True)
class SampledCombination:
    """Represent a simulated manufacturing run used for training."""

    features: Dict[str, float | str]
    targets: Dict[str, float]

    def as_row(self) -> Dict[str, float | str]:
        payload = {**self.features}
        payload.update(self.targets)
        return payload


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
    # Compute average yield ratios to scale energy/water savings.
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
# Feature and target engineering
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
        "problematic_mass_frac": float(np.clip(np.dot(weights, picks["pct_mass"].to_numpy(dtype=float)) / 100.0, 0.0, 1.0)),
        "problematic_item_frac": float(np.clip(np.dot(weights, picks["pct_volume"].to_numpy(dtype=float)) / 100.0, 0.0, 1.0)),
        "packaging_frac": _category_fraction(categories, weights, ["packaging", "food packaging"]),
    }

    for name, keywords in keyword_map.items():
        features[name] = _keyword_fraction(tokens, weights, keywords)

    gas_index = gas_metrics["mean_yield_ratio"] * (
        0.7 * features["polyethylene_frac"]
        + 0.4 * features["foam_frac"]
        + 0.5 * features["eva_frac"]
        + 0.2 * features["textile_frac"]
    )
    features["gas_recovery_index"] = float(np.clip(gas_index / 10.0, 0.0, 1.0))

    logistics_index = logistics_metrics["mean_reuse_efficiency"] * (
        features["packaging_frac"] + 0.5 * features["eva_frac"]
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
    structural_boost = 0.55 * features.get("aluminum_frac", 0.0) + 0.35 * features.get("carbon_fiber_frac", 0.0)
    foam_penalty = 0.18 * features.get("foam_frac", 0.0)
    regolith_bonus = 0.3 * float(regolith_props.get("crystalline_fraction", 0.0)) * features.get("regolith_pct", 0.0)

    rigidity = 0.25 + structural_boost + regolith_bonus - foam_penalty
    rigidity = float(np.clip(rigidity, 0.05, 1.0))

    tightness = 0.3
    tightness += 0.45 * features.get("multilayer_frac", 0.0)
    tightness += 0.2 * features.get("textile_frac", 0.0)
    tightness += 0.15 * features.get("polyethylene_frac", 0.0)
    tightness -= 0.12 * features.get("regolith_pct", 0.0)
    tightness = float(np.clip(tightness, 0.05, 1.0))

    process_energy = float(process["energy_kwh_per_kg"])
    process_water = float(process["water_l_per_kg"])
    process_crew = float(process["crew_min_per_batch"])

    difficulty = features.get("difficulty_index", 0.0)
    moisture = features.get("moisture_frac", 0.0)
    regolith_pct = features.get("regolith_pct", 0.0)
    gas_index = features.get("gas_recovery_index", 0.0)
    reuse_index = features.get("logistics_reuse_index", 0.0)

    total_mass = features.get("total_mass_kg", 0.0)

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
        picks = inventory.sample(n=rng.choice([2, 3]), replace=False, weights=inventory["mass_kg"], random_state=rng.randint(0, 10_000))
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
# Model training
# ---------------------------------------------------------------------------

def build_training_dataframe(n_samples: int = 1200, seed: int | None = 21) -> DataFrame:
    inventory = _load_inventory()
    processes = _load_process_catalog()
    samples = _generate_samples(inventory, processes, n_samples, seed)
    df = pd.DataFrame([sample.as_row() for sample in samples])
    return df


def _build_preprocessor() -> ColumnTransformer:
    categorical = ["process_id"]
    numeric = [col for col in FEATURE_COLUMNS if col not in categorical]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
        ]
    )


def train_regressor(df: DataFrame) -> Pipeline:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    pipeline = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    max_iter=600,
                    random_state=7,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(feature_matrix: np.ndarray, latent_dim: int = LATENT_DIM) -> _Autoencoder:
    tensor = torch.from_numpy(feature_matrix.astype(np.float32))
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = _Autoencoder(feature_matrix.shape[1], latent_dim=latent_dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(80):
        for (batch,) in loader:
            optimiser.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimiser.step()

    model.eval()
    return model


def persist_artifacts(df: DataFrame, pipeline: Pipeline, autoencoder: _Autoencoder, feature_matrix: np.ndarray) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, PIPELINE_PATH)
    df.to_parquet(DATASET_PATH, index=False)

    torch.save(
        {
            "state_dict": autoencoder.state_dict(),
            "input_dim": feature_matrix.shape[1],
            "latent_dim": LATENT_DIM,
        },
        AUTOENCODER_PATH,
    )

    metadata = {
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "n_samples": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "model_name": "rexai-mlp-nasa-v2",
        "dataset_path": str(DATASET_PATH.relative_to(DATASETS_ROOT)),
        "autoencoder": {
            "latent_dim": LATENT_DIM,
            "path": str(AUTOENCODER_PATH.relative_to(MODEL_DIR)),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_and_save(n_samples: int = 1200, seed: int | None = 21) -> None:
    df = build_training_dataframe(n_samples=n_samples, seed=seed)
    pipeline = train_regressor(df)
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]  # type: ignore[assignment]
    matrix = preprocessor.transform(df[FEATURE_COLUMNS])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    autoencoder = train_autoencoder(np.asarray(matrix, dtype=np.float32))
    persist_artifacts(df, pipeline, autoencoder, np.asarray(matrix, dtype=np.float32))


if __name__ == "__main__":  # pragma: no cover - script entry point
    train_and_save()
