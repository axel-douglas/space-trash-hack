"""Training pipeline for Rex-AI recycling models.

The script generates synthetic manufacturing runs from the NASA waste
inventory, trains the RandomForest ensemble used by the app, and stores
additional artefacts (XGBoost, autoencoder, TabTransformer) when the
required dependencies are available.

Historically this module suffered from repeated blocks that made
maintenance difficult.  The refactor below keeps the public behaviour
intact while providing a clean, reproducible training entry-point.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Optional dependency for boosted trees
    import xgboost as xgb

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - environments without xgboost
    xgb = None  # type: ignore[assignment]
    HAS_XGBOOST = False

try:  # Optional dependency for deep models
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:  # pragma: no cover - environments without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = TensorDataset = None  # type: ignore[assignment]
    HAS_TORCH = False

from app.modules.generator import compute_feature_vector, prepare_waste_frame, heuristic_props

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DATASETS_ROOT = ROOT / "datasets"
RAW_DIR = DATASETS_ROOT / "raw"
PROCESSED_DIR = DATASETS_ROOT / "processed"
PROCESSED_ML = DATA_ROOT / "processed" / "ml"
MODEL_DIR = DATA_ROOT / "models"

PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
AUTOENCODER_PATH = MODEL_DIR / "rexai_autoencoder.pt"
XGBOOST_PATH = MODEL_DIR / "rexai_xgboost.joblib"
TABTRANSFORMER_PATH = MODEL_DIR / "rexai_tabtransformer.pt"
METADATA_PATH = MODEL_DIR / "metadata.json"
DATASET_PATH = PROCESSED_DIR / "rexai_training_dataset.parquet"
DATASET_ML_PATH = PROCESSED_ML / "synthetic_runs.parquet"

TARGET_COLUMNS = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]
FEATURE_COLUMNS = [
    "process_id",
    "regolith_pct",
    "total_mass_kg",
    "mass_input_kg",
    "num_items",
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
    "glove_frac",
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

LATENT_DIM = 12
TABTRANSFORMER_TOKENS = 8
TABTRANSFORMER_DIM = 64


# ---------------------------------------------------------------------------
# Utility classes & helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SampledCombination:
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
    if HAS_TORCH:
        torch.manual_seed(seed)  # type: ignore[arg-type]


def _load_csv(path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return pd.read_csv(path)


def _load_inventory() -> DataFrame:
    df = _load_csv(RAW_DIR / "nasa_waste_inventory.csv")
    df = prepare_waste_frame(df)
    return df


def _load_process_catalog() -> DataFrame:
    return _load_csv(DATA_ROOT / "process_catalog.csv")


def _sample_weights(n: int, rng: random.Random) -> np.ndarray:
    raw = np.array([rng.gammavariate(1.0, 1.0) for _ in range(n)], dtype=float)
    raw = np.clip(raw, 1e-6, None)
    return raw / raw.sum()


def _compute_targets(
    picks: DataFrame,
    process: pd.Series,
    weights: Sequence[float],
    regolith_pct: float,
) -> Dict[str, float]:
    props = heuristic_props(picks, process, weights, regolith_pct)
    return props.to_targets()


def _generate_samples(n_samples: int, seed: int | None) -> List[SampledCombination]:
    inventory = _load_inventory()
    processes = _load_process_catalog()
    rng = random.Random(seed or 0)
    samples: list[SampledCombination] = []

    while len(samples) < n_samples:
        picks = inventory.sample(
            n=rng.choice([2, 3]),
            replace=False,
            weights=inventory["kg"],
            random_state=rng.randint(0, 10_000),
        )
        weights = _sample_weights(len(picks), rng)
        process = processes.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]

        regolith_pct = 0.0
        if str(process["process_id"]).upper() == "P03":
            regolith_pct = rng.uniform(0.15, 0.35)

        features = compute_feature_vector(picks, weights, process, regolith_pct)
        targets = _compute_targets(picks, process, weights, regolith_pct)
        samples.append(SampledCombination(features=features, targets=targets))

    return samples


def build_training_dataframe(n_samples: int = 1600, seed: int | None = 21) -> DataFrame:
    samples = _generate_samples(n_samples, seed)
    df = pd.DataFrame([sample.as_row() for sample in samples])
    return df


def _build_preprocessor() -> ColumnTransformer:
    categorical = ["process_id"]
    numeric = [col for col in FEATURE_COLUMNS if col not in categorical]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline(steps=[("scale", StandardScaler(with_mean=False))]), numeric),
        ],
        remainder="drop",
    )


def _train_random_forest(
    df: DataFrame, seed: int | None
) -> tuple[Pipeline, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed or 0)

    preprocessor = _build_preprocessor()
    regressor = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=240,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed or 0,
        )
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("regressor", regressor)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_valid)
    residuals = y_valid.to_numpy(dtype=float) - preds
    residual_std = residuals.std(axis=0)

    metrics = {
        target: {
            "mae": float(mean_absolute_error(y_valid[target], preds[:, idx])),
            "rmse": float(math.sqrt(mean_squared_error(y_valid[target], preds[:, idx]))),
            "r2": float(r2_score(y_valid[target], preds[:, idx])),
        }
        for idx, target in enumerate(TARGET_COLUMNS)
    }
    metrics["overall"] = {
        "mae": float(np.mean([m["mae"] for m in metrics.values()])),
        "rmse": float(np.mean([m["rmse"] for m in metrics.values()])),
        "r2": float(np.mean([m["r2"] for m in metrics.values()])),
    }

    matrix = pipeline.named_steps["preprocess"].transform(X_train)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)

    feature_means = matrix.mean(axis=0)
    feature_stds = matrix.std(axis=0) + 1e-6

    rf = pipeline.named_steps["regressor"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    importances: dict[str, list[tuple[str, float]]] = {}
    averages: list[tuple[str, float]] = []
    for idx, target in enumerate(TARGET_COLUMNS):
        estimator: RandomForestRegressor = rf.estimators_[idx]
        fi = estimator.feature_importances_
        target_pairs = [(feature_names[i], float(fi[i])) for i in np.argsort(fi)[::-1][:16]]
        importances[target] = target_pairs
        for name, weight in target_pairs:
            averages.append((name, weight))

    if averages:
        grouped: Dict[str, float] = {}
        for name, weight in averages:
            grouped[name] = grouped.get(name, 0.0) + weight
        total = sum(grouped.values()) or 1.0
        averaged = sorted(((name, weight / total) for name, weight in grouped.items()), key=lambda x: x[1], reverse=True)
    else:
        averaged = []

    rf_payload = {
        "metrics": metrics,
        "feature_importance": {
            "per_target": importances,
            "average": averaged[:16],
        },
        "n_estimators": rf.estimators_[0].n_estimators,
    }

    return pipeline, rf_payload, feature_means, feature_stds, residual_std, feature_names


def _train_xgboost(pipeline: Pipeline, df: DataFrame, seed: int | None) -> Dict[str, Any]:
    if not HAS_XGBOOST:
        return {}

    preprocessor = pipeline.named_steps["preprocess"]
    matrix = preprocessor.transform(df[FEATURE_COLUMNS])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)

    models: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for idx, target in enumerate(TARGET_COLUMNS):
        booster = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=seed or 0,
            tree_method="hist",
        )
        booster.fit(matrix, df[target])
        preds = booster.predict(matrix)
        models[target] = booster
        metrics[target] = {
            "mae": float(mean_absolute_error(df[target], preds)),
            "rmse": float(math.sqrt(mean_squared_error(df[target], preds))),
            "r2": float(r2_score(df[target], preds)),
        }

    metrics["overall"] = {
        "mae": float(np.mean([m["mae"] for m in metrics.values()])),
        "rmse": float(np.mean([m["rmse"] for m in metrics.values()])),
        "r2": float(np.mean([m["r2"] for m in metrics.values()])),
    }

    payload = {"models": models, "metrics": metrics}
    joblib.dump(payload, XGBOOST_PATH)
    return {"metrics": metrics, "path": XGBOOST_PATH.name}


class _Autoencoder(nn.Module if HAS_TORCH else object):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM) -> None:
        if not HAS_TORCH:  # pragma: no cover - executed only without torch
            raise RuntimeError("PyTorch is required to train the Rex-AI autoencoder")
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


class _TabTransformer(nn.Module if HAS_TORCH else object):
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
        if not HAS_TORCH:  # pragma: no cover - executed only without torch
            raise RuntimeError("PyTorch is required to train the Rex-AI TabTransformer")
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


def _train_autoencoder(matrix: np.ndarray) -> Dict[str, Any]:
    if not HAS_TORCH:
        return {}

    dataset = TensorDataset(torch.tensor(matrix, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = _Autoencoder(matrix.shape[1], latent_dim=LATENT_DIM)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(40):
        for (batch,) in loader:
            optim.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optim.step()

    torch.save(model.state_dict(), AUTOENCODER_PATH)
    return {"latent_dim": LATENT_DIM, "path": AUTOENCODER_PATH.name}


def _train_tabtransformer(matrix: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    if not HAS_TORCH:
        return {}

    dataset = TensorDataset(
        torch.tensor(matrix, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = _TabTransformer(matrix.shape[1])
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(60):
        for batch, target in loader:
            optim.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, target)
            loss.backward()
            optim.step()

    torch.save({"state_dict": model.state_dict(), "tokens": model.n_tokens, "d_model": model.d_model}, TABTRANSFORMER_PATH)
    return {
        "tokens": model.n_tokens,
        "d_model": model.d_model,
        "path": TABTRANSFORMER_PATH.name,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_save(n_samples: int = 1600, seed: int | None = 21) -> Dict[str, Any]:
    """Generate data, train models and persist artefacts to disk."""

    _set_seed(seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_ML.mkdir(parents=True, exist_ok=True)

    df = build_training_dataframe(n_samples=n_samples, seed=seed)
    df.to_parquet(DATASET_PATH, index=False)
    df.to_parquet(DATASET_ML_PATH, index=False)

    pipeline, rf_payload, feature_means, feature_stds, residual_std, feature_names = _train_random_forest(df, seed)
    joblib.dump(pipeline, PIPELINE_PATH)

    preprocessor = pipeline.named_steps["preprocess"]
    matrix = preprocessor.transform(df[FEATURE_COLUMNS])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)

    extras: Dict[str, Any] = {}
    extras["xgboost"] = _train_xgboost(pipeline, df, seed)
    extras["autoencoder"] = _train_autoencoder(matrix)
    extras["tabtransformer"] = _train_tabtransformer(matrix, df[TARGET_COLUMNS].to_numpy(dtype=float))

    metadata = {
        "model_name": "rexai-rf-ensemble",
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "n_samples": int(len(df)),
        "dataset": {
            "path": DATASET_PATH.relative_to(DATASETS_ROOT).as_posix(),
            "hash": _hash_file(DATASET_PATH),
        },
        "feature_columns": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "post_transform_features": preprocessor.get_feature_names_out().tolist(),
        "feature_means": {name: float(val) for name, val in zip(feature_names, feature_means)},
        "feature_stds": {name: float(val) for name, val in zip(feature_names, feature_stds)},
        "residual_std": {target: float(val) for target, val in zip(TARGET_COLUMNS, residual_std)},
        "random_forest": rf_payload,
        "artifacts": {
            "pipeline": PIPELINE_PATH.name,
            "xgboost": extras["xgboost"],
            "autoencoder": extras["autoencoder"],
            "tabtransformer": extras["tabtransformer"],
        },
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    info = train_and_save()
    print(json.dumps(info, indent=2))
