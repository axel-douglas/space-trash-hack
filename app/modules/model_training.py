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

import argparse
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
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

from app.modules.generator import compute_feature_vector, prepare_waste_frame
from app.modules.label_mapper import derive_recipe_id, load_curated_labels, lookup_labels

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
GOLD_DIR = DATASETS_ROOT / "gold"
GOLD_FEATURES_PATH = GOLD_DIR / "features.parquet"
GOLD_LABELS_PATH = GOLD_DIR / "labels.parquet"

PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
AUTOENCODER_PATH = MODEL_DIR / "rexai_autoencoder.pt"
XGBOOST_PATH = MODEL_DIR / "rexai_xgboost.joblib"
TABTRANSFORMER_PATH = MODEL_DIR / "rexai_tabtransformer.pt"
METADATA_PATH = MODEL_DIR / "metadata_gold.json"
LEGACY_METADATA_PATH = MODEL_DIR / "metadata.json"
DATASET_PATH = PROCESSED_DIR / "rexai_training_dataset.parquet"
DATASET_ML_PATH = PROCESSED_ML / "synthetic_runs.parquet"

TARGET_COLUMNS = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]
CLASS_TARGET_COLUMNS = ["tightness_pass", "rigidity_level"]
TIGHTNESS_MODEL_PATH = MODEL_DIR / "rexai_class_tightness.joblib"
RIGIDITY_MODEL_PATH = MODEL_DIR / "rexai_class_rigidity.joblib"

TIGHTNESS_SCORE_MAP = {0: 0.35, 1: 0.85}
RIGIDITY_SCORE_MAP = {1: 0.35, 2: 0.65, 3: 0.9}
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

_GOLD_FEATURES_CACHE: DataFrame | None = None
_GOLD_FEATURES_CACHE_PATH: Path | None = None
_GOLD_TARGETS_CACHE: DataFrame | None = None
_GOLD_TARGETS_CACHE_PATH: Path | None = None


def _infer_trained_on_label(df: DataFrame) -> str:
    """Infer the training dataset label based on ``label_source`` values."""

    if "label_source" not in df.columns:
        return "synthetic_v0"

    sources = (
        df["label_source"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .str.lower()
    )
    synthetic_aliases = {"simulated", "weak", "weakly_supervised"}
    normalized_sources = set()
    for value in sources.tolist():
        if not value:
            continue
        normalized_sources.add("simulated" if value in synthetic_aliases else value)

    if not normalized_sources or normalized_sources == {"simulated"}:
        return "synthetic_v0"
    if "simulated" in normalized_sources:
        return "hybrid_v1"
    return "gold_v1"


def _relative_path(path: Path) -> str:
    """Return path relative to repository root for metadata serialisation."""

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


# ---------------------------------------------------------------------------
# Utility classes & helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SampledCombination:
    features: Dict[str, float | str]
    targets: Dict[str, Any]

    def as_row(self) -> Dict[str, Any]:
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


def _load_parquet(path: Path) -> DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - propagated for visibility
        raise RuntimeError(f"No se pudo leer parquet {path}: {exc}") from exc


def _normalise_key_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def _load_gold_features(path: Path | None = None) -> DataFrame:
    global _GOLD_FEATURES_CACHE, _GOLD_FEATURES_CACHE_PATH
    target_path = Path(path) if path is not None else GOLD_FEATURES_PATH

    if _GOLD_FEATURES_CACHE is not None and _GOLD_FEATURES_CACHE_PATH == target_path:
        return _GOLD_FEATURES_CACHE

    table = _load_parquet(target_path)
    if table.empty:
        _GOLD_FEATURES_CACHE = pd.DataFrame()
        _GOLD_FEATURES_CACHE_PATH = target_path
        return _GOLD_FEATURES_CACHE

    required = {"recipe_id", "process_id"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)} en {target_path}")

    table = table.copy()
    table["recipe_id"] = _normalise_key_column(table["recipe_id"])
    table["process_id"] = _normalise_key_column(table["process_id"])
    _GOLD_FEATURES_CACHE_PATH = target_path
    _GOLD_FEATURES_CACHE = table
    return table


def _load_gold_targets(path: Path | None = None) -> DataFrame:
    global _GOLD_TARGETS_CACHE, _GOLD_TARGETS_CACHE_PATH
    target_path = Path(path) if path is not None else GOLD_LABELS_PATH

    if _GOLD_TARGETS_CACHE is not None and _GOLD_TARGETS_CACHE_PATH == target_path:
        return _GOLD_TARGETS_CACHE

    table = load_curated_labels(target_path)
    if table.empty:
        _GOLD_TARGETS_CACHE = pd.DataFrame()
    else:
        _GOLD_TARGETS_CACHE = table
    _GOLD_TARGETS_CACHE_PATH = target_path
    return _GOLD_TARGETS_CACHE


def _sample_weights(n: int, rng: random.Random) -> np.ndarray:
    raw = np.array([rng.gammavariate(1.0, 1.0) for _ in range(n)], dtype=float)
    raw = np.clip(raw, 1e-6, None)
    return raw / raw.sum()


def _support_dict(values: np.ndarray) -> Dict[str, int]:
    classes, counts = np.unique(values, return_counts=True)
    return {str(int(cls)): int(count) for cls, count in zip(classes, counts)}


def _compute_resource_targets(
    features: Dict[str, Any],
    picks: DataFrame,
    process: pd.Series,
    regolith_pct: float,
) -> Dict[str, float]:
    total_mass = max(0.001, float(picks["kg"].sum()))
    moisture = float(features.get("moisture_frac", 0.0))
    difficulty = float(features.get("difficulty_index", 0.0))
    hydrogen = float(features.get("hydrogen_rich_frac", 0.0))
    logistics = float(features.get("logistics_reuse_index", 0.0))

    base_energy = float(process.get("energy_kwh_per_kg", 0.0))
    base_water = float(process.get("water_l_per_kg", 0.0))
    base_crew = float(process.get("crew_min_per_batch", 0.0))

    energy_kwh = total_mass * (
        base_energy
        + 0.32 * difficulty
        + 0.14 * moisture
        + 0.11 * regolith_pct
        + 0.06 * hydrogen
        - 0.08 * logistics
    )
    water_l = total_mass * (
        base_water
        + 0.48 * moisture
        + 0.1 * hydrogen
        + 0.18 * regolith_pct
        - 0.05 * logistics
    )
    crew_min = (
        base_crew
        + 14.0 * difficulty
        + 4.5 * float(features.get("num_items", len(picks)))
        + 6.0 * regolith_pct * 10.0
        + 2.0 * total_mass
    )

    return {
        "energy_kwh": float(max(0.0, energy_kwh)),
        "water_l": float(max(0.0, water_l)),
        "crew_min": float(max(5.0, crew_min)),
    }


def _label_tightness(features: Dict[str, Any], process: pd.Series) -> int:
    process_id = str(process.get("process_id", "")).upper()
    lamination = process_id in {"P02", "P04"}
    packaging = float(features.get("packaging_frac", 0.0))
    multilayer = float(features.get("multilayer_frac", 0.0))
    polyethylene = float(features.get("polyethylene_frac", 0.0))
    hydrogen = float(features.get("hydrogen_rich_frac", 0.0))
    regolith_pct = float(features.get("regolith_pct", 0.0))

    continuity_score = (
        0.38 * packaging
        + 0.32 * multilayer
        + 0.22 * polyethylene
        + 0.12 * hydrogen
        + (0.12 if lamination else 0.0)
        - 0.25 * regolith_pct
    )
    return int(continuity_score >= 0.55)


def _label_rigidity(features: Dict[str, Any]) -> int:
    aluminum = float(features.get("aluminum_frac", 0.0))
    carbon = float(features.get("carbon_fiber_frac", 0.0))
    regolith_pct = float(features.get("regolith_pct", 0.0))
    density = float(features.get("density_kg_m3", 0.0))
    foam = float(features.get("foam_frac", 0.0))
    difficulty = float(features.get("difficulty_index", 0.0))

    density_norm = float(np.clip(density / 4.0, 0.0, 1.2))
    reinforcement = 0.5 * aluminum + 0.45 * carbon + 0.35 * regolith_pct + 0.25 * density_norm
    reinforcement += 0.12 * difficulty
    reinforcement -= 0.35 * foam

    if reinforcement < 0.55:
        return 1
    if reinforcement < 0.95:
        return 2
    return 3


def _compute_targets(
    picks: DataFrame,
    process: pd.Series,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    regolith_pct = float(features.get("regolith_pct", 0.0))
    resources = _compute_resource_targets(features, picks, process, regolith_pct)

    recipe_id = derive_recipe_id(picks, process, features)
    if recipe_id:
        features["recipe_id"] = recipe_id
    process_id = features.get("process_id") or process.get("process_id", "")

    payload: Dict[str, Any] = {}

    curated_targets, curated_meta = lookup_labels(
        picks,
        str(process_id),
        {"recipe_id": recipe_id, "process_id": process_id},
    )

    for target, value in curated_targets.items():
        if target in TARGET_COLUMNS:
            payload[target] = float(value)
        elif target in CLASS_TARGET_COLUMNS:
            payload[target] = int(value)

    raw_provenance = curated_meta.get("provenance") or curated_meta.get("label_source")
    provenance = str(raw_provenance or "").lower()
    use_fallback = not payload or provenance in {"weak", "weakly_supervised"}

    original_label_source = None
    if "label_source" in curated_meta:
        original_label_source = str(curated_meta["label_source"])
        payload["label_source"] = original_label_source
    if "provenance" in curated_meta and curated_meta["provenance"]:
        payload["provenance"] = str(curated_meta["provenance"])
    if "label_weight" in curated_meta:
        try:
            payload["label_weight"] = float(curated_meta["label_weight"])
        except (TypeError, ValueError):
            payload["label_weight"] = 1.0

    for name, value in resources.items():
        payload.setdefault(name, value)

    if use_fallback or "tightness_pass" not in payload:
        tight_label = _label_tightness(features, process)
        payload["tightness_pass"] = int(tight_label)
    else:
        tight_label = int(payload["tightness_pass"])
    if use_fallback or "estanqueidad" not in payload:
        payload["estanqueidad"] = float(TIGHTNESS_SCORE_MAP.get(tight_label, tight_label))

    if use_fallback or "rigidity_level" not in payload:
        rigidity_label = _label_rigidity(features)
        payload["rigidity_level"] = int(rigidity_label)
    else:
        rigidity_label = int(payload["rigidity_level"])
    if use_fallback or "rigidez" not in payload:
        payload["rigidez"] = float(RIGIDITY_SCORE_MAP.get(rigidity_label, rigidity_label))

    if use_fallback:
        if raw_provenance:
            payload.setdefault("provenance", str(raw_provenance))
        elif original_label_source:
            payload.setdefault("provenance", original_label_source)
        payload["label_source"] = "simulated"
    elif "label_source" not in payload:
        payload["label_source"] = "measured"
    payload.setdefault("label_weight", 0.7)
    return payload


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
        recipe_id = derive_recipe_id(picks, process, features)
        if recipe_id:
            features.setdefault("recipe_id", recipe_id)
        targets = _compute_targets(picks, process, features)
        samples.append(SampledCombination(features=features, targets=targets))

    return samples


def build_training_dataframe(
    n_samples: int = 1600,
    seed: int | None = 21,
    *,
    gold_features_path: Path | None = None,
    gold_labels_path: Path | None = None,
) -> DataFrame:
    gold_features = _load_gold_features(gold_features_path)
    gold_targets = _load_gold_targets(gold_labels_path)

    if gold_features.empty or gold_targets.empty:
        samples = _generate_samples(n_samples, seed)
        df = pd.DataFrame([sample.as_row() for sample in samples])
        return df

    targets_flat = gold_targets.reset_index(drop=True)
    overlap = {
        column
        for column in targets_flat.columns
        if column in gold_features.columns and column not in {"recipe_id", "process_id"}
    }
    if overlap:
        targets_flat = targets_flat.drop(columns=list(overlap))

    df_gold = gold_features.merge(
        targets_flat,
        on=["recipe_id", "process_id"],
        how="inner",
    )

    if df_gold.empty:
        samples = _generate_samples(n_samples, seed)
        df = pd.DataFrame([sample.as_row() for sample in samples])
        return df

    df_gold = df_gold.copy()

    weight_sources = [
        column
        for column in ("label_weight", "weight", "sample_weight")
        if column in df_gold.columns
    ]
    if weight_sources:
        base_weight_column = weight_sources[0]
        df_gold["label_weight"] = pd.to_numeric(
            df_gold[base_weight_column], errors="coerce"
        ).fillna(1.0)
    else:
        df_gold["label_weight"] = 1.0

    if "label_source" in df_gold.columns:
        df_gold["label_source"] = (
            df_gold["label_source"].fillna("measured").astype(str)
        )
    elif "provenance" in df_gold.columns:
        df_gold["label_source"] = (
            df_gold["provenance"].fillna("measured").astype(str)
        )
    else:
        df_gold["label_source"] = "measured"

    for column in [c for c in df_gold.columns if c.startswith("conf_")]:
        df_gold[column] = pd.to_numeric(df_gold[column], errors="coerce")

    missing_targets = [col for col in TARGET_COLUMNS if col not in df_gold.columns]
    if missing_targets:
        raise ValueError(
            "El dataset gold no contiene todas las columnas objetivo requeridas: "
            + ", ".join(missing_targets)
        )

    processes = _load_process_catalog()
    processes["process_id_norm"] = processes["process_id"].astype(str).str.upper().str.strip()
    process_map = processes.set_index("process_id_norm")

    def _process_lookup(pid: Any) -> pd.Series:
        key = str(pid).upper().strip()
        if key in process_map.index:
            return process_map.loc[key]
        return pd.Series({"process_id": key})

    if "tightness_pass" not in df_gold.columns:
        df_gold["tightness_pass"] = df_gold.apply(
            lambda row: _label_tightness(
                row.to_dict(),
                _process_lookup(row["process_id"]),
            ),
            axis=1,
        )
    else:
        mask = df_gold["tightness_pass"].isna()
        if mask.any():
            df_gold.loc[mask, "tightness_pass"] = df_gold.loc[mask].apply(
                lambda row: _label_tightness(
                    row.to_dict(),
                    _process_lookup(row["process_id"]),
                ),
                axis=1,
            )

    if "rigidity_level" not in df_gold.columns:
        df_gold["rigidity_level"] = df_gold.apply(
            lambda row: _label_rigidity(row.to_dict()),
            axis=1,
        )
    else:
        mask = df_gold["rigidity_level"].isna()
        if mask.any():
            df_gold.loc[mask, "rigidity_level"] = df_gold.loc[mask].apply(
                lambda row: _label_rigidity(row.to_dict()),
                axis=1,
            )

    df_gold["recipe_id"] = df_gold["recipe_id"].astype(str).str.upper().str.strip()
    df_gold["process_id"] = df_gold["process_id"].astype(str).str.upper().str.strip()
    df_gold["tightness_pass"] = df_gold["tightness_pass"].astype(int)
    df_gold["rigidity_level"] = df_gold["rigidity_level"].astype(int)
    return df_gold


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
    ) -> tuple[
    Pipeline,
    Dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, float]],
]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]
    meta = df[["label_source", "label_weight"]].copy()
    meta["label_source"] = meta["label_source"].astype(str)
    meta["label_weight"] = pd.to_numeric(meta["label_weight"], errors="coerce").fillna(1.0)

    X_train, X_valid, y_train, y_valid, meta_train, meta_valid = train_test_split(
        X,
        y,
        meta,
        test_size=0.2,
        random_state=seed or 0,
    )

    train_weights = meta_train["label_weight"].to_numpy(dtype=float)

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
    fit_params: Dict[str, Any] = {}
    if train_weights.size:
        fit_params["regressor__sample_weight"] = train_weights
    pipeline.fit(X_train, y_train, **fit_params)

    preds = pipeline.predict(X_valid)
    residuals = y_valid.to_numpy(dtype=float) - preds
    valid_weights = meta_valid["label_weight"].to_numpy(dtype=float)
    if valid_weights.size and np.isfinite(valid_weights).any() and float(valid_weights.sum()) > 0:
        weights = np.clip(valid_weights, 1e-6, None)
        residual_std = np.sqrt(np.average(residuals**2, weights=weights, axis=0))
    else:
        weights = None
        residual_std = residuals.std(axis=0)

    metrics = {
        target: {
            "mae": float(
                mean_absolute_error(y_valid[target], preds[:, idx], sample_weight=weights)
                if weights is not None
                else mean_absolute_error(y_valid[target], preds[:, idx])
            ),
            "rmse": float(
                math.sqrt(mean_squared_error(y_valid[target], preds[:, idx], sample_weight=weights))
                if weights is not None
                else math.sqrt(mean_squared_error(y_valid[target], preds[:, idx]))
            ),
            "r2": float(
                r2_score(y_valid[target], preds[:, idx], sample_weight=weights)
                if weights is not None
                else r2_score(y_valid[target], preds[:, idx])
            ),
        }
        for idx, target in enumerate(TARGET_COLUMNS)
    }
    metrics["overall"] = {
        "mae": float(np.mean([m["mae"] for m in metrics.values()])),
        "rmse": float(np.mean([m["rmse"] for m in metrics.values()])),
        "r2": float(np.mean([m["r2"] for m in metrics.values()])),
    }

    quantiles = np.quantile(residuals, [0.1, 0.5, 0.9], axis=0)
    residual_summary: Dict[str, Dict[str, float]] = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        series = residuals[:, idx]
        residual_summary[target] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "mae": float(np.mean(np.abs(series))),
            "p10": float(quantiles[0, idx]),
            "p50": float(quantiles[1, idx]),
            "p90": float(quantiles[2, idx]),
        }

    residuals_df = pd.DataFrame(residuals, columns=TARGET_COLUMNS)
    residuals_df["label_source"] = meta_valid["label_source"].to_numpy()
    residuals_df["label_weight"] = valid_weights
    residual_by_source: Dict[str, Dict[str, Any]] = {}
    for source, group in residuals_df.groupby("label_source"):
        group_weights = pd.to_numeric(group["label_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if group_weights.size and float(group_weights.sum()) > 0:
            w = np.clip(group_weights, 1e-6, None)
            rmse = np.sqrt(
                np.average(
                    group[TARGET_COLUMNS].to_numpy(dtype=float) ** 2,
                    weights=w,
                    axis=0,
                )
            )
        else:
            rmse = group[TARGET_COLUMNS].to_numpy(dtype=float).std(axis=0)
        residual_by_source[str(source)] = {
            "count": int(len(group)),
            "rmse": {target: float(rmse[idx]) for idx, target in enumerate(TARGET_COLUMNS)},
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

    return (
        pipeline,
        rf_payload,
        feature_means,
        feature_stds,
        residual_std,
        feature_names,
        residual_by_source,
        residual_summary,
    )


def _train_classifiers(
    pipeline: Pipeline,
    df: DataFrame,
    seed: int | None,
) -> Dict[str, Any]:
    if "tightness_pass" not in df.columns or "rigidity_level" not in df.columns:
        return {}

    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocess")
    if preprocessor is None:
        return {}

    matrix = preprocessor.transform(df[FEATURE_COLUMNS])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)

    rng_seed = seed or 0
    payload: Dict[str, Any] = {}

    # Tightness (binary)
    y_tight = df["tightness_pass"].to_numpy(dtype=int)
    stratify_tight = y_tight if len(np.unique(y_tight)) > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        matrix,
        y_tight,
        test_size=0.25,
        random_state=rng_seed,
        stratify=stratify_tight,
    )
    tight_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=rng_seed,
    )
    tight_clf.fit(X_train, y_train)
    tight_preds = tight_clf.predict(X_valid)
    tight_metrics = {
        "accuracy": float(accuracy_score(y_valid, tight_preds)),
        "precision": float(precision_score(y_valid, tight_preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_valid, tight_preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_valid, tight_preds, pos_label=1, zero_division=0)),
        "support": _support_dict(y_valid),
    }
    joblib.dump(tight_clf, TIGHTNESS_MODEL_PATH)
    payload["tightness_pass"] = {
        "path": _relative_path(TIGHTNESS_MODEL_PATH),
        "metrics": tight_metrics,
        "classes": [int(c) for c in tight_clf.classes_],
        "score_map": {int(k): float(v) for k, v in TIGHTNESS_SCORE_MAP.items()},
    }

    # Rigidity (ordinal 1-3)
    y_rigidity = df["rigidity_level"].to_numpy(dtype=int)
    stratify_rigidity = y_rigidity if len(np.unique(y_rigidity)) > 1 else None
    X_train_r, X_valid_r, y_train_r, y_valid_r = train_test_split(
        matrix,
        y_rigidity,
        test_size=0.25,
        random_state=rng_seed,
        stratify=stratify_rigidity,
    )
    rigidity_clf = RandomForestClassifier(
        n_estimators=320,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=rng_seed + 7,
    )
    rigidity_clf.fit(X_train_r, y_train_r)
    rigidity_preds = rigidity_clf.predict(X_valid_r)
    rigidity_metrics = {
        "accuracy": float(accuracy_score(y_valid_r, rigidity_preds)),
        "precision_weighted": float(
            precision_score(y_valid_r, rigidity_preds, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_valid_r, rigidity_preds, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_valid_r, rigidity_preds, average="weighted", zero_division=0)),
        "support": _support_dict(y_valid_r),
    }
    joblib.dump(rigidity_clf, RIGIDITY_MODEL_PATH)
    payload["rigidity_level"] = {
        "path": _relative_path(RIGIDITY_MODEL_PATH),
        "metrics": rigidity_metrics,
        "classes": [int(c) for c in rigidity_clf.classes_],
        "score_map": {int(k): float(v) for k, v in RIGIDITY_SCORE_MAP.items()},
    }

    return payload


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
    return {"metrics": metrics, "path": _relative_path(XGBOOST_PATH)}


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
    return {"latent_dim": LATENT_DIM, "path": _relative_path(AUTOENCODER_PATH)}


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

    torch.save(
        {"state_dict": model.state_dict(), "tokens": model.n_tokens, "d_model": model.d_model},
        TABTRANSFORMER_PATH,
    )
    return {
        "tokens": model.n_tokens,
        "d_model": model.d_model,
        "path": _relative_path(TABTRANSFORMER_PATH),
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_save(
    n_samples: int = 1600,
    seed: int | None = 21,
    *,
    gold_features_path: Path | None = None,
    gold_labels_path: Path | None = None,
) -> Dict[str, Any]:
    """Generate data, train models and persist artefacts to disk."""

    _set_seed(seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_ML.mkdir(parents=True, exist_ok=True)

    df = build_training_dataframe(
        n_samples=n_samples,
        seed=seed,
        gold_features_path=gold_features_path,
        gold_labels_path=gold_labels_path,
    )
    df.to_parquet(DATASET_PATH, index=False)
    df.to_parquet(DATASET_ML_PATH, index=False)

    (
        pipeline,
        rf_payload,
        feature_means,
        feature_stds,
        residual_std,
        feature_names,
        residual_by_source,
        residual_summary,
    ) = _train_random_forest(df, seed)
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
    extras["classifiers"] = _train_classifiers(pipeline, df, seed)

    label_summary = (
        df.groupby("label_source")["label_weight"]
        .agg(["count", "mean", "min", "max"])
        .rename(columns={"count": "n", "mean": "mean", "min": "min", "max": "max"})
    )
    label_summary_dict = {
        str(source): {
            "count": int(values["n"]),
            "mean_weight": float(values["mean"]),
            "min_weight": float(values["min"]),
            "max_weight": float(values["max"]),
        }
        for source, values in label_summary.iterrows()
    }

    trained_on = _infer_trained_on_label(df)

    metadata = {
        "model_name": "rexai-rf-ensemble",
        "trained_on": trained_on,
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "n_samples": int(len(df)),
        "dataset": {
            "path": _relative_path(DATASET_PATH),
            "hash": _hash_file(DATASET_PATH),
        },
        "feature_columns": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "classification_targets": CLASS_TARGET_COLUMNS,
        "post_transform_features": preprocessor.get_feature_names_out().tolist(),
        "feature_means": {name: float(val) for name, val in zip(feature_names, feature_means)},
        "feature_stds": {name: float(val) for name, val in zip(feature_names, feature_stds)},
        "residual_std": {target: float(val) for target, val in zip(TARGET_COLUMNS, residual_std)},
        "residuals_by_label_source": residual_by_source,
        "residuals_summary": residual_summary,
        "random_forest": rf_payload,
        "artifacts": {
            "pipeline": _relative_path(PIPELINE_PATH),
            "xgboost": extras["xgboost"],
            "autoencoder": extras["autoencoder"],
            "tabtransformer": extras["tabtransformer"],
        },
        "classifiers": extras["classifiers"],
        "labeling": {
            "columns": {"source": "label_source", "weight": "label_weight"},
            "summary": label_summary_dict,
        },
    }

    payload = json.dumps(metadata, indent=2, sort_keys=True)
    METADATA_PATH.write_text(payload, encoding="utf-8")
    try:
        LEGACY_METADATA_PATH.write_text(payload, encoding="utf-8")
    except Exception:  # pragma: no cover - legacy path optional
        pass
    return metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training pipeline para Rex-AI")
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help=(
            "Directorio que contiene features.parquet y labels.parquet para "
            "entrenamiento con datos dorados."
        ),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help=(
            "Ruta alternativa (o directorio) para features.parquet. Si se proporciona "
            "un directorio se asumirá un archivo llamado features.parquet."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1600,
        help="Número de combinaciones sintéticas a generar si faltan etiquetas doradas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Semilla usada para generación y entrenamiento.",
    )
    return parser


def _resolve_gold_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    gold_features: Path | None = None
    gold_labels: Path | None = None

    if args.gold is not None:
        gold_dir = Path(args.gold)
        gold_features = gold_dir / "features.parquet"
        gold_labels = gold_dir / "labels.parquet"

    if args.features is not None:
        features_path = Path(args.features)
        if features_path.is_dir():
            features_path = features_path / "features.parquet"
        gold_features = features_path

    return gold_features, gold_labels


def cli(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    gold_features_path, gold_labels_path = _resolve_gold_paths(args)
    return train_and_save(
        n_samples=args.samples,
        seed=args.seed,
        gold_features_path=gold_features_path,
        gold_labels_path=gold_labels_path,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    info = cli()
    print(json.dumps(info, indent=2))
