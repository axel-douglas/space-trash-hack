"""Candidate generation utilities for the Rex-AI demo.

This module converts NASA's non-metabolic waste inventory into the
structured features consumed by the machine learning models.  When
artifacts are available, predictions are served from the trained
RandomForest/XGBoost ensemble; otherwise the fallback heuristics ensure
the UI remains functional.

The code historically suffered from duplicated blocks introduced during
rapid prototyping.  The refactor below consolidates the logic into
clear, reusable helpers so that both the app runtime and the training
pipeline can rely on a single source of truth.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Sequence, Tuple

try:
    import jax.numpy as jnp
    from jax import jit
except Exception:  # pragma: no cover - JAX is optional during inference
    jnp = None  # type: ignore[assignment]

    def jit(fn):  # type: ignore[override]
        return fn

import numpy as np
import pandas as pd
import polars as pl

try:  # Optional heavy dependencies; gracefully disable logging if missing
    import pyarrow as pa
except Exception:  # pragma: no cover - pyarrow is expected in production
    pa = None  # type: ignore[assignment]

try:  # ``deltalake`` provides lightweight Delta transactions
    from deltalake.writer import write_deltalake
except Exception:  # pragma: no cover - deltalake is expected in production
    write_deltalake = None  # type: ignore[assignment]

from app.modules.label_mapper import derive_recipe_id, lookup_labels
from app.modules.ranking import derive_auxiliary_signals, score_recipe

try:  # Lazy import to avoid circular dependency during training pipelines
    from app.modules.ml_models import MODEL_REGISTRY
except Exception:  # pragma: no cover - fallback when models are not available
    MODEL_REGISTRY = None

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
LOGS_ROOT = Path(__file__).resolve().parents[2] / "data" / "logs"

_INFERENCE_LOG_LOCK = threading.Lock()


def _to_lazy_frame(
    frame: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
) -> tuple[pl.LazyFrame, str]:
    """Return a :class:`polars.LazyFrame` along with the original frame type."""

    if isinstance(frame, pl.LazyFrame):
        return frame, "lazy"
    if isinstance(frame, pl.DataFrame):
        return frame.lazy(), "polars"
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame).lazy(), "pandas"
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _from_lazy_frame(lazy: pl.LazyFrame, frame_kind: str) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
    """Convert *lazy* back to the representation described by *frame_kind*."""

    if frame_kind == "lazy":
        return lazy

    collected = lazy.collect()
    if frame_kind == "polars":
        return collected
    if frame_kind == "pandas":
        return collected.to_pandas()
    raise ValueError(f"Unsupported frame kind: {frame_kind}")


def _resolve_dataset_path(name: str) -> Path | None:
    """Return the first dataset path that exists for *name*.

    The helper checks the canonical ``datasets`` root alongside the ``raw``
    subdirectory so callers do not need to remember where a file was stored.
    """

    candidates = (
        DATASETS_ROOT / name,
        DATASETS_ROOT / "raw" / name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _slugify(value: str) -> str:
    """Convert *value* into a snake_case identifier safe for feature names."""

    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def _to_serializable(value: Any) -> Any:
    """Convert *value* into a JSON-serializable structure."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_to_serializable(v) for v in value.tolist()]
    return str(value)


def _resolve_inference_log_dir(timestamp: datetime) -> Path:
    """Return the Delta Lake directory for the given *timestamp*."""

    return LOGS_ROOT / "inference" / timestamp.strftime("%Y%m%d")


def _prepare_inference_event(
    input_features: Dict[str, Any],
    prediction: Dict[str, Any] | None,
    uncertainty: Dict[str, Any] | None,
    model_registry: Any | None,
    timestamp: datetime | None = None,
) -> tuple[datetime, Dict[str, str | None]]:
    """Build the serializable payload for an inference log event."""

    now = timestamp or datetime.now(UTC)

    model_hash = ""
    if model_registry is not None:
        metadata = getattr(model_registry, "metadata", {}) or {}
        if isinstance(metadata, dict):
            model_hash = str(metadata.get("model_hash") or metadata.get("checksum") or "")
        if not model_hash:
            for attr in ("model_hash", "checksum", "pipeline_checksum", "pipeline_hash"):
                value = getattr(model_registry, attr, None)
                if value:
                    model_hash = str(value)
                    break

    payload: Dict[str, str | None] = {
        "timestamp": now.isoformat(timespec="microseconds"),
        "input_features": json.dumps(_to_serializable(input_features or {}), sort_keys=True),
        "prediction": json.dumps(_to_serializable(prediction or {}), sort_keys=True),
        "uncertainty": json.dumps(_to_serializable(uncertainty or {}), sort_keys=True),
        "model_hash": model_hash or None,
    }

    return now, payload


def _append_inference_log(
    input_features: Dict[str, Any],
    prediction: Dict[str, Any] | None,
    uncertainty: Dict[str, Any] | None,
    model_registry: Any | None,
) -> None:
    """Persist an inference event using Delta transactions to avoid read-modify-write."""

    if pa is None or write_deltalake is None:  # pragma: no cover - dependencies should exist
        return

    try:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    event_time, event_payload = _prepare_inference_event(
        input_features=input_features,
        prediction=prediction,
        uncertainty=uncertainty,
        model_registry=model_registry,
    )

    log_dir = _resolve_inference_log_dir(event_time)

    try:
        log_dir.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    data = {
        key: pa.array([value], type=pa.string())
        for key, value in event_payload.items()
    }

    table = pa.table(data)

    try:
        with _INFERENCE_LOG_LOCK:
            write_deltalake(
                str(log_dir),
                table,
                mode="append",
                schema_mode="merge",
                engine="rust",
            )
    except Exception:
        return


def _load_regolith_vector() -> Dict[str, float]:
    path = _resolve_dataset_path("MGS-1_Martian_Regolith_Simulant_Recipe.csv")
    if path is None:
        path = DATASETS_ROOT / "raw" / "mgs1_oxides.csv"

    if path and path.exists():
        table = pd.read_csv(path)
        key_cols = [
            col
            for col in table.columns
            if col.lower() in {"oxide", "component", "phase", "mineral"}
        ]
        value_cols = [
            col
            for col in table.columns
            if any(token in col.lower() for token in ("wt", "weight", "percent"))
        ]

        key_col = key_cols[0] if key_cols else None
        value_col = value_cols[0] if value_cols else None

        if key_col and value_col:
            working = table[[key_col, value_col]].dropna()

            def _clean_label(value: Any) -> str:
                text = str(value or "").lower()
                text = re.sub(r"[^0-9a-z]+", "_", text)
                text = re.sub(r"_+", "_", text).strip("_")
                return text

            working[key_col] = working[key_col].map(_clean_label)
            weights = pd.to_numeric(working[value_col], errors="coerce")
            total = float(weights.sum())
            if total > 0:
                normalised = weights.div(total)
                return {
                    str(key): float(normalised.iloc[idx])
                    for idx, key in enumerate(working[key_col])
                    if pd.notna(normalised.iloc[idx])
                }

    return {"sio2": 0.48, "feot": 0.18, "mgo": 0.13, "cao": 0.055, "so3": 0.07, "h2o": 0.032}


def _load_gas_mean_yield() -> float:
    path = DATASETS_ROOT / "raw" / "nasa_trash_to_gas.csv"
    if path.exists():
        table = pd.read_csv(path)
        ratio = table["o2_ch4_yield_kg"] / table["water_makeup_kg"].clip(lower=1e-6)
        return float(ratio.mean())
    return 6.0


def _load_mean_reuse() -> float:
    path = DATASETS_ROOT / "raw" / "logistics_to_living.csv"
    if path.exists():
        table = pd.read_csv(path)
        efficiency = (
            (table["outfitting_replaced_kg"] - table["residual_waste_kg"]) / table["packaging_kg"].clip(lower=1e-6)
        ).clip(lower=0)
        return float(efficiency.mean())
    return 0.6


@dataclass
class _L2LParameters:
    constants: Dict[str, float]
    category_features: Dict[str, Dict[str, float]]
    item_features: Dict[str, Dict[str, float]]
    hints: Dict[str, str]


def _parse_l2l_numeric(value: Any) -> Dict[str, float]:
    """Return numeric representations for Logistics-to-Living values."""

    result: Dict[str, float] = {}
    if value is None:
        return result

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isfinite(value):
            result["value"] = float(value)
        return result

    text = str(value).strip()
    if not text:
        return result

    cleaned = text.replace(",", "").replace("—", "-").replace("–", "-").replace("−", "-")
    numbers = [
        float(match)
        for match in re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if match not in {"-"}
    ]
    if not numbers:
        return result

    lowered = cleaned.lower()
    if any(token in lowered for token in (" per ", ":", "/")) and len(numbers) >= 2:
        denominator = numbers[1]
        if denominator:
            result["value"] = numbers[0] / denominator
            result["numerator"] = numbers[0]
            result["denominator"] = denominator
        else:
            result["value"] = numbers[0]
        return result

    if "-" in cleaned and len(numbers) >= 2:
        result["min"] = numbers[0]
        result["max"] = numbers[1]
        result["value"] = float(np.mean(numbers[:2]))
        return result

    result["value"] = numbers[0]
    if len(numbers) > 1:
        result["extra"] = numbers[1]
    return result


def _feature_name_from_parts(*parts: str) -> str:
    return "_".join(part for part in (_slugify(part) for part in parts if part) if part)


def _load_l2l_parameters() -> _L2LParameters:
    path = _resolve_dataset_path("l2l_parameters.csv")
    if path is None or not path.exists():
        return _L2LParameters({}, {}, {}, {})

    table = pd.read_csv(path)
    if table.empty:
        return _L2LParameters({}, {}, {}, {})

    normalized_cols = {col.lower(): col for col in table.columns}
    category_col = normalized_cols.get("category")
    subitem_col = normalized_cols.get("subitem")

    descriptor_cols = [
        normalized_cols[name]
        for name in ("parameter", "metric", "key", "feature", "name", "field")
        if name in normalized_cols
    ]
    value_candidates = [
        column
        for column in table.columns
        if column not in {category_col, subitem_col}
        and column not in descriptor_cols
        and column.lower() not in {"page_hint", "units", "unit", "notes"}
    ]

    constants: Dict[str, float] = {}
    category_features: Dict[str, Dict[str, float]] = {}
    item_features: Dict[str, Dict[str, float]] = {}
    hints: Dict[str, str] = {}

    global_categories = {
        "geometry",
        "logistics",
        "scenario",
        "scenarios",
        "testbed",
        "ops",
        "operations",
        "materials",
        "material",
        "global",
        "constants",
    }

    for _, row in table.iterrows():
        category_value = row.get(category_col, "") if category_col else ""
        category_norm = _normalize_category(category_value)
        subitem_value = row.get(subitem_col, "") if subitem_col else ""
        subitem_norm = _normalize_item(subitem_value) if subitem_value else ""

        descriptor = ""
        for candidate in descriptor_cols:
            value = str(row.get(candidate, "")).strip()
            if value:
                descriptor = value
                break

        hint = str(row.get(normalized_cols.get("page_hint", "page_hint"), "")).strip()

        target_map: Dict[str, Dict[str, float]] | None
        key: str | None

        if category_norm in global_categories or not category_norm:
            target_map = None
            key = None
        elif subitem_norm:
            key = f"{category_norm}|{subitem_norm}"
            target_map = item_features
        else:
            key = category_norm
            target_map = category_features

        base_parts = ["l2l", category_norm]
        if subitem_norm:
            base_parts.append(subitem_norm)
        if descriptor:
            base_parts.append(descriptor)

        for column in value_candidates:
            payload = _parse_l2l_numeric(row.get(column))
            if not payload:
                continue

            for suffix, numeric_value in payload.items():
                if not np.isfinite(numeric_value):
                    continue
                name_parts = list(base_parts)
                if column:
                    name_parts.append(column)
                if suffix not in {"value"}:
                    name_parts.append(suffix)
                feature_name = _feature_name_from_parts(*name_parts)
                if not feature_name:
                    continue

                if category_norm in global_categories or not category_norm:
                    constants[feature_name] = float(numeric_value)
                elif target_map is not None and key is not None:
                    entry = target_map.setdefault(key, {})
                    entry[feature_name] = float(numeric_value)
                else:
                    constants[feature_name] = float(numeric_value)

                if hint:
                    hints[feature_name] = hint

    return _L2LParameters(constants, category_features, item_features, hints)


_REGOLITH_VECTOR = _load_regolith_vector()
_GAS_MEAN_YIELD = _load_gas_mean_yield()
_MEAN_REUSE = _load_mean_reuse()

_OFFICIAL_FEATURES_PATH = DATASETS_ROOT / "rexai_nasa_waste_features.csv"

_CATEGORY_SYNONYMS = {
    "foam": "foam packaging",
    "foam packaging": "foam packaging",
    "packaging": "other packaging glove",
    "other packaging": "other packaging glove",
    "glove": "other packaging glove",
    "other packaging glove": "other packaging glove",
    "food packaging": "food packaging",
    "structural elements": "structural element",
    "structural element": "structural element",
    "eva": "eva waste",
    "eva waste": "eva waste",
    "gloves": "other packaging glove",
}

_COMPOSITION_DENSITY_MAP = {
    "Aluminum_pct": 2700.0,
    "Carbon_Fiber_pct": 1700.0,
    "Polyethylene_pct": 950.0,
    "PVDF_pct": 1780.0,
    "Nomex_pct": 1350.0,
    "Nylon_pct": 1140.0,
    "Polyester_pct": 1380.0,
    "Cotton_Cellulose_pct": 1550.0,
    "EVOH_pct": 1250.0,
    "PET_pct": 1370.0,
    "Nitrile_pct": 1030.0,
}

_CATEGORY_DENSITY_DEFAULTS = {
    "foam packaging": 100.0,
    "food packaging": 650.0,
    "structural element": 1800.0,
    "other packaging glove": 420.0,
    "eva waste": 240.0,
    "fabric": 350.0,
}


def _merge_reference_dataset(
    base: pd.DataFrame | pl.DataFrame | pl.LazyFrame, filename: str, prefix: str
) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
    path = _resolve_dataset_path(filename)
    if path is None:
        return base

    base_lazy, base_kind = _to_lazy_frame(base)
    base_columns = list(base_lazy.columns)

    extra_lazy = pl.scan_csv(path)
    extra_columns = extra_lazy.columns

    join_cols = [col for col in ("category", "subitem") if col in base_columns and col in extra_columns]
    if not join_cols:
        return base

    existing = set(base_columns)
    rename_map: Dict[str, str] = {}
    drop_cols: list[str] = []
    for column in extra_columns:
        if column in join_cols:
            continue
        if column in existing:
            drop_cols.append(column)
            continue
        rename_map[column] = f"{prefix}_{_slugify(column)}"

    if drop_cols:
        extra_lazy = extra_lazy.drop(drop_cols)
    if rename_map:
        extra_lazy = extra_lazy.rename(rename_map)

    added_columns = [rename_map.get(col, col) for col in extra_columns if col not in join_cols and col not in drop_cols]

    merged_lazy = base_lazy.join(extra_lazy, on=join_cols, how="left")
    if added_columns:
        projection = base_columns + [col for col in added_columns if col not in base_columns]
        merged_lazy = merged_lazy.select([pl.col(name) for name in projection])

    result = _from_lazy_frame(merged_lazy, base_kind)
    if isinstance(result, pd.DataFrame):
        return result.loc[:, ~result.columns.duplicated()]
    if isinstance(result, pl.DataFrame):
        unique_cols = []
        seen: set[str] = set()
        for name in result.columns:
            if name in seen:
                continue
            seen.add(name)
            unique_cols.append(name)
        return result.select(unique_cols)
    return result


def _mission_slug(column: str) -> str:
    cleaned = column.lower()
    cleaned = cleaned.replace("summary_", "")
    cleaned = cleaned.replace("mass", "")
    cleaned = cleaned.replace("kg", "")
    cleaned = cleaned.replace("total", "")
    cleaned = cleaned.replace("__", "_")
    return _slugify(cleaned)


class _WasteSummary(NamedTuple):
    mass_by_key: Dict[str, Dict[str, float]]
    mission_totals: Dict[str, float]


def _load_waste_summary_data() -> _WasteSummary:
    path = _resolve_dataset_path("nasa_waste_summary.csv")
    if path is None:
        return _WasteSummary({}, {})

    table = pl.scan_csv(path)
    if "category" not in table.columns:
        return _WasteSummary({}, {})

    mass_columns = [
        column
        for column in table.columns
        if column.lower().endswith("mass_kg") and not column.lower().startswith("subitem_")
    ]
    if not mass_columns:
        return _WasteSummary({}, {})

    has_subitem = "subitem" in table.columns
    subitem_expr = (
        pl.when(pl.col("subitem").is_not_null())
        .then(pl.col("subitem").map_elements(_normalize_item, return_dtype=pl.String))
        .otherwise(pl.lit(""))
        .alias("subitem_norm")
        if has_subitem
        else pl.lit("").alias("subitem_norm")
    )

    melted = (
        table.with_columns(
            pl.col("category")
            .map_elements(_normalize_category, return_dtype=pl.String)
            .alias("category_norm"),
            subitem_expr,
        )
        .with_columns(
            pl.when(pl.col("subitem_norm").str.len_bytes() > 0)
            .then(pl.col("category_norm") + pl.lit("|") + pl.col("subitem_norm"))
            .otherwise(pl.col("category_norm"))
            .alias("item_key"),
            pl.col("category_norm").alias("category_key"),
        )
        .melt(
            id_vars=["category_key", "item_key"],
            value_vars=mass_columns,
            variable_name="mission_column",
            value_name="mass_value",
        )
        .with_columns(
            pl.col("mission_column")
            .map_elements(_mission_slug, return_dtype=pl.String)
            .alias("mission"),
            pl.col("mass_value").cast(pl.Float64, strict=False).alias("mass"),
        )
        .filter(pl.col("mission").is_not_null() & pl.col("mass").is_finite() & (pl.col("mass") > 0))
    )

    row_count = melted.select(pl.len().alias("rows")).collect().row(0)[0]
    if row_count == 0:
        return _WasteSummary({}, {})

    mission_totals = {
        row["mission"]: float(row["mass"])
        for row in melted.group_by("mission").agg(pl.col("mass").sum()).collect().to_dicts()
        if row["mission"]
    }

    mass_by_key: Dict[str, Dict[str, float]] = {}

    subitem_totals = (
        melted
        .filter(pl.col("item_key") != pl.col("category_key"))
        .group_by(["item_key", "mission"])
        .agg(pl.col("mass").sum())
        .collect()
        .to_dicts()
    )

    for row in subitem_totals:
        key = row.get("item_key")
        mission = row.get("mission")
        value = row.get("mass")
        if not key or not mission or value is None:
            continue
        entry = mass_by_key.setdefault(str(key), {})
        entry[str(mission)] = entry.get(str(mission), 0.0) + float(value)

    for row in (
        melted.group_by(["category_key", "mission"]).agg(pl.col("mass").sum()).collect().to_dicts()
    ):
        key = row.get("category_key")
        mission = row.get("mission")
        value = row.get("mass")
        if not key or not mission or value is None:
            continue
        entry = mass_by_key.setdefault(str(key), {})
        entry[str(mission)] = entry.get(str(mission), 0.0) + float(value)

    return _WasteSummary(mass_by_key, mission_totals)


def _extract_grouped_metrics(filename: str, prefix: str) -> Dict[str, Dict[str, float]]:
    path = _resolve_dataset_path(filename)
    if path is None:
        return {}

    table = pl.scan_csv(path)

    row_count = table.select(pl.len().alias("rows")).collect().row(0)[0]
    if row_count == 0:
        return {}

    schema = table.schema
    numeric_cols = [
        name
        for name, dtype in schema.items()
        if dtype.is_numeric()
    ]
    if not numeric_cols:
        return {}

    group_candidates = {
        "mission",
        "scenario",
        "approach",
        "vehicle",
        "propulsion",
        "architecture",
    }
    group_columns = [col for col in table.columns if col.lower() in group_candidates]

    aggregated: Dict[str, Dict[str, float]] = {}

    if not group_columns:
        summary = (
            table.select([pl.col(col).cast(pl.Float64, strict=False).mean().alias(col) for col in numeric_cols])
            .collect()
            .to_dicts()
        )
        metrics = {}
        if summary:
            metrics = {}
            for column, value in summary[0].items():
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                metrics[f"{prefix}_{_slugify(column)}"] = float(value)
        if metrics:
            aggregated[prefix] = metrics
        return aggregated

    combinations: list[tuple[str, ...]] = []
    for length in range(1, len(group_columns) + 1):
        combinations.extend(itertools.combinations(group_columns, length))

    for combo in combinations:
        grouped = (
            table.group_by(list(combo))
            .agg([pl.col(col).cast(pl.Float64, strict=False).mean().alias(col) for col in numeric_cols])
            .collect()
            .to_dicts()
        )
        for row in grouped:
            slug_parts: list[str] = []
            for column in combo:
                value = row.get(column)
                if isinstance(value, str):
                    slug = _slugify(value)
                elif value is not None:
                    slug = _slugify(str(value))
                else:
                    slug = ""
                if slug:
                    slug_parts.append(slug)
            slug = "_".join(part for part in slug_parts if part)
            if not slug:
                continue

            metrics: Dict[str, float] = {}
            for column in numeric_cols:
                value = row.get(column)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                metrics[f"{prefix}_{_slugify(column)}"] = float(value)

            if metrics:
                aggregated[slug] = metrics

    return aggregated

def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("—", " ").replace("/", " ")
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = []
    for token in text.split():
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens).strip()


def _normalize_category(value: Any) -> str:
    normalized = _normalize_text(value)
    return _CATEGORY_SYNONYMS.get(normalized, normalized)
def _build_match_key(category: Any, subitem: Any | None = None) -> str:
    """Return the canonical key used to match NASA reference tables."""

    if subitem:
        return f"{_normalize_category(category)}|{_normalize_item(subitem)}"
    return _normalize_category(category)
def _estimate_density_from_row(row: pd.Series) -> float | None:
    category = _normalize_category(row.get("category", ""))

    try:
        cat_mass = float(row.get("category_total_mass_kg"))
        cat_volume = float(row.get("category_total_volume_m3"))
    except (TypeError, ValueError):
        cat_mass = cat_volume = float("nan")

    if pd.notna(cat_mass) and pd.notna(cat_volume) and cat_volume > 0:
        return float(np.clip(cat_mass / cat_volume, 20.0, 4000.0))

    composition_weights: list[tuple[float, float]] = []
    total = 0.0
    for column, density in _COMPOSITION_DENSITY_MAP.items():
        try:
            pct = float(row.get(column, 0.0))
        except (TypeError, ValueError):
            pct = 0.0
        if pct and not np.isnan(pct):
            frac = pct / 100.0
            if frac > 0:
                composition_weights.append((frac, density))
                total += frac

    if total > 0 and composition_weights:
        weighted = sum(frac * density for frac, density in composition_weights) / total
        if category == "foam packaging":
            return float(min(weighted, _CATEGORY_DENSITY_DEFAULTS.get(category, weighted)))
        return float(np.clip(weighted, 20.0, 4000.0))

    if category in _CATEGORY_DENSITY_DEFAULTS:
        return float(_CATEGORY_DENSITY_DEFAULTS[category])

    return None
def _normalize_item(value: Any) -> str:
    return _normalize_text(value)


def _token_set(value: Any) -> frozenset[str]:
    normalized = _normalize_item(value)
    if not normalized:
        return frozenset()
    return frozenset(normalized.split())


_L2L_PARAMETERS = _load_l2l_parameters()


class _OfficialFeaturesBundle(NamedTuple):
    value_columns: tuple[str, ...]
    composition_columns: tuple[str, ...]
    direct_map: Dict[str, Dict[str, float]]
    category_tokens: Dict[str, list[tuple[frozenset[str], Dict[str, float], str]]]
    table: pl.DataFrame
    mission_mass: Dict[str, Dict[str, float]]
    mission_totals: Dict[str, float]
    processing_metrics: Dict[str, Dict[str, float]]
    leo_mass_savings: Dict[str, Dict[str, float]]
    propellant_benefits: Dict[str, Dict[str, float]]
    l2l_constants: Dict[str, float]
    l2l_category_features: Dict[str, Dict[str, float]]
    l2l_item_features: Dict[str, Dict[str, float]]
    l2l_hints: Dict[str, str]


@lru_cache(maxsize=1)
def _official_features_bundle() -> _OfficialFeaturesBundle:
    l2l = _L2L_PARAMETERS
    default = _OfficialFeaturesBundle(
        (),
        (),
        {},
        pl.DataFrame(),
        {},
        {},
        {},
        {},
        {},
        {},
        l2l.constants,
        l2l.category_features,
        l2l.item_features,
        l2l.hints,
    )

    if not _OFFICIAL_FEATURES_PATH.exists():
        return default

    table_lazy = pl.scan_csv(_OFFICIAL_FEATURES_PATH)
    duplicate_suffixes = [column for column in table_lazy.columns if column.endswith(".1")]
    if duplicate_suffixes:
        table_lazy = table_lazy.drop(duplicate_suffixes)

    table_lazy = _merge_reference_dataset(table_lazy, "nasa_waste_summary.csv", "summary")
    table_lazy = _merge_reference_dataset(table_lazy, "nasa_waste_processing_products.csv", "processing")
    table_lazy = _merge_reference_dataset(table_lazy, "nasa_leo_mass_savings.csv", "leo")
    table_lazy = _merge_reference_dataset(table_lazy, "nasa_propellant_benefits.csv", "propellant")

    table_lazy = table_lazy.with_columns(
        [
            pl.col("category")
            .map_elements(_normalize_category)
            .alias("category_norm"),
            pl.col("subitem")
            .map_elements(_normalize_item)
            .alias("subitem_norm"),
        ]
    )

    table_lazy = table_lazy.with_columns(
        pl.when(pl.col("subitem_norm").str.len_bytes() > 0)
        .then(pl.col("category_norm") + pl.lit("|") + pl.col("subitem_norm"))
        .otherwise(pl.col("category_norm"))
        .alias("key")
    )

    if isinstance(table_lazy, pd.DataFrame):  # pragma: no cover - defensive
        table_df = pl.from_pandas(table_lazy)
    elif isinstance(table_lazy, pl.DataFrame):
        table_df = table_lazy
    else:
        table_df = table_lazy.collect()

    if table_df.height == 0:
        return default

    columns = table_df.columns
    excluded = {"category", "subitem", "category_norm", "subitem_norm", "token_set", "key"}
    value_columns = tuple(col for col in columns if col not in excluded)
    composition_columns = tuple(
        col for col in value_columns if col.endswith("_pct") and not col.startswith("subitem_")
    )

    direct_map: Dict[str, Dict[str, float]] = {}
    category_tokens: Dict[str, list[tuple[frozenset[str], Dict[str, float], str]]] = {}

    for row in table_df.to_dicts():
        category_raw = row.get("category")
        subitem_raw = row.get("subitem")

        if category_raw is None:
            continue

        key = _build_match_key(category_raw, subitem_raw)
        category_norm = _normalize_category(category_raw)
        tokens = _token_set(subitem_raw)

        payload: Dict[str, float] = {}
        for column in value_columns:
            value = row.get(column)
            if value is None:
                payload[column] = float("nan")
                continue
            if isinstance(value, (int, float)):
                payload[column] = float(value)
                continue
            try:
                payload[column] = float(value)
            except (TypeError, ValueError):
                payload[column] = float("nan")

        direct_map[key] = payload
        category_tokens.setdefault(category_norm, []).append((tokens, payload, key))

    waste_summary = _load_waste_summary_data()
    processing_metrics = _extract_grouped_metrics("nasa_waste_processing_products.csv", "processing")
    leo_savings = _extract_grouped_metrics("nasa_leo_mass_savings.csv", "leo")
    propellant_metrics = _extract_grouped_metrics("nasa_propellant_benefits.csv", "propellant")

    table_join = table_df.select(
        ["category_norm", "subitem_norm", *value_columns]
    ).unique(subset=["category_norm", "subitem_norm"], maintain_order=True)

    return _OfficialFeaturesBundle(
        value_columns,
        composition_columns,
        direct_map,
        category_tokens,
        table_join,
        waste_summary.mass_by_key,
        waste_summary.mission_totals,
        processing_metrics,
        leo_savings,
        propellant_metrics,
        l2l.constants,
        l2l.category_features,
        l2l.item_features,
        l2l.hints,
    )


def _lookup_official_feature_values(row: pd.Series) -> tuple[Dict[str, float], str]:
    bundle = _official_features_bundle()
    if not bundle.value_columns:
        return {}, ""

    category = _normalize_category(row.get("category", ""))
    if not category:
        return {}, ""

    candidates = (
        row.get("material"),
        row.get("material_family"),
        row.get("key_materials"),
    )

    for candidate in candidates:
        normalized = _normalize_item(candidate)
        if not normalized:
            continue
        key = f"{category}|{normalized}"
        payload = bundle.direct_map.get(key)
        if payload:
            return payload, key

    token_candidates = [value for value in candidates if value]
    if not token_candidates:
        return {}, ""

    matches = bundle.category_tokens.get(category)
    if not matches:
        return {}, ""

    for candidate in token_candidates:
        tokens = _token_set(candidate)
        if not tokens:
            continue
        for reference_tokens, payload, match_key in matches:
            if tokens.issubset(reference_tokens):
                return payload, match_key

    return {}, ""


def _inject_official_features(frame: pd.DataFrame) -> pd.DataFrame:
    bundle = _official_features_bundle()
    if not bundle.value_columns or frame.empty:
        return frame

    working = frame.copy()
    working["__row_id__"] = np.arange(len(working))

    inventory_pl = pl.from_pandas(working)
    existing_columns = set(inventory_pl.columns)

    norm_exprs: list[pl.Expr] = []
    if "category" in existing_columns:
        norm_exprs.append(
            pl.col("category").map_elements(_normalize_category).alias("category_norm")
        )
    else:
        norm_exprs.append(pl.lit("").alias("category_norm"))

    for source, alias in (
        ("material", "material_norm"),
        ("material_family", "material_family_norm"),
        ("key_materials", "key_materials_norm"),
    ):
        if source in existing_columns:
            norm_exprs.append(
                pl.col(source).map_elements(_normalize_item).alias(alias)
            )
        else:
            norm_exprs.append(pl.lit("").alias(alias))

    inventory_lazy = inventory_pl.lazy().with_columns(norm_exprs)

    inventory_lazy = inventory_lazy.with_columns(
        pl.when(pl.col("material_norm").str.len_bytes() > 0)
        .then(pl.col("material_norm"))
        .when(pl.col("material_family_norm").str.len_bytes() > 0)
        .then(pl.col("material_family_norm"))
        .when(pl.col("key_materials_norm").str.len_bytes() > 0)
        .then(pl.col("key_materials_norm"))
        .otherwise(pl.lit(""))
        .alias("subitem_norm")
    )

    inventory_lazy = inventory_lazy.with_columns(
        pl.when(pl.col("subitem_norm").str.len_bytes() > 0)
        .then(pl.col("category_norm") + pl.lit("|") + pl.col("subitem_norm"))
        .otherwise(pl.col("category_norm"))
        .alias("_official_match_key")
    )

    official_lazy = bundle.table.lazy()
    official_columns = ["category_norm", "subitem_norm", *bundle.value_columns]
    official_lazy = official_lazy.select(official_columns)

    joined_lazy = inventory_lazy.join(
        official_lazy,
        on=["category_norm", "subitem_norm"],
        how="left",
        suffix="_official",
    )

    combine_exprs: list[pl.Expr] = []
    drop_official: list[str] = []
    left_columns = set(inventory_pl.columns)
    for column in bundle.value_columns:
        official_name = f"{column}_official"
        if official_name not in joined_lazy.columns:
            continue
        if column in left_columns:
            combine_exprs.append(
                pl.when(pl.col(official_name).is_not_null())
                .then(pl.col(official_name))
                .otherwise(pl.col(column))
                .alias(column)
            )
        else:
            combine_exprs.append(pl.col(official_name).alias(column))
        drop_official.append(official_name)

    if combine_exprs:
        joined_lazy = joined_lazy.with_columns(combine_exprs)
    if drop_official:
        joined_lazy = joined_lazy.drop(drop_official)

    match_checks = [
        pl.col(column).is_not_null() for column in bundle.value_columns if column in joined_lazy.columns
    ]
    if match_checks:
        joined_lazy = joined_lazy.with_columns(
            pl.when(pl.any_horizontal(match_checks))
            .then(pl.col("_official_match_key"))
            .otherwise(pl.lit(""))
            .alias("_official_match_key")
        )

    if bundle.l2l_category_features:
        category_rows = [
            {"category_norm": key, **{name: float(value) for name, value in values.items()}}
            for key, values in bundle.l2l_category_features.items()
        ]
        if category_rows:
            category_df = pl.DataFrame(category_rows)
            joined_lazy = joined_lazy.join(category_df.lazy(), on="category_norm", how="left")

    if bundle.l2l_item_features:
        item_rows = []
        for key, values in bundle.l2l_item_features.items():
            row = {"_official_match_key": key}
            row.update({name: float(value) for name, value in values.items()})
            item_rows.append(row)
        if item_rows:
            item_df = pl.DataFrame(item_rows)
            joined_lazy = joined_lazy.join(
                item_df.lazy(), on="_official_match_key", how="left"
            )

    if bundle.l2l_constants:
        const_exprs = []
        for name, value in bundle.l2l_constants.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                const_exprs.append(pl.lit(float(value)).alias(name))
        if const_exprs:
            joined_lazy = joined_lazy.with_columns(const_exprs)

    result_pl = joined_lazy.sort("__row_id__").collect()
    result_df = result_pl.to_pandas()

    helper_columns = {
        "__row_id__",
        "category_norm",
        "subitem_norm",
        "material_norm",
        "material_family_norm",
        "key_materials_norm",
    }
    result_df = result_df.drop(columns=[col for col in helper_columns if col in result_df.columns])
    result_df.index = frame.index

    hint_columns = [
        column for column in bundle.l2l_hints if column in result_df.columns
    ]
    if hint_columns:
        hint_map: Dict[str, list[str]] = {}
        for column in hint_columns:
            hint = bundle.l2l_hints[column]
            if hint:
                hint_map.setdefault(hint, []).append(column)

        if hint_map:
            hints_df = pd.DataFrame(index=result_df.index)
            for hint, columns in hint_map.items():
                mask = result_df[columns].notna().any(axis=1)
                hints_df[hint] = np.where(mask, hint, "")

            if not hints_df.empty:
                result_df["_l2l_page_hints"] = (
                    hints_df.apply(
                        lambda row: "; ".join(sorted(filter(None, row.tolist()))), axis=1
                    )
                )

    if bundle.value_columns:
        numeric_candidates = [
            column
            for column in bundle.value_columns
            if column.endswith(("_kg", "_pct"))
            or column.startswith("category_total")
            or column in {"difficulty_factor", "approx_moisture_pct"}
        ]

        for column in numeric_candidates:
            if column in result_df.columns:
                result_df[column] = pd.to_numeric(result_df[column], errors="coerce")

    if "approx_moisture_pct" in result_df.columns:
        mask = result_df["approx_moisture_pct"].notna()
        if mask.any():
            result_df.loc[mask, "moisture_pct"] = result_df.loc[mask, "approx_moisture_pct"]

    return result_df
@dataclass(slots=True)
class PredProps:
    """Structured container for predicted (or heuristic) properties."""

    rigidity: float
    tightness: float
    mass_final_kg: float
    energy_kwh: float
    water_l: float
    crew_min: float
    source: str = "heuristic"
    uncertainty: Dict[str, float] | None = None
    confidence_interval: Dict[str, Tuple[float, float]] | None = None
    feature_importance: list[tuple[str, float]] | None = None
    comparisons: dict[str, dict[str, float]] | None = None

    def to_targets(self) -> Dict[str, float]:
        return {
            "rigidez": float(self.rigidity),
            "estanqueidad": float(self.tightness),
            "energy_kwh": float(self.energy_kwh),
            "water_l": float(self.water_l),
            "crew_min": float(self.crew_min),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rigidez": float(self.rigidity),
            "estanqueidad": float(self.tightness),
            "mass_final_kg": float(self.mass_final_kg),
            "energy_kwh": float(self.energy_kwh),
            "water_l": float(self.water_l),
            "crew_min": float(self.crew_min),
            "source": str(self.source),
            "uncertainty": {
                str(k): float(v) for k, v in (self.uncertainty or {}).items()
            },
            "confidence_interval": {
                str(k): [float(x) for x in v] for k, v in (self.confidence_interval or {}).items()
            },
            "feature_importance": [
                (str(name), float(weight)) for name, weight in (self.feature_importance or [])
            ],
            "comparisons": {
                str(name): {str(k): float(vv) for k, vv in val.items()}
                for name, val in (self.comparisons or {}).items()
            },
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PredProps":
        def _get(name: str, alt: str | None = None, default: float = 0.0) -> float:
            if alt is not None and alt in payload:
                value = payload.get(alt)
            else:
                value = payload.get(name)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        uncertainty_raw = payload.get("uncertainty") or {}
        confidence_raw = payload.get("confidence_interval") or {}
        feature_imp_raw = payload.get("feature_importance") or []
        comparisons_raw = payload.get("comparisons") or {}

        if isinstance(uncertainty_raw, Mapping):
            uncertainty_map = {str(k): float(v) for k, v in uncertainty_raw.items()}
        else:
            uncertainty_map = {}

        confidence_map: Dict[str, Tuple[float, float]] = {}
        if isinstance(confidence_raw, Mapping):
            for key, bounds in confidence_raw.items():
                try:
                    lo, hi = bounds
                    confidence_map[str(key)] = (float(lo), float(hi))
                except Exception:
                    confidence_map[str(key)] = (0.0, 0.0)

        feature_importance_list: list[tuple[str, float]] = []
        if isinstance(feature_imp_raw, Iterable):
            for item in feature_imp_raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    feature_importance_list.append((str(item[0]), float(item[1])))

        comparisons_map: Dict[str, Dict[str, float]] = {}
        if isinstance(comparisons_raw, Mapping):
            for name, payload_map in comparisons_raw.items():
                if isinstance(payload_map, Mapping):
                    comparisons_map[str(name)] = {
                        str(k): float(v) for k, v in payload_map.items()
                    }

        return cls(
            rigidity=_get("rigidez", "rigidity", 0.0),
            tightness=_get("estanqueidad", "tightness", 0.0),
            mass_final_kg=_get("mass_final_kg", default=0.0),
            energy_kwh=_get("energy_kwh", default=0.0),
            water_l=_get("water_l", default=0.0),
            crew_min=_get("crew_min", default=0.0),
            source=str(payload.get("source", "heuristic")),
            uncertainty=uncertainty_map,
            confidence_interval=confidence_map,
            feature_importance=feature_importance_list,
            comparisons=comparisons_map,
        )


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def _first_available(df: pd.DataFrame, names: Iterable[str], default: str | None = None) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return default


def prepare_waste_frame(waste_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *waste_df* with the canonical columns used downstream."""

    if waste_df is None or waste_df.empty:
        return pd.DataFrame()

    out = waste_df.copy()

    if "id" not in out.columns:
        out["id"] = out.index.astype(str)

    cat_col = _first_available(out, ["category", "Category"])
    if cat_col != "category":
        out["category"] = out[cat_col] if cat_col else ""

    mat_col = _first_available(out, ["material", "material_family", "Material", "item", "Item"])
    if mat_col != "material":
        out["material"] = out[mat_col] if mat_col else ""

    kg_col = _first_available(out, ["kg", "mass_kg", "Mass_kg"])
    if kg_col != "kg":
        out["kg"] = pd.to_numeric(out[kg_col], errors="coerce").fillna(0.0) if kg_col else 0.0

    volume_col = _first_available(out, ["volume_l", "Volume_L", "volume_m3"])
    if volume_col == "volume_m3":
        liters = pd.to_numeric(out[volume_col], errors="coerce").fillna(0.0) * 1000.0
        out["volume_l"] = liters
    elif volume_col != "volume_l":
        out["volume_l"] = pd.to_numeric(out[volume_col], errors="coerce").fillna(0.0) if volume_col else 0.0

    moist_col = _first_available(out, ["moisture_pct", "moisture", "moisture_percent"], default=None)
    if moist_col and moist_col != "moisture_pct":
        out["moisture_pct"] = pd.to_numeric(out[moist_col], errors="coerce").fillna(0.0)
    elif "moisture_pct" not in out.columns:
        out["moisture_pct"] = 0.0

    diff_col = _first_available(out, ["difficulty_factor", "difficulty", "diff_factor"], default=None)
    if diff_col and diff_col != "difficulty_factor":
        out["difficulty_factor"] = pd.to_numeric(out[diff_col], errors="coerce").fillna(1.0)
    elif "difficulty_factor" not in out.columns:
        out["difficulty_factor"] = 1.0

    mass_pct_col = _first_available(out, ["pct_mass", "percent_mass"], default=None)
    if mass_pct_col and mass_pct_col != "pct_mass":
        out["pct_mass"] = pd.to_numeric(out[mass_pct_col], errors="coerce").fillna(0.0)
    elif "pct_mass" not in out.columns:
        out["pct_mass"] = 0.0

    vol_pct_col = _first_available(out, ["pct_volume", "percent_volume"], default=None)
    if vol_pct_col and vol_pct_col != "pct_volume":
        out["pct_volume"] = pd.to_numeric(out[vol_pct_col], errors="coerce").fillna(0.0)
    elif "pct_volume" not in out.columns:
        out["pct_volume"] = 0.0

    flags_col = _first_available(out, ["flags", "Flags"])
    if flags_col != "flags":
        out["flags"] = out[flags_col] if flags_col else ""

    if "key_materials" not in out.columns:
        out["key_materials"] = out["material"].astype(str)

    out["tokens"] = (
        out["material"].astype(str).str.lower()
        + " "
        + out["category"].astype(str).str.lower()
        + " "
        + out["flags"].astype(str).str.lower()
        + " "
        + out["key_materials"].astype(str).str.lower()
    )

    if "_problematic" not in out.columns:
        out["_problematic"] = out.apply(_is_problematic, axis=1)

    out["_source_id"] = out["id"].astype(str)
    out["_source_category"] = out["category"].astype(str)
    out["_source_flags"] = out["flags"].astype(str)

    out = _inject_official_features(out)

    mass = pd.to_numeric(out["kg"], errors="coerce").fillna(0.0)
    volume_l = pd.to_numeric(out.get("volume_l"), errors="coerce")
    volume_m3 = volume_l / 1000.0
    density = pd.Series(np.nan, index=out.index, dtype=float)
    with_volume = volume_m3.notna() & (volume_m3 > 0)
    density.loc[with_volume] = mass.loc[with_volume] / volume_m3.loc[with_volume]

    missing_density = density.isna() | ~np.isfinite(density)
    if missing_density.any():
        for idx, row in out.loc[missing_density].iterrows():
            estimate = _estimate_density_from_row(row)
            if estimate is not None:
                density.at[idx] = estimate

    default_density = float(_CATEGORY_DENSITY_DEFAULTS.get("other packaging glove", 500.0))
    density = density.fillna(default_density)
    out["density_kg_m3"] = density.clip(lower=20.0, upper=4000.0)

    return out


def _is_problematic(row: pd.Series) -> bool:
    cat = str(row.get("category", "")).lower()
    fam = str(row.get("material", "")).lower() + " " + str(row.get("material_family", "")).lower()
    flg = str(row.get("flags", "")).lower()
    rules = [
        "pouches" in cat or "multilayer" in flg or "pe-pet-al" in fam,
        "foam" in cat or "zotek" in fam or "closed_cell" in flg,
        "eva" in cat or "ctb" in flg or "nomex" in fam or "nylon" in fam or "polyester" in fam,
        "glove" in cat or "nitrile" in fam,
        "wipe" in flg or "textile" in cat,
    ]
    return any(rules)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


_KEYWORD_FEATURES: Dict[str, Tuple[str, ...]] = {
    "aluminum_frac": ("aluminum", " alloy", " al "),
    "foam_frac": ("foam", "zotek", "closed cell"),
    "eva_frac": ("eva", "ctb", "nomex"),
    "textile_frac": ("textile", "cloth", "fabric", "wipe"),
    "multilayer_frac": ("multilayer", "pe-pet-al", "pouch"),
    "glove_frac": ("glove", "nitrile"),
    "polyethylene_frac": ("polyethylene", "pvdf", "ldpe"),
    "carbon_fiber_frac": ("carbon fiber", "composite"),
    "hydrogen_rich_frac": ("polyethylene", "cotton", "pvdf"),
}

_KEYWORD_INDEX = {name: idx for idx, name in enumerate(_KEYWORD_FEATURES)}
_PACKAGING_TARGETS = ("packaging", "food packaging")


def _material_tokens(row: pd.Series) -> str:
    parts = [
        str(row.get("material", "")),
        str(row.get("category", "")),
        str(row.get("flags", "")),
        str(row.get("material_family", "")),
        str(row.get("key_materials", "")),
    ]
    return " ".join(parts).lower()


def _keyword_fraction(tokens: Iterable[str], weights: Iterable[float], keywords: Tuple[str, ...]) -> float:
    score = 0.0
    for token, weight in zip(tokens, weights):
        if any(keyword in token for keyword in keywords):
            score += weight
    return float(np.clip(score, 0.0, 1.0))


def _category_fraction(categories: Iterable[str], weights: Iterable[float], targets: Tuple[str, ...]) -> float:
    score = 0.0
    for category, weight in zip(categories, weights):
        if any(target in category for target in targets):
            score += weight
    return float(np.clip(score, 0.0, 1.0))


@dataclass
class CandidateFeatureContext:
    total_mass: float
    weights: np.ndarray
    densities: np.ndarray
    moisture: np.ndarray
    difficulty: np.ndarray
    pct_mass: np.ndarray
    pct_volume: np.ndarray
    tokens: Tuple[str, ...]
    categories: Tuple[str, ...]
    regolith_pct: float
    mission_share: Dict[str, float]
    mission_scaled_mass: Dict[str, float]
    mission_official_mass: Dict[str, float]
    official_composition: Dict[str, float]
    num_items: int


@dataclass
class FeatureTensorBatch:
    weights: Any
    densities: Any
    moisture: Any
    difficulty: Any
    pct_mass: Any
    pct_volume: Any
    keyword_hits: Any
    packaging_hits: Any
    mission_share: Any
    mission_scaled_mass: Any
    mission_official_mass: Any
    mission_totals: Any
    total_mass: Any
    regolith: Any
    keyword_names: Tuple[str, ...]
    mission_names: Tuple[str, ...]
    process_ids: Tuple[str, ...]
    num_items: Tuple[int, ...]
    official_comp: Tuple[Dict[str, float], ...]
    l2l_constants: Dict[str, float]
    bundle_processing_metrics: Dict[str, Dict[str, float]]
    bundle_leo_mass_savings: Dict[str, Dict[str, float]]
    bundle_propellant_benefits: Dict[str, Dict[str, float]]
    logistics_ratio: float


def _prepare_feature_context(
    picks: pd.DataFrame,
    weights: Iterable[float],
    regolith_pct: float,
    bundle: "_OfficialFeaturesBundle",
) -> CandidateFeatureContext:
    total_kg = max(0.001, float(picks["kg"].sum()))
    raw_weights = np.asarray(list(weights), dtype=float)
    if len(raw_weights) < len(picks):
        raw_weights = np.pad(raw_weights, (0, len(picks) - len(raw_weights)), constant_values=0.0)
    elif len(raw_weights) > len(picks):
        raw_weights = raw_weights[: len(picks)]

    tokens = tuple(_material_tokens(row) for _, row in picks.iterrows())
    categories = tuple(str(row.get("category", "")).lower() for _, row in picks.iterrows())

    pct_mass = picks.get("pct_mass", 0).to_numpy(dtype=float) / 100.0
    pct_volume = picks.get("pct_volume", 0).to_numpy(dtype=float) / 100.0
    moisture = picks.get("moisture_pct", 0).to_numpy(dtype=float) / 100.0
    difficulty = picks.get("difficulty_factor", 1).to_numpy(dtype=float) / 3.0
    densities = picks.get("density_kg_m3", 0).to_numpy(dtype=float)

    official_comp: Dict[str, float] = {}
    if bundle.composition_columns:
        for column in bundle.composition_columns:
            if column not in picks.columns:
                continue
            series = picks[column]
            if not series.notna().any():
                continue
            values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if not len(values):
                continue
            frac = float(np.dot(raw_weights[: len(values)], values / 100.0))
            official_comp[column] = frac

    mission_share: Dict[str, float] = {}
    mission_scaled_mass: Dict[str, float] = {}
    mission_official_mass: Dict[str, float] = {}

    if bundle.mission_mass and bundle.mission_totals:
        match_keys_col = picks.get("_official_match_key")
        if match_keys_col is not None:
            match_keys_list = match_keys_col.fillna("").astype(str).tolist()
        else:
            match_keys_list = [""] * len(picks)

        for idx, (_, row) in enumerate(picks.iterrows()):
            weight = float(raw_weights[idx]) if idx < len(raw_weights) else 0.0
            if weight <= 0:
                continue

            keys_to_check: list[str] = []
            match_key = match_keys_list[idx] if idx < len(match_keys_list) else ""
            if match_key:
                keys_to_check.append(match_key)

            if not match_key:
                category_key = _normalize_category(row.get("category", ""))
                if category_key:
                    keys_to_check.append(category_key)

            seen_keys: set[str] = set()
            for key in keys_to_check:
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                missions = bundle.mission_mass.get(key)
                if not missions:
                    continue

                for mission, reference_mass in missions.items():
                    total_reference = bundle.mission_totals.get(mission)
                    if not total_reference or total_reference <= 0:
                        continue
                    share = (reference_mass / total_reference) * weight
                    mission_share[mission] = mission_share.get(mission, 0.0) + share
                    mission_official_mass[mission] = mission_official_mass.get(mission, 0.0) + weight * float(reference_mass)
                    mission_scaled_mass[mission] = mission_scaled_mass.get(mission, 0.0) + share * total_kg

    return CandidateFeatureContext(
        total_mass=total_kg,
        weights=raw_weights,
        densities=densities,
        moisture=moisture,
        difficulty=difficulty,
        pct_mass=pct_mass,
        pct_volume=pct_volume,
        tokens=tokens,
        categories=categories,
        regolith_pct=float(regolith_pct),
        mission_share=mission_share,
        mission_scaled_mass=mission_scaled_mass,
        mission_official_mass=mission_official_mass,
        official_composition=official_comp,
        num_items=int(len(picks)),
    )


def _contexts_to_tensor_batch(
    contexts: Sequence[CandidateFeatureContext],
    processes: Sequence[pd.Series],
    bundle: "_OfficialFeaturesBundle",
    *,
    backend: str = "jax",
) -> FeatureTensorBatch:
    if len(contexts) != len(processes):
        raise ValueError("Number of contexts must match number of processes")

    batch = len(contexts)
    max_items = max((ctx.num_items for ctx in contexts), default=0)
    keyword_names = tuple(_KEYWORD_FEATURES.keys())
    mission_names = tuple(sorted(bundle.mission_totals.keys())) if bundle.mission_totals else tuple()

    def _alloc(shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=float)

    weights = _alloc((batch, max_items))
    densities = _alloc((batch, max_items))
    moisture = _alloc((batch, max_items))
    difficulty = _alloc((batch, max_items))
    pct_mass = _alloc((batch, max_items))
    pct_volume = _alloc((batch, max_items))
    keyword_hits = _alloc((batch, max_items, len(keyword_names)))
    packaging_hits = _alloc((batch, max_items))
    mission_share = _alloc((batch, len(mission_names)))
    mission_scaled_mass = _alloc((batch, len(mission_names)))
    mission_official_mass = _alloc((batch, len(mission_names)))

    total_mass = np.zeros(batch, dtype=float)
    regolith = np.zeros(batch, dtype=float)
    num_items: list[int] = []
    official_comp: list[Dict[str, float]] = []

    mission_totals = np.array([float(bundle.mission_totals.get(name, 0.0)) for name in mission_names], dtype=float)

    for idx, ctx in enumerate(contexts):
        count = ctx.num_items
        num_items.append(count)
        official_comp.append(dict(ctx.official_composition))

        if count:
            weights[idx, :count] = ctx.weights[:count]
            densities[idx, :count] = ctx.densities[:count]
            moisture[idx, :count] = ctx.moisture[:count]
            difficulty[idx, :count] = ctx.difficulty[:count]
            pct_mass[idx, :count] = ctx.pct_mass[:count]
            pct_volume[idx, :count] = ctx.pct_volume[:count]

            for item_idx in range(count):
                token = ctx.tokens[item_idx]
                category = ctx.categories[item_idx]
                for kw_idx, keywords in enumerate(keyword_names):
                    patterns = _KEYWORD_FEATURES[keywords]
                    if any(pattern in token for pattern in patterns):
                        keyword_hits[idx, item_idx, kw_idx] = 1.0
                if any(target in category for target in _PACKAGING_TARGETS):
                    packaging_hits[idx, item_idx] = 1.0

        for mission_idx, mission in enumerate(mission_names):
            mission_share[idx, mission_idx] = ctx.mission_share.get(mission, 0.0)
            mission_scaled_mass[idx, mission_idx] = ctx.mission_scaled_mass.get(mission, 0.0)
            mission_official_mass[idx, mission_idx] = ctx.mission_official_mass.get(mission, 0.0)

        total_mass[idx] = ctx.total_mass
        regolith[idx] = ctx.regolith_pct

    process_ids = tuple(str(proc.get("process_id")) for proc in processes)
    l2l_constants = dict(bundle.l2l_constants or {})
    logistics_ratio = float(l2l_constants.get("l2l_logistics_packaging_per_goods_ratio", float("nan")))

    def _to_backend_array(array: np.ndarray) -> Any:
        if backend == "jax" and jnp is not None:
            return jnp.asarray(array)
        return array

    return FeatureTensorBatch(
        weights=_to_backend_array(weights),
        densities=_to_backend_array(densities),
        moisture=_to_backend_array(moisture),
        difficulty=_to_backend_array(difficulty),
        pct_mass=_to_backend_array(pct_mass),
        pct_volume=_to_backend_array(pct_volume),
        keyword_hits=_to_backend_array(keyword_hits),
        packaging_hits=_to_backend_array(packaging_hits),
        mission_share=_to_backend_array(mission_share),
        mission_scaled_mass=_to_backend_array(mission_scaled_mass),
        mission_official_mass=_to_backend_array(mission_official_mass),
        mission_totals=_to_backend_array(mission_totals),
        total_mass=_to_backend_array(total_mass),
        regolith=_to_backend_array(regolith),
        keyword_names=keyword_names,
        mission_names=mission_names,
        process_ids=process_ids,
        num_items=tuple(num_items),
        official_comp=tuple(official_comp),
        l2l_constants=l2l_constants,
        bundle_processing_metrics=dict(bundle.processing_metrics or {}),
        bundle_leo_mass_savings=dict(bundle.leo_mass_savings or {}),
        bundle_propellant_benefits=dict(bundle.propellant_benefits or {}),
        logistics_ratio=logistics_ratio,
    )


def build_feature_tensor_batch(
    picks: Sequence[pd.DataFrame],
    weights: Sequence[Iterable[float]],
    processes: Sequence[pd.Series],
    regolith_pct: Sequence[float],
    *,
    backend: str = "jax",
) -> FeatureTensorBatch:
    if not picks:
        raise ValueError("picks must contain at least one candidate")

    if not (len(picks) == len(weights) == len(processes) == len(regolith_pct)):
        raise ValueError("picks, weights, processes and regolith_pct must share the same length")

    bundle = _official_features_bundle()
    contexts = [
        _prepare_feature_context(p, w, r, bundle)
        for p, w, r in zip(picks, weights, regolith_pct)
    ]
    return _contexts_to_tensor_batch(contexts, processes, bundle, backend=backend)


def _feature_kernel_body(
    xp: Any,
    weights: Any,
    densities: Any,
    moisture: Any,
    difficulty: Any,
    pct_mass: Any,
    pct_volume: Any,
    keyword_hits: Any,
    packaging_hits: Any,
    mission_share: Any,
    mission_scaled_mass: Any,
    mission_official_mass: Any,
    mission_totals: Any,
    regolith: Any,
    logistics_ratio: float,
) -> Dict[str, Any]:
    w = xp.asarray(weights)
    densities = xp.asarray(densities)
    moisture = xp.asarray(moisture)
    difficulty = xp.asarray(difficulty)
    pct_mass = xp.asarray(pct_mass)
    pct_volume = xp.asarray(pct_volume)
    keyword_hits = xp.asarray(keyword_hits)
    packaging_hits = xp.asarray(packaging_hits)
    mission_share = xp.asarray(mission_share)
    mission_scaled_mass = xp.asarray(mission_scaled_mass)
    mission_official_mass = xp.asarray(mission_official_mass)
    mission_totals = xp.asarray(mission_totals)
    regolith = xp.asarray(regolith)

    density = xp.sum(w * densities, axis=1)
    moisture_frac = xp.clip(xp.sum(w * moisture, axis=1), 0.0, 1.0)
    difficulty_index = xp.clip(xp.sum(w * difficulty, axis=1), 0.0, 1.0)
    problematic_mass_frac = xp.clip(xp.sum(w * pct_mass, axis=1), 0.0, 1.0)
    problematic_item_frac = xp.clip(xp.sum(w * pct_volume, axis=1), 0.0, 1.0)

    keyword = xp.clip(xp.sum(w[:, :, None] * keyword_hits, axis=1), 0.0, 1.0)
    packaging = xp.clip(xp.sum(w * packaging_hits, axis=1), 0.0, 1.0)

    mission_share_clipped = xp.clip(mission_share, 0.0, 1.0)
    mission_similarity_total = xp.clip(xp.sum(mission_share_clipped, axis=1), 0.0, 1.0)
    mission_reference_mass = xp.maximum(0.0, mission_share_clipped * mission_totals)
    mission_scaled_mass = xp.maximum(0.0, mission_scaled_mass)
    mission_official_mass = xp.maximum(0.0, mission_official_mass)

    polyethylene = keyword[:, _KEYWORD_INDEX["polyethylene_frac"]] if keyword.shape[1] else xp.zeros(keyword.shape[0])
    foam = keyword[:, _KEYWORD_INDEX["foam_frac"]] if keyword.shape[1] else xp.zeros(keyword.shape[0])
    eva = keyword[:, _KEYWORD_INDEX["eva_frac"]] if keyword.shape[1] else xp.zeros(keyword.shape[0])
    textile = keyword[:, _KEYWORD_INDEX["textile_frac"]] if keyword.shape[1] else xp.zeros(keyword.shape[0])

    gas_index = _GAS_MEAN_YIELD * (
        0.7 * polyethylene
        + 0.4 * foam
        + 0.5 * eva
        + 0.2 * textile
    )
    gas_recovery_index = xp.clip(gas_index / 10.0, 0.0, 1.0)

    ratio = xp.asarray(logistics_ratio)
    ratio = xp.where(xp.isfinite(ratio), ratio, xp.asarray(float("nan")))
    packaging_term = packaging + 0.5 * eva
    ratio_broadcast = xp.broadcast_to(ratio, packaging_term.shape)
    valid_ratio = xp.logical_and(xp.isfinite(ratio_broadcast), ratio_broadcast > 0)
    reuse_term = packaging_term * _MEAN_REUSE
    logistics_index = xp.where(valid_ratio, packaging_term / ratio_broadcast, reuse_term)
    logistics_index = xp.clip(logistics_index, 0.0, 2.0)

    regolith = xp.clip(regolith, 0.0, 1.0)

    return {
        "density": density,
        "moisture": moisture_frac,
        "difficulty": difficulty_index,
        "problematic_mass": problematic_mass_frac,
        "problematic_item": problematic_item_frac,
        "keyword": keyword,
        "packaging": packaging,
        "mission_similarity": mission_share_clipped,
        "mission_reference_mass": mission_reference_mass,
        "mission_scaled_mass": mission_scaled_mass,
        "mission_official_mass": mission_official_mass,
        "mission_similarity_total": mission_similarity_total,
        "gas_recovery_index": gas_recovery_index,
        "logistics_reuse_index": logistics_index,
        "regolith": regolith,
    }


if jnp is not None:

    @jit
    def _feature_kernel(
        weights: Any,
        densities: Any,
        moisture: Any,
        difficulty: Any,
        pct_mass: Any,
        pct_volume: Any,
        keyword_hits: Any,
        packaging_hits: Any,
        mission_share: Any,
        mission_scaled_mass: Any,
        mission_official_mass: Any,
        mission_totals: Any,
        regolith: Any,
        logistics_ratio: float,
    ) -> Dict[str, Any]:
        return _feature_kernel_body(
            jnp,
            weights,
            densities,
            moisture,
            difficulty,
            pct_mass,
            pct_volume,
            keyword_hits,
            packaging_hits,
            mission_share,
            mission_scaled_mass,
            mission_official_mass,
            mission_totals,
            regolith,
            logistics_ratio,
        )

else:

    def _feature_kernel(
        weights: Any,
        densities: Any,
        moisture: Any,
        difficulty: Any,
        pct_mass: Any,
        pct_volume: Any,
        keyword_hits: Any,
        packaging_hits: Any,
        mission_share: Any,
        mission_scaled_mass: Any,
        mission_official_mass: Any,
        mission_totals: Any,
        regolith: Any,
        logistics_ratio: float,
    ) -> Dict[str, Any]:
        return _feature_kernel_body(
            np,
            weights,
            densities,
            moisture,
            difficulty,
            pct_mass,
            pct_volume,
            keyword_hits,
            packaging_hits,
            mission_share,
            mission_scaled_mass,
            mission_official_mass,
            mission_totals,
            regolith,
            logistics_ratio,
        )


def _apply_official_composition_overrides(
    features: Dict[str, Any], official_comp: Mapping[str, float]
) -> None:
    if not official_comp:
        return

    clipped = {key: max(0.0, float(value)) for key, value in official_comp.items()}
    total = sum(clipped.values())
    if total > 1.0 + 1e-6 and total > 0:
        normalized = {key: value / total for key, value in clipped.items()}
    else:
        normalized = {key: float(np.clip(value, 0.0, 1.0)) for key, value in clipped.items()}

    def _set_official_fraction(name: str, *columns: str) -> None:
        total_value = 0.0
        found = False
        for column in columns:
            if column in normalized:
                total_value += normalized[column]
                found = True
        if not found:
            return
        features[name] = float(np.clip(total_value, 0.0, 1.0))

    _set_official_fraction("aluminum_frac", "Aluminum_pct")
    _set_official_fraction("carbon_fiber_frac", "Carbon_Fiber_pct")
    _set_official_fraction("polyethylene_frac", "Polyethylene_pct")
    _set_official_fraction("glove_frac", "Nitrile_pct")
    _set_official_fraction("eva_frac", "Nomex_pct")
    _set_official_fraction("foam_frac", "PVDF_pct")
    _set_official_fraction("multilayer_frac", "EVOH_pct", "PET_pct")

    textile_total = sum(
        normalized.get(column, 0.0)
        for column in ("Cotton_Cellulose_pct", "Polyester_pct", "Nylon_pct")
    )
    if textile_total > 0:
        features["textile_frac"] = float(np.clip(textile_total, 0.0, 1.0))

    hydrogen_total = sum(
        normalized.get(column, 0.0)
        for column in ("Polyethylene_pct", "Cotton_Cellulose_pct", "PVDF_pct")
    )
    if hydrogen_total > 0:
        features["hydrogen_rich_frac"] = float(np.clip(hydrogen_total, 0.0, 1.0))


def _apply_weighted_metrics(
    features: Dict[str, Any],
    mission_similarity_clipped: Mapping[str, float],
    source: Mapping[str, Mapping[str, float]],
) -> None:
    if not source or not mission_similarity_clipped:
        return
    weighted: Dict[str, float] = {}
    for mission, share in mission_similarity_clipped.items():
        metrics = source.get(mission)
        if not metrics:
            continue
        for metric_name, value in metrics.items():
            expected_name = (
                metric_name if metric_name.endswith("_expected") else f"{metric_name}_expected"
            )
            weighted[expected_name] = weighted.get(expected_name, 0.0) + share * float(value)
            features[f"{metric_name}_{mission}"] = float(value)
    for metric_name, value in weighted.items():
        features[metric_name] = float(value)


def _compute_features_from_batch(batch: FeatureTensorBatch) -> list[Dict[str, Any]]:
    if not batch.process_ids:
        return []

    kernel_output = _feature_kernel(
        batch.weights,
        batch.densities,
        batch.moisture,
        batch.difficulty,
        batch.pct_mass,
        batch.pct_volume,
        batch.keyword_hits,
        batch.packaging_hits,
        batch.mission_share,
        batch.mission_scaled_mass,
        batch.mission_official_mass,
        batch.mission_totals,
        batch.regolith,
        batch.logistics_ratio,
    )

    to_numpy = np.asarray
    density = to_numpy(kernel_output["density"])
    moisture = to_numpy(kernel_output["moisture"])
    difficulty = to_numpy(kernel_output["difficulty"])
    problematic_mass = to_numpy(kernel_output["problematic_mass"])
    problematic_item = to_numpy(kernel_output["problematic_item"])
    keyword_matrix = to_numpy(kernel_output["keyword"])
    packaging = to_numpy(kernel_output["packaging"])
    mission_similarity = to_numpy(kernel_output["mission_similarity"])
    mission_reference_mass = to_numpy(kernel_output["mission_reference_mass"])
    mission_scaled_mass = to_numpy(kernel_output["mission_scaled_mass"])
    mission_official_mass = to_numpy(kernel_output["mission_official_mass"])
    mission_similarity_total = to_numpy(kernel_output["mission_similarity_total"])
    gas_recovery = to_numpy(kernel_output["gas_recovery_index"])
    logistics_reuse = to_numpy(kernel_output["logistics_reuse_index"])
    regolith = to_numpy(kernel_output["regolith"])
    total_mass = to_numpy(batch.total_mass)

    features_list: list[Dict[str, Any]] = []
    for idx, process_id in enumerate(batch.process_ids):
        features: Dict[str, Any] = {
            "process_id": str(process_id),
            "total_mass_kg": float(total_mass[idx]),
            "mass_input_kg": float(total_mass[idx]),
            "num_items": int(batch.num_items[idx]),
            "density_kg_m3": float(density[idx]) if density.size else 0.0,
            "moisture_frac": float(moisture[idx]) if moisture.size else 0.0,
            "difficulty_index": float(difficulty[idx]) if difficulty.size else 0.0,
            "problematic_mass_frac": float(problematic_mass[idx]) if problematic_mass.size else 0.0,
            "problematic_item_frac": float(problematic_item[idx]) if problematic_item.size else 0.0,
            "regolith_pct": float(regolith[idx]) if regolith.size else 0.0,
            "packaging_frac": float(packaging[idx]) if packaging.size else 0.0,
        }

        for name, kw_idx in _KEYWORD_INDEX.items():
            if keyword_matrix.shape[1] > kw_idx:
                features[name] = float(keyword_matrix[idx, kw_idx])

        _apply_official_composition_overrides(features, batch.official_comp[idx])

        for name, value in batch.l2l_constants.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                features[name] = float(value)

        mission_similarity_row = mission_similarity[idx] if mission_similarity.size else np.array([])
        mission_similarity_dict: Dict[str, float] = {}
        if mission_similarity_row.size and np.any(mission_similarity_row > 0):
            for mission_idx, mission in enumerate(batch.mission_names):
                share = float(mission_similarity_row[mission_idx])
                if share <= 0:
                    continue
                mission_similarity_dict[mission] = share
                features[f"mission_similarity_{mission}"] = float(np.clip(share, 0.0, 1.0))
                features[f"mission_reference_mass_{mission}"] = float(mission_reference_mass[idx, mission_idx])
                features[f"mission_scaled_mass_{mission}"] = float(mission_scaled_mass[idx, mission_idx])
                features[f"mission_official_mass_{mission}"] = float(mission_official_mass[idx, mission_idx])

            features["mission_similarity_total"] = float(mission_similarity_total[idx])

            _apply_weighted_metrics(features, mission_similarity_dict, batch.bundle_processing_metrics)
            _apply_weighted_metrics(features, mission_similarity_dict, batch.bundle_leo_mass_savings)
            _apply_weighted_metrics(features, mission_similarity_dict, batch.bundle_propellant_benefits)

        features["gas_recovery_index"] = float(gas_recovery[idx]) if gas_recovery.size else 0.0
        features["logistics_reuse_index"] = float(logistics_reuse[idx]) if logistics_reuse.size else 0.0

        for oxide, value in _REGOLITH_VECTOR.items():
            features[f"oxide_{oxide}"] = float(value * features["regolith_pct"])

        features_list.append(features)

    return features_list


def compute_feature_vectors_batch(
    picks: Sequence[pd.DataFrame],
    weights: Sequence[Iterable[float]],
    processes: Sequence[pd.Series],
    regolith_pct: Sequence[float],
    *,
    backend: str = "jax",
) -> list[Dict[str, Any]]:
    batch = build_feature_tensor_batch(picks, weights, processes, regolith_pct, backend=backend)
    return _compute_features_from_batch(batch)


def compute_feature_vector(
    picks: pd.DataFrame | FeatureTensorBatch,
    weights: Iterable[float] | None = None,
    process: pd.Series | None = None,
    regolith_pct: float | None = None,
) -> Dict[str, Any] | list[Dict[str, Any]]:
    if isinstance(picks, FeatureTensorBatch):
        return _compute_features_from_batch(picks)

    if not isinstance(picks, pd.DataFrame):
        raise TypeError("picks must be a pandas.DataFrame or a FeatureTensorBatch")
    if weights is None or process is None or regolith_pct is None:
        raise ValueError("weights, process and regolith_pct are required for DataFrame inputs")

    bundle = _official_features_bundle()
    context = _prepare_feature_context(picks, weights, regolith_pct, bundle)
    batch = _contexts_to_tensor_batch([context], [process], bundle)
    features = _compute_features_from_batch(batch)
    return features[0] if features else {}


def heuristic_props(
    picks: pd.DataFrame,
    process: pd.Series,
    weights: Iterable[float],
    regolith_pct: float,
) -> PredProps:
    weights_arr = np.asarray(list(weights), dtype=float)
    total_mass = max(0.001, float(picks["kg"].sum()))
    base_weights = weights_arr if weights_arr.sum() else np.ones_like(weights_arr) / len(weights_arr)

    materials = " ".join(picks["material"].astype(str)).lower()
    categories = " ".join(picks["category"].astype(str).str.lower())
    flags = " ".join(picks["flags"].astype(str).str.lower())

    rigidity = 0.5
    if any(keyword in materials for keyword in ("al", "aluminum", "alloy")):
        rigidity += 0.2
    if regolith_pct > 0:
        rigidity += 0.1
    rigidity = float(np.clip(rigidity, 0.05, 1.0))

    tightness = 0.5
    if "pouch" in materials or "pouches" in categories or "pe-pet-al" in flags:
        tightness += 0.2
    if regolith_pct > 0:
        tightness -= 0.05
    tightness = float(np.clip(tightness, 0.05, 1.0))

    process_energy = float(process["energy_kwh_per_kg"])
    process_water = float(process["water_l_per_kg"])
    process_crew = float(process["crew_min_per_batch"])

    moisture = float(np.dot(base_weights, picks.get("moisture_pct", 0).to_numpy(dtype=float) / 100.0))
    difficulty = float(np.dot(base_weights, picks.get("difficulty_factor", 1).to_numpy(dtype=float) / 3.0))

    energy_kwh = total_mass * (process_energy + 0.25 * difficulty + 0.12 * moisture + 0.18 * regolith_pct)
    water_l = total_mass * (process_water + 0.35 * moisture + 0.08 * regolith_pct)
    crew_min = process_crew + 18.0 * difficulty + 10.0 * regolith_pct
    crew_min += 3.0 * len(picks)

    return PredProps(
        rigidity=rigidity,
        tightness=tightness,
        mass_final_kg=total_mass * 0.9,
        energy_kwh=float(max(0.0, energy_kwh)),
        water_l=float(max(0.0, water_l)),
        crew_min=float(max(1.0, crew_min)),
    )


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _pick_materials(df: pd.DataFrame, rng: random.Random, n: int = 2, bias: float = 2.0) -> pd.DataFrame:
    weights = df["kg"].clip(lower=0.01) + df["_problematic"].astype(int) * float(bias)
    return df.sample(n=min(n, len(df)), weights=weights, replace=False, random_state=rng.randint(0, 10_000))


def _select_process(
    proc_df: pd.DataFrame,
    rng: random.Random,
    used_cats: list[str],
    used_flags: list[str],
    used_mats: list[str],
    preferred: str | None = None,
) -> pd.Series:
    if preferred:
        forced = proc_df[proc_df["process_id"].astype(str) == str(preferred)]
        if not forced.empty:
            return forced.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]

    proc = proc_df.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]
    cats_join = " ".join([str(c).lower() for c in used_cats])
    flags_join = " ".join([str(f).lower() for f in used_flags])
    mats_join = " ".join(used_mats).lower()

    if ("pouches" in cats_join) or ("multilayer" in flags_join) or ("pe-pet-al" in mats_join):
        cand = proc_df[proc_df["process_id"].astype(str) == "P02"]
        if not cand.empty:
            proc = cand.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]

    if ("foam" in cats_join) or ("zotek" in mats_join):
        cand = proc_df[proc_df["process_id"].astype(str).isin(["P03", "P02"])]
        if not cand.empty:
            proc = cand.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]

    if ("eva" in cats_join) or ("ctb" in flags_join):
        cand = proc_df[proc_df["process_id"].astype(str) == "P04"]
        if not cand.empty:
            proc = cand.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]

    return proc


def _score_candidate(
    props: PredProps,
    target: Mapping[str, Any],
    picks: pd.DataFrame,
    total_kg: float,
    crew_time_low: bool,
) -> tuple[float, Dict[str, Any], Dict[str, Any]]:
    prob_mass = float((picks["_problematic"].astype(int) * picks["kg"]).sum())
    context = {
        "problematic_mass_ratio": prob_mass / max(0.1, total_kg),
        "crew_time_low": bool(crew_time_low),
    }
    auxiliary = derive_auxiliary_signals(props, target)
    score, breakdown = score_recipe(props, target, context=context, aux=auxiliary)
    return float(score), breakdown, auxiliary


@dataclass
class CandidateComponents:
    picks: pd.DataFrame
    process: pd.Series
    weights: list[float]
    regolith_pct: float
    total_mass: float
    materials_for_plan: list[str]
    weights_for_plan: list[float]
    used_ids: list[str]
    used_cats: list[str]
    used_flags: list[str]
    used_mats: list[str]


def _create_candidate_components(
    picks: pd.DataFrame,
    proc_df: pd.DataFrame,
    rng: random.Random,
    tuning: dict[str, Any] | None,
) -> CandidateComponents | None:
    if picks.empty or proc_df is None or proc_df.empty:
        return None

    tuning = tuning or {}
    total_kg = max(0.001, float(picks["kg"].sum()))
    weights = (picks["kg"] / total_kg).clip(lower=0.0).tolist()
    weights = [float(round(w, 4)) for w in weights]

    used_ids = picks["_source_id"].tolist()
    used_cats = picks["_source_category"].tolist()
    used_flags = picks["_source_flags"].tolist()
    used_mats = picks["material"].tolist()

    preferred_process = tuning.get("process_choice")
    proc = _select_process(proc_df, rng, used_cats, used_flags, used_mats, preferred_process)

    regolith_pct = float(tuning.get("regolith_pct", 0.0))
    if regolith_pct <= 0.0 and str(proc["process_id"]).upper() == "P03":
        regolith_pct = 0.2

    materials_for_plan = used_mats.copy()
    weights_for_plan = weights.copy()
    if regolith_pct > 0:
        materials_for_plan.append("MGS-1_regolith")
        weights_for_plan = [round(w * (1.0 - regolith_pct), 3) for w in weights_for_plan]
        weights_for_plan.append(round(regolith_pct, 3))
        total = sum(weights_for_plan)
        if total > 0:
            weights_for_plan = [round(w / total, 3) for w in weights_for_plan]

    return CandidateComponents(
        picks=picks,
        process=proc,
        weights=weights,
        regolith_pct=regolith_pct,
        total_mass=total_kg,
        materials_for_plan=materials_for_plan,
        weights_for_plan=weights_for_plan,
        used_ids=used_ids,
        used_cats=used_cats,
        used_flags=used_flags,
        used_mats=used_mats,
    )


def _finalize_candidate(
    components: CandidateComponents,
    features: Dict[str, Any],
    target: dict,
    crew_time_low: bool,
    use_ml: bool,
) -> dict | None:
    picks = components.picks
    proc = components.process
    weights = components.weights
    regolith_pct = components.regolith_pct
    total_kg = components.total_mass
    materials_for_plan = components.materials_for_plan
    weights_for_plan = components.weights_for_plan
    used_ids = components.used_ids
    used_cats = components.used_cats
    used_flags = components.used_flags
    used_mats = components.used_mats

    features = dict(features)
    recipe_id = derive_recipe_id(picks, proc, features)
    if recipe_id:
        features["recipe_id"] = recipe_id
    features["prediction_mode"] = "heuristic"

    heuristic = heuristic_props(picks, proc, weights, regolith_pct)
    curated_targets, curated_meta = lookup_labels(
        picks,
        str(proc.get("process_id")),
        {"recipe_id": recipe_id, "materials": used_ids},
    )
    features["curated_label_targets"] = curated_targets or {}
    features["curated_label_metadata"] = curated_meta or {}

    provenance = str(
        curated_meta.get("provenance")
        or curated_meta.get("label_source")
        or ""
    ).lower()
    use_curated = bool(curated_targets) and provenance != "weak"

    prediction: dict[str, Any] = {}
    prediction_error: str | None = None
    if use_curated:
        confidence = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in (curated_meta.get("confidence_intervals") or {}).items()
        }
        props = PredProps(
            rigidity=float(curated_targets.get("rigidez", heuristic.rigidity)),
            tightness=float(curated_targets.get("estanqueidad", heuristic.tightness)),
            mass_final_kg=heuristic.mass_final_kg,
            energy_kwh=float(curated_targets.get("energy_kwh", heuristic.energy_kwh)),
            water_l=float(curated_targets.get("water_l", heuristic.water_l)),
            crew_min=float(curated_targets.get("crew_min", heuristic.crew_min)),
            source=str(curated_meta.get("label_source") or curated_meta.get("provenance") or "curated"),
            confidence_interval=confidence or None,
        )
        prediction = {
            "source": props.source,
            "metadata": curated_meta,
            "targets": curated_targets,
            "confidence_interval": confidence,
        }
        features["prediction_model"] = props.source
        features["model_metadata"] = curated_meta
        features["confidence_interval"] = props.confidence_interval or {}
        features["uncertainty"] = {}
        features["feature_importance"] = []
        features["model_variants"] = {}
        features["prediction_mode"] = "curated"
    else:
        props = heuristic
        force_env = os.getenv("REXAI_FORCE_HEURISTIC", "").lower() in {"1", "true", "yes"}
        force_heuristic = (not use_ml) or force_env
        if not force_heuristic and MODEL_REGISTRY is not None and getattr(MODEL_REGISTRY, "ready", False):
            features_for_inference = dict(features)
            if prediction:
                # prediction se inicializa vacío; mantener la rama por claridad estructural.
                pass
            else:
                try:
                    prediction = MODEL_REGISTRY.predict(features_for_inference)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logging.getLogger(__name__).exception("MODEL_REGISTRY.predict failed")
                    prediction = {}
                    prediction_error = f"Error al invocar el modelo ML: {exc}"
                    _append_inference_log(
                        features_for_inference,
                        {"error": str(exc)},
                        {},
                        MODEL_REGISTRY,
                    )
                else:
                    logged_prediction = prediction or {"error": "MODEL_REGISTRY returned no data"}
                    _append_inference_log(
                        features_for_inference,
                        logged_prediction,
                        ((prediction or {}).get("uncertainty") if isinstance(prediction, dict) else {}),
                        MODEL_REGISTRY,
                    )
                    if prediction:
                        props = PredProps(
                            rigidity=float(prediction.get("rigidez", props.rigidity)),
                            tightness=float(prediction.get("estanqueidad", props.tightness)),
                            mass_final_kg=heuristic.mass_final_kg,
                            energy_kwh=float(prediction.get("energy_kwh", props.energy_kwh)),
                            water_l=float(prediction.get("water_l", props.water_l)),
                            crew_min=float(prediction.get("crew_min", props.crew_min)),
                            source=str(prediction.get("source", "ml")),
                            uncertainty={k: float(v) for k, v in (prediction.get("uncertainty") or {}).items()},
                            confidence_interval={
                                k: (float(v[0]), float(v[1]))
                                for k, v in (prediction.get("confidence_interval") or {}).items()
                            },
                            feature_importance=[
                                (str(k), float(v)) for k, v in (prediction.get("feature_importance") or [])
                            ],
                            comparisons={
                                k: {kk: float(vv) for kk, vv in val.items()}
                                for k, val in (prediction.get("comparisons") or {}).items()
                            },
                        )
                        features["prediction_model"] = props.source
                        features["model_metadata"] = prediction.get("metadata", {})
                        features["uncertainty"] = props.uncertainty or {}
                        features["confidence_interval"] = props.confidence_interval or {}
                        features["feature_importance"] = props.feature_importance or []
                        features["model_variants"] = props.comparisons or {}
                        features["prediction_mode"] = "ml"
                    else:
                        prediction = {}
                        prediction_error = "El modelo ML no devolvió resultados."
                        logging.getLogger(__name__).error("MODEL_REGISTRY.predict returned no data")
        if force_heuristic:
            features["prediction_mode"] = "heuristic"

    latent: Tuple[float, ...] | list[float] = []
    if MODEL_REGISTRY is not None and getattr(MODEL_REGISTRY, "ready", False):
        try:
            emb = MODEL_REGISTRY.embed(features)  # type: ignore[attr-defined]
        except Exception:
            emb = ()
        if emb:
            latent = tuple(float(x) for x in emb)
            features["latent_vector"] = latent

    score, breakdown, auxiliary = _score_candidate(props, target, picks, total_kg, crew_time_low)

    return {
        "materials": materials_for_plan,
        "weights": weights_for_plan,
        "process_id": str(proc["process_id"]),
        "process_name": str(proc.get("name", "")),
        "props": props,
        "heuristic_props": heuristic,
        "score": round(float(score), 3),
        "source_ids": used_ids,
        "source_categories": used_cats,
        "source_flags": used_flags,
        "regolith_pct": regolith_pct,
        "features": features,
        "prediction_source": props.source,
        "ml_prediction": prediction,
        "prediction_error": prediction_error,
        "latent_vector": latent,
        "uncertainty": props.uncertainty or {},
        "confidence_interval": props.confidence_interval or {},
        "feature_importance": props.feature_importance or [],
        "model_variants": props.comparisons or {},
        "score_breakdown": breakdown,
        "auxiliary": auxiliary,
    }


def _build_candidate(
    picks: pd.DataFrame,
    proc_df: pd.DataFrame,
    rng: random.Random,
    target: dict,
    crew_time_low: bool,
    use_ml: bool,
    tuning: dict[str, Any] | None,
) -> dict | None:
    components = _create_candidate_components(picks, proc_df, rng, tuning)
    if components is None:
        return None

    batch = build_feature_tensor_batch(
        [components.picks],
        [components.weights],
        [components.process],
        [components.regolith_pct],
    )
    features_batch = _compute_features_from_batch(batch)
    if not features_batch:
        return None

    return _finalize_candidate(components, features_batch[0], target, crew_time_low, use_ml)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_candidates(
    waste_df: pd.DataFrame,
    proc_df: pd.DataFrame,
    target: dict,
    n: int = 6,
    crew_time_low: bool = False,
    optimizer_evals: int = 0,
    use_ml: bool = True,
):
    """Generate *n* candidate recycling plans plus optional optimization history."""

    if waste_df is None or waste_df.empty or proc_df is None or proc_df.empty:
        return [], pd.DataFrame()

    df = prepare_waste_frame(waste_df)
    rng = random.Random()
    process_ids = sorted(proc_df["process_id"].astype(str).unique().tolist()) if not proc_df.empty else []

    def sampler(override: dict[str, Any] | None = None) -> dict | None:
        override = override or {}
        bias = float(override.get("problematic_bias", 2.0))
        picks = _pick_materials(df, rng, n=rng.choice([2, 3]), bias=bias)
        components = _create_candidate_components(picks, proc_df, rng, override)
        if components is None:
            return None
        batch = build_feature_tensor_batch(
            [components.picks],
            [components.weights],
            [components.process],
            [components.regolith_pct],
        )
        features = _compute_features_from_batch(batch)
        if not features:
            return None
        return _finalize_candidate(components, features[0], target, crew_time_low, use_ml)

    components_batch: list[CandidateComponents] = []
    for _ in range(n):
        bias = 2.0
        picks = _pick_materials(df, rng, n=rng.choice([2, 3]), bias=bias)
        components = _create_candidate_components(picks, proc_df, rng, {})
        if components:
            components_batch.append(components)

    candidates: list[dict] = []
    if components_batch:
        batch = build_feature_tensor_batch(
            [comp.picks for comp in components_batch],
            [comp.weights for comp in components_batch],
            [comp.process for comp in components_batch],
            [comp.regolith_pct for comp in components_batch],
        )
        features_list = _compute_features_from_batch(batch)
        for comp, feat in zip(components_batch, features_list):
            candidate = _finalize_candidate(comp, feat, target, crew_time_low, use_ml)
            if candidate:
                candidates.append(candidate)

    history = pd.DataFrame()
    if optimizer_evals and optimizer_evals > 0:
        try:
            from app.modules.optimizer import optimize_candidates

            pareto, history = optimize_candidates(
                initial_candidates=candidates,
                sampler=sampler,
                target=target,
                n_evals=int(optimizer_evals),
                process_ids=process_ids,
            )
            candidates = pareto
        except Exception:
            history = pd.DataFrame()

    candidates.sort(key=lambda cand: cand.get("score", 0.0), reverse=True)
    return candidates, history


__all__ = [
    "generate_candidates",
    "PredProps",
    "prepare_waste_frame",
    "compute_feature_vector",
    "compute_feature_vectors_batch",
    "build_feature_tensor_batch",
    "FeatureTensorBatch",
    "heuristic_props",
]
