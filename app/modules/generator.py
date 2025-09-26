"""Candidate generation utilities for the Rex-AI demo.

This module converts NASA's non-metabolic waste inventory into the
structured features consumed by the machine learning models.  When
artifacts are available, predictions are served from the trained
RandomForest/XGBoost ensemble; otherwise the fallback heuristics ensure
the UI remains functional.

Reference dataset loading and inference logging responsibilities now live
in :mod:`app.modules.data_sources` and :mod:`app.modules.logging_utils`
respectively so that this file focuses on feature engineering and
candidate assembly.
"""

from __future__ import annotations

import atexit
import itertools
import json
import logging
import math
import os
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Sequence, Tuple

from functools import lru_cache
from functools import lru_cache
from pathlib import Path
from datetime import UTC, datetime
from pathlib import Path
from functools import lru_cache
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
try:
    from scipy import sparse
except Exception:  # pragma: no cover - scipy is optional during inference
    sparse = None  # type: ignore[assignment]

try:  # Torch tensors may appear when the caller works in PyTorch land.
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]

from app.modules.data_sources import (
    _CATEGORY_SYNONYMS,
    GAS_MEAN_YIELD,
    MEAN_REUSE,
    REGOLITH_VECTOR,
    L2LParameters as _L2LParameters,
    load_l2l_parameters as _load_l2l_parameters,
    OfficialFeaturesBundle,
    normalize_category,
    normalize_item,
)

# The demo previously collapsed multiple NASA inventory families (Packaging,
# Other Packaging, Gloves, Foam Packaging, Food Packaging, Structural Elements
# and EVA Waste) into a single "other packaging glove" bucket.  The updated
# mapping keeps these families distinct so downstream heuristics can reason
# about them independently while still tolerating legacy spellings.
_CATEGORY_SYNONYMS.update(
    {
        "packaging": "packaging",
        "packaging material": "packaging",
        "other packaging": "other packaging",
        "other packaging glove": "other packaging",
        "glove": "gloves",
        "gloves": "gloves",
        "foam packaging": "foam packaging",
        "foam packaging for launch": "foam packaging",
        "food packaging": "food packaging",
        "structural element": "structural elements",
        "structural elements": "structural elements",
        "eva": "eva waste",
        "eva waste": "eva waste",
    }
)

_CATEGORY_FAMILY_FALLBACKS: Dict[str, tuple[str, ...]] = {
    "packaging": ("packaging", "other packaging"),
    "other packaging": ("other packaging", "packaging"),
    "gloves": ("gloves", "other packaging", "packaging"),
    "other packaging glove": ("other packaging", "gloves", "packaging"),
}
try:  # Optional heavy dependencies; gracefully disable logging if missing
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - pyarrow is expected in production
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

from app.modules.execution import (
    DEFAULT_PARALLEL_THRESHOLD,
    ExecutionBackend,
    create_backend,
)
from app.modules.label_mapper import derive_recipe_id, lookup_labels
from app.modules.ranking import derive_auxiliary_signals, score_recipe

try:  # Lazy import to avoid circular dependency during training pipelines
    from app.modules.ml_models import MODEL_REGISTRY
except Exception:  # pragma: no cover - fallback when models are not available
    MODEL_REGISTRY = None

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
LOGS_ROOT = Path(__file__).resolve().parents[2] / "data" / "logs"
_OFFICIAL_FEATURES_PATH = DATASETS_ROOT / "rexai_nasa_waste_features.csv"

_REGOLITH_OXIDE_ITEMS = tuple(REGOLITH_VECTOR.items())
_REGOLITH_OXIDE_NAMES = tuple(f"oxide_{name}" for name, _ in _REGOLITH_OXIDE_ITEMS)
_REGOLITH_OXIDE_VALUES = np.asarray([float(value) for _, value in _REGOLITH_OXIDE_ITEMS], dtype=float)


@dataclass(slots=True)
class _InferenceWriterState:
    """Book-keeping for an active Parquet writer."""

    date_token: str
    path: Path
    schema: "pa.Schema"
    writer: Any


class _InferenceLogWriterManager:
    """Manage a single append-only Parquet writer with daily rotation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: _InferenceWriterState | None = None

    def close(self) -> None:
        """Close the active writer, if any."""

        if pq is None:
            return
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        state = self._state
        if state is None:
            return
        try:
            state.writer.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
        self._state = None

    def _open_locked(
        self, timestamp: datetime, field_names: Iterable[str]
    ) -> _InferenceWriterState | None:
        if pa is None or pq is None:
            return None

        desired_fields = sorted(set(str(name) for name in field_names))
        if not desired_fields:
            return None

        log_dir = _resolve_inference_log_dir(timestamp)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        date_token = timestamp.strftime("%Y%m%d")
        path = self._resolve_log_path(log_dir, date_token)
        schema = pa.schema(pa.field(name, pa.string()) for name in desired_fields)

        try:
            writer = pq.ParquetWriter(str(path), schema=schema)
        except Exception:
            return None

        state = _InferenceWriterState(
            date_token=date_token,
            path=path,
            schema=schema,
            writer=writer,
        )
        self._state = state
        return state

    def _resolve_log_path(self, log_dir: Path, date_token: str) -> Path:
        """Return a Parquet path for *date_token* avoiding overwriting shards."""

        base = log_dir / f"inference_{date_token}.parquet"
        if not base.exists():
            return base

        counter = 1
        while True:
            candidate = log_dir / f"inference_{date_token}_{counter:04d}.parquet"
            if not candidate.exists():
                return candidate
            counter += 1

    def _ensure_state_locked(
        self, timestamp: datetime, field_names: Iterable[str]
    ) -> _InferenceWriterState | None:
        state = self._state
        desired_fields = sorted(set(str(name) for name in field_names))
        if not desired_fields:
            return None

        date_token = timestamp.strftime("%Y%m%d")
        if state is not None:
            if state.date_token != date_token or set(state.schema.names) != set(
                desired_fields
            ):
                self._close_locked()
                state = None

        if state is None:
            state = self._open_locked(timestamp, desired_fields)

        return state

    def write_event(
        self, timestamp: datetime, payload: Mapping[str, str | None]
    ) -> None:
        if pa is None or pq is None:
            return

        with self._lock:
            state = self._ensure_state_locked(timestamp, payload.keys())
            if state is None:
                return

            arrays = []
            for field in state.schema:
                arrays.append(pa.array([payload.get(field.name)], type=field.type))

            table = pa.Table.from_arrays(arrays, schema=state.schema)

            try:
                state.writer.write_table(table)
            except Exception:
                self._close_locked()


_INFERENCE_LOG_MANAGER = _InferenceLogWriterManager()


def _close_inference_log_writer() -> None:
    """Expose a test hook to close the shared inference writer."""

    _INFERENCE_LOG_MANAGER.close()


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
    """Return the directory backing inference logs for *timestamp*."""

    return LOGS_ROOT / "inference" / timestamp.strftime("%Y%m%d")


if pq is not None:  # pragma: no branch - guard for optional dependency
    atexit.register(_INFERENCE_LOG_MANAGER.close)


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


def append_inference_log(
    input_features: Dict[str, Any],
    prediction: Dict[str, Any] | None,
    uncertainty: Dict[str, Any] | None,
    model_registry: Any | None,
) -> None:
    """Persist an inference event using a streaming Parquet writer."""

    if pa is None or pq is None:  # pragma: no cover - dependencies should exist
        return

    event_time, event_payload = _prepare_inference_event(
        input_features=input_features,
        prediction=prediction,
        uncertainty=uncertainty,
        model_registry=model_registry,
    )

    _INFERENCE_LOG_MANAGER.write_event(event_time, event_payload)


def _close_inference_log_writer() -> None:
    """Close the cached Parquet writer used for inference logs."""

    _INFERENCE_LOG_MANAGER.close()


def _load_regolith_vector() -> Dict[str, float]:
    path = _resolve_dataset_path("MGS-1_Martian_Regolith_Simulant_Recipe.csv")
    if path is None:
        path = DATASETS_ROOT / "raw" / "mgs1_oxides.csv"

    if path and path.exists():
        table_lazy = pl.scan_csv(path)
        columns = table_lazy.columns

        key_cols = [
            col
            for col in columns
            if col.lower() in {"oxide", "component", "phase", "mineral"}
        ]
        value_cols = [
            col
            for col in columns
            if any(token in col.lower() for token in ("wt", "weight", "percent"))
        ]

        key_col = key_cols[0] if key_cols else None
        value_col = value_cols[0] if value_cols else None

        if key_col and value_col:

            def _clean_label(value: Any) -> str:
                text = str(value or "").lower()
                text = re.sub(r"[^0-9a-z]+", "_", text)
                text = re.sub(r"_+", "_", text).strip("_")
                return text

            working_lazy = (
                table_lazy.select(
                    pl.col(key_col).alias("key"),
                    pl.col(value_col).cast(pl.Float64, strict=False).alias("value"),
                )
                .drop_nulls()
                .with_columns(
                    pl.col("key").map_elements(_clean_label, return_dtype=pl.String)
                )
            )

            working = working_lazy.collect()
            if working.height:
                values = working.get_column("value")
                total = float(values.sum())
                if total > 0:
                    normalised = working.with_columns(
                        (pl.col("value") / pl.lit(total)).alias("weight")
                    )
                    keys = normalised.get_column("key").to_list()
                    weights = normalised.get_column("weight").to_numpy()
                    return {
                        str(key): float(weight)
                        for key, weight in zip(keys, weights, strict=False)
                        if key and weight is not None and np.isfinite(weight)
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
    "approx_moisture_pct": 1000.0,
    "Other_pct": 500.0,
    "Plastic_Resin_pct": 950.0,
}

_CATEGORY_DENSITY_DEFAULTS = {
    "foam packaging": 100.0,
    "food packaging": 650.0,
    "structural elements": 1800.0,
    "structural element": 1800.0,
    "packaging": 420.0,
    "other packaging": 420.0,
    "gloves": 420.0,
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

    mission_totals_df = melted.group_by("mission").agg(pl.col("mass").sum()).collect()
    mission_totals: Dict[str, float] = {}
    if mission_totals_df.height:
        missions = mission_totals_df.get_column("mission").to_list()
        masses = mission_totals_df.get_column("mass").to_numpy()
        for mission, mass in zip(missions, masses, strict=False):
            if mission and mass is not None and np.isfinite(mass):
                mission_totals[str(mission)] = float(mass)

    mass_by_key: Dict[str, Dict[str, float]] = {}

    subitem_totals_df = (
        melted
        .filter(pl.col("item_key") != pl.col("category_key"))
        .group_by(["item_key", "mission"])
        .agg(pl.col("mass").sum())
        .collect()
    )

    if subitem_totals_df.height:
        keys = subitem_totals_df.get_column("item_key").to_list()
        missions = subitem_totals_df.get_column("mission").to_list()
        masses = subitem_totals_df.get_column("mass").to_numpy()
        for key, mission, mass in zip(keys, missions, masses, strict=False):
            if not key or not mission or mass is None or not np.isfinite(mass):
                continue
            entry = mass_by_key.setdefault(str(key), {})
            entry[str(mission)] = entry.get(str(mission), 0.0) + float(mass)

    category_totals_df = (
        melted.group_by(["category_key", "mission"]).agg(pl.col("mass").sum()).collect()
    )

    if category_totals_df.height:
        keys = category_totals_df.get_column("category_key").to_list()
        missions = category_totals_df.get_column("mission").to_list()
        masses = category_totals_df.get_column("mass").to_numpy()
        for key, mission, mass in zip(keys, missions, masses, strict=False):
            if not key or not mission or mass is None or not np.isfinite(mass):
                continue
            entry = mass_by_key.setdefault(str(key), {})
            entry[str(mission)] = entry.get(str(mission), 0.0) + float(mass)

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
        summary_df = table.select(
            [pl.col(col).cast(pl.Float64, strict=False).mean().alias(col) for col in numeric_cols]
        ).collect()
        if summary_df.height:
            metrics: Dict[str, float] = {}
            for column in numeric_cols:
                series = summary_df.get_column(column)
                value = series[0] if series.len() else None
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                metrics[f"{prefix}_{_slugify(column)}"] = float(value)
            if metrics:
                aggregated[prefix] = metrics
        return aggregated

    combinations: list[tuple[str, ...]] = []
    for length in range(1, len(group_columns) + 1):
        combinations.extend(itertools.combinations(group_columns, length))

    for combo in combinations:
        grouped_df = (
            table.group_by(list(combo))
            .agg([pl.col(col).cast(pl.Float64, strict=False).mean().alias(col) for col in numeric_cols])
            .collect()
        )
        if not grouped_df.height:
            continue

        combo_values = [grouped_df.get_column(column).to_list() for column in combo]
        metric_arrays = [
            np.asarray(grouped_df.get_column(column).to_numpy(), dtype=np.float64)
            for column in numeric_cols
        ]

        for row_idx in range(grouped_df.height):
            slug_parts: list[str] = []
            for values, column in zip(combo_values, combo, strict=False):
                value = values[row_idx]
                if isinstance(value, str):
                    slug_part = _slugify(value)
                elif value is not None:
                    slug_part = _slugify(str(value))
                else:
                    slug_part = ""
                if slug_part:
                    slug_parts.append(slug_part)
            slug = "_".join(part for part in slug_parts if part)
            if not slug:
                continue

            metrics: Dict[str, float] = {}
            for array, column in zip(metric_arrays, numeric_cols, strict=False):
                value = array[row_idx]
                if value is None or not np.isfinite(value):
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
    return normalize_category(value)
def _build_match_key(category: Any, subitem: Any | None = None) -> str:
    """Return the canonical key used to match NASA reference tables."""

    if subitem:
        return f"{_normalize_category(category)}|{_normalize_item(subitem)}"
    return _normalize_category(category)
def _estimate_density_from_row(row: pd.Series) -> float | None:
    """Estimate a material density with packaging-aware fallbacks."""

    # The NASA features bundle exposes aggregate density and composition values
    # per category.  Now that the packaging families are disambiguated the
    # estimator first honours category-specific defaults (e.g. ``gloves`` vs.
    # ``other packaging``) before falling back to the more general packaging
    # prior used by legacy data dumps.
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
    return normalize_item(value)


def _token_set(value: Any) -> frozenset[str]:
    normalized = _normalize_item(value)
    if not normalized:
        return frozenset()
    return frozenset(normalized.split())


_L2L_PARAMETERS: _L2LParameters | Mapping[str, Any] | None = None


def _coerce_l2l_parameters(candidate: Any) -> _L2LParameters | None:
    """Return a structured Logistics-to-Living payload when possible."""

    if isinstance(candidate, _L2LParameters):
        constants = getattr(candidate, "constants", None)
        category_features = getattr(candidate, "category_features", None)
        if isinstance(constants, Mapping) and isinstance(category_features, Mapping):
            return candidate
        return _L2LParameters(
            dict(constants or {}),
            {str(k): dict(v) for k, v in (category_features or {}).items() if isinstance(v, Mapping)},
            {
                str(k): dict(v)
                for k, v in getattr(candidate, "item_features", {}).items()
                if isinstance(v, Mapping)
            },
            {str(k): str(v) for k, v in getattr(candidate, "hints", {}).items()},
        )

    if isinstance(candidate, Mapping):
        constants = candidate.get("constants", {})
        category_features = candidate.get("category_features", {})
        item_features = candidate.get("item_features", {})
        hints = candidate.get("hints", {})
        return _L2LParameters(
            dict(constants) if isinstance(constants, Mapping) else {},
            {
                str(key): dict(value)
                for key, value in (category_features if isinstance(category_features, Mapping) else {}).items()
                if isinstance(value, Mapping)
            },
            {
                str(key): dict(value)
                for key, value in (item_features if isinstance(item_features, Mapping) else {}).items()
                if isinstance(value, Mapping)
            },
            {str(key): str(value) for key, value in (hints if isinstance(hints, Mapping) else {}).items()},
        )

    return None


def _get_l2l_parameters() -> _L2LParameters:
    """Load Logistics-to-Living parameters with robust fallbacks."""

    global _L2L_PARAMETERS

    cached = _coerce_l2l_parameters(_L2L_PARAMETERS)
    if cached is not None:
        _L2L_PARAMETERS = cached
        return cached

    try:
        loaded = _load_l2l_parameters()
    except Exception:  # pragma: no cover - defensive guard for optional data
        logging.getLogger(__name__).exception("Failed to load Logistics-to-Living parameters")
        loaded = None

    coerced = _coerce_l2l_parameters(loaded)
    if coerced is None:
        coerced = _L2LParameters({}, {}, {}, {})

    _L2L_PARAMETERS = coerced
    return coerced


class _OfficialFeaturesBundle(NamedTuple):
    value_columns: tuple[str, ...]
    composition_columns: tuple[str, ...]
    direct_map: Dict[str, int]
    category_tokens: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    table: pl.DataFrame
    value_matrix: np.ndarray
    mission_mass: Dict[str, Dict[str, float]]
    mission_totals: Dict[str, float]
    mission_reference_keys: tuple[str, ...]
    mission_reference_index: Dict[str, int]
    mission_reference_matrix: Any
    mission_reference_dense: np.ndarray
    mission_names: tuple[str, ...]
    mission_totals_vector: np.ndarray
    processing_metrics: Dict[str, Dict[str, float]]
    leo_mass_savings: Dict[str, Dict[str, float]]
    propellant_benefits: Dict[str, Dict[str, float]]
    l2l_constants: Dict[str, float]
    l2l_category_features: Dict[str, Dict[str, float]]
    l2l_item_features: Dict[str, Dict[str, float]]
    l2l_hints: Dict[str, str]


def official_features_bundle() -> _OfficialFeaturesBundle:
    """Public accessor mirroring :func:`_official_features_bundle`."""

    return _official_features_bundle()


def _build_payload_from_row(row: np.ndarray, columns: Sequence[str]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for name, value in zip(columns, row, strict=False):
        if value is None:
            payload[name] = float("nan")
            continue
        if isinstance(value, (float, np.floating)):
            if math.isnan(value):
                payload[name] = float("nan")
            else:
                payload[name] = float(value)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        payload[name] = numeric
    return payload


def _tokenize_subitems(subitems: Sequence[str]) -> np.ndarray:
    array = np.asarray(list(subitems), dtype=object)
    tokenizer = np.frompyfunc(lambda text: frozenset(str(text).split()) if text else frozenset(), 1, 1)
    return tokenizer(array)


def _vectorized_feature_maps(
    table_df: pl.DataFrame, value_columns: Sequence[str]
) -> tuple[Dict[str, int], Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    value_frame = table_df.select(
        [pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in value_columns]
    )

    if value_frame.height != table_df.height:
        value_matrix = np.empty((0, len(value_columns)), dtype=np.float64)
    elif pa is not None:
        arrow_block = value_frame.to_arrow()
        arrays = [
            np.asarray(arrow_block.column(i).to_numpy(zero_copy_only=False), dtype=np.float64)
            for i in range(arrow_block.num_columns)
        ]
        value_matrix = (
            np.column_stack(arrays)
            if arrays
            else np.empty((table_df.height, 0), dtype=np.float64)
        )
    else:
        value_matrix = value_frame.to_numpy()

    string_columns = ["key", "category_norm", "subitem_norm"]
    if pa is not None:
        string_block = table_df.select(string_columns).to_arrow()
        keys_raw = string_block.column("key").to_pylist()
        categories_raw = string_block.column("category_norm").to_pylist()
        subitems_raw = string_block.column("subitem_norm").to_pylist()
    else:
        keys_raw = table_df.get_column("key").to_list()
        categories_raw = table_df.get_column("category_norm").to_list()
        subitems_raw = table_df.get_column("subitem_norm").to_list()

    key_array = np.asarray([str(value) if value is not None else "" for value in keys_raw], dtype=object)
    category_list = [str(value) if value is not None else "" for value in categories_raw]
    subitem_list = [str(value) if value is not None else "" for value in subitems_raw]

    direct_map = {key: idx for idx, key in enumerate(key_array) if key}

    token_array = _tokenize_subitems(subitem_list)
    row_indices = np.arange(len(key_array), dtype=np.int32)

    category_tokens: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if category_list:
        category_array = np.asarray(category_list, dtype=object)
        unique_categories, inverse = np.unique(category_array, return_inverse=True)
        for pos, category in enumerate(unique_categories):
            if not category:
                continue
            mask = inverse == pos
            matched_indices = row_indices[mask]
            if matched_indices.size == 0:
                continue
            category_tokens[str(category)] = (
                token_array[mask],
                key_array[mask],
                matched_indices,
            )

    return direct_map, category_tokens, value_matrix


def _build_mission_reference_tables(
    mass_by_key: Mapping[str, Mapping[str, float]],
    mission_totals: Mapping[str, float],
) -> tuple[tuple[str, ...], Dict[str, int], Any, tuple[str, ...], np.ndarray]:
    mission_names = tuple(sorted(mission_totals.keys()))
    mission_index = {name: idx for idx, name in enumerate(mission_names)}
    totals_vector = np.asarray(
        [float(mission_totals.get(name, 0.0)) for name in mission_names], dtype=np.float64
    )

    mission_reference_keys = tuple(sorted(mass_by_key.keys()))
    mission_reference_index = {key: idx for idx, key in enumerate(mission_reference_keys)}

    if not mission_reference_keys or not mission_names:
        if sparse is not None:
            matrix = sparse.csr_matrix((0, 0), dtype=np.float64)
        else:
            matrix = np.zeros((0, 0), dtype=np.float64)
        return mission_reference_keys, mission_reference_index, matrix, mission_names, totals_vector

    data: list[float] = []
    rows: list[int] = []
    cols: list[int] = []

    for key, row_idx in mission_reference_index.items():
        missions = mass_by_key.get(key) or {}
        if not missions:
            continue
        for mission, mass in missions.items():
            col_idx = mission_index.get(mission)
            if col_idx is None:
                continue
            try:
                mass_value = float(mass)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(mass_value) or mass_value <= 0:
                continue
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(mass_value)

    shape = (len(mission_reference_keys), len(mission_names))
    if sparse is not None:
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float64)
    else:
        matrix = np.zeros(shape, dtype=np.float64)
        if data:
            matrix[rows, cols] = data

    return mission_reference_keys, mission_reference_index, matrix, mission_names, totals_vector


@lru_cache(maxsize=1)
def _official_features_bundle() -> _OfficialFeaturesBundle:
    l2l = _get_l2l_parameters()
    if not isinstance(getattr(l2l, "constants", None), Mapping) or not isinstance(
        getattr(l2l, "category_features", None), Mapping
    ):
        l2l = _L2LParameters({}, {}, {}, {})
    if sparse is not None:
        empty_reference = sparse.csr_matrix((0, 0), dtype=np.float64)
    else:
        empty_reference = np.zeros((0, 0), dtype=np.float64)
    default = _OfficialFeaturesBundle(
        (),
        (),
        {},
        pl.DataFrame(),
        np.empty((0, 0), dtype=np.float64),
        {},
        {},
        (),
        {},
        empty_reference,
        np.zeros((0, 0), dtype=np.float64),
        (),
        np.zeros(0, dtype=np.float64),
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
            .map_elements(_normalize_category, return_dtype=pl.String)
            .alias("category_norm"),
            pl.col("subitem")
            .map_elements(_normalize_item, return_dtype=pl.String)
            .alias("subitem_norm"),
        ]
    )

    table_lazy = table_lazy.with_columns(
        pl.when(pl.col("category_norm") == "other packaging")
        .then(
            pl.when(pl.col("subitem_norm").str.contains("glove"))
            .then(pl.lit("gloves"))
            .otherwise(pl.col("category_norm"))
        )
        .otherwise(pl.col("category_norm"))
        .alias("category_norm")
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
    if not value_columns:
        return default
    composition_columns = tuple(
        col for col in value_columns if col.endswith("_pct") and not col.startswith("subitem_")
    )

    direct_map, category_tokens, value_matrix = _vectorized_feature_maps(table_df, value_columns)

    waste_summary = _load_waste_summary_data()
    processing_metrics = _extract_grouped_metrics("nasa_waste_processing_products.csv", "processing")
    leo_savings = _extract_grouped_metrics("nasa_leo_mass_savings.csv", "leo")
    propellant_metrics = _extract_grouped_metrics("nasa_propellant_benefits.csv", "propellant")

    table_join = table_df.select(
        ["category_norm", "subitem_norm", *value_columns]
    ).unique(subset=["category_norm", "subitem_norm"], maintain_order=True)

    (
        mission_reference_keys,
        mission_reference_index,
        mission_reference_matrix,
        mission_names,
        mission_totals_vector,
    ) = _build_mission_reference_tables(
        waste_summary.mass_by_key, waste_summary.mission_totals
    )

    if sparse is not None and sparse.issparse(mission_reference_matrix):
        mission_reference_dense = mission_reference_matrix.toarray()
    else:
        mission_reference_dense = np.asarray(mission_reference_matrix, dtype=np.float64)

    return _OfficialFeaturesBundle(
        value_columns,
        composition_columns,
        direct_map,
        category_tokens,
        table_join,
        value_matrix,
        waste_summary.mass_by_key,
        waste_summary.mission_totals,
        mission_reference_keys,
        mission_reference_index,
        mission_reference_matrix,
        mission_reference_dense,
        mission_names,
        mission_totals_vector,
        processing_metrics,
        leo_savings,
        propellant_metrics,
        l2l.constants,
        l2l.category_features,
        l2l.item_features,
        l2l.hints,
    )


official_features_bundle = _official_features_bundle
def official_features_bundle() -> _OfficialFeaturesBundle:
    """Public accessor for tests and external callers."""

    return _official_features_bundle()


def _lookup_official_feature_values(row: pd.Series) -> tuple[Dict[str, float], str]:
    """Resolve NASA reference features with packaging-aware fallbacks."""

    bundle = _official_features_bundle()
    if not bundle.value_columns:
        return {}, ""

    category = _normalize_category(row.get("category", ""))
    if not category:
        return {}, ""

    family_candidates = _CATEGORY_FAMILY_FALLBACKS.get(category, (category,))
    # Preserve order while dropping duplicates so legacy aliases remain usable.
    seen_families: set[str] = set()
    families: tuple[str, ...] = tuple(
        family for family in family_candidates if not (family in seen_families or seen_families.add(family))
    )
    if not families:
        families = (category,)

    candidates = (
        row.get("material"),
        row.get("material_family"),
        row.get("key_materials"),
    )

    for candidate in candidates:
        normalized = _normalize_item(candidate)
        if not normalized:
            continue
        for family in families:
            key = f"{family}|{normalized}"
            index = bundle.direct_map.get(key)
            if index is not None and 0 <= index < bundle.value_matrix.shape[0]:
                payload = _build_payload_from_row(bundle.value_matrix[index], bundle.value_columns)
                return payload, key

    for family in families:
        category_index = bundle.direct_map.get(family)
        if category_index is not None and 0 <= category_index < bundle.value_matrix.shape[0]:
            payload = _build_payload_from_row(bundle.value_matrix[category_index], bundle.value_columns)
            return payload, family

    token_candidates = [value for value in candidates if value]
    if not token_candidates:
        return {}, ""

    for family in families:
        matches = bundle.category_tokens.get(family)
        if not matches:
            continue

        token_array, key_array, row_indices = matches
        reference_tokens = list(token_array.tolist())
        match_keys = list(key_array.tolist())
        indices = list(row_indices.tolist())

        for candidate in token_candidates:
            tokens = _token_set(candidate)
            if not tokens:
                continue
            for reference, match_key, index in zip(reference_tokens, match_keys, indices, strict=False):
                if not isinstance(reference, frozenset):
                    continue
                if tokens.issubset(reference) and 0 <= index < bundle.value_matrix.shape[0]:
                    payload = _build_payload_from_row(bundle.value_matrix[index], bundle.value_columns)
                    return payload, str(match_key)

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
            pl.col("category")
            .map_elements(normalize_category)
            .map_elements(_normalize_category, return_dtype=pl.String)
            .alias("category_norm")
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
                pl.col(source)
                .map_elements(normalize_item)
                .map_elements(_normalize_item, return_dtype=pl.String)
                .alias(alias)
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
            .otherwise(pl.col("_official_match_key"))
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
        for column in bundle.value_columns:
            if column not in result_df.columns:
                result_df[column] = np.nan

        numeric_candidates = [
            column
            for column in bundle.value_columns
            if column.endswith(("_kg", "_pct"))
            or column.startswith("category_total")
            or column in {"difficulty_factor", "approx_moisture_pct"}
        ]

        if bundle.direct_map and bundle.value_matrix.size:
            matrix = bundle.value_matrix
            columns = bundle.value_columns
            match_series = result_df.get("_official_match_key")
            for row_idx in result_df.index:
                candidates: list[str] = []
                if isinstance(match_series, pd.Series):
                    raw_key = match_series.get(row_idx)
                    if raw_key:
                        candidates.append(str(raw_key))

                category_norm = _normalize_category(result_df.at[row_idx, "category"] if "category" in result_df.columns else "")
                if category_norm:
                    for source in ("material", "key_materials", "material_family"):
                        value = result_df.at[row_idx, source] if source in result_df.columns else ""
                        normalized = _normalize_item(value)
                        if normalized:
                            candidates.append(f"{category_norm}|{normalized}")
                    candidates.append(category_norm)

                resolved_index: int | None = None
                for candidate in candidates:
                    if not candidate:
                        continue
                    index = bundle.direct_map.get(candidate)
                    if index is not None and 0 <= index < matrix.shape[0]:
                        resolved_index = index
                        break

                if resolved_index is None:
                    continue

                row_values = matrix[resolved_index]
                for col_pos, column in enumerate(columns):
                    if column not in result_df.columns:
                        continue
                    try:
                        value = float(row_values[col_pos])
                    except (IndexError, TypeError, ValueError):
                        continue
                    if np.isnan(value):
                        continue
                    result_df.at[row_idx, column] = value

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

    # ------------------------------------------------------------------
    # Vectorised token assembly and heuristic flags
    # ------------------------------------------------------------------
    for column in ("material", "category", "flags", "key_materials"):
        if column not in out.columns:
            out[column] = ""
    material_lower = out["material"].astype(str).str.lower()
    category_lower = out["category"].astype(str).str.lower()
    flags_lower = out["flags"].astype(str).str.lower()
    key_materials_lower = out["key_materials"].astype(str).str.lower()

    tokens = material_lower.str.cat(category_lower, sep=" ", na_rep="")
    tokens = tokens.str.cat(flags_lower, sep=" ", na_rep="")
    tokens = tokens.str.cat(key_materials_lower, sep=" ", na_rep="")
    out["tokens"] = tokens

    if "_problematic" not in out.columns:
        material_family_series = out.get("material_family")
        if isinstance(material_family_series, pd.Series):
            material_family_series = material_family_series.astype(str)
        else:
            material_family_series = pd.Series("", index=out.index, dtype=str)
        family_tokens = material_lower.str.cat(
            material_family_series.fillna("").str.lower(),
            sep=" ",
            na_rep="",
        )

        problematic_rules = (
            category_lower.str.contains("pouches", na=False)
            | flags_lower.str.contains("multilayer", na=False)
            | family_tokens.str.contains("pe-pet-al", na=False)
            | category_lower.str.contains("foam", na=False)
            | family_tokens.str.contains("zotek", na=False)
            | flags_lower.str.contains("closed_cell", na=False)
            | category_lower.str.contains("eva", na=False)
            | flags_lower.str.contains("ctb", na=False)
            | family_tokens.str.contains("nomex", na=False)
            | family_tokens.str.contains("nylon", na=False)
            | family_tokens.str.contains("polyester", na=False)
            | category_lower.str.contains("glove", na=False)
            | family_tokens.str.contains("nitrile", na=False)
            | flags_lower.str.contains("wipe", na=False)
            | category_lower.str.contains("textile", na=False)
        )
        out["_problematic"] = problematic_rules.astype(bool)
    else:
        out["_problematic"] = out["_problematic"].astype(bool)

    out["_source_id"] = out["id"].astype(str)
    out["_source_category"] = out["category"].astype(str)
    out["_source_flags"] = out["flags"].astype(str)

    bundle = _official_features_bundle()
    out = _inject_official_features(out)

    mass = pd.to_numeric(out["kg"], errors="coerce").fillna(0.0)
    volume_l = pd.to_numeric(out.get("volume_l"), errors="coerce")
    volume_m3 = volume_l / 1000.0

    density = mass.divide(volume_m3).where((volume_m3 > 0) & volume_m3.notna())

    cat_mass = pd.to_numeric(out.get("category_total_mass_kg"), errors="coerce")
    if not isinstance(cat_mass, pd.Series):
        cat_mass = pd.Series(cat_mass, index=out.index, dtype=float)
    cat_volume = pd.to_numeric(out.get("category_total_volume_m3"), errors="coerce")
    if not isinstance(cat_volume, pd.Series):
        cat_volume = pd.Series(cat_volume, index=out.index, dtype=float)
    cat_density = cat_mass.divide(cat_volume).where((cat_volume > 0) & cat_volume.notna())
    density = density.fillna(cat_density)

    selected_columns: list[str] = []
    if getattr(bundle, "composition_columns", None):
        selected_columns.extend(
            [
                column
                for column in bundle.composition_columns
                if column in out.columns and column in _COMPOSITION_DENSITY_MAP
            ]
        )
    fallback_columns = [
        column
        for column in _COMPOSITION_DENSITY_MAP
        if column in out.columns and column not in selected_columns
    ]
    selected_columns.extend(fallback_columns)

    if selected_columns:
        composition_numeric = {
            column: pd.to_numeric(out[column], errors="coerce").fillna(0.0)
            for column in selected_columns
        }
        composition_frac = pd.DataFrame(composition_numeric, index=out.index).div(100.0)
        frac_total = composition_frac.sum(axis=1)
        density_lookup = pd.Series(
            {column: float(_COMPOSITION_DENSITY_MAP[column]) for column in selected_columns},
            index=selected_columns,
            dtype=float,
        )
        density_weights = composition_frac.multiply(density_lookup, axis=1)
        weighted_density = density_weights.sum(axis=1)
        weighted_density = weighted_density.divide(frac_total).where(frac_total > 0)
    else:
        weighted_density = pd.Series(np.nan, index=out.index, dtype=float)

    normalized_category = _vectorized_normalize_category(out["category"])
    foam_mask = normalized_category == "foam packaging"
    foam_default = _CATEGORY_DENSITY_DEFAULTS.get("foam packaging")
    if foam_default is not None:
        weighted_density = weighted_density.where(
            ~foam_mask,
            weighted_density.clip(upper=float(foam_default)),
        )

    density = density.fillna(weighted_density)

    logistic_density_map: Dict[str, float] = {}
    category_features = getattr(bundle, "l2l_category_features", None)
    if isinstance(category_features, Mapping):
        for key, features in category_features.items():
            if not isinstance(features, Mapping):
                continue
            density_value: float | None = None
            for name, value in features.items():
                if not isinstance(value, (int, float)):
                    continue
                if not np.isfinite(value):
                    continue
                name_lower = str(name).lower()
                if "density" not in name_lower:
                    continue
                density_value = float(value)
                break
            if density_value is None:
                continue
            normalized_key = normalize_category(key)
            if not normalized_key:
                continue
            logistic_density_map.setdefault(normalized_key, density_value)

    if logistic_density_map:
        logistic_density = normalized_category.map(logistic_density_map)
        logistic_density = pd.to_numeric(logistic_density, errors="coerce")
        density = density.fillna(logistic_density)

    category_defaults = normalized_category.map(_CATEGORY_DENSITY_DEFAULTS).astype(float)
    density = density.fillna(category_defaults)

    default_density = float(_CATEGORY_DENSITY_DEFAULTS.get("packaging", 500.0))
    density = density.fillna(default_density)
    out["density_kg_m3"] = density.clip(lower=20.0, upper=4000.0)

    return out


def _vectorized_normalize_category(series: pd.Series) -> pd.Series:
    """Return :func:`normalize_category` values without row-wise ``apply``."""

    if not isinstance(series, pd.Series) or series.empty:
        return pd.Series([], dtype=str)

    values = series.astype(str)
    unique_values = pd.unique(values)
    mapping = {value: normalize_category(value) for value in unique_values}
    normalized = values.map(mapping)
    return normalized.fillna("")


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
    keyword_hits: np.ndarray
    packaging_hits: np.ndarray
    regolith_pct: float
    mission_indices: np.ndarray
    composition_matrix: np.ndarray
    composition_presence: np.ndarray
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
    mission_indices: Any
    mission_reference: Any
    mission_totals: Any
    composition_values: Any
    total_mass: Any
    regolith: Any
    keyword_names: Tuple[str, ...]
    mission_names: Tuple[str, ...]
    composition_names: Tuple[str, ...]
    process_ids: Tuple[str, ...]
    num_items: Tuple[int, ...]
    composition_presence: np.ndarray
    l2l_constants: Dict[str, float]
    bundle_processing_metrics: Dict[str, Dict[str, float]]
    bundle_leo_mass_savings: Dict[str, Dict[str, float]]
    bundle_propellant_benefits: Dict[str, Dict[str, float]]
    logistics_ratio: float


def _coerce_feature_tensor_batch_like(
    value: FeatureTensorBatch | Mapping[str, Any]
) -> FeatureTensorBatch:
    if isinstance(value, FeatureTensorBatch):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        tuple_keys = {
            "keyword_names",
            "mission_names",
            "composition_names",
            "process_ids",
            "num_items",
        }
        for key in tuple_keys:
            if key in data and not isinstance(data[key], tuple):
                data[key] = tuple(data[key])
        if "composition_presence" in data and not isinstance(
            data["composition_presence"], np.ndarray
        ):
            data["composition_presence"] = np.asarray(data["composition_presence"], dtype=bool)

        required = set(FeatureTensorBatch.__annotations__.keys())
        missing = sorted(required.difference(data.keys()))
        if missing:
            raise ValueError(
                "FeatureTensorBatch mapping is missing required fields: " + ", ".join(missing)
            )
        return FeatureTensorBatch(**data)
    raise TypeError("Unsupported feature tensor batch input")


def _coerce_weight_array(values: Iterable[float] | Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = values.astype(float, copy=False)
    elif torch is not None and isinstance(values, torch.Tensor):  # type: ignore[arg-type]
        array = values.detach().cpu().numpy().astype(float, copy=False)
    elif jnp is not None and hasattr(jnp, "ndarray") and isinstance(values, jnp.ndarray):  # type: ignore[attr-defined]
        array = np.asarray(values, dtype=float)
    elif isinstance(values, pd.Series):
        array = values.to_numpy(dtype=float, copy=False)
    else:
        array = np.asarray(list(values), dtype=float)

    if array.ndim == 0:
        array = np.asarray([float(array)], dtype=float)

    return array


def _prepare_feature_context(
    picks: pd.DataFrame,
    weights: Iterable[float] | Any,
    regolith_pct: float,
    bundle: _OfficialFeaturesBundle,
) -> CandidateFeatureContext:
    total_kg = max(0.001, float(picks["kg"].sum()))
    raw_weights = _coerce_weight_array(weights)
    if len(raw_weights) < len(picks):
        raw_weights = np.pad(raw_weights, (0, len(picks) - len(raw_weights)), constant_values=0.0)
    elif len(raw_weights) > len(picks):
        raw_weights = raw_weights[: len(picks)]

    item_count = len(picks)

    default_series = lambda default: pd.Series(default, index=picks.index, dtype=float)
    pct_mass = picks.get("pct_mass", default_series(0.0)).to_numpy(dtype=float) / 100.0
    pct_volume = picks.get("pct_volume", default_series(0.0)).to_numpy(dtype=float) / 100.0
    moisture = picks.get("moisture_pct", default_series(0.0)).to_numpy(dtype=float) / 100.0
    difficulty = picks.get("difficulty_factor", default_series(1.0)).to_numpy(dtype=float) / 3.0
    densities = picks.get("density_kg_m3", default_series(0.0)).to_numpy(dtype=float)
    if item_count:
        token_arrays: list[np.ndarray] = []
        for column in ("material", "category", "flags", "material_family", "key_materials"):
            if column in picks.columns:
                values = np.asarray(
                    picks[column].astype(str).fillna("").to_numpy(), dtype=str
                )
            else:
                values = np.full(item_count, "", dtype=str)
            token_arrays.append(values)

        tokens_array = token_arrays[0].astype(str)
        for part in token_arrays[1:]:
            tokens_array = np.char.add(np.char.add(tokens_array, " "), part.astype(str))
        tokens_array = np.asarray(np.char.lower(np.char.strip(tokens_array)))
    else:
        tokens_array = np.empty(0, dtype=str)

    if "category" in picks.columns:
        category_raw = np.asarray(
            picks["category"].astype(str).fillna("").to_numpy(), dtype=str
        )
    else:
        category_raw = np.full(item_count, "", dtype=str)
    category_lower = np.asarray(np.char.lower(category_raw)) if item_count else np.empty(0, dtype=str)

    keyword_hits = np.zeros((item_count, len(_KEYWORD_FEATURES)), dtype=float)
    if item_count and len(_KEYWORD_FEATURES):
        for kw_idx, patterns in enumerate(_KEYWORD_FEATURES.values()):
            if not patterns:
                continue
            mask = np.zeros(item_count, dtype=bool)
            for pattern in patterns:
                if not pattern:
                    continue
                mask |= np.char.find(tokens_array, pattern.lower()) >= 0
            keyword_hits[:, kw_idx] = mask.astype(float)

    packaging_hits = np.zeros(item_count, dtype=float)
    if item_count and _PACKAGING_TARGETS:
        mask = np.zeros(item_count, dtype=bool)
        for target in _PACKAGING_TARGETS:
            if not target:
                continue
            mask |= np.char.find(category_lower, target.lower()) >= 0
        packaging_hits = mask.astype(float)

    def _series_or_default(column: str, default: float) -> pd.Series:
        series = picks.get(column)
        if isinstance(series, pd.Series):
            return pd.to_numeric(series, errors="coerce").fillna(default)
        return pd.Series(default, index=picks.index, dtype=float)

    pct_mass = _series_or_default("pct_mass", 0.0).to_numpy(dtype=float) / 100.0
    pct_volume = _series_or_default("pct_volume", 0.0).to_numpy(dtype=float) / 100.0
    moisture = _series_or_default("moisture_pct", 0.0).to_numpy(dtype=float) / 100.0
    difficulty = _series_or_default("difficulty_factor", 1.0).to_numpy(dtype=float) / 3.0
    densities = _series_or_default("density_kg_m3", 0.0).to_numpy(dtype=float)

    composition_count = len(bundle.composition_columns)
    if composition_count:
        composition_matrix = np.zeros((item_count, composition_count), dtype=float)
        composition_presence = np.zeros(composition_count, dtype=bool)
    else:
        composition_matrix = np.zeros((item_count, 0), dtype=float)
        composition_presence = np.zeros(0, dtype=bool)

    if composition_count:
        for col_idx, column in enumerate(bundle.composition_columns):
            if column not in picks.columns:
                continue
            series = picks[column]
            numeric = pd.to_numeric(series, errors="coerce")
            if not numeric.notna().any():
                continue
            composition_presence[col_idx] = True
            values = numeric.fillna(0.0).to_numpy(dtype=float) / 100.0
            if not len(values):
                continue
            limit = min(len(values), item_count)
            composition_matrix[:limit, col_idx] = values[:limit]

    mission_indices = np.full(item_count, -1, dtype=np.int32)
    if (
        item_count
        and bundle.mission_reference_keys
        and bundle.mission_names
        and (bundle.mission_reference_matrix.shape[0] > 0)
    ):
        if "_official_match_key" in picks.columns:
            match_keys = (
                picks["_official_match_key"].astype(str).fillna("").to_numpy(dtype=object)
            )
        else:
            match_keys = np.full(item_count, "", dtype=object)

        normalized_categories = np.asarray(
            [normalize_category(value) for value in category_raw], dtype=object
        )

        effective_keys = match_keys.astype(object)
        empty_mask = (effective_keys == "") | pd.isna(effective_keys)
        if empty_mask.any():
            effective_keys[empty_mask] = normalized_categories[empty_mask]

        weights_arr = raw_weights[:item_count]
        if weights_arr.size:
            positive_mask = np.isfinite(weights_arr) & (weights_arr > 0)
            if not positive_mask.all():
                effective_keys = effective_keys.astype(object)
                effective_keys[~positive_mask] = ""

            key_indices = np.array(
                [bundle.mission_reference_index.get(str(key), -1) for key in effective_keys],
                dtype=np.int32,
            )
            valid_mask = key_indices >= 0
            if np.any(valid_mask):
                mission_indices = key_indices

    return CandidateFeatureContext(
        total_mass=total_kg,
        weights=raw_weights,
        densities=densities,
        moisture=moisture,
        difficulty=difficulty,
        pct_mass=pct_mass,
        pct_volume=pct_volume,
        keyword_hits=keyword_hits,
        packaging_hits=packaging_hits,
        regolith_pct=float(regolith_pct),
        mission_indices=mission_indices,
        composition_matrix=composition_matrix,
        composition_presence=composition_presence,
        num_items=int(len(picks)),
    )


def _contexts_to_tensor_batch(
    contexts: Sequence[CandidateFeatureContext],
    processes: Sequence[pd.Series],
    bundle: _OfficialFeaturesBundle,
    *,
    backend: str = "jax",
) -> FeatureTensorBatch:
    if len(contexts) != len(processes):
        raise ValueError("Number of contexts must match number of processes")

    batch = len(contexts)
    max_items = max((ctx.num_items for ctx in contexts), default=0)
    keyword_names = tuple(_KEYWORD_FEATURES.keys())
    mission_names = bundle.mission_names
    composition_names = bundle.composition_columns

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
    composition_values = _alloc((batch, max_items, len(composition_names)))
    mission_indices = np.full((batch, max_items), -1, dtype=np.int32)

    total_mass = np.zeros(batch, dtype=float)
    regolith = np.zeros(batch, dtype=float)
    num_items: list[int] = []
    composition_presence_matrix = np.zeros((batch, len(composition_names)), dtype=bool)

    mission_totals = np.asarray(bundle.mission_totals_vector, dtype=float)

    for idx, ctx in enumerate(contexts):
        count = ctx.num_items
        num_items.append(count)

        if count:
            weights[idx, :count] = ctx.weights[:count]
            densities[idx, :count] = ctx.densities[:count]
            moisture[idx, :count] = ctx.moisture[:count]
            difficulty[idx, :count] = ctx.difficulty[:count]
            pct_mass[idx, :count] = ctx.pct_mass[:count]
            pct_volume[idx, :count] = ctx.pct_volume[:count]
            if ctx.keyword_hits.size:
                kw_cols = min(ctx.keyword_hits.shape[1], len(keyword_names))
                keyword_hits[idx, :count, :kw_cols] = ctx.keyword_hits[:count, :kw_cols]
            if ctx.packaging_hits.size:
                packaging_hits[idx, :count] = ctx.packaging_hits[:count]
            if ctx.composition_matrix.size:
                comp_cols = min(ctx.composition_matrix.shape[1], len(composition_names))
                composition_values[idx, :count, :comp_cols] = ctx.composition_matrix[
                    :count, :comp_cols
                ]
            if ctx.mission_indices.size:
                mission_count = min(len(ctx.mission_indices), count)
                mission_indices[idx, :mission_count] = ctx.mission_indices[:mission_count]

        if ctx.composition_presence.size:
            comp_len = min(len(ctx.composition_presence), len(composition_names))
            composition_presence_matrix[idx, :comp_len] = ctx.composition_presence[:comp_len]

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
        mission_indices=_to_backend_array(mission_indices),
        mission_reference=_to_backend_array(np.asarray(bundle.mission_reference_dense, dtype=float)),
        mission_totals=_to_backend_array(mission_totals),
        composition_values=_to_backend_array(composition_values),
        total_mass=_to_backend_array(total_mass),
        regolith=_to_backend_array(regolith),
        keyword_names=keyword_names,
        mission_names=mission_names,
        composition_names=composition_names,
        process_ids=process_ids,
        num_items=tuple(num_items),
        composition_presence=composition_presence_matrix,
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
    mission_indices: Any,
    mission_reference: Any,
    mission_totals: Any,
    total_mass: Any,
    composition_values: Any,
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
    mission_totals = xp.asarray(mission_totals)
    mission_indices = xp.asarray(mission_indices)
    mission_reference = xp.asarray(mission_reference)
    total_mass = xp.asarray(total_mass)
    composition_values = xp.asarray(composition_values)
    regolith = xp.asarray(regolith)

    density = xp.sum(w * densities, axis=1)
    moisture_frac = xp.clip(xp.sum(w * moisture, axis=1), 0.0, 1.0)
    difficulty_index = xp.clip(xp.sum(w * difficulty, axis=1), 0.0, 1.0)
    problematic_mass_frac = xp.clip(xp.sum(w * pct_mass, axis=1), 0.0, 1.0)
    problematic_item_frac = xp.clip(xp.sum(w * pct_volume, axis=1), 0.0, 1.0)

    keyword = xp.clip(xp.sum(w[:, :, None] * keyword_hits, axis=1), 0.0, 1.0)
    packaging = xp.clip(xp.sum(w * packaging_hits, axis=1), 0.0, 1.0)

    batch_size = w.shape[0]
    item_dim = w.shape[1] if w.ndim > 1 else 0
    mission_dim = mission_totals.shape[0]
    key_dim = mission_reference.shape[0] if mission_reference.ndim else 0

    valid_indices = mission_indices >= 0
    if mission_dim == 0 or key_dim == 0:
        reference_rows = xp.zeros((batch_size, item_dim, mission_dim))
        weighted_mass = xp.zeros((batch_size, mission_dim))
    else:
        safe_indices = xp.where(valid_indices, mission_indices, 0)
        reference_rows = xp.take(mission_reference, safe_indices, axis=0)
        reference_rows = xp.where(valid_indices[:, :, None], reference_rows, 0.0)
        weighted_mass = xp.sum(w[:, :, None] * reference_rows, axis=1)

    totals_positive = mission_totals > 0
    totals_safe = xp.where(totals_positive, mission_totals, 1.0)
    mission_share = xp.where(totals_positive, weighted_mass / totals_safe, 0.0)
    mission_share_clipped = xp.clip(mission_share, 0.0, 1.0)
    mission_similarity_total = xp.clip(xp.sum(mission_share_clipped, axis=1), 0.0, 1.0)
    mission_reference_mass = xp.maximum(0.0, mission_share_clipped * mission_totals)
    mission_scaled_mass = xp.maximum(0.0, mission_share_clipped * total_mass[:, None])
    mission_official_mass = xp.maximum(0.0, weighted_mass)

    composition = xp.sum(w[:, :, None] * composition_values, axis=1)

    polyethylene = keyword[:, _KEYWORD_INDEX["polyethylene_frac"]] if keyword.shape[1] else xp.zeros(keyword.shape[0])
    foam = keyword[:, _KEYWORD_INDEX["foam_frac"]] if keyword.shape[1] else xp.zeros(keyword.shape[0])
    eva = keyword[:, _KEYWORD_INDEX["eva_frac"]] if keyword.shape[1] else xp.zeros(keyword.shape[0])
    textile = keyword[:, _KEYWORD_INDEX["textile_frac"]] if keyword.shape[1] else xp.zeros(keyword.shape[0])

    gas_index = GAS_MEAN_YIELD * (
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
    reuse_term = packaging_term * MEAN_REUSE
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
        "composition": composition,
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
        mission_indices: Any,
        mission_reference: Any,
        mission_totals: Any,
        total_mass: Any,
        composition_values: Any,
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
            mission_indices,
            mission_reference,
            mission_totals,
            total_mass,
            composition_values,
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
        mission_indices: Any,
        mission_reference: Any,
        mission_totals: Any,
        total_mass: Any,
        composition_values: Any,
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
            mission_indices,
            mission_reference,
            mission_totals,
            total_mass,
            composition_values,
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
        batch.mission_indices,
        batch.mission_reference,
        batch.mission_totals,
        batch.total_mass,
        batch.composition_values,
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
    composition_total = to_numpy(kernel_output["composition"])
    total_mass = to_numpy(batch.total_mass)

    keyword_pairs = tuple(
        (name, idx)
        for name, idx in _KEYWORD_INDEX.items()
        if keyword_matrix.shape[1] > idx
    )

    mission_arrays: tuple[
        tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ...,
    ] = tuple(
        (
            mission,
            mission_similarity[:, mission_idx],
            mission_reference_mass[:, mission_idx],
            mission_scaled_mass[:, mission_idx],
            mission_official_mass[:, mission_idx],
        )
        for mission_idx, mission in enumerate(batch.mission_names)
    )

    l2l_items = [
        (name, float(value))
        for name, value in batch.l2l_constants.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    ]
    l2l_names = tuple(name for name, _ in l2l_items)
    l2l_values = (
        np.asarray([value for _, value in l2l_items], dtype=float)
        if l2l_items
        else np.empty(0, dtype=float)
    )

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

        for name, kw_idx in keyword_pairs:
            features[name] = float(keyword_matrix[idx, kw_idx])

        official_comp: Dict[str, float] = {}
        if batch.composition_names and composition_total.size:
            presence_row = (
                batch.composition_presence[idx]
                if idx < len(batch.composition_presence)
                else np.zeros(len(batch.composition_names), dtype=bool)
            )
            for name_idx, name in enumerate(batch.composition_names):
                if presence_row.size > name_idx and not presence_row[name_idx]:
                    continue
                if composition_total.ndim >= 2 and composition_total.shape[0] > idx and composition_total.shape[1] > name_idx:
                    official_comp[name] = float(composition_total[idx, name_idx])
                elif composition_total.ndim == 1 and composition_total.size > name_idx:
                    official_comp[name] = float(composition_total[name_idx])
        _apply_official_composition_overrides(features, official_comp)

        if l2l_values.size:
            for name_idx, name in enumerate(l2l_names):
                features[name] = float(l2l_values[name_idx])

        mission_similarity_row = mission_similarity[idx] if mission_similarity.size else np.array([])
        mission_similarity_dict: Dict[str, float] = {}
        if mission_similarity_row.size and np.any(mission_similarity_row > 0):
            for (
                mission,
                share_values,
                reference_values,
                scaled_values,
                official_values,
            ) in mission_arrays:
                share = float(share_values[idx])
                if share <= 0:
                    continue
                mission_similarity_dict[mission] = share
                features[f"mission_similarity_{mission}"] = float(np.clip(share, 0.0, 1.0))
                features[f"mission_reference_mass_{mission}"] = float(reference_values[idx])
                features[f"mission_scaled_mass_{mission}"] = float(scaled_values[idx])
                features[f"mission_official_mass_{mission}"] = float(official_values[idx])

            features["mission_similarity_total"] = float(mission_similarity_total[idx])

            _apply_weighted_metrics(features, mission_similarity_dict, batch.bundle_processing_metrics)
            _apply_weighted_metrics(features, mission_similarity_dict, batch.bundle_leo_mass_savings)
            _apply_weighted_metrics(features, mission_similarity_dict, batch.bundle_propellant_benefits)

        features["gas_recovery_index"] = float(gas_recovery[idx]) if gas_recovery.size else 0.0
        features["logistics_reuse_index"] = float(logistics_reuse[idx]) if logistics_reuse.size else 0.0

        if _REGOLITH_OXIDE_VALUES.size:
            oxide_values = _REGOLITH_OXIDE_VALUES * features["regolith_pct"]
            for oxide_name, oxide_value in zip(_REGOLITH_OXIDE_NAMES, oxide_values, strict=False):
                features[oxide_name] = float(oxide_value)

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
    picks: pd.DataFrame | pl.DataFrame | FeatureTensorBatch | Sequence[pd.DataFrame | pl.DataFrame],
    weights: Iterable[float] | Sequence[Iterable[float]] | None = None,
    process: pd.Series | Sequence[pd.Series] | None = None,
    regolith_pct: float | Sequence[float] | None = None,
    *,
    backend: str | None = None,
) -> Dict[str, Any] | list[Dict[str, Any]]:
    if isinstance(picks, (FeatureTensorBatch, Mapping)):
        batch = _coerce_feature_tensor_batch_like(picks)  # type: ignore[arg-type]
        return _compute_features_from_batch(batch)

    backend = backend or "jax"

    def _to_pandas(frame: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        if isinstance(frame, pd.DataFrame):
            return frame
        if isinstance(frame, pl.DataFrame):
            return frame.to_pandas()
        raise TypeError(
            "picks must be a pandas.DataFrame, a polars.DataFrame or a FeatureTensorBatch"
        )

    is_sequence_input = False
    if isinstance(picks, (pd.DataFrame, pl.DataFrame)):
        picks_list = [_to_pandas(picks)]
    else:
        if not isinstance(picks, Sequence) or isinstance(picks, (str, bytes)):
            raise TypeError(
                "picks must be a pandas.DataFrame, a polars.DataFrame, a FeatureTensorBatch, or a sequence of DataFrames"
            )
        picks_list = [_to_pandas(frame) for frame in picks]
        is_sequence_input = True

    if not picks_list:
        return [] if is_sequence_input else {}

    candidate_count = len(picks_list)

    if weights is None or process is None or regolith_pct is None:
        raise ValueError("weights, process and regolith_pct are required for DataFrame inputs")

    if is_sequence_input:
        if not isinstance(weights, Sequence) or len(weights) != candidate_count:
            raise ValueError("weights must be a sequence matching the number of candidates")
        if not isinstance(process, Sequence) or len(process) != candidate_count:
            raise ValueError("process must be a sequence matching the number of candidates")
        if not isinstance(regolith_pct, Sequence) or len(regolith_pct) != candidate_count:
            raise ValueError("regolith_pct must be a sequence matching the number of candidates")
        weights_list = list(weights)
        process_list = list(process)
        regolith_list = list(regolith_pct)
    else:
        weights_list = [weights]
        process_list = [process]
        regolith_list = [regolith_pct]

    batch = build_feature_tensor_batch(
        picks_list,
        weights_list,
        process_list,
        regolith_list,
        backend=backend,
    )
    features = _compute_features_from_batch(batch)
    if is_sequence_input:
        return features
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

    material_series = picks.get("material", pd.Series("", index=picks.index))
    category_series = picks.get("category", pd.Series("", index=picks.index))
    flags_series = picks.get("flags", pd.Series("", index=picks.index))

    materials = " ".join(material_series.astype(str)).lower()
    categories = " ".join(category_series.astype(str).str.lower())
    flags = " ".join(flags_series.astype(str).str.lower())

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

    process_energy = float(process.get("energy_kwh_per_kg", 0.0))
    process_water = float(process.get("water_l_per_kg", 0.0))
    process_crew = float(process.get("crew_min_per_batch", 0.0))

    moisture = float(
        np.dot(
            base_weights,
            picks.get("moisture_pct", pd.Series(0.0, index=picks.index)).to_numpy(dtype=float) / 100.0,
        )
    )
    difficulty = float(
        np.dot(
            base_weights,
            picks.get("difficulty_factor", pd.Series(1.0, index=picks.index)).to_numpy(dtype=float) / 3.0,
        )
    )
    def _vector(column: str, default: float, scale: float) -> np.ndarray:
        series = picks.get(column)
        if isinstance(series, pd.Series):
            values = pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)
        else:
            values = np.full(len(picks), default, dtype=float)
        return values * scale

    moisture = float(np.dot(base_weights, _vector("moisture_pct", 0.0, 1.0 / 100.0)))
    difficulty = float(np.dot(base_weights, _vector("difficulty_factor", 1.0, 1.0 / 3.0)))

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
    curated_targets = curated_targets or {}
    curated_meta = curated_meta or {}
    features["curated_label_targets"] = curated_targets
    features["curated_label_metadata"] = curated_meta

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
                    append_inference_log(
                        features_for_inference,
                        {"error": str(exc)},
                        {},
                        MODEL_REGISTRY,
                    )
                else:
                    logged_prediction = prediction or {"error": "MODEL_REGISTRY returned no data"}
                    append_inference_log(
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


_PARALLEL_THRESHOLD = DEFAULT_PARALLEL_THRESHOLD


def generate_candidates(
    waste_df: pd.DataFrame,
    proc_df: pd.DataFrame,
    target: dict,
    n: int = 6,
    crew_time_low: bool = False,
    optimizer_evals: int = 0,
    use_ml: bool = True,
    backend: ExecutionBackend | None = None,
    backend_kind: str | None = None,
):
    """Generate *n* candidate recycling plans plus optional optimization history."""

    if waste_df is None or waste_df.empty or proc_df is None or proc_df.empty:
        return [], pd.DataFrame()

    df = prepare_waste_frame(waste_df)
    rng = random.Random()
    process_ids = sorted(proc_df["process_id"].astype(str).unique().tolist()) if not proc_df.empty else []

    base_seed = rng.randint(0, 2**31 - 1)
    seed_counter = itertools.count()
    seed_lock = threading.Lock()

    def _next_seed() -> int:
        with seed_lock:
            idx = next(seed_counter)
        return (base_seed + idx * 9973) % (2**32 - 1)

    def sampler(
        override: dict[str, Any] | None = None,
        *,
        seed: int | None = None,
    ) -> dict | None:
        override = override or {}
        seed_value = seed if seed is not None else _next_seed()
        local_rng = random.Random(seed_value)
        bias = float(override.get("problematic_bias", 2.0))
        picks = _pick_materials(df, local_rng, n=local_rng.choice([2, 3]), bias=bias)
        return _build_candidate(picks, proc_df, local_rng, target, crew_time_low, use_ml, override)

    candidates: list[dict] = []
    seeds = [_next_seed() for _ in range(n)]
    local_backend = backend
    owns_backend = False
    if local_backend is None:
        task_count = max(n, 1)
        local_backend = create_backend(
            task_count,
            preferred=backend_kind,
            threshold=_PARALLEL_THRESHOLD,
        )
        owns_backend = True
    try:
        results = local_backend.map(lambda seed: sampler({}, seed=seed), seeds)
        for candidate in results:
            if candidate:
                candidates.append(candidate)

        components_batch: list[CandidateComponents] = []
        for _ in range(n):
            bias = 2.0
            picks = _pick_materials(df, rng, n=rng.choice([2, 3]), bias=bias)
            components = _create_candidate_components(picks, proc_df, rng, {})
            if components:
                components_batch.append(components)

        if components_batch:
            _ = compute_feature_vector(
                [component.picks for component in components_batch],
                weights=[component.weights for component in components_batch],
                process=[component.process for component in components_batch],
                regolith_pct=[component.regolith_pct for component in components_batch],
            )

        attempts = 0
        max_attempts = max(n * 2, 4)
        while len(candidates) < n and attempts < max_attempts:
            attempts += 1
            candidate = sampler({})
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
                    backend=local_backend,
                    backend_kind=backend_kind,
                )
                candidates = pareto
            except Exception:
                history = pd.DataFrame()

    finally:
        if owns_backend:
            local_backend.shutdown()

    candidates.sort(key=lambda cand: cand.get("score", 0.0), reverse=True)
    return candidates, history


__all__ = [
    "generate_candidates",
    "PredProps",
    "append_inference_log",
    "prepare_waste_frame",
    "official_features_bundle",
    "compute_feature_vector",
    "compute_feature_vectors_batch",
    "build_feature_tensor_batch",
    "FeatureTensorBatch",
    "heuristic_props",
]
