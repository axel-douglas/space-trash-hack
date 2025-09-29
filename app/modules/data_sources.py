"""Centralised data ingestion helpers for Rex-AI reference datasets.

The :mod:`app.modules.generator` module depends on a fairly eclectic mix of
CSV/Delta inputs curated by NASA.  Historically these helpers were sprinkled
throughout ``generator.py`` which made the core candidate-building logic hard
to audit.  This module collects the read/parse utilities so that both the
runtime and training pipelines have a consistent contract for obtaining
reference data.

Responsibilities handled here:

* resolving dataset locations inside :mod:`datasets`
* loading CSV artifacts into :class:`polars` or :class:`pandas` structures
* preparing cached bundles with official NASA feature metadata
* exposing normalisation helpers shared across generator routines
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NamedTuple

import numpy as np
import pandas as pd
import polars as pl

from .paths import DATA_ROOT

DATASETS_ROOT = DATA_ROOT.parent / "datasets"

__all__ = [
    "DATASETS_ROOT",
    "to_lazy_frame",
    "from_lazy_frame",
    "resolve_dataset_path",
    "slugify",
    "normalize_text",
    "normalize_category",
    "normalize_item",
    "token_set",
    "merge_reference_dataset",
    "extract_grouped_metrics",
    "L2LParameters",
    "load_l2l_parameters",
    "OfficialFeaturesBundle",
    "official_features_bundle",
    "lookup_official_feature_values",
    "REGOLITH_VECTOR",
    "GAS_MEAN_YIELD",
    "MEAN_REUSE",
    "RegolithThermalBundle",
    "load_regolith_granulometry",
    "load_regolith_spectral_curves",
    "load_regolith_thermal_profiles",
]


def to_lazy_frame(
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


def from_lazy_frame(lazy: pl.LazyFrame, frame_kind: str) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
    """Convert *lazy* back to the representation described by *frame_kind*."""

    if frame_kind == "lazy":
        return lazy

    collected = lazy.collect()
    if frame_kind == "polars":
        return collected
    if frame_kind == "pandas":
        return collected.to_pandas()
    raise ValueError(f"Unsupported frame kind: {frame_kind}")


def resolve_dataset_path(name: str) -> Path | None:
    """Return the first dataset path that exists for *name*."""

    candidates = (
        DATASETS_ROOT / name,
        DATASETS_ROOT / "raw" / name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def slugify(value: str) -> str:
    """Convert *value* into a snake_case identifier safe for feature names."""

    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def _feature_name_from_parts(*parts: str) -> str:
    return "_".join(part for part in (slugify(part) for part in parts if part) if part)


def normalize_text(value: Any) -> str:
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


_CATEGORY_SYNONYMS = {
    "foam": "foam packaging",
    "foam packaging": "foam packaging",
    "foam packaging for launch": "foam packaging",
    "packaging": "packaging",
    "other packaging": "other packaging",
    "other packaging glove": "other packaging",
    "glove": "gloves",
    "gloves": "gloves",
    "food packaging": "food packaging",
    "structural elements": "structural elements",
    "structural element": "structural elements",
    "eva": "eva waste",
    "eva waste": "eva waste",
}


def normalize_category(value: Any) -> str:
    normalized = normalize_text(value)
    return _CATEGORY_SYNONYMS.get(normalized, normalized)


def normalize_item(value: Any) -> str:
    return normalize_text(value)


def token_set(value: Any) -> frozenset[str]:
    normalized = normalize_item(value)
    if not normalized:
        return frozenset()
    return frozenset(normalized.split())


def merge_reference_dataset(
    base: pd.DataFrame | pl.DataFrame | pl.LazyFrame, filename: str, prefix: str
) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
    path = resolve_dataset_path(filename)
    if path is None:
        return base

    base_lazy, base_kind = to_lazy_frame(base)
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
        rename_map[column] = f"{prefix}_{slugify(column)}"

    if drop_cols:
        extra_lazy = extra_lazy.drop(drop_cols)
    if rename_map:
        extra_lazy = extra_lazy.rename(rename_map)

    added_columns = [rename_map.get(col, col) for col in extra_columns if col not in join_cols and col not in drop_cols]

    merged_lazy = base_lazy.join(extra_lazy, on=join_cols, how="left")
    if added_columns:
        projection = base_columns + [col for col in added_columns if col not in base_columns]
        merged_lazy = merged_lazy.select([pl.col(name) for name in projection])

    result = from_lazy_frame(merged_lazy, base_kind)
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


class _WasteSummary(NamedTuple):
    mass_by_key: Dict[str, Dict[str, float]]
    mission_totals: Dict[str, float]


def _mission_slug(column: str) -> str:
    cleaned = column.lower()
    cleaned = cleaned.replace("summary_", "")
    cleaned = cleaned.replace("mass", "")
    cleaned = cleaned.replace("kg", "")
    cleaned = cleaned.replace("total", "")
    cleaned = cleaned.replace("__", "_")
    return slugify(cleaned)


def _load_waste_summary_data() -> _WasteSummary:
    path = resolve_dataset_path("nasa_waste_summary.csv")
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
        .then(pl.col("subitem").map_elements(normalize_item, return_dtype=pl.String))
        .otherwise(pl.lit(""))
        .alias("subitem_norm")
        if has_subitem
        else pl.lit("").alias("subitem_norm")
    )

    melted = (
        table.with_columns(
            pl.col("category")
            .map_elements(normalize_category, return_dtype=pl.String)
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
        )
        .with_columns(
            pl.col("variable")
            .map_elements(_mission_slug, return_dtype=pl.String)
            .alias("mission"),
            pl.col("value").cast(pl.Float64, strict=False).alias("mass"),
        )
        .drop_nulls("mission")
        .drop_nulls("mass")
    )

    mass_by_key: Dict[str, Dict[str, float]] = {}
    mission_totals: Dict[str, float] = {}
    for row in melted.collect().to_dicts():
        key = str(row["item_key"])
        mission = str(row["mission"])
        mass = float(row["mass"])
        if not mission:
            continue
        mission_totals[mission] = mission_totals.get(mission, 0.0) + mass
        entry = mass_by_key.setdefault(key, {})
        entry[mission] = entry.get(mission, 0.0) + mass
        category_key = str(row["category_key"])
        if key != category_key:
            category_entry = mass_by_key.setdefault(category_key, {})
            category_entry[mission] = category_entry.get(mission, 0.0) + mass

    return _WasteSummary(mass_by_key, mission_totals)


def extract_grouped_metrics(filename: str, prefix: str) -> Dict[str, Dict[str, float]]:
    path = resolve_dataset_path(filename)
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
                metrics[f"{prefix}_{slugify(column)}"] = float(value)
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
                    slug = slugify(value)
                elif value is not None:
                    slug = slugify(str(value))
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
                metrics[f"{prefix}_{slugify(column)}"] = float(value)

            if metrics:
                aggregated[slug] = metrics

    return aggregated


def _load_regolith_vector() -> Dict[str, float]:
    path = resolve_dataset_path("MGS-1_Martian_Regolith_Simulant_Recipe.csv")
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


REGOLITH_VECTOR = _load_regolith_vector()
GAS_MEAN_YIELD = _load_gas_mean_yield()
MEAN_REUSE = _load_mean_reuse()


@dataclass
class L2LParameters:
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


def load_l2l_parameters() -> L2LParameters:
    path = resolve_dataset_path("l2l_parameters.csv")
    if path is None or not path.exists():
        return L2LParameters({}, {}, {}, {})

    table = pd.read_csv(path)
    if table.empty:
        return L2LParameters({}, {}, {}, {})

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
        category_norm = normalize_category(category_value)
        subitem_value = row.get(subitem_col, "") if subitem_col else ""
        subitem_norm = normalize_item(subitem_value) if subitem_value else ""

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

    return L2LParameters(constants, category_features, item_features, hints)


class OfficialFeaturesBundle(NamedTuple):
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


@dataclass(frozen=True)
class RegolithThermalBundle:
    """Container for MGS-1 thermogravimetric / EGA reference curves."""

    tg_curve: pd.DataFrame
    ega_curve: pd.DataFrame
    ega_long: pd.DataFrame
    gas_peaks: pd.DataFrame
    mass_events: pd.DataFrame


_L2L_PARAMETERS = load_l2l_parameters()
_OFFICIAL_FEATURES_PATH = DATASETS_ROOT / "rexai_nasa_waste_features.csv"


@lru_cache(maxsize=1)
def official_features_bundle() -> OfficialFeaturesBundle:
    l2l = _L2L_PARAMETERS
    default = OfficialFeaturesBundle(
        (),
        (),
        {},
        {},
        pl.DataFrame(),
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

    table_lazy = merge_reference_dataset(table_lazy, "nasa_waste_summary.csv", "summary")
    table_lazy = merge_reference_dataset(table_lazy, "nasa_waste_processing_products.csv", "processing")
    table_lazy = merge_reference_dataset(table_lazy, "nasa_leo_mass_savings.csv", "leo")
    table_lazy = merge_reference_dataset(table_lazy, "nasa_propellant_benefits.csv", "propellant")

    table_lazy = table_lazy.with_columns(
        [
            pl.col("category")
            .map_elements(normalize_category)
            .alias("category_norm"),
            pl.col("subitem")
            .map_elements(normalize_item)
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

        key = build_match_key(category_raw, subitem_raw)
        category_norm = normalize_category(category_raw)
        tokens = token_set(subitem_raw)

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
    processing_metrics = extract_grouped_metrics("nasa_waste_processing_products.csv", "processing")
    leo_savings = extract_grouped_metrics("nasa_leo_mass_savings.csv", "leo")
    propellant_metrics = extract_grouped_metrics("nasa_propellant_benefits.csv", "propellant")

    table_join = table_df.select(
        ["category_norm", "subitem_norm", *value_columns]
    ).unique(subset=["category_norm", "subitem_norm"], maintain_order=True)

    return OfficialFeaturesBundle(
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


def build_match_key(category: Any, subitem: Any | None = None) -> str:
    """Return the canonical key used to match NASA reference tables."""

    if subitem:
        return f"{normalize_category(category)}|{normalize_item(subitem)}"
    return normalize_category(category)


def lookup_official_feature_values(row: pd.Series) -> tuple[Dict[str, float], str]:
    bundle = official_features_bundle()
    if not bundle.value_columns:
        return {}, ""

    category = normalize_category(row.get("category", ""))
    if not category:
        return {}, ""

    candidates = (
        row.get("material"),
        row.get("material_family"),
        row.get("key_materials"),
    )

    for candidate in candidates:
        normalized = normalize_item(candidate)
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
        tokens = token_set(candidate)
        if not tokens:
            continue
        for reference_tokens, payload, match_key in matches:
            if tokens.issubset(reference_tokens):
                return payload, match_key

    return {}, ""


__all__.extend([
    "build_match_key",
])


def _empty_dataframe(columns: Iterable[str] | None = None) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame({col: [] for col in columns})


@lru_cache(maxsize=1)
def load_regolith_granulometry() -> pd.DataFrame:
    """Return particle size distribution for the MGS-1 simulant."""

    path = resolve_dataset_path("fig3_psizeData.csv")
    if path is None:
        return _empty_dataframe(["diameter_microns", "pct_retained", "pct_channel", "cumulative_retained", "pct_passing"])

    data = pd.read_csv(path, encoding="latin-1")
    rename_map = {
        "Diameter (microns)": "diameter_microns",
        "% Retained": "pct_retained",
        "% Channel": "pct_channel",
    }
    data = data.rename(columns=rename_map)

    for column in ("diameter_microns", "pct_retained", "pct_channel"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["diameter_microns"]).sort_values("diameter_microns", ascending=False).reset_index(drop=True)
    data["pct_retained"] = data["pct_retained"].fillna(0.0)
    data["pct_channel"] = data["pct_channel"].fillna(0.0)
    data["cumulative_retained"] = data["pct_retained"].cumsum().clip(upper=100.0)
    data["pct_passing"] = (100.0 - data["cumulative_retained"]).clip(lower=0.0, upper=100.0)
    return data


@lru_cache(maxsize=1)
def load_regolith_spectral_curves() -> pd.DataFrame:
    """Return VNIR reflectance curves for Martian soil simulants."""

    path = resolve_dataset_path("fig4_spectralData.csv")
    if path is None:
        return _empty_dataframe(["wavelength_nm", "sample", "reflectance", "reflectance_pct", "sample_slug"])

    table = pd.read_csv(path, encoding="latin-1")
    table = table.rename(columns={"Wavelength (nm)": "wavelength_nm"})
    table["wavelength_nm"] = pd.to_numeric(table["wavelength_nm"], errors="coerce")
    table = table.dropna(subset=["wavelength_nm"])

    samples = [column for column in table.columns if column != "wavelength_nm"]
    for column in samples:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    melted = table.melt(id_vars=["wavelength_nm"], var_name="sample", value_name="reflectance").dropna(subset=["reflectance"])
    melted["sample"] = melted["sample"].astype(str).str.strip()
    melted["sample_slug"] = melted["sample"].map(slugify)
    melted["reflectance_pct"] = melted["reflectance"] * 100.0
    melted = melted.sort_values(["sample", "wavelength_nm"]).reset_index(drop=True)
    return melted


@lru_cache(maxsize=1)
def load_regolith_thermal_profiles() -> RegolithThermalBundle:
    """Return thermogravimetric (TG) and EGA curves for MGS-1."""

    tg_path = resolve_dataset_path("fig5_tgData.csv")
    ega_path = resolve_dataset_path("fig5_egaData.csv")

    empty = RegolithThermalBundle(
        tg_curve=_empty_dataframe(["temperature_c", "mass_pct", "mass_loss_pct"]),
        ega_curve=_empty_dataframe(["temperature_c", "mz_18_h2o", "mz_32_o2", "mz_44_co2", "mz_64_so2"]),
        ega_long=_empty_dataframe(["temperature_c", "species", "signal", "signal_ppb", "species_label"]),
        gas_peaks=_empty_dataframe(["species", "species_label", "temperature_c", "signal", "signal_ppb"]),
        mass_events=_empty_dataframe(["event", "temperature_c", "mass_pct"]),
    )

    if tg_path is None or ega_path is None:
        return empty

    tg_raw = pd.read_csv(tg_path, encoding="latin-1")
    tg_raw = tg_raw.rename(columns={"Temperature (¡C)": "temperature_c", "Mass (%)": "mass_pct"})
    tg_raw["temperature_c"] = pd.to_numeric(tg_raw["temperature_c"], errors="coerce")
    tg_raw["mass_pct"] = pd.to_numeric(tg_raw["mass_pct"], errors="coerce")
    tg_raw = tg_raw.dropna(subset=["temperature_c", "mass_pct"]).sort_values("temperature_c").reset_index(drop=True)
    tg_raw["mass_loss_pct"] = (100.0 - tg_raw["mass_pct"]).clip(lower=0.0)

    if len(tg_raw) > 1200:
        step = max(1, len(tg_raw) // 1200)
        tg_curve = tg_raw.iloc[::step].reset_index(drop=True)
    else:
        tg_curve = tg_raw.copy()

    ega_raw = pd.read_csv(ega_path, encoding="latin-1")
    ega_raw = ega_raw.rename(
        columns={
            "Temperature (¡C)": "temperature_c",
            "m/z 18 (H2O)": "mz_18_h2o",
            "m/z 32 (O2)": "mz_32_o2",
            "m/z 44 (CO2)": "mz_44_co2",
            "m/z 64 (SO2)": "mz_64_so2",
        }
    )
    ega_raw["temperature_c"] = pd.to_numeric(ega_raw["temperature_c"], errors="coerce")
    gas_columns = [col for col in ega_raw.columns if col != "temperature_c"]
    for column in gas_columns:
        ega_raw[column] = pd.to_numeric(ega_raw[column], errors="coerce")
    ega_raw = ega_raw.dropna(subset=["temperature_c"]).sort_values("temperature_c").reset_index(drop=True)

    ega_long = ega_raw.melt(id_vars=["temperature_c"], var_name="species", value_name="signal").dropna(subset=["signal"])
    species_labels = {
        "mz_18_h2o": "H₂O (m/z 18)",
        "mz_32_o2": "O₂ (m/z 32)",
        "mz_44_co2": "CO₂ (m/z 44)",
        "mz_64_so2": "SO₂ (m/z 64)",
    }
    ega_long["species_label"] = ega_long["species"].map(species_labels).fillna(ega_long["species"])
    ega_long["signal_ppb"] = ega_long["signal"] * 1e9

    gas_peaks: list[dict[str, float | str]] = []
    for column in gas_columns:
        series = ega_raw[column]
        if series.isnull().all():
            continue
        idx = series.idxmax()
        temperature = float(ega_raw.loc[idx, "temperature_c"])
        signal = float(series.loc[idx])
        gas_peaks.append(
            {
                "species": column,
                "species_label": species_labels.get(column, column),
                "temperature_c": temperature,
                "signal": signal,
                "signal_ppb": signal * 1e9,
            }
        )

    peaks_df = pd.DataFrame(gas_peaks).sort_values("temperature_c").reset_index(drop=True)

    mass_events: list[dict[str, float | str]] = []
    thresholds = [99.5, 99.0, 98.0, 97.0]
    for threshold in thresholds:
        mask = tg_raw["mass_pct"] <= threshold
        if mask.any():
            temp = float(tg_raw.loc[mask, "temperature_c"].iloc[0])
            mass_events.append(
                {
                    "event": f"mass_{threshold}",
                    "temperature_c": temp,
                    "mass_pct": float(threshold),
                }
            )

    if "mass_pct" in tg_raw.columns and tg_raw.shape[0] > 2:
        diff = tg_raw[["temperature_c", "mass_pct"]].copy()
        diff["mass_pct_next"] = diff["mass_pct"].shift(-1)
        diff["temperature_next"] = diff["temperature_c"].shift(-1)
        diff["mass_loss_rate"] = (diff["mass_pct_next"] - diff["mass_pct"]) / (
            diff["temperature_next"] - diff["temperature_c"]
        )
        diff["mass_loss_rate"] = diff["mass_loss_rate"].abs()
        diff = diff.dropna(subset=["mass_loss_rate"])
        if not diff.empty:
            idx = diff["mass_loss_rate"].idxmax()
            mass_events.append(
                {
                    "event": "max_mass_loss_rate",
                    "temperature_c": float(diff.loc[idx, "temperature_c"]),
                    "mass_pct": float(diff.loc[idx, "mass_pct"]),
                }
            )

    events_df = pd.DataFrame(mass_events).sort_values("temperature_c").reset_index(drop=True)

    return RegolithThermalBundle(
        tg_curve=tg_curve,
        ega_curve=ega_raw,
        ega_long=ega_long,
        gas_peaks=peaks_df,
        mass_events=events_df,
    )
