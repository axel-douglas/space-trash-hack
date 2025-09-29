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
    "load_regolith_particle_size",
    "load_regolith_spectra",
    "load_regolith_thermogravimetry",
    "RegolithCharacterization",
    "load_regolith_characterization",
    "L2LParameters",
    "load_l2l_parameters",
    "OfficialFeaturesBundle",
    "official_features_bundle",
    "lookup_official_feature_values",
    "REGOLITH_VECTOR",
    "GAS_MEAN_YIELD",
    "MEAN_REUSE",
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


def _log_interp_percentile(diameters: np.ndarray, cdf: np.ndarray, target: float) -> float:
    """Return the diameter at *target* cumulative percent finer using log interpolation."""

    if diameters.size == 0 or cdf.size == 0:
        return float("nan")

    mask = np.isfinite(diameters) & np.isfinite(cdf)
    if not np.any(mask):
        return float("nan")

    diameters = diameters[mask]
    cdf = cdf[mask]

    if not np.all(np.diff(cdf) >= 0):
        order = np.argsort(cdf)
        cdf = cdf[order]
        diameters = diameters[order]

    unique_cdf, unique_idx = np.unique(cdf, return_index=True)
    diameters = diameters[unique_idx]

    if target <= unique_cdf[0]:
        return float(diameters[0])
    if target >= unique_cdf[-1]:
        return float(diameters[-1])

    log_diam = np.log(diameters)
    interpolated = np.interp(target, unique_cdf, log_diam)
    return float(np.exp(interpolated))


def _log_size_slope(diameters: np.ndarray, cdf: np.ndarray) -> float:
    """Return the slope of log10(size) vs. log10(percent finer) for the central distribution."""

    mask = (
        np.isfinite(diameters)
        & np.isfinite(cdf)
        & (diameters > 0)
        & (cdf > 0)
        & (cdf < 100)
    )
    if not np.any(mask):
        return float("nan")

    diameters = diameters[mask]
    cdf = cdf[mask] / 100.0

    central = (cdf >= 0.1) & (cdf <= 0.9)
    if np.count_nonzero(central) >= 2:
        diameters = diameters[central]
        cdf = cdf[central]

    if diameters.size < 2:
        return float("nan")

    x = np.log10(diameters)
    y = np.log10(np.clip(cdf, 1e-6, 1.0))
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _mass_loss_between(
    temperatures: np.ndarray, mass: np.ndarray, start: float, stop: float
) -> float:
    """Return the mass loss percentage between *start* and *stop* temperatures."""

    if temperatures.size == 0 or mass.size == 0:
        return float("nan")

    ordered = np.argsort(temperatures)
    temperatures = temperatures[ordered]
    mass = mass[ordered]

    lower = float(np.interp(start, temperatures, mass))
    upper = float(np.interp(stop, temperatures, mass))
    return max(0.0, lower - upper)


@lru_cache(maxsize=1)
def load_regolith_particle_size() -> tuple[pl.DataFrame, Dict[str, float]]:
    """Return the MGS-1 particle size distribution and derived metrics."""

    path = resolve_dataset_path("fig3_psizeData.csv")
    if path is None or not path.exists():
        empty = pl.DataFrame(
            {
                "diameter_microns": pl.Series(dtype=pl.Float64),
                "percent_retained": pl.Series(dtype=pl.Float64),
                "percent_channel": pl.Series(dtype=pl.Float64),
            }
        )
        return empty, {}

    frame = pl.read_csv(path).rename(
        {
            "Diameter (microns)": "diameter_microns",
            "% Retained": "percent_retained",
            "% Channel": "percent_channel",
        }
    )

    frame = frame.select(
        [
            pl.col("diameter_microns").cast(pl.Float64),
            pl.col("percent_retained").cast(pl.Float64),
            pl.col("percent_channel").cast(pl.Float64),
        ]
    ).sort("diameter_microns", descending=True)

    frame = frame.with_columns(
        [
            (pl.col("percent_channel") / 100.0).alias("fraction_channel"),
            pl.col("percent_channel").cum_sum().alias("cumulative_percent_finer"),
            pl.col("percent_retained").alias("cumulative_percent_retained"),
            (100.0 - pl.col("percent_retained")).alias("percent_finer_than"),
        ]
    )

    metric_frame = frame.filter(pl.col("percent_channel") > 0).select(
        "diameter_microns", "cumulative_percent_finer"
    )

    metrics: Dict[str, float] = {}
    if metric_frame.height > 0:
        diameters = metric_frame.get_column("diameter_microns").to_numpy()
        cdf = metric_frame.get_column("cumulative_percent_finer").to_numpy()
        metrics.update(
            {
                "d10_microns": _log_interp_percentile(diameters, cdf, 90.0),
                "d50_microns": _log_interp_percentile(diameters, cdf, 50.0),
                "d90_microns": _log_interp_percentile(diameters, cdf, 10.0),
                "log_slope_fraction_finer": _log_size_slope(diameters, cdf),
            }
        )

    return frame, metrics


@lru_cache(maxsize=1)
def load_regolith_spectra() -> tuple[pl.DataFrame, Dict[str, float]]:
    """Return reflectance spectra for the regolith simulants with summary metrics."""

    path = resolve_dataset_path("fig4_spectralData.csv")
    if path is None or not path.exists():
        return pl.DataFrame(), {}

    frame = pl.read_csv(path).rename(
        {
            "Wavelength (nm)": "wavelength_nm",
            "MMS1": "reflectance_mms1",
            "MMS2": "reflectance_mms2",
            "JSC Mars-1": "reflectance_jsc_mars_1",
            "MGS-1 Prototype": "reflectance_mgs_1",
        }
    )

    frame = frame.select(
        [
            pl.col("wavelength_nm").cast(pl.Float64),
            pl.col("reflectance_mms1").cast(pl.Float64),
            pl.col("reflectance_mms2").cast(pl.Float64),
            pl.col("reflectance_jsc_mars_1").cast(pl.Float64),
            pl.col("reflectance_mgs_1").cast(pl.Float64),
        ]
    ).sort("wavelength_nm")

    metrics: Dict[str, float] = {}
    for column in frame.columns:
        if column == "wavelength_nm":
            continue
        metrics[f"mean_{column}"] = float(frame.get_column(column).mean())

    window = frame.filter(
        (pl.col("wavelength_nm") >= 700.0) & (pl.col("wavelength_nm") <= 1000.0)
    )
    if window.height >= 2:
        wavelengths = window.get_column("wavelength_nm").to_numpy()
        for column in window.columns:
            if column == "wavelength_nm":
                continue
            values = window.get_column(column).to_numpy()
            slope = float(np.polyfit(wavelengths, values, 1)[0])
            metrics[f"slope_{column}_700_1000"] = slope

    return frame, metrics


@lru_cache(maxsize=1)
def load_regolith_thermogravimetry() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    Dict[str, float],
    Dict[str, float],
]:
    """Return thermogravimetric and evolved gas analysis data with summaries."""

    tg_path = resolve_dataset_path("fig5_tgData.csv")
    ega_path = resolve_dataset_path("fig5_egaData.csv")

    if tg_path is None or not tg_path.exists():
        return pl.DataFrame(), pl.DataFrame(), {}, {}

    tg_frame = (
        pl.read_csv(tg_path, encoding="latin1")
        .rename({"Temperature (¡C)": "temperature_c", "Mass (%)": "mass_percent"})
        .select(
            [
                pl.col("temperature_c").cast(pl.Float64),
                pl.col("mass_percent").cast(pl.Float64),
            ]
        )
        .sort("temperature_c")
    )

    ega_metrics: Dict[str, float] = {}
    ega_frame = pl.DataFrame()
    if ega_path is not None and ega_path.exists():
        ega_frame = (
            pl.read_csv(ega_path, encoding="latin1")
            .rename({"Temperature (¡C)": "temperature_c"})
            .select([pl.all().cast(pl.Float64)])
            .sort("temperature_c")
        )

        if ega_frame.height > 0:
            temperatures = ega_frame.get_column("temperature_c").to_numpy()
            for column in ega_frame.columns:
                if column == "temperature_c":
                    continue
                series = ega_frame.get_column(column).to_numpy()
                if series.size == 0:
                    continue
                peak_idx = int(np.argmax(series))
                ega_metrics[f"peak_temperature_{slugify(column)}"] = float(
                    temperatures[peak_idx]
                )

    temperatures = tg_frame.get_column("temperature_c").to_numpy()
    mass = tg_frame.get_column("mass_percent").to_numpy()

    thermal_metrics: Dict[str, float] = {}
    if temperatures.size > 0 and mass.size > 0:
        initial_mass = float(mass[0])
        final_mass = float(mass[-1])
        thermal_metrics["mass_loss_total_percent"] = max(0.0, initial_mass - final_mass)
        thermal_metrics["residual_mass_percent"] = final_mass

        ranges = (
            (30.0, 200.0),
            (200.0, 400.0),
            (400.0, 600.0),
            (600.0, min(800.0, float(temperatures[-1]))),
        )
        for start, stop in ranges:
            if stop <= start:
                continue
            loss = _mass_loss_between(temperatures, mass, start, stop)
            key = f"mass_loss_{int(start)}_{int(stop)}_c"
            thermal_metrics[key] = loss

    return tg_frame, ega_frame, thermal_metrics, ega_metrics


@dataclass(frozen=True)
class RegolithCharacterization:
    particle_size: pl.DataFrame
    particle_metrics: Mapping[str, float]
    spectra: pl.DataFrame
    spectral_metrics: Mapping[str, float]
    thermogravimetry: pl.DataFrame
    evolved_gas: pl.DataFrame
    thermal_metrics: Mapping[str, float]
    gas_release_peaks: Mapping[str, float]


@lru_cache(maxsize=1)
def load_regolith_characterization() -> RegolithCharacterization:
    """Return a cached bundle with regolith particle, spectral and thermal summaries."""

    particle_size, particle_metrics = load_regolith_particle_size()
    spectra, spectral_metrics = load_regolith_spectra()
    tg_frame, ega_frame, thermal_metrics, ega_metrics = load_regolith_thermogravimetry()

    return RegolithCharacterization(
        particle_size=particle_size,
        particle_metrics=particle_metrics,
        spectra=spectra,
        spectral_metrics=spectral_metrics,
        thermogravimetry=tg_frame,
        evolved_gas=ega_frame,
        thermal_metrics=thermal_metrics,
        gas_release_peaks=ega_metrics,
    )


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
