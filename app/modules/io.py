# app/modules/io.py
from __future__ import annotations

import copy
import json
import math
import errno
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence
from datetime import datetime

import pandas as pd
import polars as pl

from .generator import prepare_waste_frame
from .data_sources import official_features_bundle
from .paths import DATA_ROOT
from .problematic import problematic_mask
from .schema import ALUMINIUM_SAMPLE_COLUMNS, POLYMER_SAMPLE_COLUMNS

DATA_DIR = DATA_ROOT


class MissingDatasetError(FileNotFoundError):
    """Raised when a required dataset file is missing from disk."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        super().__init__(f"No se encontró el dataset requerido: {self.path}")


INSTALL_DATA_HINT = "Instalá los datasets ejecutando `python scripts/download_datasets.py`."


def format_missing_dataset_message(error: MissingDatasetError) -> str:
    return f"{error} {INSTALL_DATA_HINT}"


def get_last_modified(path: Path | str) -> datetime | None:
    """Return the last modification timestamp for ``path`` if it exists."""

    candidate = Path(path)
    try:
        stat_result = candidate.stat()
    except FileNotFoundError:
        return None
    return datetime.fromtimestamp(stat_result.st_mtime)

# Archivos que proporcionó NASA (ustedes)
WASTE_CSV   = DATA_DIR / "waste_inventory_sample.csv"
PROC_CSV    = DATA_DIR / "process_catalog.csv"
TARGETS_JSON= DATA_DIR / "targets_presets.json"

PROBLEM_TAGS = {
    "multilayer": "Lámina multicapa (PE/PET/Al)",
    "thermal": "Pouches térmicos",
    "ctb": "EVA / Cargo Transfer Bag",
    "closed_cell": "Espuma técnica (ZOTEK F30)",
    "nitrile": "Guantes de nitrilo",
    "struts": "Estructuras/estrús Al",
}

def _ensure_exists():
    for p in [WASTE_CSV, PROC_CSV, TARGETS_JSON]:
        if not p.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(p))


def _to_missing_dataset_error(
    exc: FileNotFoundError, default_path: Path
) -> MissingDatasetError:
    filename = getattr(exc, "filename", None)
    if not filename and exc.args:
        filename = exc.args[0]
    path = Path(filename) if filename else default_path
    return MissingDatasetError(path)

@lru_cache(maxsize=1)
def _load_waste_df_cached() -> pd.DataFrame:
    """
    Lee el CSV NASA `waste_inventory_sample.csv` con columnas:
      id,category,material_family,mass_kg,volume_l,flags
    Devuelve un DF ampliado con columnas internas que usa la UI:
      material, kg, notes
    y preserva la proveniencia en columnas _source_*.
    """
    try:
        _ensure_exists()
        base_df = pd.read_csv(WASTE_CSV)
    except FileNotFoundError as exc:  # pragma: no cover - exercised in error tests
        raise _to_missing_dataset_error(exc, WASTE_CSV) from exc
    source_snapshot = base_df.copy(deep=True)

    string_columns = [
        "id",
        "category",
        "material",
        "material_family",
        "flags",
        "key_materials",
        "notes",
    ]
    for column in string_columns:
        if column in base_df.columns:
            base_df[column] = base_df[column].fillna("").astype(str)

    numeric_columns = [
        "mass_kg",
        "volume_l",
        "moisture_pct",
        "pct_mass",
        "pct_volume",
        "difficulty_factor",
    ]
    for column in numeric_columns:
        if column in base_df.columns:
            base_df[column] = pd.to_numeric(base_df[column], errors="coerce").fillna(0.0)

    if "mass_kg" in base_df.columns:
        base_df["kg"] = base_df["mass_kg"].fillna(0.0)
    else:
        base_df["kg"] = 0.0

    if "volume_l" in base_df.columns:
        base_df["volume_l"] = base_df["volume_l"].fillna(0.0)

    category_display = base_df.get("category", "").astype(str).str.strip()
    family_display = base_df.get("material_family", "").astype(str).str.strip()
    material_display = category_display.where(family_display.eq(""), category_display + " — " + family_display)
    base_df["material_display"] = material_display.replace({" — ": ""})

    prepared = prepare_waste_frame(base_df)
    result = prepared.copy(deep=True)

    try:
        bundle = official_features_bundle()
    except Exception:  # pragma: no cover - defensive guard in case datasets are missing
        bundle = None

    if bundle and getattr(bundle, "direct_map", None):
        direct_map = bundle.direct_map
        polymer_columns: set[str] = set()
        for payload in direct_map.values():
            if not isinstance(payload, dict):
                continue
            for column in payload.keys():
                if column.startswith("pc_") or column.startswith("aluminium_"):
                    polymer_columns.add(column)

        if polymer_columns:
            profile_df = pd.DataFrame.from_dict(direct_map, orient="index")
            profile_df.index.name = "_official_match_key"
            available = [column for column in polymer_columns if column in profile_df.columns]

            if available:
                profile_df = profile_df.reset_index()[["_official_match_key", *available]]
                merged = result.merge(profile_df, on="_official_match_key", how="left", suffixes=("", "__bundle"))

                for column in available:
                    bundle_column = f"{column}__bundle"
                    if bundle_column not in merged.columns:
                        continue
                    if column in merged.columns:
                        merged[column] = merged[column].fillna(merged[bundle_column])
                    else:
                        merged[column] = merged[bundle_column]
                    merged = merged.drop(columns=[bundle_column])

                result = merged

    for column, values in source_snapshot.items():
        source_column = f"_source_{column}"
        if source_column in result.columns:
            continue
        if values.dtype == object:
            result[source_column] = values.fillna("").astype(str)
        else:
            result[source_column] = values

    if "_problematic" not in result.columns:
        result["_problematic"] = problematic_mask(result)

    return result


def load_waste_df() -> pd.DataFrame:
    """Return a defensive copy of the cached waste inventory."""

    try:
        cached = _load_waste_df_cached()
    except MissingDatasetError:
        raise
    except FileNotFoundError as exc:
        raise _to_missing_dataset_error(exc, WASTE_CSV) from exc
    return cached.copy(deep=True)

def save_waste_df(df: pd.DataFrame | pl.DataFrame) -> None:
    """
    Persiste en el MISMO esquema NASA, para que el jurado vea que seguimos
    usando su formato. Tomamos de la UI: material (parseado), kg, notes.
    """
    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df, include_index=False)
    elif isinstance(df, pl.DataFrame):
        pl_df = df.clone()
    else:
        raise TypeError("save_waste_df requiere pandas.DataFrame o polars.DataFrame")

    def _expr_for(names: list[str], default: pl.Expr) -> pl.Expr:
        for name in names:
            if name in pl_df.columns:
                return pl.col(name)
        return default

    out = pl_df.select(
        [
            _expr_for(["_source_id", "id"], pl.lit(None, dtype=pl.Utf8)).alias("id"),
            _expr_for(["_source_category", "category"], pl.lit(None, dtype=pl.Utf8)).alias(
                "category"
            ),
            _expr_for(
                ["_source_material_family", "material_family"],
                pl.lit(None, dtype=pl.Utf8),
            ).alias("material_family"),
            _expr_for(["kg", "mass_kg"], pl.lit(None, dtype=pl.Float64))
            .cast(pl.Float64, strict=False)
            .alias("mass_kg"),
            _expr_for(["_source_volume_l", "volume_l"], pl.lit(None, dtype=pl.Float64))
            .cast(pl.Float64, strict=False)
            .alias("volume_l"),
            _expr_for(["notes", "flags", "_source_flags"], pl.lit("", dtype=pl.Utf8))
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .alias("flags"),
            _expr_for(["material"], pl.lit(None, dtype=pl.Utf8)).alias("_material"),
        ]
    )

    out = out.with_columns(
        [
            pl.col("category").cast(pl.Utf8, strict=False),
            pl.col("material_family").cast(pl.Utf8, strict=False),
            pl.col("flags").cast(pl.Utf8, strict=False).fill_null(""),
            pl.col("mass_kg").cast(pl.Float64, strict=False),
            pl.col("volume_l").cast(pl.Float64, strict=False),
        ]
    )

    material_clean = pl.col("_material").cast(pl.Utf8, strict=False).fill_null("")
    parts = material_clean.str.split("—")
    parts_len = parts.list.len()

    category_needs = pl.col("category").is_null() | pl.col("category").str.strip_chars().eq("")
    material_needs = (
        pl.col("material_family").is_null()
        | pl.col("material_family").str.strip_chars().eq("")
    )

    parsed_category = pl.when(parts_len >= 2).then(
        parts.list.get(0, null_on_oob=True).str.strip_chars()
    ).otherwise(pl.lit("unknown"))
    parsed_material = pl.when(parts_len >= 2).then(
        parts.list.get(1, null_on_oob=True).str.strip_chars()
    ).otherwise(material_clean.str.strip_chars())

    out = out.with_columns(
        [
            pl.when(category_needs).then(parsed_category).otherwise(pl.col("category")).alias(
                "category"
            ),
            pl.when(material_needs)
            .then(parsed_material)
            .otherwise(pl.col("material_family"))
            .alias("material_family"),
        ]
    )

    out = out.drop("_material")

    id_str = pl.col("id").cast(pl.Utf8, strict=False)
    id_missing = id_str.is_null() | id_str.str.strip_chars().eq("")
    out = out.with_columns(
        [
            pl.when(id_missing)
            .then(pl.format("W{}", 1000 + id_missing.cast(pl.Int64).cum_sum()))
            .otherwise(id_str)
            .alias("id"),
        ]
    )

    out.write_csv(WASTE_CSV, include_header=True)
    invalidate_waste_cache()


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def format_polymer_profile(row: pd.Series) -> str:
    """Compact textual description for external polymer properties."""

    parts: list[str] = []

    for column in POLYMER_SAMPLE_COLUMNS:
        label = str(row.get(column) or "").strip()
        if label:
            parts.append(f"Ref {label}")
            break

    density = _safe_float(row.get("pc_density_density_g_per_cm3"))
    if density:
        parts.append(f"ρ {density:.2f} g/cm³")

    tensile = _safe_float(row.get("pc_mechanics_tensile_strength_mpa"))
    if tensile:
        parts.append(f"σₜ {tensile:.0f} MPa")

    modulus = _safe_float(row.get("pc_mechanics_modulus_gpa"))
    if modulus:
        parts.append(f"E {modulus:.1f} GPa")

    glass_transition = _safe_float(row.get("pc_thermal_glass_transition_c"))
    if glass_transition:
        parts.append(f"Tg {glass_transition:.0f} °C")

    ignition = _safe_float(row.get("pc_ignition_ignition_temperature_c"))
    if ignition:
        parts.append(f"Ign. {ignition:.0f} °C")

    burn_time = _safe_float(row.get("pc_ignition_burn_time_min"))
    if burn_time:
        parts.append(f"Burn {burn_time:.1f} min")

    return " · ".join(parts)


def format_aluminium_profile(row: pd.Series) -> str:
    """Compact textual description for aluminium properties."""

    parts: list[str] = []

    route = str(row.get("aluminium_processing_route") or "").strip()
    class_id = str(row.get("aluminium_class_id") or "").strip()
    if route and class_id:
        parts.append(f"{route} · Clase {class_id}")
    elif route:
        parts.append(route)
    elif class_id:
        parts.append(f"Clase {class_id}")

    tensile = _safe_float(row.get("aluminium_tensile_strength_mpa"))
    if tensile:
        parts.append(f"σₜ {tensile:.0f} MPa")

    yield_strength = _safe_float(row.get("aluminium_yield_strength_mpa"))
    if yield_strength:
        parts.append(f"σᵧ {yield_strength:.0f} MPa")

    elongation = _safe_float(row.get("aluminium_elongation_pct"))
    if elongation:
        parts.append(f"ε {elongation:.0f}%")

    return " · ".join(parts)


def format_composition_summary(row: pd.Series, columns: Sequence[str]) -> str:
    """Return a short composition summary using NASA percentage columns."""

    parts: list[str] = []
    for column in columns:
        value = _safe_float(row.get(column))
        if value is None or value <= 0:
            continue
        label = column.replace("_pct", "").replace("_", " ")
        parts.append(f"{label} {value:.0f}%")
    return ", ".join(parts)


def format_mission_bundle(
    row: pd.Series,
    columns: Sequence[str],
    labels: Mapping[str, str] | None = None,
) -> str:
    """Return a joined mission mass summary for the provided row."""

    parts: list[str] = []
    for column in columns:
        value = _safe_float(row.get(column))
        if value is None or value <= 0:
            continue
        if labels and column in labels:
            label = labels[column]
        else:
            label = column.replace("summary_", "").replace("_", " ")
        parts.append(f"{label}: {value:.1f} kg")
    return " · ".join(parts)

@lru_cache(maxsize=1)
def _load_process_df_cached() -> pd.DataFrame:
    try:
        _ensure_exists()
        return pd.read_csv(PROC_CSV)
    except FileNotFoundError as exc:  # pragma: no cover - exercised in error tests
        raise _to_missing_dataset_error(exc, PROC_CSV) from exc


def load_process_catalog() -> pd.DataFrame:
    """Alias legada para compatibilidad."""
    return load_process_df()

@lru_cache(maxsize=1)
def _load_targets_cached() -> list[dict]:
    try:
        _ensure_exists()
        return json.loads(TARGETS_JSON.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - exercised in error tests
        raise _to_missing_dataset_error(exc, TARGETS_JSON) from exc


def load_targets() -> list[dict]:
    try:
        cached = _load_targets_cached()
    except MissingDatasetError:
        raise
    except FileNotFoundError as exc:
        raise _to_missing_dataset_error(exc, TARGETS_JSON) from exc
    return copy.deepcopy(cached)


def load_process_df() -> pd.DataFrame:
    try:
        cached = _load_process_df_cached()
    except MissingDatasetError:
        raise
    except FileNotFoundError as exc:
        raise _to_missing_dataset_error(exc, PROC_CSV) from exc
    return cached.copy()


def invalidate_waste_cache() -> None:
    _load_waste_df_cached.cache_clear()


def invalidate_process_cache() -> None:
    _load_process_df_cached.cache_clear()


def invalidate_targets_cache() -> None:
    _load_targets_cached.cache_clear()


def invalidate_all_io_caches() -> None:
    invalidate_waste_cache()
    invalidate_process_cache()
    invalidate_targets_cache()


__all__ = [
    "MissingDatasetError",
    "format_missing_dataset_message",
    "INSTALL_DATA_HINT",
    "load_waste_df",
    "save_waste_df",
    "load_process_df",
    "load_process_catalog",
    "load_targets",
    "invalidate_all_io_caches",
    "format_polymer_profile",
    "format_aluminium_profile",
    "format_composition_summary",
    "format_mission_bundle",
]
