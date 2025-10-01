# app/modules/io.py
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import polars as pl

from .generator import prepare_waste_frame
from .paths import DATA_ROOT

DATA_DIR = DATA_ROOT

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
            raise FileNotFoundError(f"Falta archivo de datos: {p}")

@lru_cache(maxsize=1)
def _load_waste_df_cached() -> pd.DataFrame:
    """
    Lee el CSV NASA `waste_inventory_sample.csv` con columnas:
      id,category,material_family,mass_kg,volume_l,flags
    Devuelve un DF ampliado con columnas internas que usa la UI:
      material, kg, notes
    y preserva la proveniencia en columnas _source_*.
    """
    _ensure_exists()

    base_df = pd.read_csv(WASTE_CSV)
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

    for column, values in source_snapshot.items():
        source_column = f"_source_{column}"
        if source_column in result.columns:
            continue
        if values.dtype == object:
            result[source_column] = values.fillna("").astype(str)
        else:
            result[source_column] = values

    if "_problematic" not in result.columns:
        flags_lower = result.get("flags", "").astype(str).str.lower()
        category_lower = result.get("category", "").astype(str).str.lower()
        material_lower = result.get("material", "").astype(str).str.lower()
        problem_mask = (
            category_lower.str.contains("pouches")
            | category_lower.str.contains("foam")
            | category_lower.str.contains("eva")
            | category_lower.str.contains("glove")
            | material_lower.str.contains("aluminum")
        )
        for tag in PROBLEM_TAGS.keys():
            problem_mask = problem_mask | flags_lower.str.contains(tag)
        result["_problematic"] = problem_mask

    return result


def load_waste_df() -> pd.DataFrame:
    """Return a defensive copy of the cached waste inventory."""
    return _load_waste_df_cached().copy(deep=True)

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

@lru_cache(maxsize=1)
def _load_process_df_cached() -> pd.DataFrame:
    _ensure_exists()
    return pd.read_csv(PROC_CSV)


def load_process_catalog() -> pd.DataFrame:
    """Alias legada para compatibilidad."""
    return load_process_df()

@lru_cache(maxsize=1)
def _load_targets_cached() -> list[dict]:
    _ensure_exists()
    return json.loads(TARGETS_JSON.read_text(encoding="utf-8"))


def load_targets() -> list[dict]:
    return copy.deepcopy(_load_targets_cached())


def load_process_df() -> pd.DataFrame:
    return _load_process_df_cached().copy()


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
