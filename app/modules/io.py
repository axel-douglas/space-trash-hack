# app/modules/io.py
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

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

    flags_str = pl.col("flags").cast(pl.Utf8, strict=False).fill_null("")
    category_str = pl.col("category").cast(pl.Utf8, strict=False).fill_null("")
    material_family_str = (
        pl.col("material_family").cast(pl.Utf8, strict=False).fill_null("")
    )

    category_lower = category_str.str.to_lowercase()
    flags_lower = flags_str.str.to_lowercase()

    problem_exprs = [
        category_lower.str.contains("pouches"),
        category_lower.str.contains("foam"),
        category_lower.str.contains("eva"),
        category_lower.str.contains("glove"),
        category_lower.str.contains("aluminum"),
    ]
    problem_exprs.extend(
        flags_lower.str.contains(tag) for tag in PROBLEM_TAGS.keys()
    )

    problematic_expr = pl.any_horizontal(*problem_exprs)

    lf = pl.scan_csv(WASTE_CSV)
    result = (
        lf.with_columns(
            [
                pl.concat_str(
                    [category_str, pl.lit(" — "), material_family_str],
                    separator="",
                ).alias("material"),
                pl.col("mass_kg").cast(pl.Float64, strict=False).alias("kg"),
                flags_str.alias("notes"),
                problematic_expr.alias("_problematic"),
                pl.col("id").alias("_source_id"),
                pl.col("category").alias("_source_category"),
                pl.col("material_family").alias("_source_material_family"),
                pl.col("volume_l").alias("_source_volume_l"),
                pl.col("flags").alias("_source_flags"),
            ]
        )
        .select(
            [
                "material",
                "kg",
                "notes",
                "_source_id",
                "_source_category",
                "_source_material_family",
                "_source_volume_l",
                "_source_flags",
                "_problematic",
            ]
        )
        .collect()
    )

    return result.to_pandas(use_pyarrow_extension_array=False)


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
