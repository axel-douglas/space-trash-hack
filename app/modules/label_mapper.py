"""Utilities to map curated labels (gold) to generated recipes."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple

import pandas as pd

# Paths -----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = ROOT / "datasets"
GOLD_DIR = DATASETS_ROOT / "gold"
GOLD_LABELS_PATH = GOLD_DIR / "labels.parquet"

# Targets tracked in curated datasets
TARGET_COLUMNS: Tuple[str, ...] = (
    "rigidez",
    "estanqueidad",
    "energy_kwh",
    "water_l",
    "crew_min",
)
CLASS_TARGET_COLUMNS: Tuple[str, ...] = (
    "tightness_pass",
    "rigidity_level",
)

_LABELS_CACHE: pd.DataFrame | None = None
_LABELS_CACHE_PATH: Path | None = None

LOGGER = logging.getLogger(__name__)


def _normalise_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.upper().strip()


def _load_labels_table(path: Path | None = None) -> pd.DataFrame:
    """Read and cache the curated labels table."""

    global _LABELS_CACHE, _LABELS_CACHE_PATH
    target_path = Path(path) if path is not None else GOLD_LABELS_PATH

    if path is None and not target_path.exists():
        try:
            from app.modules import data_build

            data_build.ensure_gold_dataset()
        except Exception:  # pragma: no cover - visibility of bootstrap errors
            LOGGER.exception(
                "Failed to ensure gold dataset at %s", target_path.parent
            )
            _LABELS_CACHE = pd.DataFrame()
            _LABELS_CACHE_PATH = target_path
            return _LABELS_CACHE

    if _LABELS_CACHE is not None and _LABELS_CACHE_PATH == target_path:
        return _LABELS_CACHE

    if not target_path.exists():
        _LABELS_CACHE = pd.DataFrame()
        _LABELS_CACHE_PATH = target_path
        return _LABELS_CACHE

    try:
        table = pd.read_parquet(target_path)
    except Exception as exc:  # pragma: no cover - visibility of IO errors
        raise RuntimeError(f"No se pudo leer parquet {target_path}: {exc}") from exc

    if table.empty:
        _LABELS_CACHE = pd.DataFrame()
        _LABELS_CACHE_PATH = target_path
        return _LABELS_CACHE

    required = {"recipe_id", "process_id"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas {sorted(missing)} en {target_path}"
        )

    table = table.copy()
    table["recipe_id"] = table["recipe_id"].apply(_normalise_key)
    table["process_id"] = table["process_id"].apply(_normalise_key)

    numeric_columns = [
        *TARGET_COLUMNS,
        "label_weight",
        "weight",
        "sample_weight",
        *[col for col in table.columns if col.startswith("conf_lo_")],
        *[col for col in table.columns if col.startswith("conf_hi_")],
    ]
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    for column in CLASS_TARGET_COLUMNS:
        if column in table.columns:
            table[column] = (
                pd.to_numeric(table[column], errors="coerce").round().astype("Int64")
            )

    if "label_source" in table.columns:
        table["label_source"] = table["label_source"].fillna("measured").astype(str)
    elif "provenance" in table.columns:
        table["label_source"] = table["provenance"].fillna("measured").astype(str)
    else:
        table["label_source"] = "measured"

    if "provenance" not in table.columns and "label_source" in table.columns:
        table["provenance"] = table["label_source"].astype(str)

    table = table.drop_duplicates(subset=["recipe_id", "process_id"], keep="last")
    table = table.set_index(["recipe_id", "process_id"], drop=False)
    _LABELS_CACHE = table
    _LABELS_CACHE_PATH = target_path
    return table


def load_curated_labels(path: Path | None = None) -> pd.DataFrame:
    """Return a copy of the curated labels table."""

    table = _load_labels_table(path)
    return table.copy(deep=False)


def derive_recipe_id(
    materials: pd.DataFrame | Iterable[Any] | None,
    process: Mapping[str, Any] | pd.Series | str | None,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Derive a deterministic recipe identifier based on materials and process."""

    params = params or {}
    recipe_token = params.get("recipe_id")
    if recipe_token:
        token = _normalise_key(recipe_token)
        if token:
            return token

    if isinstance(process, pd.Series):
        process_id = _normalise_key(process.get("process_id"))
    elif isinstance(process, Mapping):
        process_id = _normalise_key(process.get("process_id"))
    else:
        process_id = _normalise_key(process)

    if not process_id:
        process_id = _normalise_key(params.get("process_id"))

    if isinstance(materials, pd.DataFrame):
        source_ids = materials.get("_source_id")
        if source_ids is None:
            source_ids = materials.index.astype(str)
        tokens = "|".join(sorted(map(str, source_ids)))
    elif isinstance(materials, Iterable) and not isinstance(materials, (str, bytes)):
        tokens = "|".join(sorted(map(str, materials)))
    else:
        tokens = ""

    if not tokens:
        extra = params.get("materials")
        if isinstance(extra, Iterable) and not isinstance(extra, (str, bytes)):
            tokens = "|".join(sorted(map(str, extra)))

    if not tokens or not process_id:
        return ""

    raw = f"{process_id}|{tokens}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return digest.upper()


def lookup_labels(
    materials: pd.DataFrame | Iterable[Any] | None,
    process_id: str | None,
    params: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return curated targets and metadata for the provided combination."""

    params = dict(params or {})
    recipe_id = derive_recipe_id(materials, process_id, params)

    process_norm = _normalise_key(process_id or params.get("process_id"))
    if not recipe_id or not process_norm:
        return {}, {}

    table = _load_labels_table()
    if table.empty:
        return {}, {}

    key = (recipe_id, process_norm)
    if key not in table.index:
        return {}, {}

    row = table.loc[key]
    if isinstance(row, pd.DataFrame):  # pragma: no cover - defensive: duplicate entries
        row = row.iloc[-1]

    targets: dict[str, Any] = {}
    for column in TARGET_COLUMNS:
        if column in row and pd.notna(row[column]):
            targets[column] = float(row[column])

    for column in CLASS_TARGET_COLUMNS:
        if column in row and pd.notna(row[column]):
            targets[column] = int(row[column])

    metadata: dict[str, Any] = {}
    for name in ("label_source", "label_weight", "provenance", "sample_weight", "weight"):
        if name in row and pd.notna(row[name]):
            value = row[name]
            if name.endswith("weight"):
                try:
                    metadata[name] = float(value)
                except (TypeError, ValueError):
                    continue
            else:
                metadata[name] = str(value)

    confidence: dict[str, Tuple[float, float]] = {}
    for column in row.index:
        if not isinstance(column, str) or not column.startswith("conf_lo_"):
            continue
        target = column.removeprefix("conf_lo_")
        lo_val = row[column]
        hi_val = row.get(f"conf_hi_{target}")
        if pd.notna(lo_val) and pd.notna(hi_val):
            confidence[target] = (float(lo_val), float(hi_val))
        elif pd.notna(lo_val):
            confidence[target] = (float(lo_val), float(lo_val))
    if confidence:
        metadata["confidence_intervals"] = confidence

    if "provenance" not in metadata and "label_source" in metadata:
        metadata["provenance"] = str(metadata["label_source"])
    if "label_source" not in metadata and "provenance" in metadata:
        metadata["label_source"] = str(metadata["provenance"])

    metadata["recipe_id"] = recipe_id
    metadata["process_id"] = process_norm
    return targets, metadata


__all__ = [
    "lookup_labels",
    "derive_recipe_id",
    "load_curated_labels",
    "TARGET_COLUMNS",
    "CLASS_TARGET_COLUMNS",
]
