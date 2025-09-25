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

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Tuple

import numpy as np
import pandas as pd

from app.modules.label_mapper import derive_recipe_id, lookup_labels
from app.modules.ranking import derive_auxiliary_signals, score_recipe

try:  # Lazy import to avoid circular dependency during training pipelines
    from app.modules.ml_models import MODEL_REGISTRY
except Exception:  # pragma: no cover - fallback when models are not available
    MODEL_REGISTRY = None

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
LOGS_ROOT = Path(__file__).resolve().parents[2] / "data" / "logs"


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


def _append_inference_log(
    input_features: Dict[str, Any],
    prediction: Dict[str, Any] | None,
    uncertainty: Dict[str, Any] | None,
    model_registry: Any | None,
) -> None:
    """Persist an inference event as a Parquet append."""

    try:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

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

    now = datetime.utcnow()
    event = {
        "timestamp": now.isoformat(),
        "input_features": json.dumps(_to_serializable(input_features or {}), sort_keys=True),
        "prediction": json.dumps(_to_serializable(prediction or {}), sort_keys=True),
        "uncertainty": json.dumps(_to_serializable(uncertainty or {}), sort_keys=True),
        "model_hash": model_hash,
    }

    log_path = LOGS_ROOT / f"inference_{now.strftime('%Y%m%d')}.parquet"

    try:
        new_row = pd.DataFrame([event])
        if log_path.exists():
            existing = pd.read_parquet(log_path)
            new_row = pd.concat([existing, new_row], ignore_index=True)
        new_row.to_parquet(log_path, index=False)
    except Exception:
        return


def _load_regolith_vector() -> Dict[str, float]:
    path = DATASETS_ROOT / "raw" / "mgs1_oxides.csv"
    if path.exists():
        table = pd.read_csv(path)
        return (
            table.assign(oxide=lambda df: df["oxide"].str.lower())
            .set_index("oxide")["wt_percent"]
            .div(100.0)
            .to_dict()
        )
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
    "packaging": "other packaging glove",
    "foam": "other packaging glove",
    "glove": "other packaging glove",
    "other packaging": "other packaging glove",
    "other packaging glove": "other packaging glove",
}


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


class _OfficialFeaturesBundle(NamedTuple):
    value_columns: tuple[str, ...]
    composition_columns: tuple[str, ...]
    direct_map: Dict[str, Dict[str, float]]
    category_tokens: Dict[str, list[tuple[frozenset[str], Dict[str, float]]]]


@lru_cache(maxsize=1)
def _official_features_bundle() -> _OfficialFeaturesBundle:
    if not _OFFICIAL_FEATURES_PATH.exists():
        return _OfficialFeaturesBundle((), (), {}, {})

    table = pd.read_csv(_OFFICIAL_FEATURES_PATH)
    duplicate_suffixes = [column for column in table.columns if column.endswith(".1")]
    if duplicate_suffixes:
        table = table.drop(columns=duplicate_suffixes)
    table = table.loc[:, ~table.columns.duplicated()].copy()
    table["category_norm"] = table["category"].map(_normalize_category)
    table["subitem_norm"] = table["subitem"].map(_normalize_item)
    table["token_set"] = table["subitem_norm"].map(_token_set)
    table["key"] = table["category_norm"] + "|" + table["subitem_norm"]

    excluded = {"category", "subitem", "category_norm", "subitem_norm", "token_set", "key"}
    value_columns = tuple(col for col in table.columns if col not in excluded)
    composition_columns = tuple(
        col
        for col in value_columns
        if col.endswith("_pct") and not col.startswith("subitem_")
    )

    direct_map: Dict[str, Dict[str, float]] = {}
    category_tokens: Dict[str, list[tuple[frozenset[str], Dict[str, float]]]] = {}

    for _, row in table.iterrows():
        payload: Dict[str, float] = {}
        for column in value_columns:
            value = row[column]
            payload[column] = float(value) if pd.notna(value) else float("nan")

        key = str(row["key"])
        direct_map[key] = payload

        category = str(row["category_norm"])
        tokens = row["token_set"]
        category_tokens.setdefault(category, []).append((tokens, payload))

    return _OfficialFeaturesBundle(value_columns, composition_columns, direct_map, category_tokens)


def _lookup_official_feature_values(row: pd.Series) -> Dict[str, float]:
    bundle = _official_features_bundle()
    if not bundle.value_columns:
        return {}

    category = _normalize_category(row.get("category", ""))
    if not category:
        return {}

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
            return payload

    token_candidates = [value for value in candidates if value]
    if not token_candidates:
        return {}

    matches = bundle.category_tokens.get(category)
    if not matches:
        return {}

    for candidate in token_candidates:
        tokens = _token_set(candidate)
        if not tokens:
            continue
        for reference_tokens, payload in matches:
            if tokens.issubset(reference_tokens):
                return payload

    return {}


def _inject_official_features(frame: pd.DataFrame) -> pd.DataFrame:
    bundle = _official_features_bundle()
    if not bundle.value_columns or frame.empty:
        return frame

    records = [_lookup_official_feature_values(row) for _, row in frame.iterrows()]
    if not any(records):
        return frame

    official_df = pd.DataFrame.from_records(records, index=frame.index)
    for column in official_df.columns:
        if column not in frame.columns:
            frame[column] = official_df[column]
        else:
            mask = official_df[column].notna()
            if mask.any():
                frame.loc[mask, column] = official_df.loc[mask, column]

    numeric_candidates = [
        column
        for column in official_df.columns
        if column.endswith(("_kg", "_pct"))
        or column.startswith("category_total")
        or column in {"difficulty_factor", "approx_moisture_pct"}
    ]

    for column in numeric_candidates:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "approx_moisture_pct" in frame.columns:
        mask = frame["approx_moisture_pct"].notna()
        if mask.any():
            frame.loc[mask, "moisture_pct"] = frame.loc[mask, "approx_moisture_pct"]

    return frame


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


def compute_feature_vector(
    picks: pd.DataFrame,
    weights: Iterable[float],
    process: pd.Series,
    regolith_pct: float,
) -> Dict[str, Any]:
    total_kg = max(0.001, float(picks["kg"].sum()))
    base_weights = np.asarray(list(weights), dtype=float)

    tokens = [_material_tokens(row) for _, row in picks.iterrows()]
    categories = [str(row.get("category", "")).lower() for _, row in picks.iterrows()]

    pct_mass = picks.get("pct_mass", 0).to_numpy(dtype=float) / 100.0
    pct_volume = picks.get("pct_volume", 0).to_numpy(dtype=float) / 100.0
    moisture = picks.get("moisture_pct", 0).to_numpy(dtype=float) / 100.0
    difficulty = picks.get("difficulty_factor", 1).to_numpy(dtype=float) / 3.0
    densities = picks.get("density_kg_m3", 0).to_numpy(dtype=float)

    features: Dict[str, Any] = {
        "process_id": str(process["process_id"]),
        "total_mass_kg": total_kg,
        "mass_input_kg": total_kg,
        "num_items": int(len(picks)),
        "density_kg_m3": float(np.dot(base_weights, densities)),
        "moisture_frac": float(np.clip(np.dot(base_weights, moisture), 0.0, 1.0)),
        "difficulty_index": float(np.clip(np.dot(base_weights, difficulty), 0.0, 1.0)),
        "problematic_mass_frac": float(np.clip(np.dot(base_weights, pct_mass), 0.0, 1.0)),
        "problematic_item_frac": float(np.clip(np.dot(base_weights, pct_volume), 0.0, 1.0)),
        "regolith_pct": float(np.clip(regolith_pct, 0.0, 1.0)),
        "packaging_frac": _category_fraction(tuple(categories), base_weights, ("packaging", "food packaging")),
    }

    keyword_map: Dict[str, Tuple[str, ...]] = {
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

    for name, keywords in keyword_map.items():
        features[name] = _keyword_fraction(tuple(tokens), base_weights, keywords)

    bundle = _official_features_bundle()
    official_comp: Dict[str, float] = {}
    if bundle.composition_columns:
        for column in bundle.composition_columns:
            if column not in picks.columns:
                continue
            values = pd.to_numeric(picks[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if not len(values):
                continue
            frac = float(np.dot(base_weights, values / 100.0))
            official_comp[column] = frac

    if official_comp:
        clipped = {key: max(0.0, float(value)) for key, value in official_comp.items()}
        total = sum(clipped.values())
        if total > 1.0 + 1e-6:
            official_comp = {key: value / total for key, value in clipped.items() if total > 0}
        else:
            official_comp = clipped
            official_comp[column] = float(np.clip(frac, 0.0, 1.0))

    def _set_official_fraction(name: str, *columns: str) -> None:
        total = 0.0
        found = False
        for column in columns:
            if column in official_comp:
                total += official_comp[column]
                found = True
        if not found:
            return
        features[name] = float(np.clip(total, 0.0, 1.0))

    if official_comp:
        _set_official_fraction("aluminum_frac", "Aluminum_pct")
        _set_official_fraction("carbon_fiber_frac", "Carbon_Fiber_pct")
        _set_official_fraction("polyethylene_frac", "Polyethylene_pct")
        _set_official_fraction("glove_frac", "Nitrile_pct")
        _set_official_fraction("eva_frac", "Nomex_pct")
        _set_official_fraction("foam_frac", "PVDF_pct")
        _set_official_fraction("multilayer_frac", "EVOH_pct", "PET_pct")

        textile_total = sum(
            official_comp.get(column, 0.0)
            for column in ("Cotton_Cellulose_pct", "Polyester_pct", "Nylon_pct")
        )
        if textile_total > 0:
            features["textile_frac"] = float(np.clip(textile_total, 0.0, 1.0))

        hydrogen_total = sum(
            official_comp.get(column, 0.0)
            for column in ("Polyethylene_pct", "Cotton_Cellulose_pct", "PVDF_pct")
        )
        if hydrogen_total > 0:
            features["hydrogen_rich_frac"] = float(np.clip(hydrogen_total, 0.0, 1.0))

    gas_index = _GAS_MEAN_YIELD * (
        0.7 * features.get("polyethylene_frac", 0.0)
        + 0.4 * features.get("foam_frac", 0.0)
        + 0.5 * features.get("eva_frac", 0.0)
        + 0.2 * features.get("textile_frac", 0.0)
    )
    features["gas_recovery_index"] = float(np.clip(gas_index / 10.0, 0.0, 1.0))

    logistics_index = _MEAN_REUSE * (
        features.get("packaging_frac", 0.0) + 0.5 * features.get("eva_frac", 0.0)
    )
    features["logistics_reuse_index"] = float(np.clip(logistics_index, 0.0, 2.0))

    for oxide, value in _REGOLITH_VECTOR.items():
        features[f"oxide_{oxide}"] = float(value * regolith_pct)

    return features


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


def _build_candidate(
    picks: pd.DataFrame,
    proc_df: pd.DataFrame,
    rng: random.Random,
    target: dict,
    crew_time_low: bool,
    use_ml: bool,
    tuning: dict[str, Any] | None,
) -> dict | None:
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

    features = compute_feature_vector(picks, weights, proc, regolith_pct)
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
        return _build_candidate(picks, proc_df, rng, target, crew_time_low, use_ml, override)

    candidates: list[dict] = []
    for _ in range(n):
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
    "heuristic_props",
]
