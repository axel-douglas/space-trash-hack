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

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from app.modules.label_mapper import derive_recipe_id, lookup_labels

try:  # Lazy import to avoid circular dependency during training pipelines
    from app.modules.ml_models import MODEL_REGISTRY
except Exception:  # pragma: no cover - fallback when models are not available
    MODEL_REGISTRY = None

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"


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

    volume_m3 = (out["volume_l"].replace(0, np.nan) / 1000.0).fillna(0.001)
    out["density_kg_m3"] = out["kg"].astype(float) / volume_m3

    if "_problematic" not in out.columns:
        out["_problematic"] = out.apply(_is_problematic, axis=1)

    out["_source_id"] = out["id"].astype(str)
    out["_source_category"] = out["category"].astype(str)
    out["_source_flags"] = out["flags"].astype(str)

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

    for name, keywords in keyword_map.items():
        features[name] = _keyword_fraction(tuple(tokens), base_weights, keywords)

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


def _score_candidate(props: PredProps, target: dict, picks: pd.DataFrame, total_kg: float, crew_time_low: bool) -> float:
    score = 0.0
    score += 1.0 - abs(props.rigidity - float(target.get("rigidity", 0.75)))
    score += 1.0 - abs(props.tightness - float(target.get("tightness", 0.75)))

    def _pen(value: float, limit: float, eps: float) -> float:
        return max(0.0, (value - float(limit)) / max(eps, float(limit)))

    crew_eps = 0.5 if crew_time_low else 1.0
    score -= _pen(props.energy_kwh, float(target.get("max_energy_kwh", 10.0)), 0.1)
    score -= _pen(props.water_l, float(target.get("max_water_l", 5.0)), 0.1)
    score -= _pen(props.crew_min, float(target.get("max_crew_min", 60.0)), crew_eps)

    prob_mass = float((picks["_problematic"].astype(int) * picks["kg"]).sum())
    score += 0.5 * (prob_mass / max(0.1, total_kg))
    return float(score)


def _build_candidate(
    picks: pd.DataFrame,
    proc_df: pd.DataFrame,
    rng: random.Random,
    target: dict,
    crew_time_low: bool,
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
    else:
        props = heuristic
        force_heuristic = os.getenv("REXAI_FORCE_HEURISTIC", "").lower() in {"1", "true", "yes"}
        if not force_heuristic and MODEL_REGISTRY is not None and getattr(MODEL_REGISTRY, "ready", False):
            prediction = MODEL_REGISTRY.predict(features)
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
                        k: (float(v[0]), float(v[1])) for k, v in (prediction.get("confidence_interval") or {}).items()
                    },
                    feature_importance=[(str(k), float(v)) for k, v in (prediction.get("feature_importance") or [])],
                    comparisons={
                        k: {kk: float(vv) for kk, vv in val.items()} for k, val in (prediction.get("comparisons") or {}).items()
                    },
                )
                features["prediction_model"] = props.source
                features["model_metadata"] = prediction.get("metadata", {})
                features["uncertainty"] = props.uncertainty or {}
                features["confidence_interval"] = props.confidence_interval or {}
                features["feature_importance"] = props.feature_importance or []
                features["model_variants"] = props.comparisons or {}
            else:
                prediction = {}

    latent: Tuple[float, ...] | list[float] = []
    if MODEL_REGISTRY is not None and getattr(MODEL_REGISTRY, "ready", False):
        try:
            emb = MODEL_REGISTRY.embed(features)  # type: ignore[attr-defined]
        except Exception:
            emb = ()
        if emb:
            latent = tuple(float(x) for x in emb)
            features["latent_vector"] = latent

    score = _score_candidate(props, target, picks, total_kg, crew_time_low)

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
        "latent_vector": latent,
        "uncertainty": props.uncertainty or {},
        "confidence_interval": props.confidence_interval or {},
        "feature_importance": props.feature_importance or [],
        "model_variants": props.comparisons or {},
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
        return _build_candidate(picks, proc_df, rng, target, crew_time_low, override)

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
