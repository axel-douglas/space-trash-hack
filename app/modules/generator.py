# app/modules/generator.py
from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import os

try:  # Lazy import to avoid circular dependency during training pipelines
    from app.modules.ml_models import MODEL_REGISTRY
except Exception:  # pragma: no cover - fallback when models are not available
    MODEL_REGISTRY = None

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"

try:  # Preload regolith composition and mission coefficients for feature parity
    _REGOLITH_VECTOR = (
        pd.read_csv(DATASETS_ROOT / "raw" / "mgs1_oxides.csv")
        .assign(oxide=lambda df: df["oxide"].str.lower())
        .set_index("oxide")["wt_percent"]
        .div(100.0)
        .to_dict()
    )
except Exception:  # pragma: no cover - fallback constants
    _REGOLITH_VECTOR = {
        "sio2": 0.48,
        "feot": 0.18,
        "mgo": 0.13,
        "cao": 0.055,
        "so3": 0.07,
        "h2o": 0.032,
    }

try:
    _GAS_METRICS = pd.read_csv(DATASETS_ROOT / "raw" / "nasa_trash_to_gas.csv")
    _GAS_MEAN_YIELD = float(
        (_GAS_METRICS["o2_ch4_yield_kg"] / _GAS_METRICS["water_makeup_kg"].clip(lower=1e-6)).mean()
    )
except Exception:  # pragma: no cover - fallback constant
    _GAS_MEAN_YIELD = 6.0

try:
    _LOGISTICS = pd.read_csv(DATASETS_ROOT / "raw" / "logistics_to_living.csv")
    _LOGISTICS["reuse_efficiency"] = (
        (_LOGISTICS["outfitting_replaced_kg"] - _LOGISTICS["residual_waste_kg"]) / _LOGISTICS["packaging_kg"].clip(lower=1e-6)
    ).clip(lower=0)
    _MEAN_REUSE = float(_LOGISTICS["reuse_efficiency"].mean())
except Exception:  # pragma: no cover - fallback constant
    _MEAN_REUSE = 0.6

@dataclass
class PredProps:
    rigidity: float
    tightness: float
    mass_final_kg: float
    energy_kwh: float
    water_l: float
    crew_min: float
    source: str = "heuristic"
    uncertainty: dict[str, float] | None = None
    confidence_interval: dict[str, tuple[float, float]] | None = None
    feature_importance: list[tuple[str, float]] | None = None
    comparisons: dict[str, dict[str, float]] | None = None

# --- utilidades de compatibilidad con DF (nombres de columnas) ---
def _col(df: pd.DataFrame, candidates: list[str], default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def _ensure_compat(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve copia con columnas estandarizadas:
       id, category, material, kg, volume_l, flags, _problematic
    """
    out = df.copy()
    # ID
    if "id" not in out.columns:
        out["id"] = out.index.astype(str)
    # Category
    cat_col = _col(out, ["category","Category"])
    if cat_col != "category":
        out["category"] = out[cat_col] if cat_col else ""
    # Material
    mat_col = _col(out, ["material","material_family","Material"])
    if mat_col != "material":
        out["material"] = out[mat_col] if mat_col else ""
    # Masa (kg)
    kg_col = _col(out, ["kg","mass_kg","Mass_kg"])
    if kg_col != "kg":
        out["kg"] = pd.to_numeric(out[kg_col], errors="coerce").fillna(0.0) if kg_col else 0.0
    # Volumen (L)
    vol_col = _col(out, ["volume_l","Volume_L"])
    if vol_col != "volume_l":
        out["volume_l"] = pd.to_numeric(out[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0
    # Moisture
    moist_col = _col(out, ["moisture_pct", "moisture", "moisture_percent"], default=None)
    if moist_col and moist_col != "moisture_pct":
        out["moisture_pct"] = pd.to_numeric(out[moist_col], errors="coerce").fillna(0.0)
    elif "moisture_pct" not in out.columns:
        out["moisture_pct"] = 0.0
    # Difficulty factor
    diff_col = _col(out, ["difficulty_factor", "difficulty", "diff_factor"], default=None)
    if diff_col and diff_col != "difficulty_factor":
        out["difficulty_factor"] = pd.to_numeric(out[diff_col], errors="coerce").fillna(1.0)
    elif "difficulty_factor" not in out.columns:
        out["difficulty_factor"] = 1.0
    # Percentages NASA references
    pmass_col = _col(out, ["pct_mass", "percent_mass"], default=None)
    if pmass_col and pmass_col != "pct_mass":
        out["pct_mass"] = pd.to_numeric(out[pmass_col], errors="coerce").fillna(0.0)
    elif "pct_mass" not in out.columns:
        out["pct_mass"] = 0.0
    pvol_col = _col(out, ["pct_volume", "percent_volume"], default=None)
    if pvol_col and pvol_col != "pct_volume":
        out["pct_volume"] = pd.to_numeric(out[pvol_col], errors="coerce").fillna(0.0)
    elif "pct_volume" not in out.columns:
        out["pct_volume"] = 0.0
    # Key materials text for embeddings
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
    # Flags
    flg_col = _col(out, ["flags","Flags"])
    if flg_col != "flags":
        out["flags"] = out[flg_col] if flg_col else ""
    # Problemáticos
    if "_problematic" not in out.columns:
        out["_problematic"] = out.apply(_is_problematic, axis=1)
    # Origen/huella NASA
    out["_source_id"] = out["id"].astype(str)
    out["_source_category"] = out["category"].astype(str)
    out["_source_flags"] = out["flags"].astype(str)
    return out

def _is_problematic(row: pd.Series) -> bool:
    cat = str(row.get("category", "")).lower()
    fam = str(row.get("material", "")).lower() + " " + str(row.get("material_family","")).lower()
    flg = str(row.get("flags", "")).lower()
    rules = [
        "pouches" in cat or "multilayer" in flg or "pe-pet-al" in fam,
        "foam" in cat or "zotek" in fam or "closed_cell" in flg,
        "eva" in cat or "ctb" in flg or "nomex" in fam or "nylon" in fam or "polyester" in fam,
        "glove" in cat or "nitrile" in fam,
        "wipe" in flg or "textile" in cat
    ]
    return any(rules)

def _pick_materials(
    df: pd.DataFrame,
    n: int = 2,
    problematic_bias: float = 2.0,
) -> pd.DataFrame:
    # Preferimos masa y problemáticos (boost configurable)
    w = df["kg"].clip(lower=0.01) + df["_problematic"].astype(int) * float(problematic_bias)
    return df.sample(n=min(n, len(df)), weights=w, replace=False, random_state=None)

def generate_candidates(waste_df: pd.DataFrame, proc_df: pd.DataFrame,
                        target: dict, n: int = 6, crew_time_low: bool = False,
                        optimizer_evals: int = 0):
    """
    Genera candidatos priorizando:
    - Consumir masa de ítems 'problemáticos' definidos por NASA.
    - Elegir procesos coherentes: P02 (laminado multicapa), P03 (sinter + MGS-1), P04 (reuso CTB).
    - Insertar explícitamente **regolito MGS-1** cuando el proceso sea P03.

    Devuelve una tupla con el listado de candidatos y, si se habilita el
    optimizador, un DataFrame con el historial de convergencia.
    """
    if waste_df is None or proc_df is None or len(waste_df) == 0 or len(proc_df) == 0:
        return [], pd.DataFrame()

    df = _ensure_compat(waste_df)
    process_ids = (
        sorted(proc_df["process_id"].astype(str).unique().tolist())
        if proc_df is not None and not proc_df.empty
        else []
    )
    def _sampler(override: dict[str, Any] | None = None) -> dict | None:
        override = override or {}
        bias = float(override.get("problematic_bias", 2.0))
        picks = _pick_materials(df, n=random.choice([2, 3]), problematic_bias=bias)
        tuning: dict[str, Any] = {}
        if "regolith_pct" in override:
            tuning["regolith_pct"] = float(override["regolith_pct"])
        if "process_choice" in override:
            tuning["process_choice"] = override["process_choice"]
        return _build_candidate(picks, proc_df, target, crew_time_low, tuning)

    out = []
    for _ in range(n):
        cand = _sampler()
        if cand:
            out.append(cand)

    history = pd.DataFrame()

    if optimizer_evals and optimizer_evals > 0:
        try:
            from app.modules.optimizer import optimize_candidates

            pareto, history = optimize_candidates(
                initial_candidates=out,
                sampler=_sampler,
                target=target,
                n_evals=int(optimizer_evals),
                process_ids=process_ids,
            )
            out = pareto
        except Exception:
            history = pd.DataFrame()

    out.sort(key=lambda x: x["score"], reverse=True)
    return out, history


def _material_tokens(row: pd.Series) -> str:
    """Concatenate informative strings to detect material families."""

    parts = [
        str(row.get("material", "")),
        str(row.get("category", "")),
        str(row.get("flags", "")),
        str(row.get("material_family", "")),
        str(row.get("key_materials", "")),
    ]
    return " ".join(parts).lower()


def _derive_features(
    picks: pd.DataFrame,
    base_weights: list[float],
    proc: pd.Series,
    regolith_pct: float,
) -> Dict[str, Any]:
    total_kg = max(0.001, float(picks["kg"].sum()))

    tokens = [_material_tokens(row) for _, row in picks.iterrows()]
    categories = [str(row.get("category", "")).lower() for _, row in picks.iterrows()]
    weights = np.asarray(base_weights, dtype=float)
    pct_mass = picks.get("pct_mass", 0).to_numpy(dtype=float) / 100.0
    pct_volume = picks.get("pct_volume", 0).to_numpy(dtype=float) / 100.0
    moisture = picks.get("moisture_pct", 0).to_numpy(dtype=float) / 100.0
    difficulty = picks.get("difficulty_factor", 1).to_numpy(dtype=float) / 3.0
    densities = picks.get("density_kg_m3", 0).to_numpy(dtype=float)

    def _weight_for(keywords: tuple[str, ...]) -> float:
        weight = 0.0
        for token, frac in zip(tokens, base_weights):
            if any(keyword in token for keyword in keywords):
                weight += frac
        return float(np.clip(weight, 0.0, 1.0))

    def _category_weight(targets: tuple[str, ...]) -> float:
        weight = 0.0
        for category, frac in zip(categories, base_weights):
            if any(target in category for target in targets):
                weight += frac
        return float(np.clip(weight, 0.0, 1.0))

    keyword_map = {
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
        "process_id": str(proc["process_id"]),
        "total_mass_kg": total_kg,
        "density_kg_m3": float(np.dot(weights, densities)),
        "num_items": int(len(picks)),
        "moisture_frac": float(np.clip(np.dot(weights, moisture), 0.0, 1.0)),
        "difficulty_index": float(np.clip(np.dot(weights, difficulty), 0.0, 1.0)),
        "problematic_mass_frac": float(np.clip(np.dot(weights, pct_mass), 0.0, 1.0)),
        "problematic_item_frac": float(np.clip(np.dot(weights, pct_volume), 0.0, 1.0)),
        "regolith_pct": float(np.clip(regolith_pct, 0.0, 1.0)),
        "packaging_frac": _category_weight(("packaging", "food packaging")),
    }

    for name, keywords in keyword_map.items():
        features[name] = _weight_for(keywords)

    gas_index = _GAS_MEAN_YIELD * (
        0.7 * features.get("polyethylene_frac", 0.0)
        + 0.4 * features.get("foam_frac", 0.0)
        + 0.5 * features.get("eva_frac", 0.0)
        + 0.2 * features.get("textile_frac", 0.0)
    )
    features["gas_recovery_index"] = float(np.clip(gas_index / 10.0, 0.0, 1.0))

    logistics_index = _MEAN_REUSE * (features.get("packaging_frac", 0.0) + 0.5 * features.get("eva_frac", 0.0))
    features["logistics_reuse_index"] = float(np.clip(logistics_index, 0.0, 2.0))

    for oxide, value in _REGOLITH_VECTOR.items():
        features[f"oxide_{oxide}"] = float(value * regolith_pct)

    return features


def _build_candidate(
    picks: pd.DataFrame,
    proc_df: pd.DataFrame,
    target: dict,
    crew_time_low: bool = False,
    tuning: dict[str, Any] | None = None,
) -> dict | None:
    if picks is None or picks.empty or proc_df is None or proc_df.empty:
        return None

    tuning = tuning or {}
    total_kg = max(0.001, float(picks["kg"].sum()))
    weights = (picks["kg"] / total_kg).round(2).tolist()
    base_weights = weights.copy()

    used_ids = picks["_source_id"].tolist()
    used_cats = picks["_source_category"].tolist()
    used_flags = picks["_source_flags"].tolist()
    used_mats = picks["material"].tolist()

    preferred_process = tuning.get("process_choice")
    proc = _select_process(proc_df, used_cats, used_flags, used_mats, preferred_process)

    mass_final = total_kg * 0.90
    energy = float(proc["energy_kwh_per_kg"]) * mass_final
    water = float(proc["water_l_per_kg"]) * mass_final
    crew = float(proc["crew_min_per_batch"])

    mats_join = " ".join(used_mats).lower()
    cats_join = " ".join([str(c).lower() for c in used_cats])
    flags_join = " ".join([str(f).lower() for f in used_flags])
    rigidity = min(1.0, 0.5 + (0.2 if "al" in mats_join or "aluminum" in mats_join else 0.0))
    tightness = min(1.0, 0.5 + (0.2 if "pouches" in cats_join or "pe-pet-al" in mats_join else 0.0))

    regolith_pct = float(tuning.get("regolith_pct", 0.0))
    materials_for_plan = used_mats.copy()
    weights_for_plan = weights.copy()

    if regolith_pct <= 0.0 and str(proc["process_id"]).upper() == "P03":
        regolith_pct = 0.2
    if regolith_pct > 0:
        materials_for_plan.append("MGS-1_regolith")
        weights_for_plan = [round(w * (1.0 - regolith_pct), 2) for w in weights_for_plan]
        weights_for_plan.append(round(regolith_pct, 2))
        s = sum(weights_for_plan)
        if s > 0:
            weights_for_plan = [round(w / s, 2) for w in weights_for_plan]
        rigidity = min(1.0, rigidity + 0.1)
        tightness = max(0.0, tightness - 0.05)

    features = _derive_features(picks, base_weights, proc, regolith_pct)

    heuristic_props = PredProps(
        rigidity=rigidity,
        tightness=tightness,
        mass_final_kg=mass_final,
        energy_kwh=energy,
        water_l=water,
        crew_min=crew,
    )

    props = heuristic_props

    force_heuristic = os.getenv("REXAI_FORCE_HEURISTIC", "").lower() in {"1", "true", "yes"}
    prediction: dict[str, Any] = {}
    if not force_heuristic and MODEL_REGISTRY is not None and MODEL_REGISTRY.ready:
        prediction = MODEL_REGISTRY.predict(features)
        if prediction:
            prediction = dict(prediction)
            prediction["sources"] = {
                "ids": used_ids,
                "categories": used_cats,
                "flags": used_flags,
            }
            uncertainty = prediction.get("uncertainty") or {}
            confidence_interval = prediction.get("confidence_interval") or {}
            importance = prediction.get("feature_importance") or []
            comparisons = prediction.get("comparisons") or {}
            props = PredProps(
                rigidity=float(prediction.get("rigidez", rigidity)),
                tightness=float(prediction.get("estanqueidad", tightness)),
                mass_final_kg=mass_final,
                energy_kwh=float(prediction.get("energy_kwh", energy)),
                water_l=float(prediction.get("water_l", water)),
                crew_min=float(prediction.get("crew_min", crew)),
                source=str(prediction.get("source", "ml")),
                uncertainty={k: float(v) for k, v in uncertainty.items()},
                confidence_interval={k: tuple(v) for k, v in confidence_interval.items()},
                feature_importance=[(str(k), float(v)) for k, v in importance],
                comparisons=comparisons,
            )
            # Guardamos info adicional para trazabilidad de la UI
            features["prediction_model"] = props.source
            features["model_metadata"] = prediction.get("metadata", {})
            features["uncertainty"] = props.uncertainty
            features["confidence_interval"] = props.confidence_interval
            features["feature_importance"] = props.feature_importance
            features["model_variants"] = props.comparisons
    else:
        prediction = {}

    if MODEL_REGISTRY is not None and MODEL_REGISTRY.ready:
        latent = MODEL_REGISTRY.embed(features)
        if latent:
            features["latent_vector"] = latent
    else:
        latent = []

    score = _score_candidate(props, target, picks, total_kg, crew_time_low)

    return {
        "materials": materials_for_plan,
        "weights": weights_for_plan,
        "process_id": str(proc["process_id"]),
        "process_name": str(proc["name"]),
        "props": props,
        "heuristic_props": heuristic_props,
        "score": round(float(score), 3),
        "source_ids": used_ids,
        "source_categories": used_cats,
        "source_flags": used_flags,
        "regolith_pct": regolith_pct,
        "features": features,
        "prediction_source": props.source,
        "ml_prediction": prediction,
        "latent_vector": latent,
        "uncertainty": props.uncertainty,
        "confidence_interval": props.confidence_interval,
        "feature_importance": props.feature_importance,
        "model_variants": props.comparisons,
    }


def _select_process(
    proc_df: pd.DataFrame,
    used_cats: list[str],
    used_flags: list[str],
    used_mats: list[str],
    preferred: str | None = None,
) -> pd.Series:
    if preferred:
        forced = proc_df[proc_df["process_id"].astype(str) == str(preferred)]
        if not forced.empty:
            return forced.iloc[0]

    proc = proc_df.sample(1).iloc[0]
    cats_join = " ".join([str(c).lower() for c in used_cats])
    flags_join = " ".join([str(f).lower() for f in used_flags])
    mats_join = " ".join(used_mats).lower()

    if ("pouches" in cats_join) or ("multilayer" in flags_join) or ("pe-pet-al" in mats_join):
        cand = proc_df[proc_df["process_id"] == "P02"]
        if not cand.empty:
            proc = cand.iloc[0]

    if ("foam" in cats_join) or ("zotek" in mats_join):
        cand = proc_df[proc_df["process_id"].isin(["P03", "P02"])]
        if not cand.empty:
            proc = cand.sample(1).iloc[0]

    if ("eva" in cats_join) or ("ctb" in flags_join):
        cand = proc_df[proc_df["process_id"] == "P04"]
        if not cand.empty:
            proc = cand.iloc[0]

    return proc


def _score_candidate(props: PredProps, target: dict, picks: pd.DataFrame, total_kg: float,
                     crew_time_low: bool = False) -> float:
    score = 0.0
    score += 1.0 - abs(props.rigidity - float(target["rigidity"]))
    score += 1.0 - abs(props.tightness - float(target["tightness"]))

    def _pen(v, lim, eps):
        return max(0.0, (v - float(lim)) / max(eps, float(lim)))

    crew_eps = 0.5 if crew_time_low else 1.0
    score -= _pen(props.energy_kwh, target["max_energy_kwh"], 0.1)
    score -= _pen(props.water_l, target["max_water_l"], 0.1)
    score -= _pen(props.crew_min, target["max_crew_min"], crew_eps)

    prob_mass = float((picks["_problematic"].astype(int) * picks["kg"]).sum())
    score += 0.5 * (prob_mass / max(0.1, total_kg))
    return score
