"""Scoring and ranking utilities for Rex-AI candidate generation.

This module centralises the multi-objective scoring logic so it can be
reused by the Streamlit UI, CLI scripts and offline experimentation.
The implementation keeps the heuristics light-weight while surfacing
intermediate signals (auxiliary classifiers, contributions, penalties)
that the operator can inspect.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Dict, Iterable, Mapping

_DEFAULT_LIMITS = {
    "max_energy_kwh": 8.0,
    "max_water_l": 5.0,
    "max_crew_min": 60.0,
    "rigidity": 0.75,
    "tightness": 0.75,
}

DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "rigidez": 1.2,
    "estanqueidad": 1.2,
    "energy_kwh": 0.8,
    "water_l": 0.6,
    "crew_min": 0.6,
    "problematic_bonus": 0.5,
}

DEFAULT_PENALTY_WEIGHTS: Dict[str, float] = {
    "energy_kwh": 1.0,
    "water_l": 0.8,
    "crew_min": 0.8,
    "passes_seal": 1.1,
    "process_risk": 1.0,
}

_PROP_MAP = {
    "rigidity": ("rigidity", "rigidez"),
    "tightness": ("tightness", "estanqueidad"),
    "energy_kwh": ("energy_kwh",),
    "water_l": ("water_l",),
    "crew_min": ("crew_min",),
    "mass_final_kg": ("mass_final_kg",),
}


def _get_prop(props: Any, key: str, default: float = 0.0) -> float:
    keys = _PROP_MAP.get(key, (key,))
    for name in keys:
        if hasattr(props, name):
            try:
                return float(getattr(props, name))
            except (TypeError, ValueError):
                continue
        if isinstance(props, Mapping) and name in props:
            try:
                return float(props[name])
            except (TypeError, ValueError):
                continue
    return default


def _extract_comparisons(props: Any) -> Mapping[str, Mapping[str, Any]]:
    if hasattr(props, "comparisons") and getattr(props, "comparisons"):
        comp = getattr(props, "comparisons")
        if isinstance(comp, Mapping):
            return comp
    if isinstance(props, Mapping):
        comp = props.get("comparisons")
        if isinstance(comp, Mapping):
            return comp
    return {}


def _ratio(value: float, limit: float) -> float:
    if not isfinite(value) or limit <= 0 or not isfinite(limit):
        return 0.0
    return max(0.0, value / max(limit, 1e-6))


def _resource_reward(value: float, limit: float) -> float:
    ratio = _ratio(value, limit)
    return max(0.0, 1.0 - ratio)


def _resource_overuse(value: float, limit: float) -> float:
    ratio = _ratio(value, limit)
    return max(0.0, ratio - 1.0)


def _align_score(value: float, target: float) -> float:
    if not isfinite(value) or not isfinite(target):
        return 0.0
    return max(0.0, 1.0 - abs(value - target))


def _clamp01(value: float) -> float:
    if not isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def derive_auxiliary_signals(props: Any, target: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    target = target or {}
    comparisons = _extract_comparisons(props)

    tight_payload = comparisons.get("tightness_classifier", {})
    pass_prob = None
    if isinstance(tight_payload, Mapping):
        if pass_prob is None and "pass_prob" in tight_payload:
            try:
                pass_prob = float(tight_payload["pass_prob"])
            except (TypeError, ValueError):
                pass_prob = None
        if pass_prob is None and "expected" in tight_payload:
            try:
                pass_prob = float(tight_payload["expected"])
            except (TypeError, ValueError):
                pass_prob = None
    if pass_prob is None:
        pass_prob = _get_prop(props, "tightness", 0.0)
    pass_prob = _clamp01(pass_prob)
    passes_seal = pass_prob >= 0.55

    rigid_payload = comparisons.get("rigidity_classifier", {})
    rigidity_level = None
    rigidity_conf = None
    if isinstance(rigid_payload, Mapping):
        if "level" in rigid_payload:
            try:
                rigidity_level = float(rigid_payload["level"])
            except (TypeError, ValueError):
                rigidity_level = None
        if "confidence" in rigid_payload:
            try:
                rigidity_conf = float(rigid_payload["confidence"])
            except (TypeError, ValueError):
                rigidity_conf = None

    limits = {
        "max_energy_kwh": float(target.get("max_energy_kwh", _DEFAULT_LIMITS["max_energy_kwh"])),
        "max_water_l": float(target.get("max_water_l", _DEFAULT_LIMITS["max_water_l"])),
        "max_crew_min": float(target.get("max_crew_min", _DEFAULT_LIMITS["max_crew_min"])),
    }
    energy_ratio = _ratio(_get_prop(props, "energy_kwh", 0.0), limits["max_energy_kwh"])
    water_ratio = _ratio(_get_prop(props, "water_l", 0.0), limits["max_water_l"])
    crew_ratio = _ratio(_get_prop(props, "crew_min", 0.0), limits["max_crew_min"])

    risk_payload = comparisons.get("process_risk", {})
    risk_score = None
    if isinstance(risk_payload, Mapping) and "risk" in risk_payload:
        try:
            risk_score = float(risk_payload["risk"])
        except (TypeError, ValueError):
            risk_score = None

    if risk_score is None:
        # Weighted average emphasising crew time (operational bottleneck)
        risk_score = 0.35 * energy_ratio + 0.25 * water_ratio + 0.4 * crew_ratio
    risk_score = _clamp01(risk_score)

    if risk_score >= 0.66:
        risk_label = "alto"
    elif risk_score >= 0.33:
        risk_label = "medio"
    else:
        risk_label = "bajo"

    return {
        "passes_seal": bool(passes_seal),
        "seal_probability": float(pass_prob),
        "process_risk": float(risk_score),
        "process_risk_label": risk_label,
        "rigidity_level": rigidity_level,
        "rigidity_confidence": rigidity_conf,
        "limits": limits,
    }


def score_recipe(
    props: Any,
    target: Mapping[str, Any] | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    penalties: Mapping[str, float] | None = None,
    context: Mapping[str, Any] | None = None,
    aux: Mapping[str, Any] | None = None,
) -> tuple[float, Dict[str, Any]]:
    target = target or {}
    context = dict(context or {})

    weights_map = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    if weights:
        weights_map.update({str(k): float(v) for k, v in weights.items()})
    penalty_map = dict(DEFAULT_PENALTY_WEIGHTS)
    if penalties:
        penalty_map.update({str(k): float(v) for k, v in penalties.items()})

    if context.get("crew_time_low"):
        weights_map["crew_min"] = weights_map.get("crew_min", 0.6) * 1.4

    aux_signals = dict(aux or derive_auxiliary_signals(props, target))

    limits = {
        "rigidity": float(target.get("rigidity", _DEFAULT_LIMITS["rigidity"])),
        "tightness": float(target.get("tightness", _DEFAULT_LIMITS["tightness"])),
        "max_energy_kwh": float(target.get("max_energy_kwh", _DEFAULT_LIMITS["max_energy_kwh"])),
        "max_water_l": float(target.get("max_water_l", _DEFAULT_LIMITS["max_water_l"])),
        "max_crew_min": float(target.get("max_crew_min", _DEFAULT_LIMITS["max_crew_min"])),
    }

    contributions: Dict[str, float] = {}
    penalties_out: Dict[str, float] = {}

    contributions["rigidez"] = weights_map.get("rigidez", 0.0) * _align_score(
        _get_prop(props, "rigidity", 0.0), limits["rigidity"]
    )
    contributions["estanqueidad"] = weights_map.get("estanqueidad", 0.0) * _align_score(
        _get_prop(props, "tightness", 0.0), limits["tightness"]
    )

    energy_value = _get_prop(props, "energy_kwh", 0.0)
    water_value = _get_prop(props, "water_l", 0.0)
    crew_value = _get_prop(props, "crew_min", 0.0)

    contributions["energy_kwh"] = weights_map.get("energy_kwh", 0.0) * _resource_reward(
        energy_value, limits["max_energy_kwh"]
    )
    contributions["water_l"] = weights_map.get("water_l", 0.0) * _resource_reward(
        water_value, limits["max_water_l"]
    )
    contributions["crew_min"] = weights_map.get("crew_min", 0.0) * _resource_reward(
        crew_value, limits["max_crew_min"]
    )

    over_energy = _resource_overuse(energy_value, limits["max_energy_kwh"])
    over_water = _resource_overuse(water_value, limits["max_water_l"])
    over_crew = _resource_overuse(crew_value, limits["max_crew_min"])
    if over_energy > 0:
        penalties_out["energy_kwh"] = penalty_map.get("energy_kwh", 0.0) * over_energy
    if over_water > 0:
        penalties_out["water_l"] = penalty_map.get("water_l", 0.0) * over_water
    if over_crew > 0:
        penalties_out["crew_min"] = penalty_map.get("crew_min", 0.0) * over_crew

    bonus_ratio = context.get("problematic_mass_ratio")
    if bonus_ratio is None:
        bonus_ratio = _get_problematic_ratio_from_features(context, props)
    try:
        bonus_ratio_f = float(bonus_ratio)
    except (TypeError, ValueError):
        bonus_ratio_f = 0.0
    context["problematic_mass_ratio"] = bonus_ratio_f
    if bonus_ratio_f > 0:
        contributions["problematic_bonus"] = (
            weights_map.get("problematic_bonus", 0.0) * _clamp01(bonus_ratio_f)
        )

    if not aux_signals.get("passes_seal", True):
        seal_prob = _clamp01(float(aux_signals.get("seal_probability", 0.0)))
        penalties_out["passes_seal"] = penalty_map.get("passes_seal", 0.0) * (1.0 - seal_prob)

    process_risk = _clamp01(float(aux_signals.get("process_risk", 0.0)))
    if process_risk > 0.5:
        penalties_out["process_risk"] = penalty_map.get("process_risk", 0.0) * (process_risk - 0.5)

    total = sum(contributions.values()) - sum(penalties_out.values())

    breakdown = {
        "total": float(total),
        "contributions": {k: float(v) for k, v in contributions.items()},
        "penalties": {k: float(v) for k, v in penalties_out.items()},
        "weights": weights_map,
        "penalty_weights": penalty_map,
        "auxiliary": aux_signals,
        "context": context,
        "limits": limits,
    }
    return float(total), breakdown


def _get_problematic_ratio_from_features(context: Mapping[str, Any], props: Any) -> float:
    ratio = context.get("problematic_mass_ratio")
    if isinstance(ratio, (int, float)):
        return float(ratio)
    if isinstance(props, Mapping):
        features = props.get("features")
        if isinstance(features, Mapping) and "problematic_mass_frac" in features:
            try:
                return float(features["problematic_mass_frac"])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def rank_candidates(
    candidates: Iterable[Mapping[str, Any]],
    target: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
    penalties: Mapping[str, float] | None = None,
    top_n: int | None = 20,
    as_dict: bool = False,
) -> list[Dict[str, Any]]:
    ranked: list[Dict[str, Any]] = []
    for cand in candidates:
        base = dict(cand)
        props = base.get("props")
        context = dict(base.get("score_breakdown", {}).get("context", {}))
        if "crew_time_low" not in context and target.get("crew_time_low") is not None:
            context["crew_time_low"] = bool(target.get("crew_time_low"))
        if "problematic_mass_ratio" not in context:
            features = base.get("features")
            if isinstance(features, Mapping) and "problematic_mass_frac" in features:
                try:
                    context["problematic_mass_ratio"] = float(features["problematic_mass_frac"])
                except (TypeError, ValueError):
                    pass
        score, breakdown = score_recipe(
            props,
            target,
            weights=weights,
            penalties=penalties,
            context=context,
            aux=base.get("auxiliary"),
        )
        base["score"] = round(float(score), 6)
        base["score_breakdown"] = breakdown
        if base.get("auxiliary") is None:
            base["auxiliary"] = breakdown.get("auxiliary")
        ranked.append(base)

    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    if top_n is not None:
        ranked = ranked[: int(top_n)]

    if as_dict:
        for item in ranked:
            props = item.get("props")
            if hasattr(props, "as_dict"):
                try:
                    item["props"] = props.as_dict()
                except Exception:
                    pass
        return ranked

    return ranked

__all__ = [
    "derive_auxiliary_signals",
    "score_recipe",
    "rank_candidates",
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "DEFAULT_PENALTY_WEIGHTS",
]
