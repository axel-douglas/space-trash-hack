"""Helpers to validate candidate properties against mission constraints.

This module coordinates three sources of information:

* Aggregated material properties provided by :class:`CandidateAssembler`.
* Mission level targets (rigidez, límites de densidad, conductividad, etc.).
* Estadísticos de entrenamiento del modelo ML (σ) para bandas de seguridad.

The resulting :class:`PropertyPlanner` can be reused by the latent explorer as
well as the Streamlit UI to surface constraint-aware insights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import math

import numpy as np

from app.modules.generator import CandidateAssembler
from app.modules.ml_models import TARGET_COLUMNS, ModelRegistry, get_model_registry
from app.modules.utils import safe_float

MetricConstraint = tuple[float | None, float | None]

_PROPERTY_LABELS: MutableMapping[str, tuple[str, str]] = {
    "material_density_kg_m3": ("Densidad", "kg/m³"),
    "material_modulus_gpa": ("Módulo", "GPa"),
    "material_thermal_conductivity_w_mk": ("Conductividad", "W/m·K"),
    "material_tensile_strength_mpa": ("σₜ", "MPa"),
    "material_elongation_pct": ("Elongación", "%"),
}

_TARGET_LABELS: MutableMapping[str, tuple[str, str]] = {
    "rigidity": ("Rigidez", ""),
    "tightness": ("Estanqueidad", ""),
}


def parse_property_constraints(
    target: Mapping[str, Any],
    property_columns: Iterable[str] | None = None,
) -> dict[str, MetricConstraint]:
    """Extract explicit min/max constraints from *target* for material metrics."""

    constraints: dict[str, MetricConstraint] = {}
    columns = tuple(property_columns) if property_columns is not None else tuple(_PROPERTY_LABELS)
    for column in columns:
        min_key = f"min_{column}"
        max_key = f"max_{column}"
        min_value = safe_float(target.get(min_key)) if min_key in target else None
        max_value = safe_float(target.get(max_key)) if max_key in target else None
        if min_value is None and max_value is None:
            continue
        constraints[column] = (min_value, max_value)
    return constraints


def _display_label(key: str) -> tuple[str, str]:
    if key in _PROPERTY_LABELS:
        return _PROPERTY_LABELS[key]
    if key in _TARGET_LABELS:
        return _TARGET_LABELS[key]
    return key, ""


@dataclass(slots=True)
class EvaluationEntry:
    key: str
    value: float
    sigma: float
    meets: bool
    minimum: float | None = None
    maximum: float | None = None

    def as_dict(self) -> dict[str, Any]:
        label, unit = _display_label(self.key)
        lower = float(self.value - self.sigma)
        upper = float(self.value + self.sigma)
        payload: dict[str, Any] = {
            "key": self.key,
            "label": label,
            "unit": unit,
            "value": float(self.value),
            "sigma": float(self.sigma),
            "lower": lower,
            "upper": upper,
            "meets": bool(self.meets),
        }
        if self.minimum is not None:
            payload["minimum"] = float(self.minimum)
        if self.maximum is not None:
            payload["maximum"] = float(self.maximum)
        return payload


class PropertyPlanner:
    """Coordinate ML predictions and material aggregates to validate targets."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        assembler: CandidateAssembler | None = None,
    ) -> None:
        if registry is None:
            try:
                registry = get_model_registry()
            except Exception:  # pragma: no cover - optional cache bootstrap
                registry = None
        if assembler is None:
            try:
                assembler = CandidateAssembler()
            except Exception:  # pragma: no cover - defensive in minimal envs
                assembler = None

        self.registry = registry
        self.assembler = assembler
        if assembler is not None:
            self.property_columns: tuple[str, ...] = tuple(
                getattr(assembler.material_reference, "property_columns", ())
            )
        else:
            self.property_columns = tuple()
        self._property_stats = self._compute_property_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_property_stats(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        if self.assembler is None:
            return stats

        reference = getattr(self.assembler.material_reference, "properties", {})
        if not isinstance(reference, Mapping):
            return stats

        for column in self.property_columns:
            values: list[float] = []
            for payload in reference.values():
                if not isinstance(payload, Mapping):
                    continue
                candidate = safe_float(payload.get(column))
                if candidate is None or not math.isfinite(candidate):
                    continue
                values.append(float(candidate))
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            if array.size > 1:
                std = float(np.std(array, ddof=1))
            else:
                std = float(np.std(array))
            stats[column] = {
                "mean": float(np.mean(array)),
                "std": float(std),
                "count": float(array.size),
            }
        return stats

    def _sigma_from_metadata(self, key: str) -> float:
        if self.registry is None:
            return 0.0
        residual = getattr(self.registry, "residual_std", None)
        if residual is None:
            return 0.0
        try:
            idx = TARGET_COLUMNS.index(key)
        except ValueError:
            return 0.0
        if idx >= residual.size:
            return 0.0
        value = float(residual[idx])
        return value if math.isfinite(value) else 0.0

    def _prediction_payload(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        props = candidate.get("props")
        if props is not None:
            if hasattr(props, "as_dict"):
                try:
                    return props.as_dict()
                except Exception:  # pragma: no cover - defensive
                    pass
            if isinstance(props, Mapping):
                return props

        prediction = candidate.get("prediction")
        if isinstance(prediction, Mapping):
            return prediction

        return {}

    def _feature_payload(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        features = candidate.get("features")
        if isinstance(features, Mapping):
            return features
        return {}

    def _extract_value(self, payload: Mapping[str, Any], *keys: str) -> float | None:
        for key in keys:
            if key not in payload:
                continue
            value = safe_float(payload.get(key))
            if value is None:
                continue
            return float(value)
        return None

    def _metric_sigma(self, candidate: Mapping[str, Any], key: str) -> float:
        prediction_unc = candidate.get("uncertainty")
        if isinstance(prediction_unc, Mapping):
            direct = safe_float(prediction_unc.get(key))
            if direct is None and key in {"rigidity", "tightness"}:
                alias = "rigidez" if key == "rigidity" else "estanqueidad"
                direct = safe_float(prediction_unc.get(alias))
            if direct is not None and math.isfinite(direct):
                return float(direct)

        features = self._feature_payload(candidate)
        if "uncertainty" in features:
            nested = features.get("uncertainty")
            if isinstance(nested, Mapping):
                val = safe_float(nested.get(key))
                if val is None and key in {"rigidity", "tightness"}:
                    alias = "rigidez" if key == "rigidity" else "estanqueidad"
                    val = safe_float(nested.get(alias))
                if val is not None and math.isfinite(val):
                    return float(val)

        if key in _PROPERTY_LABELS:
            stat = self._property_stats.get(key)
            if stat and math.isfinite(stat.get("std", float("nan"))):
                return float(stat.get("std", 0.0))

        if key in ("rigidez", "estanqueidad"):
            english = "rigidity" if key == "rigidez" else "tightness"
            return self._metric_sigma(candidate, english)

        if key in ("rigidity", "tightness"):
            return self._sigma_from_metadata("rigidez" if key == "rigidity" else "estanqueidad")

        return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def available(self) -> bool:
        return self.registry is not None or self.assembler is not None

    def evaluate_candidate(
        self,
        candidate: Mapping[str, Any],
        target: Mapping[str, Any],
    ) -> dict[str, Any]:
        prediction = self._prediction_payload(candidate)
        features = self._feature_payload(candidate)

        entries: list[EvaluationEntry] = []
        feasible = True

        rigidity_target = safe_float(target.get("rigidity"))
        if rigidity_target is not None:
            value = self._extract_value(prediction, "rigidez", "rigidity")
            if value is None:
                feasible = False
            else:
                sigma = self._metric_sigma(candidate, "rigidity")
                meets = (value - sigma) >= float(rigidity_target)
                feasible &= meets
                entries.append(
                    EvaluationEntry(
                        key="rigidity",
                        value=float(value),
                        sigma=float(sigma),
                        meets=meets,
                        minimum=float(rigidity_target),
                    )
                )

        tightness_target = safe_float(target.get("tightness"))
        if tightness_target is not None:
            value = self._extract_value(prediction, "estanqueidad", "tightness")
            if value is None:
                feasible = False
            else:
                sigma = self._metric_sigma(candidate, "tightness")
                meets = (value - sigma) >= float(tightness_target)
                feasible &= meets
                entries.append(
                    EvaluationEntry(
                        key="tightness",
                        value=float(value),
                        sigma=float(sigma),
                        meets=meets,
                        minimum=float(tightness_target),
                    )
                )

        constraints = parse_property_constraints(target, self.property_columns or None)
        for column, (minimum, maximum) in constraints.items():
            value = safe_float(features.get(column))
            if value is None:
                feasible = False
                entries.append(
                    EvaluationEntry(
                        key=column,
                        value=float("nan"),
                        sigma=0.0,
                        meets=False,
                        minimum=minimum,
                        maximum=maximum,
                    )
                )
                continue

            sigma = self._metric_sigma(candidate, column)
            meets_min = True
            meets_max = True
            if minimum is not None:
                meets_min = (value - sigma) >= float(minimum)
            if maximum is not None:
                meets_max = (value + sigma) <= float(maximum)
            meets = meets_min and meets_max
            feasible &= meets
            entries.append(
                EvaluationEntry(
                    key=column,
                    value=float(value),
                    sigma=float(sigma),
                    meets=meets,
                    minimum=minimum,
                    maximum=maximum,
                )
            )

        payload = {
            "feasible": bool(feasible),
            "entries": [entry.as_dict() for entry in entries],
        }
        return payload

    def is_feasible(self, candidate: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
        return bool(self.evaluate_candidate(candidate, target).get("feasible"))

    def filter_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        target: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        feasible: list[Mapping[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            if self.is_feasible(candidate, target):
                feasible.append(candidate)
        return feasible


__all__ = ["EvaluationEntry", "PropertyPlanner", "parse_property_constraints"]

