"""Helpers to derive slider bounds for target presets.

These utilities were extracted from el módulo legacy de luxe para mantener el
Target Designer liviano pero conservando los límites basados en datos que la
tripulación usa en laboratorio.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from app.modules.io import load_waste_df

__all__ = [
    "CREW_SIZE_BASELINE",
    "CREW_MINUTES_PER_MEMBER",
    "ENERGY_KWH_PER_KG_BASELINE",
    "WATER_L_PER_VOLUME_L_BASELINE",
    "compute_target_limits",
]

CREW_SIZE_BASELINE = 8
CREW_MINUTES_PER_MEMBER = 7.5
ENERGY_KWH_PER_KG_BASELINE = 0.0032
WATER_L_PER_VOLUME_L_BASELINE = 1 / 3000


def _numeric_values(presets: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for preset in presets:
        raw_value = preset.get(key)
        if raw_value is None:
            continue
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            continue
    return values


def _ensure_span(min_val: float, max_val: float, *, span: float) -> tuple[float, float]:
    if max_val - min_val < 1e-6:
        midpoint = min_val or max_val
        min_val = max(0.0, midpoint - span / 2)
        max_val = min_val + span
    return min_val, max_val


def compute_target_limits(
    presets: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Derive slider bounds and contextual help from presets and NASA baselines."""

    try:
        waste_df = load_waste_df()
    except Exception:  # pragma: no cover - defensive against optional IO deps
        waste_df = None

    if waste_df is not None:
        volume_series = (
            waste_df.get("_source_volume_l")
            if "_source_volume_l" in waste_df
            else waste_df.get("volume_l")
        )
        mass_series = (
            waste_df.get("kg") if "kg" in waste_df else waste_df.get("mass_kg")
        )
        volume_l = (
            volume_series.dropna().to_numpy(dtype=float)
            if volume_series is not None
            else np.array([])
        )
        mass_kg = (
            mass_series.dropna().to_numpy(dtype=float)
            if mass_series is not None
            else np.array([])
        )
    else:
        volume_l = np.array([])
        mass_kg = np.array([])

    bounds: Dict[str, Dict[str, Any]] = {}

    rigidity_values = _numeric_values(presets, "rigidity")
    rigidity_min = min(rigidity_values) if rigidity_values else 0.0
    rigidity_min = min(rigidity_min, 0.0)
    rigidity_max = max(rigidity_values + [1.0]) if rigidity_values else 1.0
    rigidity_min, rigidity_max = _ensure_span(rigidity_min, rigidity_max, span=0.1)
    rigidity_min = max(0.0, rigidity_min)
    rigidity_max = min(1.0, rigidity_max)
    bounds["rigidity"] = {
        "min": round(rigidity_min, 2),
        "max": round(rigidity_max, 2),
        "step": 0.05,
        "help": f"Rango derivado de presets NASA (máximo {rigidity_max:.2f}).",
    }

    tightness_values = _numeric_values(presets, "tightness")
    tightness_min = min(tightness_values) if tightness_values else 0.0
    tightness_min = min(tightness_min, 0.0)
    tightness_max = max(tightness_values + [1.0]) if tightness_values else 1.0
    tightness_min, tightness_max = _ensure_span(tightness_min, tightness_max, span=0.1)
    tightness_min = max(0.0, tightness_min)
    tightness_max = min(1.0, tightness_max)
    bounds["tightness"] = {
        "min": round(tightness_min, 2),
        "max": round(tightness_max, 2),
        "step": 0.05,
        "help": f"Rango derivado de presets NASA (máximo {tightness_max:.2f}).",
    }

    water_values = _numeric_values(presets, "max_water_l")
    water_min = min(water_values) if water_values else 0.0
    water_min = min(water_min, 0.0)
    water_max_preset = max(water_values) if water_values else 0.0
    water_max = water_max_preset
    water_help_parts = []
    if volume_l.size:
        volume_p90 = float(np.quantile(volume_l, 0.9))
        baseline = volume_p90 * WATER_L_PER_VOLUME_L_BASELINE
        water_max = max(water_max, baseline)
        water_help_parts.append(
            "NASA baseline: P90 volumen inventario "
            f"{volume_p90:,.0f} L → {baseline:.2f} L de proceso"
        )
    water_help_parts.append(f"Máximo preset actual: {water_max_preset:.2f} L")
    water_min, water_max = _ensure_span(water_min, water_max, span=0.2)
    water_min = max(0.0, water_min)
    water_max = max(water_min + 0.05, water_max)
    bounds["max_water_l"] = {
        "min": round(water_min, 2),
        "max": round(water_max, 2),
        "step": 0.1,
        "help": "; ".join(water_help_parts),
    }

    energy_values = _numeric_values(presets, "max_energy_kwh")
    energy_min = min(energy_values) if energy_values else 0.0
    energy_min = min(energy_min, 0.0)
    energy_max_preset = max(energy_values) if energy_values else 0.0
    energy_max = energy_max_preset
    energy_help_parts = []
    if mass_kg.size:
        mass_p90 = float(np.quantile(mass_kg, 0.9))
        baseline = mass_p90 * ENERGY_KWH_PER_KG_BASELINE
        energy_max = max(energy_max, baseline)
        energy_help_parts.append(
            "NASA baseline: P90 masa inventario "
            f"{mass_p90:,.0f} kg × {ENERGY_KWH_PER_KG_BASELINE:.4f} kWh/kg"
            f" = {baseline:.2f} kWh"
        )
    energy_help_parts.append(f"Máximo preset actual: {energy_max_preset:.2f} kWh")
    energy_min, energy_max = _ensure_span(energy_min, energy_max, span=0.2)
    energy_min = max(0.0, energy_min)
    energy_max = max(energy_min + 0.05, energy_max)
    bounds["max_energy_kwh"] = {
        "min": round(energy_min, 2),
        "max": round(energy_max, 2),
        "step": 0.1,
        "help": "; ".join(energy_help_parts),
    }

    crew_values = _numeric_values(presets, "max_crew_min")
    crew_min = min(crew_values) if crew_values else 0.0
    crew_min = min(crew_min, 0.0)
    crew_max_preset = max(crew_values) if crew_values else 0.0
    crew_max = max(crew_max_preset, CREW_SIZE_BASELINE * CREW_MINUTES_PER_MEMBER)
    crew_help = (
        f"Crew de {CREW_SIZE_BASELINE}: {CREW_SIZE_BASELINE} × "
        f"{CREW_MINUTES_PER_MEMBER:.1f} min = {crew_max:.0f} min"
    )
    if crew_max_preset:
        crew_help = f"{crew_help}; Máximo preset actual: {crew_max_preset:.0f} min"
    crew_min, crew_max = _ensure_span(crew_min, crew_max, span=5.0)
    crew_min = max(0.0, crew_min)
    crew_max = max(crew_min + 1.0, crew_max)
    bounds["max_crew_min"] = {
        "min": int(max(0, round(crew_min))),
        "max": int(round(crew_max)),
        "step": 1,
        "help": crew_help,
    }

    return bounds
