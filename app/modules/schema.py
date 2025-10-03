"""Shared schema constants and helpers for material profile data."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


POLYMER_SAMPLE_COLUMNS: tuple[str, ...] = (
    "pc_density_sample_label",
    "pc_mechanics_sample_label",
    "pc_thermal_sample_label",
    "pc_ignition_sample_label",
)

POLYMER_NUMERIC_COLUMNS: tuple[str, ...] = (
    "pc_density_density_g_per_cm3",
    "pc_density_density_kg_m3",
    "pc_mechanics_tensile_strength_mpa",
    "pc_mechanics_stress_mpa",
    "pc_mechanics_yield_strength_mpa",
    "pc_mechanics_modulus_gpa",
    "pc_mechanics_strain_pct",
    "pc_thermal_glass_transition_c",
    "pc_thermal_onset_temperature_c",
    "pc_thermal_heat_capacity_j_per_g_k",
    "pc_thermal_heat_flow_w_per_g",
    "pc_ignition_ignition_temperature_c",
    "pc_ignition_burn_time_min",
)

POLYMER_METRIC_COLUMNS: tuple[str, ...] = (
    "pc_density_density_g_per_cm3",
    "pc_mechanics_tensile_strength_mpa",
    "pc_mechanics_modulus_gpa",
    "pc_thermal_glass_transition_c",
    "pc_ignition_ignition_temperature_c",
    "pc_ignition_burn_time_min",
)

POLYMER_LABEL_COLUMNS: tuple[str, ...] = (
    "pc_density_sample_label",
    "pc_mechanics_sample_label",
    "pc_thermal_sample_label",
    "pc_ignition_sample_label",
)

ALUMINIUM_SAMPLE_COLUMNS: tuple[str, ...] = (
    "aluminium_processing_route",
    "aluminium_class_id",
)

ALUMINIUM_NUMERIC_COLUMNS: tuple[str, ...] = (
    "aluminium_tensile_strength_mpa",
    "aluminium_yield_strength_mpa",
    "aluminium_elongation_pct",
)

ALUMINIUM_LABEL_COLUMNS: tuple[str, ...] = (
    "aluminium_processing_route",
    "aluminium_class_id",
)


POLYMER_LABEL_MAP: dict[str, str] = {
    "density_g_cm3": "ρ ref (g/cm³)",
    "tensile_mpa": "σₜ ref (MPa)",
    "modulus_gpa": "E ref (GPa)",
    "glass_c": "Tg (°C)",
    "ignition_c": "Ignición (°C)",
    "burn_min": "Burn (min)",
}

ALUMINIUM_LABEL_MAP: dict[str, str] = {
    "tensile_mpa": "σₜ ref (MPa)",
    "yield_mpa": "σᵧ ref (MPa)",
    "elongation_pct": "ε ref (%)",
}


def numeric_series(
    df: pd.DataFrame | Mapping[str, object] | None, column: str
) -> pd.Series:
    """Return a cleaned numeric series for the requested ``column``.

    The helper accepts either a :class:`pandas.DataFrame` or a mapping that
    contains a DataFrame for the given key. Non-numeric entries are coerced and
    missing values are dropped to simplify downstream visualisations.
    """

    if isinstance(df, Mapping):
        candidate = df.get(column)
        if isinstance(candidate, pd.DataFrame):
            df = candidate
        else:
            return pd.Series(dtype=float)

    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return pd.Series(dtype=float)

    series = pd.to_numeric(df[column], errors="coerce")
    return series.dropna()

