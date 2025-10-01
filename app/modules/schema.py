"""Shared schema constants for external reference columns."""

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

