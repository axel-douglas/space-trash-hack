from app.modules import schema


def test_schema_exposes_polymer_columns() -> None:
    assert schema.POLYMER_SAMPLE_COLUMNS == (
        "pc_density_sample_label",
        "pc_mechanics_sample_label",
        "pc_thermal_sample_label",
        "pc_ignition_sample_label",
    )

    assert schema.POLYMER_NUMERIC_COLUMNS == (
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

    assert schema.POLYMER_METRIC_COLUMNS == (
        "pc_density_density_g_per_cm3",
        "pc_mechanics_tensile_strength_mpa",
        "pc_mechanics_modulus_gpa",
        "pc_thermal_glass_transition_c",
        "pc_ignition_ignition_temperature_c",
        "pc_ignition_burn_time_min",
    )


def test_schema_exposes_aluminium_columns() -> None:
    assert schema.ALUMINIUM_SAMPLE_COLUMNS == (
        "aluminium_processing_route",
        "aluminium_class_id",
    )

    assert schema.ALUMINIUM_NUMERIC_COLUMNS == (
        "aluminium_tensile_strength_mpa",
        "aluminium_yield_strength_mpa",
        "aluminium_elongation_pct",
    )

    assert schema.ALUMINIUM_LABEL_COLUMNS == (
        "aluminium_processing_route",
        "aluminium_class_id",
    )
