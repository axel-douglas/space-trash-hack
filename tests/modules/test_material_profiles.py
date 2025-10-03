import pandas as pd

from app.modules import schema


def test_numeric_series_from_dataframe_handles_non_numeric_values():
    df = pd.DataFrame({"value": ["1", 2, None, "nan", "3.5"]})

    series = schema.numeric_series(df, "value")

    assert series.tolist() == [1.0, 2.0, 3.5]
    assert series.dtype == float


def test_numeric_series_from_mapping_extracts_dataframe():
    df = pd.DataFrame({"value": ["10", " ", "8.5"]})

    series = schema.numeric_series({"value": df}, "value")

    assert series.tolist() == [10.0, 8.5]


def test_numeric_series_missing_dataframe_returns_empty_series():
    series = schema.numeric_series({}, "missing")

    assert series.empty
    assert series.dtype == float


def test_polymer_label_map_contains_expected_labels():
    expected = {
        "density_g_cm3": "ρ ref (g/cm³)",
        "tensile_mpa": "σₜ ref (MPa)",
        "modulus_gpa": "E ref (GPa)",
        "glass_c": "Tg (°C)",
        "ignition_c": "Ignición (°C)",
        "burn_min": "Burn (min)",
    }

    for key, value in expected.items():
        assert schema.POLYMER_LABEL_MAP[key] == value


def test_aluminium_label_map_contains_expected_labels():
    expected = {
        "tensile_mpa": "σₜ ref (MPa)",
        "yield_mpa": "σᵧ ref (MPa)",
        "elongation_pct": "ε ref (%)",
    }

    for key, value in expected.items():
        assert schema.ALUMINIUM_LABEL_MAP[key] == value
