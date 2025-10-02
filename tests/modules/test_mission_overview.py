from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

from app.modules import mission_overview


def test_compute_mass_volume_metrics_aggregates_expected_values() -> None:
    df = pd.DataFrame(
        {
            "kg": [10.0, 5.0],
            "volume_l": [500.0, 250.0],
            "moisture_pct": [20.0, 10.0],
            "difficulty_factor": [1.0, 3.0],
        }
    )

    metrics = mission_overview.compute_mass_volume_metrics(df)

    assert metrics["mass_kg"] == 15.0
    assert metrics["volume_m3"] == (500.0 + 250.0) / 1000.0

    # Water is moisture percentage applied to mass
    expected_water = 10.0 * 0.20 + 5.0 * 0.10
    assert metrics["water_l"] == expected_water

    # Energy should reflect difficulty interpolation
    base = 0.12
    max_energy = 0.70
    diff_high = base + (3.0 - 1.0) / 2.0 * (max_energy - base)
    expected_energy = 10.0 * base + 5.0 * diff_high
    assert metrics["energy_kwh"] == expected_energy


def test_summarize_model_state_reports_age_and_samples() -> None:
    trained_dt = datetime.now(timezone.utc) - timedelta(days=10)
    metadata = {
        "ready": True,
        "trained_at": trained_dt.isoformat(),
        "n_samples": 1200,
    }

    summary = mission_overview.summarize_model_state(metadata)

    assert summary["status_label"] == "✅ Modelo listo"
    assert summary["tone"] == "positive"
    assert summary["sample_count"] == 1200
    assert any("Muestras" in note for note in summary["notes"])
    assert any("Entrenamiento reciente" in note for note in summary["notes"])


def test_prepare_material_summary_formats_problematic_and_external_columns() -> None:
    df = pd.DataFrame(
        {
            "material_display": ["Aluminio reciclado", "Polímero"],
            "category": ["Metal", "Plástico"],
            "kg": [12.5, 7.5],
            "volume_l": [1000.0, 500.0],
            "_problematic": [1, 0],
            "pc_mass_kg": [2.5, 1.0],
            "aluminium_mass_kg": [5.0, 0.0],
        }
    )

    summary_df, column_config = mission_overview.prepare_material_summary(df)

    assert summary_df["material_display"].tolist() == ["Aluminio reciclado", "Polímero"]
    assert summary_df["_problematic"].tolist() == [True, False]
    assert "pc_mass_kg" in summary_df.columns
    assert "aluminium_mass_kg" in summary_df.columns

    assert column_config["material_display"]["type_config"]["type"] == "text"
    assert column_config["_problematic"]["type_config"]["type"] == "checkbox"
    assert column_config["pc_mass_kg"]["type_config"]["type"] == "number"

