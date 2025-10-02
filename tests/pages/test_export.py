import json
from types import SimpleNamespace

import pandas as pd
import pytest

from app.modules.exporters import candidate_to_csv, candidate_to_json
from app.modules.page_data import (
    build_candidate_export_table,
    build_material_summary_table,
)


def _sample_candidate():
    return {
        "process_id": "P01",
        "process_name": "Laminar",
        "materials": ["Regolith", "PET flakes"],
        "weights": [0.6, 0.4],
        "score": 0.87,
        "props": SimpleNamespace(
            energy_kwh=1.2,
            water_l=0.35,
            crew_min=32.0,
            mass_final_kg=1.05,
            rigidity=0.78,
            tightness=0.65,
        ),
        "source_ids": ["A-1"],
    }


def test_build_candidate_export_table_includes_reference_metrics():
    inventory = pd.DataFrame(
        {
            "id": ["A-1"],
            "pc_density_density_g_per_cm3": [1.12],
            "pc_mechanics_tensile_strength_mpa": [48.0],
            "pc_mechanics_modulus_gpa": [3.5],
            "pc_thermal_glass_transition_c": [145.0],
            "pc_ignition_ignition_temperature_c": [280.0],
            "pc_ignition_burn_time_min": [2.5],
        }
    )

    df = build_candidate_export_table([_sample_candidate()], inventory)

    assert "ρ ref (g/cm³)" in df.columns
    assert "σₜ ref (MPa)" in df.columns
    assert df.iloc[0]["ρ ref (g/cm³)"] == pytest.approx(1.12)
    assert df.iloc[0]["Score"] == pytest.approx(0.87)


def test_material_summary_table_aggregates_weights():
    candidates = [
        _sample_candidate(),
        {
            "materials": ["Regolith", "Glass"],
            "weights": [0.4, 0.6],
            "props": SimpleNamespace(),
        },
    ]

    summary = build_material_summary_table(candidates)
    assert "Regolith" in summary["Material"].values
    regolith_row = summary[summary["Material"] == "Regolith"].iloc[0]
    assert regolith_row["Peso total"] == pytest.approx(1.0)
    assert regolith_row["Participaciones"] == 2


def test_candidate_export_serialization_roundtrip():
    candidate = _sample_candidate()
    target = {"name": "Mission", "max_energy_kwh": 2.0}
    safety = {"level": "OK", "detail": "Sin hallazgos."}

    json_payload = json.loads(candidate_to_json(candidate, target, safety).decode("utf-8"))
    csv_payload = candidate_to_csv(candidate).decode("utf-8")

    assert json_payload["candidate"]["process"]["id"] == "P01"
    assert json_payload["candidate"]["predictions"]["energy_kwh"] == pytest.approx(1.2)
    assert "process_id" in csv_payload
    assert "Laminar" in csv_payload
