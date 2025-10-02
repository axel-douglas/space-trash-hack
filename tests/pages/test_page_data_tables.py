from types import SimpleNamespace

import pandas as pd
import pytest

from app.modules.page_data import (
    build_candidate_metric_table,
    build_export_kpi_table,
    build_ranking_table,
    build_resource_table,
)


def test_candidate_metric_table_includes_uncertainty_and_ci():
    props = SimpleNamespace(
        rigidity=0.82,
        tightness=0.64,
        energy_kwh=1.25,
        water_l=0.42,
        crew_min=38.0,
    )
    heur = {
        "rigidity": 0.75,
        "tightness": 0.6,
        "energy_kwh": 1.4,
        "water_l": 0.5,
        "crew_min": 45.0,
    }
    ci = {"rigidity": (0.7, 0.9)}
    uncertainty = {"rigidity": 0.055}

    df = build_candidate_metric_table(props, heur, score=0.78, confidence=ci, uncertainty=uncertainty)

    assert "Indicador" in df.columns
    rigidity = df[df["Indicador"] == "Rigidez"].iloc[0]
    assert rigidity["σ"] == pytest.approx(0.055)
    assert rigidity["CI 95%"] == "0.700 – 0.900"
    score_row = df[df["Indicador"] == "Score total"].iloc[0]
    assert score_row["IA Rex-AI"] == pytest.approx(0.78)


def test_resource_table_tracks_utilisation():
    props = SimpleNamespace(energy_kwh=1.0, water_l=0.5, crew_min=30)
    limits = {"max_energy_kwh": 2.0, "max_water_l": 1.0, "max_crew_min": 60}

    df = build_resource_table(props, limits)
    assert set(df["Recurso"]) == {"Energía (kWh)", "Agua (L)", "Crew (min)"}
    energy_row = df[df["Recurso"] == "Energía (kWh)"].iloc[0]
    assert energy_row["Utilización (%)"] == pytest.approx(50.0)


def test_ranking_table_orders_by_score():
    candidates = [
        {"score": 0.6, "process_id": "P02", "process_name": "Laminar", "props": SimpleNamespace(rigidity=0.7, tightness=0.5, energy_kwh=1.4, water_l=0.6, crew_min=42)},
        {"score": 0.8, "process_id": "P03", "process_name": "Sinter", "props": SimpleNamespace(rigidity=0.9, tightness=0.7, energy_kwh=1.1, water_l=0.4, crew_min=36), "auxiliary": {"passes_seal": False}},
    ]

    df = build_ranking_table(candidates)
    assert df.iloc[0]["Score"] == pytest.approx(0.8)
    assert df.iloc[0]["Seal"] == "⚠️"


def test_export_kpi_table_collects_metrics():
    df_plot = pd.DataFrame(
        {
            "Score": [0.7, 0.8],
            "Energía (kWh)": [1.2, 1.0],
            "Agua (L)": [0.5, 0.4],
            "Crew (min)": [38, 36],
            "ρ ref (g/cm³)": [1.1, 1.2],
        }
    )

    kpi_df = build_export_kpi_table(df_plot)
    assert "Opciones válidas" in kpi_df["KPI"].values
    assert kpi_df[kpi_df["KPI"] == "Opciones válidas"]["Valor"].iloc[0] == pytest.approx(2.0)
