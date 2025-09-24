from __future__ import annotations

import pytest

from app.modules.generator import PredProps
from app.modules.ranking import derive_auxiliary_signals, rank_candidates, score_recipe


def _make_props(**kwargs):
    defaults = dict(
        rigidity=0.8,
        tightness=0.8,
        mass_final_kg=3.0,
        energy_kwh=6.0,
        water_l=2.5,
        crew_min=45.0,
    )
    defaults.update(kwargs)
    return PredProps(**defaults)


def test_score_recipe_penalizes_resource_overuse():
    props = _make_props(energy_kwh=12.0, water_l=6.0, crew_min=80.0)
    target = {"rigidity": 0.85, "tightness": 0.8, "max_energy_kwh": 10.0, "max_water_l": 5.0, "max_crew_min": 60.0}

    score, breakdown = score_recipe(props, target)
    assert breakdown["penalties"]["energy_kwh"] > 0
    assert breakdown["penalties"]["water_l"] > 0
    assert breakdown["penalties"]["crew_min"] > 0
    assert score < breakdown["contributions"]["rigidez"] + breakdown["contributions"]["estanqueidad"]


def test_derive_auxiliary_signals_uses_classifier_probability():
    props = _make_props(
        comparisons={"tightness_classifier": {"pass_prob": 0.3}, "process_risk": {"risk": 0.7}},
    )
    aux = derive_auxiliary_signals(props, target={"max_energy_kwh": 10.0, "max_water_l": 5.0, "max_crew_min": 60.0})
    assert aux["seal_probability"] == pytest.approx(0.3)
    assert aux["passes_seal"] is False
    assert aux["process_risk"] == pytest.approx(0.7)
    assert aux["process_risk_label"] == "alto"


def test_rank_candidates_respects_custom_weights():
    target = {"rigidity": 0.85, "tightness": 0.8, "max_energy_kwh": 10.0, "max_water_l": 5.0, "max_crew_min": 60.0}

    high_perf = {"props": _make_props(rigidity=0.9, tightness=0.85, energy_kwh=9.8)}
    low_energy = {"props": _make_props(rigidity=0.78, tightness=0.82, energy_kwh=3.5)}

    ranked_default = rank_candidates([high_perf, low_energy], target)
    assert ranked_default[0]["props"].energy_kwh <= ranked_default[1]["props"].energy_kwh

    ranked_rigidity = rank_candidates(
        [high_perf, low_energy],
        target,
        weights={
            "rigidez": 4.5,
            "estanqueidad": 0.1,
            "energy_kwh": 0.0,
            "water_l": 0.0,
            "crew_min": 0.0,
        },
        penalties={"energy_kwh": 0.1, "water_l": 0.1, "crew_min": 0.1, "process_risk": 0.0},
    )
    assert ranked_rigidity[0]["props"].rigidity >= ranked_rigidity[1]["props"].rigidity
