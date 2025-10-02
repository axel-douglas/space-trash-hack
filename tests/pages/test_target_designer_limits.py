import numpy as np
import pandas as pd

import app.modules.luxe_components as luxe


def _sample_presets():
    return [
        {
            "name": "Preset A",
            "rigidity": 0.4,
            "tightness": 0.5,
            "max_water_l": 0.45,
            "max_energy_kwh": 0.52,
            "max_crew_min": 45,
        },
        {
            "name": "Preset B",
            "rigidity": 0.6,
            "tightness": 0.7,
            "max_water_l": 0.5,
            "max_energy_kwh": 0.65,
            "max_crew_min": 60,
        },
    ]


def _waste_df():
    return pd.DataFrame(
        {
            "_source_volume_l": [1200, 2600, 8000, 10000],
            "kg": [100.0, 240.0, 600.0, 180.0],
        }
    )


def test_water_limits_follow_baseline(monkeypatch):
    monkeypatch.setattr(luxe, "load_waste_df", lambda: _waste_df())

    limits = luxe._compute_target_limits(_sample_presets())

    volume_q90 = np.quantile(_waste_df()["_source_volume_l"], 0.9)
    expected = round(volume_q90 * luxe._WATER_L_PER_VOLUME_L_BASELINE, 2)

    assert limits["max_water_l"]["max"] == expected
    assert "NASA baseline" in limits["max_water_l"]["help"]


def test_energy_limits_follow_baseline(monkeypatch):
    monkeypatch.setattr(luxe, "load_waste_df", lambda: _waste_df())

    limits = luxe._compute_target_limits(_sample_presets())

    mass_q90 = np.quantile(_waste_df()["kg"], 0.9)
    expected = round(mass_q90 * luxe._ENERGY_KWH_PER_KG_BASELINE, 2)

    assert limits["max_energy_kwh"]["max"] == expected
    assert "NASA baseline" in limits["max_energy_kwh"]["help"]
