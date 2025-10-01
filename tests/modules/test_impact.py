from __future__ import annotations

import pandas as pd

from app.modules.data_sources import RegolithThermalBundle, regolith_observation_lines
from app.modules.impact import parse_extra_blob


def test_parse_extra_blob_passthrough_dict():
    payload = {"hello": "world"}
    assert parse_extra_blob(payload) is payload


def test_parse_extra_blob_from_json_text():
    blob = '{"pressure": 17, "notes": "stable"}'
    assert parse_extra_blob(blob) == {"pressure": 17, "notes": "stable"}


def test_parse_extra_blob_from_key_value_pairs():
    blob = "pressure=17; notes=stable; extra"
    parsed = parse_extra_blob(blob)
    assert parsed["pressure"] == "17"
    assert parsed["notes"] == "stable"
    assert parsed["raw"] == "extra"


def test_parse_extra_blob_non_string_returns_empty():
    assert parse_extra_blob(42) == {}


def test_regolith_observation_lines_from_mapping():
    thermo = {
        "peaks": [
            {"temperature_c": 150.4, "species": "H2O", "signal_ppb": 0.123},
            {"temperature_c": 425.0, "species_label": "CO₂", "signal_ppb": 1.5},
            {"temperature_c": 800.0, "species": "SO₂"},
        ],
        "events": [
            {"event": "mass_99", "mass_pct": 99.0, "temperature_c": 180.0},
            {"event": "mass_95", "mass_pct": 95.0, "temperature_c": 500.0},
            {"event": "mass_90"},
        ],
    }

    lines = regolith_observation_lines(0.6, thermo)
    assert lines[0].startswith("60% de MGS-1")
    assert "H2O" in lines[1]
    assert "CO₂" in lines[2]
    assert len(lines) == 5  # mensaje base + dos picos y dos eventos


def test_regolith_observation_lines_from_bundle():
    peaks = pd.DataFrame(
        [
            {"temperature_c": 200.0, "species_label": "H₂O", "signal_ppb": 0.5},
            {"temperature_c": 700.0, "species": "CO₂"},
        ]
    )
    events = pd.DataFrame(
        [
            {"event": "mass_99", "mass_pct": 99.0, "temperature_c": 200.0},
        ]
    )
    bundle = RegolithThermalBundle(
        tg_curve=pd.DataFrame(),
        ega_curve=pd.DataFrame(),
        ega_long=pd.DataFrame(),
        gas_peaks=peaks,
        mass_events=events,
    )

    lines = regolith_observation_lines(0.4, bundle)
    assert lines[0].startswith("40% de MGS-1")
    assert any("H₂O" in line for line in lines)
    assert any("TG:" in line for line in lines)
