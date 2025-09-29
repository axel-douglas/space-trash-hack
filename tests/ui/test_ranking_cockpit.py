import re
import sys
import types
from pathlib import Path

stub_joblib = types.ModuleType("joblib")
stub_joblib.load = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("joblib", stub_joblib)

modules_pkg = types.ModuleType("app.modules")
modules_pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "app" / "modules")]
sys.modules.setdefault("app.modules", modules_pkg)

try:
    from app.modules.luxe_components import MetricSpec, RankingCockpit
finally:
    sys.modules.pop("app.modules", None)


def test_ranking_cockpit_markup() -> None:
    entries = [
        {
            "Rank": 1,
            "Score": 0.91,
            "Proceso": "P04 · Sinter EVA",
            "Rigidez": 12.4,
            "Estanqueidad": 0.88,
            "Energía (kWh)": 14.2,
            "Agua (L)": 9.0,
            "Crew (min)": 36.0,
            "Seal": "✅",
            "Riesgo": "Bajo",
        },
        {
            "Rank": 2,
            "Score": 0.83,
            "Proceso": "P02 · Laminado CTB",
            "Rigidez": 11.6,
            "Estanqueidad": 0.9,
            "Energía (kWh)": 12.8,
            "Agua (L)": 6.0,
            "Crew (min)": 42.0,
            "Seal": "⚠️",
            "Riesgo": "Medio",
        },
        {
            "Rank": 3,
            "Score": 0.78,
            "Proceso": "P03 · Regolito",
            "Rigidez": 10.9,
            "Estanqueidad": 0.84,
            "Energía (kWh)": 11.3,
            "Agua (L)": 3.2,
            "Crew (min)": 50.0,
            "Seal": "✅",
            "Riesgo": "Crítico",
        },
    ]

    specs = [
        MetricSpec("Rigidez", "Rigidez", "{:.1f}"),
        MetricSpec("Estanqueidad", "Estanqueidad", "{:.2f}"),
        MetricSpec("Energía (kWh)", "Energía", "{:.1f}", unit="kWh"),
        MetricSpec("Agua (L)", "Agua", "{:.1f}", unit="L", higher_is_better=False),
    ]

    cockpit = RankingCockpit(entries=entries, metric_specs=specs, key="test_ranking")
    scales = cockpit._metric_scales(entries)
    prepared = cockpit._prepare_entries(entries, scales)
    markup = cockpit._build_cards(prepared, selected_idx=1)

    assert markup.count("ranking-card__rank") == 3
    assert "selected" in markup
    assert "seal-warn" in markup
    assert "tone-high" in markup
    assert "kWh" in markup

    agua_fills = [
        float(value)
        for value in re.findall(
            r"data-metric='Agua \(L\)'.*?--fill:([0-9.]+)%;",
            markup,
            flags=re.S,
        )
    ]
    assert len(agua_fills) == 3
    assert agua_fills[0] < agua_fills[1] < agua_fills[2]
