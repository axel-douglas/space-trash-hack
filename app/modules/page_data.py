"""Helper utilities for building data tables shown in Streamlit pages.

These helpers centralise the transformation of candidate payloads into
``pandas`` objects so that Streamlit views can rely on simple, well-tested
dataframes.  They are intentionally lightweight and avoid any Streamlit
imports in order to remain easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import pandas as pd


MetricSource = Mapping[str, object] | SimpleNamespace | None


def _safe_float(value: object | None) -> float | None:
    """Convert ``value`` into a float when possible.

    ``None`` and invalid inputs return ``None`` instead of propagating ``NaN``
    so the calling code can decide how to display missing data.
    """

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):  # type: ignore[arg-type]
        return None
    return float(number)


def _value_from(source: MetricSource, attr: str) -> float | None:
    """Fetch ``attr`` from ``source`` supporting mappings and namespaces."""

    if source is None:
        return None
    if isinstance(source, Mapping):
        return _safe_float(source.get(attr))
    if hasattr(source, attr):
        return _safe_float(getattr(source, attr))
    return None


def _format_interval(bounds: Sequence[object] | None) -> str:
    """Return a compact string for a two-value confidence interval."""

    if not isinstance(bounds, Sequence) or len(bounds) < 2:
        return ""
    lo = _safe_float(bounds[0])
    hi = _safe_float(bounds[1])
    if lo is None or hi is None:
        return ""
    return f"{lo:.3f} – {hi:.3f}"


@dataclass
class CandidateMetricConfig:
    label: str
    key: str


_CANDIDATE_METRIC_CONFIG: tuple[CandidateMetricConfig, ...] = (
    CandidateMetricConfig("Rigidez", "rigidity"),
    CandidateMetricConfig("Estanqueidad", "tightness"),
    CandidateMetricConfig("Energía (kWh)", "energy_kwh"),
    CandidateMetricConfig("Agua (L)", "water_l"),
    CandidateMetricConfig("Crew (min)", "crew_min"),
)


def build_candidate_metric_table(
    props: MetricSource,
    heuristics: MetricSource,
    score: float | None,
    confidence: Mapping[str, Sequence[object]] | None,
    uncertainty: Mapping[str, object] | None,
) -> pd.DataFrame:
    """Return a dataframe summarising key metrics for the selected recipe."""

    rows: list[dict[str, object]] = []

    if score is not None:
        rows.append(
            {
                "Indicador": "Score total",
                "IA Rex-AI": float(score),
                "Heurística": float("nan"),
                "Δ (IA - Heurística)": float("nan"),
                "CI 95%": "",
                "σ": float("nan"),
            }
        )

    confidence = confidence or {}
    uncertainty = uncertainty or {}

    for cfg in _CANDIDATE_METRIC_CONFIG:
        ml_value = _value_from(props, cfg.key)
        heur_value = _value_from(heuristics, cfg.key)
        delta = None
        if ml_value is not None and heur_value is not None:
            delta = ml_value - heur_value

        ci_text = _format_interval(confidence.get(cfg.key))
        sigma = _safe_float(uncertainty.get(cfg.key))

        rows.append(
            {
                "Indicador": cfg.label,
                "IA Rex-AI": ml_value if ml_value is not None else float("nan"),
                "Heurística": heur_value if heur_value is not None else float("nan"),
                "Δ (IA - Heurística)": delta if delta is not None else float("nan"),
                "CI 95%": ci_text,
                "σ": sigma if sigma is not None else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    return df


def build_resource_table(
    props: MetricSource,
    target_limits: Mapping[str, object] | None,
) -> pd.DataFrame:
    """Return a dataframe with resource usage against mission limits."""

    limits = target_limits or {}

    mapping = (
        ("Energía (kWh)", "energy_kwh", limits.get("max_energy_kwh")),
        ("Agua (L)", "water_l", limits.get("max_water_l")),
        ("Crew (min)", "crew_min", limits.get("max_crew_min")),
    )

    rows: list[dict[str, object]] = []
    for label, key, limit in mapping:
        usage = _value_from(props, key)
        limit_value = _safe_float(limit)
        utilisation = float("nan")
        if usage is not None and limit_value not in {None, 0}:
            utilisation = (usage / limit_value) * 100
        rows.append(
            {
                "Recurso": label,
                "Uso": usage if usage is not None else float("nan"),
                "Límite": limit_value if limit_value is not None else float("nan"),
                "Utilización (%)": utilisation,
            }
        )

    return pd.DataFrame(rows)


def build_ranking_table(candidates: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Create a ranking dataframe for generator candidates."""

    rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates, start=1):
        props = candidate.get("props")
        aux = candidate.get("auxiliary") or {}
        rows.append(
            {
                "Rank": idx,
                "Score": _safe_float(candidate.get("score")),
                "Proceso": f"{candidate.get('process_id', '')} · {candidate.get('process_name', '')}".strip(
                    " ·"
                ),
                "Rigidez": _value_from(props, "rigidity"),
                "Estanqueidad": _value_from(props, "tightness"),
                "Energía (kWh)": _value_from(props, "energy_kwh"),
                "Agua (L)": _value_from(props, "water_l"),
                "Crew (min)": _value_from(props, "crew_min"),
                "Seal": "✅" if aux.get("passes_seal", True) else "⚠️",
                "Riesgo": aux.get("process_risk_label", "—"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "Rank",
            "Score",
            "Proceso",
            "Rigidez",
            "Estanqueidad",
            "Energía (kWh)",
            "Agua (L)",
            "Crew (min)",
            "Seal",
            "Riesgo",
        ])

    df = pd.DataFrame(rows)
    df.sort_values("Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_export_kpi_table(df_plot: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KPIs for the export page."""

    if df_plot.empty:
        return pd.DataFrame(
            [
                {
                    "KPI": "Opciones válidas",
                    "Valor": 0.0,
                }
            ]
        )

    rows = [
        {"KPI": "Opciones válidas", "Valor": float(len(df_plot))},
        {"KPI": "Score máximo", "Valor": float(df_plot["Score"].max())},
        {"KPI": "Mín. Agua", "Valor": float(df_plot["Agua (L)"].min())},
        {"KPI": "Mín. Energía", "Valor": float(df_plot["Energía (kWh)"].min())},
    ]

    for column, label in [
        ("ρ ref (g/cm³)", "ρ ref promedio"),
        ("σₜ ref (MPa)", "σₜ ref promedio"),
        ("σₜ Al (MPa)", "σₜ Al promedio"),
    ]:
        if column in df_plot:
            series = pd.to_numeric(df_plot[column], errors="coerce").dropna()
            if not series.empty:
                rows.append({"KPI": label, "Valor": float(series.mean())})

    return pd.DataFrame(rows)

