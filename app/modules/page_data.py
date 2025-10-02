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


_POLYMER_METRIC_ALIAS = {
    "pc_density_density_g_per_cm3": "density_g_cm3",
    "pc_mechanics_tensile_strength_mpa": "tensile_mpa",
    "pc_mechanics_modulus_gpa": "modulus_gpa",
    "pc_thermal_glass_transition_c": "glass_c",
    "pc_ignition_ignition_temperature_c": "ignition_c",
    "pc_ignition_burn_time_min": "burn_min",
}

_ALUMINIUM_METRIC_ALIAS = {
    "aluminium_tensile_strength_mpa": "tensile_mpa",
    "aluminium_yield_strength_mpa": "yield_mpa",
    "aluminium_elongation_pct": "elongation_pct",
}

_POLYMER_REFERENCE_LABELS = {
    "density_g_cm3": "ρ ref (g/cm³)",
    "tensile_mpa": "σₜ ref (MPa)",
    "modulus_gpa": "E ref (GPa)",
    "glass_c": "Tg (°C)",
    "ignition_c": "Ignición (°C)",
    "burn_min": "Burn (min)",
}

_ALUMINIUM_REFERENCE_LABELS = {
    "tensile_mpa": "σₜ Al (MPa)",
    "yield_mpa": "σᵧ Al (MPa)",
    "elongation_pct": "ε Al (%)",
}

_FEEDBACK_LABELS = {
    "feedback_overall": "Satisfacción general",
    "feedback_rigidity": "Rigidez percibida",
    "feedback_porosity": "Porosidad",
    "feedback_surface": "Acabado superficial",
    "feedback_bonding": "Union / bonding",
    "feedback_ease": "Facilidad operativa",
}


def collect_reference_profiles(candidate: Mapping[str, object] | None, inventory: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    """Return averaged external reference metrics for ``candidate``.

    The resulting mapping contains one entry per material family (``polymer``
    and ``aluminium``) with two keys:

    - ``metrics``: Dict[str, float] – averaged numeric attributes.
    - ``labels``: list[str] – distinct label fields found in the inventory.
    """

    if not candidate or not isinstance(candidate, Mapping):
        return {}
    if not isinstance(inventory, pd.DataFrame) or inventory.empty:
        return {}

    raw_ids = candidate.get("source_ids") or []
    ids = {str(value).strip() for value in raw_ids if str(value).strip()}
    if not ids:
        return {}

    mask = pd.Series(False, index=inventory.index)
    if "id" in inventory.columns:
        mask |= inventory["id"].astype(str).isin(ids)
    if "_source_id" in inventory.columns:
        mask |= inventory["_source_id"].astype(str).isin(ids)

    subset = inventory.loc[mask].copy()
    if subset.empty:
        return {}

    payload: dict[str, dict[str, object]] = {}

    def _section(
        metric_alias: Mapping[str, str],
        label_columns: Sequence[str],
    ) -> dict[str, object] | None:
        metrics: dict[str, float] = {}
        for column, alias in metric_alias.items():
            if column not in subset.columns:
                continue
            series = pd.to_numeric(subset[column], errors="coerce").dropna()
            if series.empty:
                continue
            metrics[alias] = float(series.mean())

        labels: list[str] = []
        for column in label_columns:
            if column not in subset.columns:
                continue
            series = subset[column].dropna().astype(str).str.strip()
            labels.extend(value for value in series if value)

        if not metrics and not labels:
            return None

        return {"metrics": metrics, "labels": sorted(dict.fromkeys(labels))}

    polymer_section = _section(
        _POLYMER_METRIC_ALIAS,
        (
            "pc_density_sample_label",
            "pc_mechanics_sample_label",
            "pc_thermal_sample_label",
            "pc_ignition_sample_label",
        ),
    )
    if polymer_section:
        payload["polymer"] = polymer_section

    aluminium_section = _section(
        _ALUMINIUM_METRIC_ALIAS,
        (
            "aluminium_processing_route",
            "aluminium_class_id",
        ),
    )
    if aluminium_section:
        payload["aluminium"] = aluminium_section

    return payload


def build_candidate_export_table(
    candidates: Iterable[Mapping[str, object]],
    inventory: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return a normalised dataframe used across export-related pages."""

    rows: list[dict[str, object]] = []
    inventory = inventory if isinstance(inventory, pd.DataFrame) else None

    for idx, candidate in enumerate(candidates, start=1):
        props = candidate.get("props") if isinstance(candidate, Mapping) else None
        profiles = collect_reference_profiles(candidate, inventory)

        materials = []
        if isinstance(candidate, Mapping):
            for material in candidate.get("materials", []) or []:
                text = str(material).strip()
                if text:
                    materials.append(text)

        row = {
            "Opción": idx,
            "Proceso": (
                f"{candidate.get('process_id', '')} · {candidate.get('process_name', '')}".strip(" ·")
                if isinstance(candidate, Mapping)
                else ""
            ),
            "Score": _safe_float(candidate.get("score")) if isinstance(candidate, Mapping) else None,
            "Energía (kWh)": _value_from(props, "energy_kwh"),
            "Agua (L)": _value_from(props, "water_l"),
            "Crew (min)": _value_from(props, "crew_min"),
            "Masa (kg)": _value_from(props, "mass_final_kg"),
            "Rigidez": _value_from(props, "rigidity"),
            "Estanqueidad": _value_from(props, "tightness"),
            "Materiales": ", ".join(materials),
        }

        for family, section in profiles.items():
            metrics = section.get("metrics", {}) if isinstance(section, Mapping) else {}
            for key, value in metrics.items():
                if family == "aluminium":
                    label_map = _ALUMINIUM_REFERENCE_LABELS
                else:
                    label_map = _POLYMER_REFERENCE_LABELS
                label = label_map.get(key)
                if not label:
                    continue
                row[label] = float(value)

        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "Opción",
                "Proceso",
                "Score",
                "Energía (kWh)",
                "Agua (L)",
                "Crew (min)",
                "Masa (kg)",
                "Rigidez",
                "Estanqueidad",
                "Materiales",
            ]
        )

    df = pd.DataFrame(rows)
    numeric_columns = [
        "Score",
        "Energía (kWh)",
        "Agua (L)",
        "Crew (min)",
        "Masa (kg)",
        "Rigidez",
        "Estanqueidad",
    ]
    reference_columns = list(_POLYMER_REFERENCE_LABELS.values()) + list(_ALUMINIUM_REFERENCE_LABELS.values())
    numeric_columns.extend(column for column in reference_columns if column in df.columns)
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def build_material_summary_table(
    candidates: Iterable[Mapping[str, object]],
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Aggregate material usage across generated candidates."""

    rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            continue
        materials = candidate.get("materials") or []
        weights = candidate.get("weights") or []
        for material, weight in zip(materials, weights):
            amount = _safe_float(weight)
            if amount is None:
                continue
            rows.append({
                "Material": str(material),
                "Peso": amount,
                "Opción": idx,
            })

    if not rows:
        return pd.DataFrame(columns=["Material", "Peso total", "Participaciones", "Peso promedio", "% sobre total"])

    df = pd.DataFrame(rows)
    grouped = df.groupby("Material", as_index=False).agg(
        Peso_total=("Peso", "sum"),
        Participaciones=("Opción", "nunique"),
        Peso_promedio=("Peso", "mean"),
    )
    total_weight = grouped["Peso_total"].sum()
    grouped["% sobre total"] = grouped["Peso_total"].apply(
        lambda value: float(value) / total_weight * 100 if total_weight else float("nan")
    )
    grouped.sort_values("Peso_total", ascending=False, inplace=True)
    grouped.rename(columns={"Peso_total": "Peso total", "Peso_promedio": "Peso promedio"}, inplace=True)
    return grouped.head(top_n).reset_index(drop=True)


def build_feedback_summary_table(feedback_df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated feedback scores for decision making."""

    if not isinstance(feedback_df, pd.DataFrame) or feedback_df.empty:
        return pd.DataFrame(columns=["Métrica", "Promedio", "Observaciones"])

    rows: list[dict[str, object]] = []
    for column, label in _FEEDBACK_LABELS.items():
        if column not in feedback_df.columns:
            continue
        numeric = pd.to_numeric(feedback_df[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        rows.append(
            {
                "Métrica": label,
                "Promedio": float(numeric.mean()),
                "Observaciones": int(numeric.count()),
            }
        )

    return pd.DataFrame(rows)

