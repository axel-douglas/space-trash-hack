"""Helper utilities for the mission overview section of the app.

The goal of this module is to centralise logic that was previously spread
across several pages.  It provides lightweight helpers to:

* Load the enriched inventory (delegating to :func:`load_waste_df`).
* Compute operational metrics derived from the mass/volume data.
* Summarise the model state in a consumable structure for the UI.
* Prepare compact tables that expose external mission signals
  (``pc_*``/``aluminium_*`` columns) with sensible Streamlit configuration.

The rendering helpers intentionally keep their Streamlit footprint minimal so
that they can be reused from multiple pages without duplicating markup.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
import streamlit as st

from app.modules.io import (
    MissingDatasetError,
    format_missing_dataset_message,
    get_last_modified,
    load_waste_df,
)


# ---------------------------------------------------------------------------
# Inventory loading & metrics


def load_inventory_overview() -> pd.DataFrame:
    """Return a defensive copy of the enriched inventory dataframe."""

    try:
        df = load_waste_df()
    except MissingDatasetError as error:
        st.error(format_missing_dataset_message(error))
        st.stop()
        raise  # pragma: no cover - st.stop halts execution
    result = df.copy(deep=True)
    if "_problematic" in result.columns:
        result["_problematic"] = result["_problematic"].astype(bool)
    return result


def compute_mass_volume_metrics(df: pd.DataFrame | None) -> dict[str, float]:
    """Aggregate key operational metrics from the inventory dataframe."""

    if df is None or df.empty:
        return {"mass_kg": 0.0, "water_l": 0.0, "energy_kwh": 0.0, "volume_m3": 0.0}

    if "kg" in df.columns:
        mass = pd.to_numeric(df["kg"], errors="coerce").fillna(0.0)
    else:
        mass = pd.to_numeric(df.get("mass_kg"), errors="coerce").fillna(0.0)

    volume_l = pd.to_numeric(df.get("volume_l"), errors="coerce").fillna(0.0)
    moisture = pd.to_numeric(df.get("moisture_pct"), errors="coerce").fillna(0.0) / 100.0
    difficulty = (
        pd.to_numeric(df.get("difficulty_factor"), errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0, upper=3.0)
    )

    base_energy = 0.12  # kWh/kg para operaciones base
    max_energy = 0.70   # kWh/kg para operaciones complejas
    energy_per_kg = base_energy + (difficulty - 1.0) / 2.0 * (max_energy - base_energy)

    return {
        "mass_kg": float(mass.sum()),
        "water_l": float((mass * moisture).sum()),
        "energy_kwh": float((mass * energy_per_kg).sum()),
        "volume_m3": float(volume_l.sum()) / 1000.0,
    }


def compute_mission_summary(df: pd.DataFrame | None) -> dict[str, float]:
    """Return extended mission metrics derived from the inventory."""

    metrics = compute_mass_volume_metrics(df)
    if df is None or df.empty:
        metrics.update({"item_count": 0, "problematic_count": 0})
        return metrics

    metrics["item_count"] = int(df.shape[0])
    problematic_series = df.get("_problematic")
    if problematic_series is None:
        metrics["problematic_count"] = 0
    else:
        problematic_bool = pd.Series(problematic_series).astype(bool)
        metrics["problematic_count"] = int(problematic_bool.sum())
    return metrics


# ---------------------------------------------------------------------------
# Model state helpers


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def summarize_model_state(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Produce a compact summary of the model registry metadata."""

    metadata = metadata or {}
    ready = bool(metadata.get("ready", False))
    trained_at = (
        metadata.get("trained_at")
        or metadata.get("trained_on")
        or metadata.get("trained_label")
    )
    trained_dt = _parse_datetime(trained_at)
    n_samples = metadata.get("n_samples")
    try:
        sample_count = int(n_samples)
    except (TypeError, ValueError):
        sample_count = 0

    tone = "positive"
    notes: list[str] = []

    if trained_dt is None:
        tone = "danger"
        notes.append("Sin fecha de entrenamiento registrada")
        trained_display = "sin metadata"
    else:
        trained_dt = trained_dt.astimezone(timezone.utc)
        age_days = max((datetime.now(timezone.utc) - trained_dt).days, 0)
        trained_display = trained_dt.strftime("%Y-%m-%d %H:%M UTC")
        notes.append(f"Edad del modelo: {age_days} días")
        if age_days > 180:
            tone = "danger"
            notes.append("⚠️ Reentrená: supera 6 meses")
        elif age_days > 90:
            tone = "warning"
            notes.append("Sugerido reentrenar en <90 días")
        elif age_days <= 30:
            notes.append("Entrenamiento reciente (<30 días)")

    if sample_count <= 0:
        tone = "danger"
        notes.append("Sin muestras declaradas")
    else:
        notes.append(f"Muestras: {sample_count:,}")
        if sample_count < 400 and tone != "danger":
            tone = "warning"
            notes.append("Amplía dataset: <400 muestras")
        elif sample_count >= 1000:
            notes.append("Cobertura sólida (≥1k)")

    status_label = "✅ Modelo listo" if ready else "⚠️ Entrená localmente"

    return {
        "status_label": status_label,
        "trained_display": trained_display,
        "tone": tone,
        "notes": notes,
        "sample_count": sample_count,
    }


# ---------------------------------------------------------------------------
# Table preparation


def _format_number_column(name: str) -> st.column_config.NumberColumn:
    label = name.replace("_", " ").title()
    if name.endswith("_kg"):
        format_spec = "%.1f kg"
    elif name.endswith("_m3"):
        format_spec = "%.3f m³"
    else:
        format_spec = "%.2f"
    return st.column_config.NumberColumn(label, format=format_spec)


def prepare_material_summary(
    df: pd.DataFrame | None, *, max_rows: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a condensed dataframe and Streamlit column configuration."""

    if df is None or df.empty:
        return pd.DataFrame(), {}

    candidate_columns: list[str] = []
    for column in ("material_display", "category", "kg", "volume_l", "_problematic"):
        if column in df.columns:
            candidate_columns.append(column)

    extra_columns = [
        column
        for column in df.columns
        if column.startswith("pc_") or column.startswith("aluminium_")
    ]
    candidate_columns.extend(extra_columns)

    summary = df.loc[:, candidate_columns].copy()

    if "kg" in summary.columns:
        summary["kg"] = pd.to_numeric(summary["kg"], errors="coerce").fillna(0.0)
    if "volume_l" in summary.columns:
        summary["volume_l"] = pd.to_numeric(summary["volume_l"], errors="coerce").fillna(0.0) / 1000.0
        summary = summary.rename(columns={"volume_l": "volume_m3"})

    if "_problematic" in summary.columns:
        summary["_problematic"] = summary["_problematic"].astype(bool)

    if max_rows is not None and max_rows > 0:
        summary = summary.head(max_rows)

    column_config: dict[str, Any] = {}
    if "material_display" in summary.columns:
        column_config["material_display"] = st.column_config.TextColumn("Material")
    if "category" in summary.columns:
        column_config["category"] = st.column_config.TextColumn("Categoría")
    if "kg" in summary.columns:
        column_config["kg"] = st.column_config.NumberColumn("Masa (kg)", format="%.1f kg")
    if "volume_m3" in summary.columns:
        column_config["volume_m3"] = st.column_config.NumberColumn("Volumen (m³)", format="%.3f m³")
    if "_problematic" in summary.columns:
        column_config["_problematic"] = st.column_config.CheckboxColumn("Problemático", disabled=True)

    for column in extra_columns:
        if column in summary.columns:
            column_config[column] = _format_number_column(column)

    return summary, column_config


# ---------------------------------------------------------------------------
# Rendering helpers


def _format_metric(value: float, unit: str, *, precision: int = 1) -> str:
    if unit == "kg" and value >= 1000:
        return f"{value / 1000:.{precision}f} t"
    if unit in {"L", "kWh"} and value >= 1000:
        return f"{value / 1000:.{precision}f} {'m³' if unit == 'L' else 'MWh'}"
    if unit == "m³" and value >= 1:
        return f"{value:.{precision}f} m³"
    suffix = unit
    return f"{value:.{precision}f} {suffix}".strip()


def _format_delta(current: float, baseline: float | None, unit: str) -> str | None:
    if baseline is None:
        return None
    diff = current - baseline
    if abs(diff) < 1e-6:
        return None
    arrow = "↑" if diff > 0 else "↓"
    return f"{arrow} {abs(diff):.1f}{unit}"


def render_mission_objective(
    metrics: Mapping[str, float],
    *,
    baseline: Mapping[str, float] | None = None,
) -> None:
    """Render mission objective metrics using Streamlit primitives."""

    baseline = baseline or {}
    with st.container():
        col_mass, col_volume, col_water, col_energy = st.columns(4)
        mass = float(metrics.get("mass_kg", 0.0))
        volume = float(metrics.get("volume_m3", 0.0))
        water = float(metrics.get("water_l", 0.0))
        energy = float(metrics.get("energy_kwh", 0.0))

        col_mass.metric(
            "Masa total",
            _format_metric(mass, "kg"),
            delta=_format_delta(mass, baseline.get("mass_kg"), " kg"),
        )
        col_volume.metric(
            "Volumen total",
            _format_metric(volume, "m³", precision=2),
            delta=_format_delta(volume, baseline.get("volume_m3"), " m³"),
        )
        col_water.metric(
            "Agua estimada",
            _format_metric(water, "L"),
            delta=_format_delta(water, baseline.get("water_l"), " L"),
        )
        col_energy.metric(
            "Energía estimada",
            _format_metric(energy, "kWh"),
            delta=_format_delta(energy, baseline.get("energy_kwh"), " kWh"),
        )


def render_model_health(summary: Mapping[str, Any]) -> None:
    """Render the model health metric block."""

    with st.container():
        st.metric(
            label="Estado del modelo",
            value=str(summary.get("status_label", "—")),
            delta=f"Entrenado: {summary.get('trained_display', 'sin metadata')}",
        )
        notes = summary.get("notes")
        if isinstance(notes, Iterable):
            bullets = [str(note) for note in notes if str(note).strip()]
            if bullets:
                st.markdown("\n".join(f"• {note}" for note in bullets))


def render_material_summary(
    df: pd.DataFrame | None, *, max_rows: int | None = None
) -> None:
    """Render a compact dataframe with mission material signals."""

    summary, column_config = prepare_material_summary(df, max_rows=max_rows)
    with st.container():
        st.dataframe(
            summary,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
        )


def _resolve_inventory_loader(
    inventory_loader: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if inventory_loader is None:
        return load_inventory_overview()
    return inventory_loader()


def render_overview_dashboard(
    *, inventory_loader: Callable[[], pd.DataFrame] | None = None
) -> None:
    """Render the combined mission overview dashboard used on the home page."""

    from app.modules.ml_models import get_model_registry
    from app.modules.navigation import render_breadcrumbs, render_stepper, set_active_step
    from app.modules.paths import DATA_ROOT
    from app.modules.ui_blocks import initialise_frontend, load_theme

    st.set_page_config(page_title="Mission Overview", page_icon="🛰️", layout="wide")
    initialise_frontend()

    current_step = set_active_step("home")
    load_theme(show_hud=False)

    render_breadcrumbs(current_step)
    render_stepper(current_step)

    st.title("0) Mission Overview")

    try:
        inventory_df = _resolve_inventory_loader(inventory_loader)
    except MissingDatasetError as error:
        st.error(format_missing_dataset_message(error))
        st.stop()
        return

    mission_metrics = compute_mission_summary(inventory_df)
    render_mission_objective(mission_metrics)

    registry = get_model_registry()
    metadata = getattr(registry, "metadata", {}) if registry else {}
    model_summary = summarize_model_state(metadata)
    render_model_health(model_summary)

    if inventory_df is None or inventory_df.empty:
        st.info("No se encontró inventario para la misión actual.")
        return

    categories_column = inventory_df.get("category")
    unique_categories: list[str] = []
    if categories_column is not None:
        categories: Iterable[str] = (
            str(value).strip() for value in categories_column if str(value).strip()
        )
        unique_categories = sorted({category for category in categories})

    problematic_column = inventory_df.get("_problematic")
    problematic = 0
    if problematic_column is not None:
        problematic = int(pd.Series(problematic_column).astype(bool).sum())

    if unique_categories:
        categories_label = ", ".join(unique_categories[:5])
        if len(unique_categories) > 5:
            categories_label += "…"
        st.caption(f"Categorías: {categories_label}")
    st.caption(f"Problemáticos detectados: {problematic}")

    data_path = DATA_ROOT / "waste_inventory_sample.csv"
    last_modified = get_last_modified(data_path)
    if last_modified:
        st.caption(last_modified.strftime("Actualizado: %Y-%m-%d %H:%M"))

    render_material_summary(inventory_df, max_rows=20)

