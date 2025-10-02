# app/Home.py
import _bootstrap  # noqa: F401
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.modules.ml_models import get_model_registry
from app.modules.navigation import set_active_step
from app.modules.ui_blocks import load_theme

st.set_page_config(
    page_title="Rex-AI • Mission Copilot",
    page_icon="🛰️",
    layout="wide",
)

_current_step = set_active_step("brief")

load_theme()
model_registry = get_model_registry()


@st.cache_data
def load_inventory_sample() -> pd.DataFrame | None:
    sample_path = Path("data") / "waste_inventory_sample.csv"
    if not sample_path.exists():
        return None
    try:
        return pd.read_csv(sample_path)
    except Exception:
        return None


@st.cache_data
def load_feedback_preview(max_rows: int = 5) -> tuple[pd.DataFrame | None, int]:
    csv_path = Path("data") / "feedback_log.csv"
    if not csv_path.exists():
        return None, 0

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, 0

    if df.empty:
        return None, 0

    if "ts_iso" in df.columns:
        df["ts_iso"] = pd.to_datetime(df["ts_iso"], errors="coerce", utc=True)
        df = df.sort_values("ts_iso", ascending=False, na_position="last")
    else:
        df = df.sort_values(df.columns[0], ascending=False)

    total_rows = int(df.shape[0])
    preview = df.head(max_rows).copy()
    display_columns = [
        column
        for column in ["ts_iso", "astronaut", "scenario", "target_name", "issues", "notes"]
        if column in preview.columns
    ]
    if display_columns:
        preview = preview[display_columns]

    if "ts_iso" in preview.columns:
        preview["ts_iso"] = preview["ts_iso"].dt.strftime("%Y-%m-%d %H:%M UTC")

    return preview, total_rows


def format_mass(value: float | int | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value/1000:.1f} t"
    return f"{value:.0f} kg"


def format_water(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value/1000:.2f} m³"
    return f"{value:.0f} L"


def format_energy(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value/1000:.2f} MWh"
    return f"{value:.0f} kWh"


def training_health_summary(
    trained_dt: datetime | str | None, n_samples: int | None
) -> tuple[str, list[str]]:
    """Return tone and bullet points describing training freshness."""

    tone = "positive"
    notes: list[str] = []

    normalized_dt: datetime | None = None

    if not trained_dt:
        notes.append("Sin fecha de entrenamiento registrada")
        tone = "danger"
    else:
        parsed_note: str | None = None
        if isinstance(trained_dt, datetime):
            normalized_dt = trained_dt
        elif isinstance(trained_dt, str):
            iso_str = trained_dt.strip()
            if iso_str.endswith("Z"):
                iso_str = iso_str[:-1] + "+00:00"
            try:
                normalized_dt = datetime.fromisoformat(iso_str)
            except ValueError:
                parsed_note = "Formato de fecha no reconocido"
        else:
            parsed_note = f"Tipo de fecha inesperado: {type(trained_dt).__name__}"

        if normalized_dt is None:
            tone = "warning"
            notes.append(parsed_note or "Fecha de entrenamiento no interpretable")
        else:
            try:
                if normalized_dt.tzinfo is None:
                    normalized_dt = normalized_dt.replace(tzinfo=timezone.utc)
                else:
                    normalized_dt = normalized_dt.astimezone(timezone.utc)
                age_days = max((datetime.now(timezone.utc) - normalized_dt).days, 0)
                notes.append(f"Edad del modelo: {age_days} días")
                if age_days > 180:
                    tone = "danger"
                    notes.append("⚠️ Reentrená: supera 6 meses")
                elif age_days > 90:
                    tone = "warning"
                    notes.append("Sugerido reentrenar en <90 días")
                elif age_days <= 30:
                    notes.append("Entrenamiento reciente (<30 días)")
            except Exception as exc:  # pragma: no cover - defensive
                tone = "warning"
                notes.append(f"No se pudo normalizar fecha: {exc}")

    sample_count = int(n_samples or 0)
    if sample_count <= 0:
        tone = "danger"
        notes.append("Sin muestras declaradas")
    else:
        notes.append(f"Muestras: {sample_count:,}")
        if sample_count < 400:
            tone = "warning"
            notes.append("Amplía dataset: <400 muestras")
        elif sample_count >= 1000:
            notes.append("Cobertura sólida (≥1k)")

    return tone, notes


def uncertainty_health_summary(metadata: dict[str, Any]) -> tuple[str, list[str]]:
    """Compute tone and highlights from residual_std metadata."""

    residual_std = metadata.get("residual_std")
    if not isinstance(residual_std, dict) or not residual_std:
        return "danger", ["Sin residuales reportados"]

    def _get_float(key: str, default: float = 0.0) -> float:
        try:
            return float(residual_std.get(key, default))
        except (TypeError, ValueError):
            return default

    energy_std = _get_float("energy_kwh")
    water_std = _get_float("water_l")
    crew_std = _get_float("crew_min")
    rigidity_std = _get_float("rigidez")
    tightness_std = _get_float("estanqueidad")

    tone = "positive"
    if any(value == 0 for value in (energy_std, water_std, crew_std)):
        tone = "warning"

    if energy_std > 22000 or crew_std > 6000 or water_std > 220:
        tone = "danger"
    elif energy_std > 16000 or crew_std > 4500 or water_std > 160:
        tone = "warning"

    if rigidity_std > 0.45 or tightness_std > 0.12:
        tone = "warning"

    notes = [
        f"σ energía: {energy_std:.0f} kWh",
        f"σ agua: {water_std:.0f} L",
        f"σ crew: {crew_std:.0f} min",
        f"σ rigidez: {rigidity_std:.3f}",
        f"σ estanqueidad: {tightness_std:.3f}",
    ]

    return tone, notes


def compute_inventory_totals(df: pd.DataFrame | None) -> dict[str, float]:
    if df is None or df.empty:
        return {}

    mass = pd.to_numeric(df.get("mass_kg"), errors="coerce").fillna(0.0)
    moisture = pd.to_numeric(df.get("moisture_pct"), errors="coerce").fillna(0.0) / 100.0
    difficulty = (
        pd.to_numeric(df.get("difficulty_factor"), errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0, upper=3.0)
    )

    base_energy = 0.12  # kWh/kg para operaciones base (shredder)
    max_energy = 0.70   # kWh/kg para operaciones complejas (sinterizado)
    energy_per_kg = base_energy + (difficulty - 1.0) / 2.0 * (max_energy - base_energy)

    totals = {
        "mass_kg": float(mass.sum()),
        "water_l": float((mass * moisture).sum()),
        "energy_kwh": float((mass * energy_per_kg).sum()),
    }
    return totals


def compute_delta_strings(
    key: str,
    current: float,
    baseline: dict[str, float] | None,
    unit: str,
    *,
    precision: int = 0,
) -> tuple[str | None, str]:
    if baseline is None or key not in baseline:
        return None, "Sin histórico"

    diff = current - baseline[key]
    tolerance = 1.0 if precision == 0 else 0.1
    if abs(diff) < tolerance:
        return None, "Sin cambios vs. baseline"

    arrow = "↑" if diff > 0 else "↓"
    formatted = f"{abs(diff):.{precision}f}"
    label = f"{arrow} {formatted}{unit}".rstrip()
    detail = f"{label} vs. baseline guardado"
    return label, detail


def describe_baseline_caption(state: dict | None) -> str:
    if not state:
        return "Baseline pendiente: guardá inventario para registrar histórico."
    saved_at = state.get("saved_at") if isinstance(state, dict) else None
    if isinstance(saved_at, datetime):
        timestamp = saved_at.astimezone(timezone.utc).strftime("%d %b %Y %H:%M UTC")
        return f"Baseline desde último save_waste_df ({timestamp})."
    return "Baseline según último save_waste_df disponible."


# ──────────── Lectura segura de metadata del modelo ────────────
trained_at_raw = model_registry.metadata.get("trained_at")
trained_label_value = (
    model_registry.metadata.get("trained_label")
    or model_registry.metadata.get("trained_on")
)

try:
    trained_dt = datetime.fromisoformat(trained_at_raw) if trained_at_raw else None
except Exception:
    trained_dt = None

trained_at_display = (
    trained_dt.strftime("%d %b %Y %H:%M UTC") if trained_dt else "sin metadata"
)

trained_combo = model_registry.trained_label()
if trained_at_display == "sin metadata" and trained_combo and trained_combo != "—":
    trained_at_display = trained_combo

if not trained_label_value and trained_combo and trained_combo != "—":
    trained_label_value = trained_combo.split(" · ", 1)[0]

trained_label_value = trained_label_value or "—"

n_samples = model_registry.metadata.get("n_samples")
model_name = model_registry.metadata.get("model_name", "rexai-rf-ensemble")
feature_count = len(getattr(model_registry, "feature_names", []) or [])

ready_label = "✅ Modelo listo" if model_registry.ready else "⚠️ Entrená localmente"

inventory_session_df = st.session_state.get("inventory_data")
inventory_loaded = isinstance(inventory_session_df, pd.DataFrame) and not inventory_session_df.empty

if inventory_loaded:
    inventory_reference_df: pd.DataFrame | None = inventory_session_df
else:
    inventory_reference_df = load_inventory_sample()

inventory_count = int(inventory_reference_df.shape[0]) if isinstance(inventory_reference_df, pd.DataFrame) else 0

baseline_state = st.session_state.get("_inventory_baseline")
if not isinstance(baseline_state, dict):
    baseline_state = None

baseline_totals: dict[str, float] | None = None
if baseline_state is not None:
    baseline_df_candidate = baseline_state.get("df")
    if isinstance(baseline_df_candidate, pd.DataFrame):
        baseline_totals = compute_inventory_totals(baseline_df_candidate)

inventory_totals = compute_inventory_totals(inventory_reference_df)

try:
    n_samples_int = int(n_samples) if n_samples is not None else None
except (TypeError, ValueError):
    n_samples_int = None

training_tone, training_notes = training_health_summary(trained_dt, n_samples_int)
uncertainty_tone, uncertainty_notes = uncertainty_health_summary(model_registry.metadata)

feedback_preview, feedback_total = load_feedback_preview()

latest_feedback_label = "Esperando feedback"
if feedback_total and feedback_preview is not None and not feedback_preview.empty:
    first_row = feedback_preview.iloc[0]
    ts_value = first_row.get("ts_iso")
    if isinstance(ts_value, str) and ts_value:
        latest_feedback_label = f"Último: {ts_value}"
    else:
        latest_feedback_label = "Últimos registros listos"
elif feedback_total:
    latest_feedback_label = "Logs disponibles"

inventory_metric_value = (
    f"{inventory_count} ítem{'s' if inventory_count != 1 else ''}"
    if inventory_reference_df is not None and not inventory_reference_df.empty
    else "Sin inventario"
)
inventory_metric_delta = (
    "Fuente: sesión actual" if inventory_loaded else "Usando muestra NASA"
)

feedback_metric_value = (
    f"{feedback_total} registro{'s' if feedback_total != 1 else ''}"
    if feedback_total
    else "Sin registros"
)

st.title("Panel operativo Rex-AI")
st.caption(
    "Monitoreá el estado del modelo, verificá tu inventario normalizado y seguí los últimos "
    "feedbacks de la tripulación sin salir del brief."
)

col_model, col_inventory, col_feedback = st.columns(3)

with col_model:
    st.metric(
        label="Estado del modelo",
        value=ready_label,
        delta=f"Entrenado: {trained_at_display}",
    )
    st.caption(f"Modelo {model_name} · {feature_count} features")
    if training_notes:
        st.markdown("\n".join(f"• {note}" for note in training_notes))

with col_inventory:
    st.metric(
        label="Inventario normalizado",
        value=inventory_metric_value,
        delta=inventory_metric_delta,
    )
    st.caption(describe_baseline_caption(baseline_state))

with col_feedback:
    st.metric(
        label="Feedback de crew",
        value=feedback_metric_value,
        delta=latest_feedback_label,
    )
    if feedback_total:
        st.caption("Consolida registros en 8) Feedback & Impact.")
    else:
        st.caption("Capturá feedback tras cada corrida para habilitar retraining.")

st.divider()

st.subheader("Inventario en curso")
if inventory_totals:
    mass_delta, mass_detail = compute_delta_strings(
        "mass_kg", inventory_totals.get("mass_kg", 0.0), baseline_totals, " kg"
    )
    water_delta, water_detail = compute_delta_strings(
        "water_l", inventory_totals.get("water_l", 0.0), baseline_totals, " L"
    )
    energy_delta, energy_detail = compute_delta_strings(
        "energy_kwh", inventory_totals.get("energy_kwh", 0.0), baseline_totals, " kWh"
    )

    tot_mass = inventory_totals.get("mass_kg")
    tot_water = inventory_totals.get("water_l")
    tot_energy = inventory_totals.get("energy_kwh")

    col_mass, col_water, col_energy = st.columns(3)
    col_mass.metric(
        "Masa total",
        format_mass(tot_mass),
        delta=mass_delta if mass_delta is not None else None,
    )
    col_mass.caption(mass_detail)

    col_water.metric(
        "Agua estimada",
        format_water(tot_water),
        delta=water_delta if water_delta is not None else None,
    )
    col_water.caption(water_detail)

    col_energy.metric(
        "Energía estimada",
        format_energy(tot_energy),
        delta=energy_delta if energy_delta is not None else None,
    )
    col_energy.caption(energy_detail)
else:
    st.info("Cargá y normalizá tu inventario para calcular consumos operativos.")

with st.expander(
    "Ver inventario normalizado", expanded=inventory_reference_df is not None and not inventory_reference_df.empty
):
    if inventory_reference_df is not None and not inventory_reference_df.empty:
        display_columns = [
            column
            for column in [
                "id",
                "category",
                "material",
                "mass_kg",
                "difficulty_factor",
                "flags",
            ]
            if column in inventory_reference_df.columns
        ]
        st.dataframe(
            inventory_reference_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No hay inventario disponible todavía.")

st.divider()

st.subheader("Feedback operativo")
if feedback_preview is not None and not feedback_preview.empty:
    st.caption(f"{feedback_total} registro{'s' if feedback_total != 1 else ''} totales")
else:
    st.caption("Sin registros cargados en feedback_log.csv")

with st.expander("Últimos envíos de crew", expanded=feedback_preview is not None and not feedback_preview.empty):
    if feedback_preview is not None and not feedback_preview.empty:
        st.dataframe(feedback_preview, use_container_width=True, hide_index=True)
    else:
        st.info("Aún no hay feedback registrado. Capturá la próxima corrida en 8) Feedback & Impact.")

with st.expander("Notas de salud del modelo", expanded=training_tone != "positive" or uncertainty_tone != "positive"):
    st.markdown("**Entrenamiento**")
    st.markdown("\n".join(f"• {note}" for note in training_notes))
    st.markdown("**Incertidumbre**")
    st.markdown("\n".join(f"• {note}" for note in uncertainty_notes))
