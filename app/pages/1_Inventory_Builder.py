import _bootstrap  # noqa: F401

from datetime import datetime, timezone

import altair as alt
import math
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from app.modules.io import load_waste_df, save_waste_df
from app.modules.navigation import set_active_step
from app.modules.ui_blocks import load_theme, minimal_button
from app.modules.problematic import problematic_mask

_SAVE_SUCCESS_FLAG = "_inventory_save_success"


def _trigger_rerun() -> None:
    """Trigger a Streamlit rerun regardless of version."""
    try:
        st.rerun()
    except AttributeError:  # pragma: no cover - legacy fallback
        st.experimental_rerun()

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Inventario", page_icon="🧱", layout="wide")

set_active_step("inventory")

load_theme()

save_success = st.session_state.pop(_SAVE_SUCCESS_FLAG, False)
if save_success:
    st.success("Inventario guardado.")

st.title("1) Inventario de residuos")
st.caption(
    "Este inventario refleja ítems no-metabólicos problemáticos para NASA: "
    "pouches multicapa, espumas (ZOTEK F30), bolsas EVA/CTB, textiles/wipes y guantes de nitrilo. "
    "El generador prioriza valorizarlos y, cuando corresponde, mezcla **regolito MGS-1**."
)

# --------------------- helpers robustos de columnas ---------------------
POLYMER_SAMPLE_COLUMNS = (
    "pc_density_sample_label",
    "pc_mechanics_sample_label",
    "pc_thermal_sample_label",
    "pc_ignition_sample_label",
)

POLYMER_NUMERIC_COLUMNS = (
    "pc_density_density_g_per_cm3",
    "pc_density_density_kg_m3",
    "pc_mechanics_tensile_strength_mpa",
    "pc_mechanics_stress_mpa",
    "pc_mechanics_yield_strength_mpa",
    "pc_mechanics_modulus_gpa",
    "pc_mechanics_strain_pct",
    "pc_thermal_glass_transition_c",
    "pc_thermal_onset_temperature_c",
    "pc_thermal_heat_capacity_j_per_g_k",
    "pc_thermal_heat_flow_w_per_g",
    "pc_ignition_ignition_temperature_c",
    "pc_ignition_burn_time_min",
)

ALUMINIUM_SAMPLE_COLUMNS = (
    "aluminium_processing_route",
    "aluminium_class_id",
)

ALUMINIUM_NUMERIC_COLUMNS = (
    "aluminium_tensile_strength_mpa",
    "aluminium_yield_strength_mpa",
    "aluminium_elongation_pct",
)

EXTERNAL_STRING_COLUMNS = POLYMER_SAMPLE_COLUMNS + ALUMINIUM_SAMPLE_COLUMNS
EXTERNAL_NUMERIC_COLUMNS = POLYMER_NUMERIC_COLUMNS + ALUMINIUM_NUMERIC_COLUMNS
EXTERNAL_COLUMNS = EXTERNAL_STRING_COLUMNS + EXTERNAL_NUMERIC_COLUMNS


def _resolve_column(df: pd.DataFrame, names: tuple[str, ...], *, numeric: bool = False) -> pd.Series:
    for name in names:
        if name in df.columns:
            series = df[name]
            if numeric:
                return pd.to_numeric(series, errors="coerce").fillna(0.0)
            if series.dtype == object:
                return series.fillna("").astype(str)
            return series
    if numeric:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.Series("", index=df.index, dtype=str)


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


@st.cache_data
def _load_processing_products() -> pd.DataFrame:
    try:
        return pd.read_csv("datasets/nasa_waste_processing_products.csv")
    except Exception:
        return pd.DataFrame()

# --------------------- cargar y normalizar a esquema estándar ---------------------
raw = load_waste_df().copy()


def _build_editable_df(source: pd.DataFrame) -> pd.DataFrame:
    columns_map: dict[str, tuple[str, ...]] = {
        "id": ("_source_id", "id"),
        "category": ("_source_category", "category", "Category"),
        "material": ("_source_material", "material"),
        "material_family": ("_source_material_family", "material_family"),
        "mass_kg": ("_source_mass_kg", "mass_kg", "kg", "Mass_kg"),
        "volume_l": ("_source_volume_l", "volume_l", "Volume_L"),
        "moisture_pct": ("_source_moisture_pct", "moisture_pct"),
        "difficulty_factor": ("_source_difficulty_factor", "difficulty_factor"),
        "pct_mass": ("_source_pct_mass", "pct_mass"),
        "pct_volume": ("_source_pct_volume", "pct_volume"),
        "flags": ("_source_flags", "flags", "Flags"),
        "key_materials": ("_source_key_materials", "key_materials"),
        "notes": ("_source_notes", "notes"),
    }

    for column in EXTERNAL_COLUMNS:
        columns_map[column] = (column,)

    numeric_fields = {
        "mass_kg",
        "volume_l",
        "moisture_pct",
        "pct_mass",
        "pct_volume",
        "difficulty_factor",
        *EXTERNAL_NUMERIC_COLUMNS,
    }

    editable_data: dict[str, pd.Series] = {}
    for target, candidates in columns_map.items():
        editable_data[target] = _resolve_column(source, candidates, numeric=target in numeric_fields)

    editable = pd.DataFrame(editable_data)
    editable["id"] = editable["id"].astype(str)
    editable["category"] = editable["category"].astype(str)
    editable["material"] = editable["material"].astype(str)
    editable["material_family"] = editable["material_family"].astype(str)
    editable["flags"] = editable["flags"].astype(str)
    editable["key_materials"] = editable["key_materials"].astype(str)
    editable["notes"] = editable["notes"].astype(str)
    if editable["id"].eq("").all():
        editable["id"] = source.index.astype(str)
    editable["_problematic"] = problematic_mask(editable)
    return editable


def _get_baseline_state() -> dict | None:
    baseline_state = st.session_state.get("_inventory_baseline")
    if isinstance(baseline_state, dict):
        df_candidate = baseline_state.get("df")
        if isinstance(df_candidate, pd.DataFrame):
            return baseline_state
    return None


def _format_baseline_caption(state: dict | None) -> str:
    if not state:
        return "Sin histórico de guardado."
    saved_at = state.get("saved_at")
    if isinstance(saved_at, datetime):
        timestamp = saved_at.astimezone(timezone.utc).strftime("%d %b %Y %H:%M UTC")
        return f"Baseline desde save_waste_df ({timestamp})."
    return "Baseline calculado desde el último save_waste_df conocido."


def _format_sidebar_delta(current: float, baseline: float | None, unit: str) -> str:
    if baseline is None:
        return "Sin histórico"
    diff = current - baseline
    tolerance = 1e-6
    if abs(diff) < tolerance:
        return "Sin cambios"
    arrow = "⬆️" if diff > 0 else "⬇️"
    if abs(baseline) > tolerance:
        pct = diff / baseline * 100.0
        return f"{arrow} {pct:+.1f}% vs. histórico"
    return f"{arrow} {diff:+.1f} {unit} vs. histórico"


if "_inventory_baseline" not in st.session_state:
    st.session_state["_inventory_baseline"] = {"df": raw.copy(deep=True), "saved_at": None}

df = _build_editable_df(raw)

derived_columns: dict[str, pd.Series] = {}
for column in raw.columns:
    if column in df.columns:
        continue
    if column.startswith("summary_"):
        derived_columns[column] = raw[column]
        continue
    if column.startswith("_source_"):
        derived_columns[column] = raw[column]
        continue
    if column.endswith("_pct") and column not in {"pct_mass", "pct_volume"}:
        derived_columns[column] = raw[column]
        continue
    if column in {"kg", "density_kg_m3", "category_total_mass_kg", "category_total_volume_m3", "material_display"}:
        derived_columns[column] = raw[column]

for column in EXTERNAL_COLUMNS:
    if column in raw.columns and column not in df.columns:
        derived_columns[column] = raw[column]

if derived_columns:
    df = pd.concat([df, pd.DataFrame(derived_columns)], axis=1)

composition_cols = [
    column
    for column in df.columns
    if column.endswith("_pct") and column not in {"pct_mass", "pct_volume"} and not column.startswith("_source_")
]

if composition_cols:
    def _format_composition(row: pd.Series) -> str:
        parts: list[str] = []
        for column in composition_cols:
            value = row.get(column)
            if pd.isna(value) or float(value) <= 0:
                continue
            label = column.replace("_pct", "").replace("_", " ")
            parts.append(f"{label} {float(value):.0f}%")
        return ", ".join(parts)

    df["composition_summary"] = df[composition_cols].apply(_format_composition, axis=1)
else:
    df["composition_summary"] = ""

df["polymer_profile"] = df.apply(_format_polymer_profile, axis=1)
df["aluminium_profile"] = df.apply(_format_aluminium_profile, axis=1)


def _format_polymer_profile(row: pd.Series) -> str:
    parts: list[str] = []

    for column in POLYMER_SAMPLE_COLUMNS:
        label = str(row.get(column) or "").strip()
        if label:
            parts.append(f"Ref {label}")
            break

    density = _safe_float(row.get("pc_density_density_g_per_cm3"))
    if density:
        parts.append(f"ρ {density:.2f} g/cm³")

    tensile = _safe_float(row.get("pc_mechanics_tensile_strength_mpa"))
    if tensile:
        parts.append(f"σₜ {tensile:.0f} MPa")

    modulus = _safe_float(row.get("pc_mechanics_modulus_gpa"))
    if modulus:
        parts.append(f"E {modulus:.1f} GPa")

    glass_transition = _safe_float(row.get("pc_thermal_glass_transition_c"))
    if glass_transition:
        parts.append(f"Tg {glass_transition:.0f} °C")

    ignition = _safe_float(row.get("pc_ignition_ignition_temperature_c"))
    if ignition:
        parts.append(f"Ign. {ignition:.0f} °C")

    burn_time = _safe_float(row.get("pc_ignition_burn_time_min"))
    if burn_time:
        parts.append(f"Burn {burn_time:.1f} min")

    return "||".join(parts)


def _format_aluminium_profile(row: pd.Series) -> str:
    parts: list[str] = []

    route = str(row.get("aluminium_processing_route") or "").strip()
    class_id = str(row.get("aluminium_class_id") or "").strip()
    if route and class_id:
        parts.append(f"{route} · Clase {class_id}")
    elif route:
        parts.append(route)
    elif class_id:
        parts.append(f"Clase {class_id}")

    tensile = _safe_float(row.get("aluminium_tensile_strength_mpa"))
    if tensile:
        parts.append(f"σₜ {tensile:.0f} MPa")

    yield_strength = _safe_float(row.get("aluminium_yield_strength_mpa"))
    if yield_strength:
        parts.append(f"σᵧ {yield_strength:.0f} MPa")

    elongation = _safe_float(row.get("aluminium_elongation_pct"))
    if elongation:
        parts.append(f"ε {elongation:.0f}%")

    return "||".join(parts)

mission_columns = [column for column in df.columns if column.startswith("summary_")]
mission_labels = {
    "summary_gateway_phase_i_mass_kg": "Gateway I",
    "summary_gateway_phase_ii_mass_kg": "Gateway II",
    "summary_mars_transit_mass_kg": "Mars Transit",
    "summary_total_mass_kg": "Total NASA",
}

if mission_columns:
    def _format_mission_bundle(row: pd.Series) -> str:
        parts: list[str] = []
        for column in mission_columns:
            value = row.get(column)
            if pd.isna(value) or float(value) <= 0:
                continue
            label = mission_labels.get(column, column.replace("summary_", "").replace("_", " "))
            parts.append(f"{label}: {float(value):.1f} kg")
        return " · ".join(parts)

    df["nasa_mission_bundle"] = df[mission_columns].apply(_format_mission_bundle, axis=1)
else:
    df["nasa_mission_bundle"] = ""

# --------------------- métricas “sabor laboratorio” ---------------------
inventory_metrics = [
    ("Ítems", f"{len(df)}", None),
    ("Masa total (kg)", f"{float(df['mass_kg'].sum()):.2f}", None),
    ("Volumen (L)", f"{float(df['volume_l'].sum()):.1f}", None),
    ("Problemáticos", f"{int(df['_problematic'].sum())}", None),
]

metric_columns = st.columns(len(inventory_metrics))
for column, (label, value, delta) in zip(metric_columns, inventory_metrics):
    column.metric(label, value, delta=delta)

mission_totals: dict[str, float] = {}
for column in mission_columns:
    mission_totals[column] = float(pd.to_numeric(df[column], errors="coerce").sum())

if mission_totals:
    mission_order = [
        "summary_gateway_phase_i_mass_kg",
        "summary_gateway_phase_ii_mass_kg",
        "summary_mars_transit_mass_kg",
        "summary_total_mass_kg",
    ]
    ordered_totals = [
        (mission_labels.get(column, column), mission_totals[column])
        for column in mission_order
        if column in mission_totals
    ]
    mission_cols = st.columns(len(ordered_totals))
    for column, (label, value) in zip(mission_cols, ordered_totals):
        column.metric(f"NASA · {label}", f"{value:.1f} kg")
    st.caption("Referencias NASA: masas estimadas por misión usando `nasa_waste_summary.csv`.")

polymer_density_series = pd.to_numeric(df.get("pc_density_density_g_per_cm3"), errors="coerce")
polymer_tensile_series = pd.to_numeric(df.get("pc_mechanics_tensile_strength_mpa"), errors="coerce")
aluminium_tensile_series = pd.to_numeric(df.get("aluminium_tensile_strength_mpa"), errors="coerce")
aluminium_yield_series = pd.to_numeric(df.get("aluminium_yield_strength_mpa"), errors="coerce")
aluminium_elong_series = pd.to_numeric(df.get("aluminium_elongation_pct"), errors="coerce")

polymer_density_clean = polymer_density_series.dropna()
polymer_tensile_clean = polymer_tensile_series.dropna()
aluminium_tensile_clean = aluminium_tensile_series.dropna()
aluminium_yield_clean = aluminium_yield_series.dropna()
aluminium_elong_clean = aluminium_elong_series.dropna()

if (
    not polymer_density_clean.empty
    or not polymer_tensile_clean.empty
    or not aluminium_tensile_clean.empty
    or not aluminium_yield_clean.empty
    or not aluminium_elong_clean.empty
):
    st.subheader("Propiedades externas (NASA + industria)")

    metric_stats: list[tuple[str, float, str]] = []
    if not polymer_density_clean.empty:
        metric_stats.append(("ρ media (g/cm³)", float(polymer_density_clean.mean()), ".2f"))
    if not polymer_tensile_clean.empty:
        metric_stats.append(("σₜ polímero (MPa)", float(polymer_tensile_clean.median()), ".0f"))
    if not aluminium_tensile_clean.empty:
        metric_stats.append(("σₜ aluminio (MPa)", float(aluminium_tensile_clean.median()), ".0f"))
    if not aluminium_elong_clean.empty:
        metric_stats.append(("ε aluminio (%)", float(aluminium_elong_clean.median()), ".0f"))

    if metric_stats:
        metric_columns = st.columns(len(metric_stats))
        for column, (label, value, fmt) in zip(metric_columns, metric_stats, strict=False):
            column.metric(label, f"{value:{fmt}}")

    tabs = st.tabs(["Polímeros", "Aluminio"])

    polymer_chart_data = df[[
        "category",
        "material",
        "pc_density_density_g_per_cm3",
        "pc_mechanics_tensile_strength_mpa",
        "pc_density_sample_label",
    ]].copy()

    with tabs[0]:
        density_data = polymer_chart_data.dropna(subset=["pc_density_density_g_per_cm3"])
        if not density_data.empty:
            density_chart = (
                alt.Chart(density_data)
                .transform_aggregate(
                    mean_density="mean(pc_density_density_g_per_cm3)",
                    groupby=["category"],
                )
                .mark_bar(color="#22d3ee")
                .encode(
                    x=alt.X("mean_density:Q", title="Densidad promedio (g/cm³)"),
                    y=alt.Y("category:N", sort="-x", title="Categoría NASA"),
                    tooltip=[
                        alt.Tooltip("mean_density:Q", format=".2f", title="Densidad promedio"),
                        alt.Tooltip("category:N", title="Categoría"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(density_chart, use_container_width=True)
        else:
            st.info("Sin densidades de polímeros disponibles en el inventario.")

        tensile_scatter = polymer_chart_data.dropna(
            subset=["pc_mechanics_tensile_strength_mpa", "pc_density_density_g_per_cm3"]
        )
        if not tensile_scatter.empty:
            scatter_chart = (
                alt.Chart(tensile_scatter)
                .mark_circle(opacity=0.75, size=70)
                .encode(
                    x=alt.X("pc_mechanics_tensile_strength_mpa:Q", title="σₜ (MPa)"),
                    y=alt.Y("pc_density_density_g_per_cm3:Q", title="ρ (g/cm³)"),
                    color=alt.Color("category:N", title="Categoría", legend=None),
                    tooltip=[
                        alt.Tooltip("category:N", title="Categoría"),
                        alt.Tooltip("material:N", title="Subitem"),
                        alt.Tooltip("pc_mechanics_tensile_strength_mpa:Q", format=".0f", title="σₜ (MPa)"),
                        alt.Tooltip("pc_density_density_g_per_cm3:Q", format=".2f", title="ρ (g/cm³)"),
                    ],
                )
                .properties(height=220, title="Relación densidad vs. resistencia (polímeros)")
            )
            st.altair_chart(scatter_chart, use_container_width=True)

        label_counts = (
            polymer_chart_data["pc_density_sample_label"].dropna().astype(str).str.strip().replace("", pd.NA)
        )
        if not label_counts.empty:
            top_refs = label_counts.value_counts().head(5).reset_index()
            top_refs.columns = ["Referencia", "Ítems"]
            st.caption("Principales referencias de laboratorio/industria")
            st.dataframe(top_refs, hide_index=True, use_container_width=True)

    aluminium_chart_data = df[[
        "category",
        "material",
        "aluminium_tensile_strength_mpa",
        "aluminium_yield_strength_mpa",
        "aluminium_processing_route",
    ]].copy()

    with tabs[1]:
        aluminium_stats = aluminium_chart_data.dropna(subset=["aluminium_tensile_strength_mpa"])
        if not aluminium_stats.empty:
            alu_chart = (
                alt.Chart(aluminium_stats)
                .transform_aggregate(
                    mean_sigma="mean(aluminium_tensile_strength_mpa)",
                    groupby=["category"],
                )
                .mark_bar(color="#f97316")
                .encode(
                    x=alt.X("mean_sigma:Q", title="σₜ promedio (MPa)"),
                    y=alt.Y("category:N", sort="-x", title="Categoría NASA"),
                    tooltip=[
                        alt.Tooltip("mean_sigma:Q", format=".0f", title="σₜ promedio"),
                        alt.Tooltip("category:N", title="Categoría"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(alu_chart, use_container_width=True)
        else:
            st.info("Sin ensayos de aluminio asociados a este inventario.")

        yield_data = aluminium_chart_data.dropna(subset=["aluminium_yield_strength_mpa"])
        if not yield_data.empty:
            yield_chart = (
                alt.Chart(yield_data)
                .mark_circle(opacity=0.75, size=70, color="#fb923c")
                .encode(
                    x=alt.X("aluminium_yield_strength_mpa:Q", title="σᵧ (MPa)"),
                    y=alt.Y("aluminium_tensile_strength_mpa:Q", title="σₜ (MPa)"),
                    tooltip=[
                        alt.Tooltip("category:N", title="Categoría"),
                        alt.Tooltip("material:N", title="Subitem"),
                        alt.Tooltip("aluminium_processing_route:N", title="Ruta"),
                        alt.Tooltip("aluminium_yield_strength_mpa:Q", format=".0f", title="σᵧ"),
                        alt.Tooltip("aluminium_tensile_strength_mpa:Q", format=".0f", title="σₜ"),
                    ],
                )
                .properties(height=220, title="Resistencias aluminio (fluencia vs. tracción)")
            )
            st.altair_chart(yield_chart, use_container_width=True)

        route_counts = (
            aluminium_chart_data["aluminium_processing_route"].dropna().astype(str).str.strip().replace("", pd.NA)
        )
        if not route_counts.empty:
            top_routes = route_counts.value_counts().head(5).reset_index()
            top_routes.columns = ["Proceso", "Ítems"]
            st.caption("Rutas metalúrgicas más frecuentes")
            st.dataframe(top_routes, hide_index=True, use_container_width=True)

    st.caption(
        "Datos externos integrados desde `polymer_composite_*` y `aluminium_alloys.csv`."
    )

with st.expander("¿Por qué estos ítems? (resumen rápido)", expanded=False):
    st.markdown(
        "- **Pouches PE–PET–Al** (multicapa) → capas incompatibles, difícil reciclado.\n"
        "- **Espumas ZOTEK F30 (PE reticulado)** → celdas cerradas, voluminosas.\n"
        "- **EVA/CTB (Nomex/Nylon/Polyester)** → textiles técnicos.\n"
        "- **Guantes de nitrilo** → elastómeros con aditivos.\n"
        "- **Aluminio** → valioso para **reuso/refuerzo**.\n"
        "- **MGS-1** → carga mineral para *Sinter with MGS-1*.\n"
    )

st.markdown("### Editar inventario")
st.caption("Los **problemáticos** se visualizan con gradientes tipo Tesla. Agrupá por `category`, ajustá `flags` y usá la vista lateral para controles rápidos.")

# ------------------------------------------------------------------
# Estado inicial y helpers de sesión
if "inventory_data" not in st.session_state:
    st.session_state["inventory_data"] = df.copy()

if "inventory_quick_filters" not in st.session_state:
    st.session_state["inventory_quick_filters"] = []

if "inventory_grid_state" not in st.session_state:
    st.session_state["inventory_grid_state"] = {}

# ------------------------------------------------------------------
# Barra lateral de análisis instantáneo
sidebar = st.sidebar
sidebar.header("Análisis instantáneo")

session_df = st.session_state["inventory_data"].copy()
session_df["_problematic"] = problematic_mask(session_df)
st.session_state["inventory_data"] = session_df.copy()

baseline_state = _get_baseline_state()
baseline_df = _build_editable_df(baseline_state["df"]) if baseline_state else None

baseline_mass = float(baseline_df["mass_kg"].sum()) if baseline_df is not None else None
baseline_volume = float(baseline_df["volume_l"].sum()) if baseline_df is not None else None
baseline_problematic = int(baseline_df["_problematic"].sum()) if baseline_df is not None else None

current_mass = float(session_df["mass_kg"].sum())
current_volume = float(session_df["volume_l"].sum())
current_problematic = int(session_df["_problematic"].sum())

sidebar.metric(
    "Masa total",
    f"{current_mass:.2f} kg",
    delta=_format_sidebar_delta(current_mass, baseline_mass, "kg"),
)
sidebar.metric(
    "Volumen agregado",
    f"{current_volume:.1f} L",
    delta=_format_sidebar_delta(current_volume, baseline_volume, "L"),
)
sidebar.metric(
    "Problemáticos",
    current_problematic,
    delta=_format_sidebar_delta(float(current_problematic),
                               float(baseline_problematic) if baseline_problematic is not None else None,
                               "ítems"),
)
sidebar.caption(_format_baseline_caption(baseline_state))

sidebar.subheader("Recomendaciones IA")
problematic_tags = session_df.loc[session_df["_problematic"], "flags"].fillna("")
top_flags = (
    problematic_tags.str.split(r"[,;]\s*")
    .explode()
    .str.strip()
    .replace("", pd.NA)
    .dropna()
)
flag_counts = top_flags.value_counts().head(3)
if not flag_counts.empty:
    for flag, count in flag_counts.items():
        sidebar.write(f"• Concentrar segregación en `{flag}` ({count} lotes)")
else:
    sidebar.caption("Sin flags críticos detectados. ✨")

sidebar.subheader("Filtros rápidos")
sidebar.markdown(
    "<style>div[data-baseweb='toggle']{margin-bottom:0.4rem;}"
    "div[data-baseweb='toggle'] label{background:#111827;color:#f9fafb;padding:0.35rem 0.9rem;"
    "border-radius:999px;font-size:0.85rem;box-shadow:0 0 0 1px #4b5563 inset;}"
    "div[data-baseweb='toggle'] input:checked+label{background:#00a19d;color:#06141b;font-weight:600;}"
    "</style>",
    unsafe_allow_html=True,
)

all_flags = (
    session_df["flags"].fillna("").str.split(r"[,;]\s*").explode().str.strip().replace("", pd.NA).dropna().unique()
)
selected_filters = st.session_state["inventory_quick_filters"]

for flag in all_flags:
    toggle_key = f"inventory_flag_{flag}"
    current_value = flag in selected_filters
    toggled = sidebar.toggle(flag, value=current_value, key=toggle_key)
    if toggled and not current_value:
        selected_filters.append(flag)
    if not toggled and current_value:
        selected_filters = [f for f in selected_filters if f != flag]

st.session_state["inventory_quick_filters"] = selected_filters

filtered_df = session_df.copy()
if selected_filters:
    mask = filtered_df["flags"].fillna("").apply(
        lambda value: all(flag.lower() in value.lower() for flag in selected_filters)
    )
    filtered_df = filtered_df[mask]

# ------------------------------------------------------------------
# Configuración de AG Grid
filtered_df = filtered_df.reset_index(drop=True)

st.markdown(
    """
    <style>
    .property-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(14,165,233,0.16);
        color: #e0f2fe;
        border-radius: 999px;
        padding: 0.12rem 0.55rem;
        margin: 0.05rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .property-badge:nth-child(2n) {
        background: rgba(16,185,129,0.18);
        color: #d1fae5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

flag_chip_renderer = JsCode(
    """
    function(params) {
        if (!params.value) { return ''; }
        const chips = String(params.value)
            .split(/[;,]/)
            .map(v => v.trim())
            .filter(Boolean)
            .map(flag => `<span class="flag-chip">${flag}</span>`);
        return chips.join(' ');
    }
    """
)

property_badge_renderer = JsCode(
    """
    function(params){
        if (!params.value) { return ''; }
        const parts = String(params.value)
            .split('||')
            .map(v => v.trim())
            .filter(Boolean);
        return parts.map(part => `<span class="property-badge">${part}</span>`).join(' ');
    }
    """
)

tesla_gradient_style = JsCode(
    """
    function(params){
        const base = {borderRadius: '8px', fontWeight: 500, paddingLeft: '6px'};
        if (params.value === null || params.value === undefined || params.value === '') {
            return {...base, backgroundColor: '#1f2937', color: '#9ca3af'};
        }
        const value = Number(params.value);
        if (isNaN(value)) {
            return {...base, backgroundColor: '#1f2937', color: '#f9fafb'};
        }
        if (value < 0) {
            return {...base, backgroundColor: '#ffedef', color: '#cc0000'};
        }
        const clamp = Math.min(value / 10.0, 1.0);
        const teal = Math.round(80 + 90 * clamp);
        return {
            ...base,
            backgroundColor: `rgba(0, 161, 157, ${0.15 + clamp * 0.35})`,
            color: `rgb(${50 + (1 - clamp) * 60}, ${teal}, ${150 + clamp * 80})`
        };
    }
    """
)

problematic_row_style = JsCode(
    """
    function(params){
        if (params.data && params.data._problematic) {
            return {'backgroundColor': 'rgba(204,0,0,0.12)'};
        }
        return {};
    }
    """
)

grid_column_order = [
    "id",
    "category",
    "material",
    "material_display",
    "material_family",
    "mass_kg",
    "volume_l",
    "moisture_pct",
    "difficulty_factor",
    "flags",
    "key_materials",
    "polymer_profile",
    "aluminium_profile",
    "notes",
    "composition_summary",
    "nasa_mission_bundle",
    "density_kg_m3",
    "_problematic",
]

for column in mission_columns:
    if column not in grid_column_order:
        grid_column_order.append(column)

for column in composition_cols:
    if column not in grid_column_order:
        grid_column_order.append(column)

for column in EXTERNAL_COLUMNS:
    if column not in grid_column_order:
        grid_column_order.append(column)

source_columns = [column for column in filtered_df.columns if column.startswith("_source_")]
for column in source_columns:
    if column not in grid_column_order:
        grid_column_order.append(column)

grid_columns = [column for column in grid_column_order if column in filtered_df.columns]
grid_df = filtered_df[grid_columns].copy()

gb = GridOptionsBuilder.from_dataframe(grid_df)
gb.configure_default_column(
    editable=True,
    groupable=True,
    resizable=True,
    sortable=True,
    filter=True,
    tooltipField="flags",
)
gb.configure_column("id", header_name="ID", pinned="left")
gb.configure_column("category", header_name="Categoría NASA", rowGroup=True, hide=True)
gb.configure_column("material", header_name="Subitem NASA")
gb.configure_column("material_display", header_name="Resumen", editable=False)
gb.configure_column("material_family", header_name="Familia material")
gb.configure_column("mass_kg", header_name="Masa (kg)", type=["numericColumn"], cellStyle=tesla_gradient_style)
gb.configure_column("volume_l", header_name="Volumen (L)", type=["numericColumn"], cellStyle=tesla_gradient_style)
gb.configure_column("moisture_pct", header_name="Humedad (%)", type=["numericColumn"], cellStyle=tesla_gradient_style)
gb.configure_column("difficulty_factor", header_name="Dificultad", type=["numericColumn"], cellStyle=tesla_gradient_style)
gb.configure_column("flags", header_name="Flags", cellRenderer=flag_chip_renderer, editable=True)
gb.configure_column("key_materials", header_name="Key materials", wrapText=True, autoHeight=True)
gb.configure_column(
    "polymer_profile",
    header_name="Polímeros ref",
    editable=False,
    cellRenderer=property_badge_renderer,
    autoHeight=True,
    wrapText=True,
)
gb.configure_column(
    "aluminium_profile",
    header_name="Aluminio ref",
    editable=False,
    cellRenderer=property_badge_renderer,
    autoHeight=True,
    wrapText=True,
)
gb.configure_column("notes", header_name="Notas", wrapText=True, autoHeight=True)
gb.configure_column("composition_summary", header_name="Composición NASA", editable=False, wrapText=True, autoHeight=True)
gb.configure_column("nasa_mission_bundle", header_name="Masas por misión", editable=False, wrapText=True, autoHeight=True)
gb.configure_column("density_kg_m3", header_name="Densidad (kg/m³)", type=["numericColumn"], editable=False)
gb.configure_column("_problematic", header_name="Problemático", editable=False, hide=True)

for column in mission_columns:
    if column in grid_df.columns:
        header = f"{mission_labels.get(column, column)} (kg)"
        gb.configure_column(column, header_name=header, editable=False, type=["numericColumn"], hide=True)

for column in composition_cols:
    if column in grid_df.columns:
        header = column.replace("_pct", " (%)").replace("_", " ")
        gb.configure_column(column, header_name=header, editable=False, hide=True)

for column in EXTERNAL_COLUMNS:
    if column in grid_df.columns:
        gb.configure_column(column, header_name=column, editable=False, hide=True)

for column in source_columns:
    if column in grid_df.columns:
        gb.configure_column(column, header_name=column, editable=False, hide=True)

gb.configure_grid_options(
    animateRows=True,
    enableRangeSelection=True,
    enableCellTextSelection=True,
    rowClassRules={"problematic": "data._problematic === true"},
    rowGroupPanelShow="always",
)
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()

grid_options = gb.build()

if st.session_state["inventory_grid_state"]:
    stored_state = st.session_state["inventory_grid_state"]
    for key in ["columnState", "sortModel", "filterModel", "rowGroupPanelShow", "grouping"]:
        if stored_state.get(key):
            grid_options[key] = stored_state[key]

st.markdown(
    """
    <style>
    .flag-chip {
        background: linear-gradient(120deg, rgba(0,161,157,0.25), rgba(17,24,39,0.15));
        border-radius: 999px;
        padding: 2px 8px;
        margin-right: 4px;
        color: #00d1c1;
        font-weight: 600;
        display: inline-block;
        font-size: 0.75rem;
    }
    .ag-theme-streamlit .ag-row.problematic {
        border-left: 3px solid #cc0000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

grid_col, preview_col = st.columns((2.2, 1))

with grid_col:
    grid_response = AgGrid(
        grid_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
        data_return_mode="AS_INPUT",
        theme="streamlit",
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
    )

    st.session_state["inventory_grid_state"] = grid_response.get("grid_state", st.session_state["inventory_grid_state"])

    updated_df = pd.DataFrame(grid_response.get("data", grid_df))
    if not updated_df.empty:
        numeric_candidates = [
            "mass_kg",
            "volume_l",
            "moisture_pct",
            "pct_mass",
            "pct_volume",
            "difficulty_factor",
            "density_kg_m3",
        ]
        for column in numeric_candidates:
            if column in updated_df.columns:
                updated_df[column] = pd.to_numeric(updated_df[column], errors="coerce").fillna(0.0)
        updated_df["_problematic"] = problematic_mask(updated_df)
        if composition_cols and all(column in updated_df.columns for column in composition_cols):
            updated_df["composition_summary"] = updated_df[composition_cols].apply(_format_composition, axis=1)
        if mission_columns and all(column in updated_df.columns for column in mission_columns):
            updated_df["nasa_mission_bundle"] = updated_df[mission_columns].apply(_format_mission_bundle, axis=1)
        st.session_state["inventory_data"] = updated_df

selected_rows = grid_response.get("selected_rows", []) if "grid_response" in locals() else []
selected_ids = [row.get("id") for row in selected_rows]

with preview_col:
    st.subheader("Vista lateral")
    if selected_rows:
        st.caption("Lotes seleccionados — edición contextual")
        preview_df = pd.DataFrame(selected_rows)
        preview_df["_problematic"] = problematic_mask(preview_df)
        preview_columns = [
            "id",
            "category",
            "material",
            "mass_kg",
            "volume_l",
            "composition_summary",
            "nasa_mission_bundle",
        ]
        preview_columns = [column for column in preview_columns if column in preview_df.columns]
        st.dataframe(
            preview_df[preview_columns]
            .rename(columns={"mass_kg": "kg", "volume_l": "L", "composition_summary": "Composición", "nasa_mission_bundle": "Misiones"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Flags representados**")
        chips = []
        for flags in preview_df.get("flags", pd.Series(dtype=str)).fillna(""):
            chips.extend([f.strip() for f in str(flags).split(",") if f.strip()])
        chips = sorted(set(chips))
        if chips:
            st.markdown(
                " ".join([f"<span class='flag-chip'>{chip}</span>" for chip in chips]),
                unsafe_allow_html=True,
            )
        else:
            st.caption("Sin flags adicionales en selección.")

        if composition_cols and "composition_summary" in preview_df:
            st.markdown("**Composición NASA**")
            for _, row in preview_df.iterrows():
                summary = row.get("composition_summary") or "—"
                st.markdown(f"• **{row.get('material', row.get('category', 'Lote'))}**: {summary}")

        if mission_columns:
            mission_summary = {
                mission_labels.get(column, column): float(pd.to_numeric(preview_df.get(column), errors="coerce").sum())
                for column in mission_columns
                if column in preview_df.columns
            }
            if mission_summary:
                st.markdown("**Masa proyectada por misión (selección)**")
                mission_table = pd.DataFrame(
                    {"Misión": mission_summary.keys(), "kg": mission_summary.values()}
                )
                st.dataframe(mission_table, hide_index=True, use_container_width=True)

        processing_df = _load_processing_products()
        if not processing_df.empty:
            st.markdown("**Procesamiento NASA (Trash-to-Gas / Trash-to-Supply Gas)**")
            st.dataframe(
                processing_df[
                    [
                        "approach",
                        "propellant_per_cm_day_kg",
                        "gateway_phase_i_propellant_kg",
                        "gateway_phase_ii_propellant_kg",
                        "mars_outbound_propellant_kg",
                        "total_propellant_kg",
                    ]
                ]
                .rename(
                    columns={
                        "approach": "Enfoque",
                        "propellant_per_cm_day_kg": "kg prop./cm·día",
                        "gateway_phase_i_propellant_kg": "Gateway I (kg)",
                        "gateway_phase_ii_propellant_kg": "Gateway II (kg)",
                        "mars_outbound_propellant_kg": "Mars outbound (kg)",
                        "total_propellant_kg": "Total propulsor (kg)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Seleccioná filas para ver detalles y acciones en lote.")

    st.markdown("---")
    st.subheader("Edición en lote")
    help_text = "Aplicá cambios a todas las filas seleccionadas. Tooltips indican impacto." 
    st.caption(help_text)
    batch_disabled = not selected_ids

    with st.form("batch_edit_form"):
        mass_delta = st.number_input(
            "Ajustar masa (kg)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.25,
            help="Usá valores negativos para descontar masa del lote",
            disabled=batch_disabled,
        )
        new_flag = st.text_input(
            "Añadir flag", "", help="Se añade a todos los lotes seleccionados", disabled=batch_disabled
        )
        submitted = st.form_submit_button("Aplicar", disabled=batch_disabled)

    if submitted and selected_ids:
        live_df = st.session_state["inventory_data"].copy()
        mask = live_df["id"].isin(selected_ids)
        if mass_delta != 0:
            live_df.loc[mask, "mass_kg"] = (live_df.loc[mask, "mass_kg"] + mass_delta).clip(lower=0)
        if new_flag:
            live_df.loc[mask, "flags"] = live_df.loc[mask, "flags"].fillna("").apply(
                lambda text: ", ".join(sorted(set([*filter(None, map(str.strip, text.split(","))), new_flag.strip()])))
            )
        live_df["_problematic"] = problematic_mask(live_df)
        st.session_state["inventory_data"] = live_df
        st.toast("Edición en lote aplicada.")
        _trigger_rerun()

colA, colB = st.columns(2)
with colA:
    button_state = "success" if save_success else "idle"
    if minimal_button(
        "💾 Guardar inventario",
        key="inventory_save",
        state=button_state,
        width="full",
        help_text="Se exporta a data/waste_inventory_sample.csv",
        success_label="Inventario guardado",
        status_hints={
            "idle": "",
            "loading": "Guardando dataset",
            "success": "Inventario actualizado",
            "error": "",
        },
    ):
        save_columns = [
            "id",
            "category",
            "material",
            "material_family",
            "mass_kg",
            "volume_l",
            "moisture_pct",
            "difficulty_factor",
            "pct_mass",
            "pct_volume",
            "flags",
            "key_materials",
            "notes",
        ]
        inventory_df = st.session_state["inventory_data"]
        save_columns = [column for column in save_columns if column in inventory_df.columns]
        out = inventory_df[save_columns].copy()
        save_waste_df(out)
        st.session_state["_inventory_baseline"] = {
            "df": load_waste_df().copy(deep=True),
            "saved_at": datetime.now(timezone.utc),
        }
        st.session_state[_SAVE_SUCCESS_FLAG] = True
        _trigger_rerun()

with colB:
    st.caption("Vista y orden guardados automáticamente en esta sesión.")

st.info(
    "**Siguiente paso** → Abrí **2) Objetivo**. "
    "El generador prioriza ítems problemáticos y usa **P03 (Sinter with MGS-1)** con regolito cuando corresponde."
)
