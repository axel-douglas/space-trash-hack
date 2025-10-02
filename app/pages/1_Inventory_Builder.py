import _bootstrap  # noqa: F401

from datetime import datetime, timezone

import altair as alt
import streamlit as st
import pandas as pd
from app.modules.io import (
    format_aluminium_profile,
    format_composition_summary,
    format_mission_bundle,
    format_polymer_profile,
    load_waste_df,
    save_waste_df,
)
from app.modules.navigation import set_active_step
from app.modules.ui_blocks import action_button, load_theme
from app.modules.problematic import problematic_mask
from app.modules.schema import (
    ALUMINIUM_NUMERIC_COLUMNS,
    ALUMINIUM_SAMPLE_COLUMNS,
    POLYMER_NUMERIC_COLUMNS,
    POLYMER_SAMPLE_COLUMNS,
)

_SAVE_SUCCESS_FLAG = "_inventory_save_success"


def _trigger_rerun() -> None:
    """Trigger a Streamlit rerun regardless of version."""
    try:
        st.rerun()
    except AttributeError:  # pragma: no cover - legacy fallback
        st.experimental_rerun()

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Inventario", page_icon="🧱", layout="wide")

_current_step = set_active_step("inventory")

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

mission_columns = [column for column in df.columns if column.startswith("summary_")]
mission_labels = {
    "summary_gateway_phase_i_mass_kg": "Gateway I",
    "summary_gateway_phase_ii_mass_kg": "Gateway II",
    "summary_mars_transit_mass_kg": "Mars Transit",
    "summary_total_mass_kg": "Total NASA",
}

def _enrich_inventory_df(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()

    string_columns = [
        "id",
        "category",
        "material",
        "material_family",
        "flags",
        "key_materials",
        "notes",
    ]
    for column in string_columns:
        if column in enriched.columns:
            enriched[column] = enriched[column].fillna("").astype(str)

    numeric_columns = {
        "mass_kg",
        "volume_l",
        "moisture_pct",
        "pct_mass",
        "pct_volume",
        "difficulty_factor",
        *EXTERNAL_NUMERIC_COLUMNS,
    }
    for column in numeric_columns:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)

    if composition_cols:
        enriched["composition_summary"] = enriched.apply(
            lambda row: format_composition_summary(row, composition_cols),
            axis=1,
        )
    else:
        enriched["composition_summary"] = ""

    enriched["polymer_profile"] = enriched.apply(format_polymer_profile, axis=1)
    enriched["aluminium_profile"] = enriched.apply(format_aluminium_profile, axis=1)

    if mission_columns:
        enriched["nasa_mission_bundle"] = enriched.apply(
            lambda row: format_mission_bundle(row, mission_columns, mission_labels),
            axis=1,
        )
    else:
        enriched["nasa_mission_bundle"] = ""

    enriched["_problematic"] = problematic_mask(enriched)
    return enriched


df = _enrich_inventory_df(df)

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

if "inventory_selection" not in st.session_state:
    st.session_state["inventory_selection"] = {}

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

mass_by_category = (
    session_df.groupby("category")["mass_kg"].sum().sort_values(ascending=False).head(6)
)
if not mass_by_category.empty:
    sidebar.subheader("Distribución de masa (kg)")
    sidebar.bar_chart(mass_by_category)

volume_by_category = (
    session_df.groupby("category")["volume_l"].sum().sort_values(ascending=False).head(6)
)
if not volume_by_category.empty:
    sidebar.subheader("Volumen por categoría (L)")
    sidebar.bar_chart(volume_by_category)

if composition_cols:
    composition_masses: dict[str, float] = {}
    base_mass = pd.to_numeric(session_df.get("mass_kg"), errors="coerce").fillna(0.0)
    for column in composition_cols:
        fractions = pd.to_numeric(session_df.get(column), errors="coerce").fillna(0.0) / 100.0
        component_mass = float((fractions * base_mass).sum())
        if component_mass <= 0:
            continue
        label = column.replace("_pct", "").replace("_", " ")
        composition_masses[label] = component_mass
    if composition_masses:
        composition_series = (
            pd.Series(composition_masses).sort_values(ascending=False).head(6)
        )
        sidebar.subheader("Composición estimada (kg)")
        sidebar.bar_chart(composition_series)

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
all_flags = (
    session_df["flags"].fillna("")
    .str.split(r"[,;]\s*")
    .explode()
    .str.strip()
    .replace("", pd.NA)
    .dropna()
    .unique()
)
available_flags = sorted(all_flags.tolist()) if len(all_flags) else []
default_flags = [flag for flag in st.session_state["inventory_quick_filters"] if flag in available_flags]
selected_filters = sidebar.multiselect(
    "Filtrar por flags",
    options=available_flags,
    default=default_flags,
)
st.session_state["inventory_quick_filters"] = selected_filters

filtered_df = session_df.copy()
if selected_filters:
    mask = filtered_df["flags"].fillna("").apply(
        lambda value: all(flag.lower() in value.lower() for flag in selected_filters)
    )
    filtered_df = filtered_df[mask]

# ------------------------------------------------------------------
# Editor interactivo
filtered_df = filtered_df.reset_index(drop=True)
filtered_df["id"] = filtered_df["id"].astype(str)

valid_ids = set(session_df["id"].astype(str))
selection_state = {
    key: bool(value)
    for key, value in st.session_state["inventory_selection"].items()
    if key in valid_ids
}
st.session_state["inventory_selection"] = selection_state

filtered_df["_selected"] = filtered_df["id"].map(selection_state).fillna(False)

editor_column_order = [
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
    if column not in editor_column_order:
        editor_column_order.append(column)

for column in composition_cols:
    if column not in editor_column_order:
        editor_column_order.append(column)

for column in EXTERNAL_COLUMNS:
    if column not in editor_column_order:
        editor_column_order.append(column)

source_columns = [column for column in filtered_df.columns if column.startswith("_source_")]
for column in source_columns:
    if column not in editor_column_order:
        editor_column_order.append(column)

editor_column_order = ["_selected", *editor_column_order]
editor_columns = [column for column in editor_column_order if column in filtered_df.columns]
editor_df = filtered_df[editor_columns].copy()

column_config = {
    "_selected": st.column_config.CheckboxColumn("Seleccionar", help="Marcar lotes para edición en lote"),
    "id": st.column_config.TextColumn("ID", disabled=True),
    "mass_kg": st.column_config.NumberColumn("Masa (kg)", min_value=0.0, step=0.25, format="%.2f"),
    "volume_l": st.column_config.NumberColumn("Volumen (L)", min_value=0.0, step=0.1, format="%.2f"),
    "moisture_pct": st.column_config.NumberColumn("Humedad (%)", min_value=0.0, max_value=100.0, step=1.0),
    "difficulty_factor": st.column_config.NumberColumn("Dificultad", min_value=0.0, step=0.5),
    "flags": st.column_config.TextColumn("Flags"),
    "key_materials": st.column_config.TextColumn("Materiales clave"),
    "composition_summary": st.column_config.TextColumn(
        "Composición NASA",
        disabled=True,
        help="Se calcula con los porcentajes NASA.*",
    ),
    "nasa_mission_bundle": st.column_config.TextColumn(
        "Misiones NASA",
        disabled=True,
        help="Resumen de masa proyectada por misión",
    ),
    "polymer_profile": st.column_config.TextColumn(
        "Perfil polímero",
        disabled=True,
        help="Datos de densidad, resistencia y laboratorio",
    ),
    "aluminium_profile": st.column_config.TextColumn("Perfil aluminio", disabled=True),
    "_problematic": st.column_config.CheckboxColumn("Problemático", disabled=True),
}

editor_col, preview_col = st.columns((2.2, 1))

with editor_col:
    editor_result = st.data_editor(
        editor_df,
        column_config=column_config,
        column_order=editor_columns,
        hide_index=True,
        key="inventory_editor",
        num_rows="fixed",
        use_container_width=True,
    )

if not editor_result.empty:
    editor_result = editor_result.copy()
    editor_result["id"] = editor_result["id"].astype(str)

    selection_updates = editor_result.set_index("id")["_selected"].fillna(False).astype(bool).to_dict()
    st.session_state["inventory_selection"].update(selection_updates)

    editable_result = editor_result.drop(
        columns=[
            column
            for column in (
                "_selected",
                "composition_summary",
                "nasa_mission_bundle",
                "polymer_profile",
                "aluminium_profile",
                "_problematic",
            )
            if column in editor_result.columns
        ]
    )

    live_df = st.session_state["inventory_data"].copy()
    live_df["id"] = live_df["id"].astype(str)

    live_df = live_df.set_index("id")
    update_payload = editable_result.set_index("id")
    live_df.update(update_payload)
    live_df = live_df.reset_index()
    live_df = _enrich_inventory_df(live_df)
    st.session_state["inventory_data"] = live_df
    session_df = live_df

selection_state = st.session_state["inventory_selection"]
selected_ids = [key for key, value in selection_state.items() if value]

with preview_col:
    st.subheader("Vista lateral")
    preview_df = session_df[session_df["id"].isin(selected_ids)].copy()
    if not preview_df.empty:
        st.caption("Lotes seleccionados — edición contextual")

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
            .rename(
                columns={
                    "mass_kg": "kg",
                    "volume_l": "L",
                    "composition_summary": "Composición",
                    "nasa_mission_bundle": "Misiones",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        selection_mass = float(preview_df.get("mass_kg", pd.Series(dtype=float)).sum())
        selection_volume = float(preview_df.get("volume_l", pd.Series(dtype=float)).sum())
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Masa selección (kg)", f"{selection_mass:.2f}")
        metric_b.metric("Volumen selección (L)", f"{selection_volume:.1f}")

        if "flags" in preview_df.columns:
            unique_flags = (
                preview_df["flags"].fillna("").str.split(r"[,;]\s*")
                .explode()
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
            )
        else:
            unique_flags = []

        if len(unique_flags) > 0:
            st.markdown("**Flags representados**")
            for flag in sorted(unique_flags):
                st.write(f"• {flag}")
        else:
            st.caption("Sin flags adicionales en selección.")

        if composition_cols and "composition_summary" in preview_df.columns:
            st.markdown("**Composición NASA**")
            for _, row in preview_df.iterrows():
                material_label = row.get("material") or row.get("category") or "Lote"
                summary = row.get("composition_summary") or "—"
                st.write(f"• {material_label}: {summary}")

        if mission_columns:
            mission_summary = {
                mission_labels.get(column, column): float(pd.to_numeric(preview_df.get(column), errors="coerce").sum())
                for column in mission_columns
                if column in preview_df.columns
            }
            if mission_summary:
                st.markdown("**Masa proyectada por misión (selección)**")
                mission_table = pd.DataFrame({"Misión": mission_summary.keys(), "kg": mission_summary.values()})
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
    st.caption("Aplicá cambios a todas las filas seleccionadas. Tooltips indican impacto.")
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
        live_df["id"] = live_df["id"].astype(str)
        mask = live_df["id"].isin(selected_ids)
        if mass_delta != 0:
            live_df.loc[mask, "mass_kg"] = (live_df.loc[mask, "mass_kg"] + mass_delta).clip(lower=0)
        if new_flag:
            live_df.loc[mask, "flags"] = live_df.loc[mask, "flags"].fillna("").apply(
                lambda text: ", ".join(
                    sorted(
                        set(
                            [
                                *filter(None, map(str.strip, str(text).split(","))),
                                new_flag.strip(),
                            ]
                        )
                    )
                )
            )
        live_df = _enrich_inventory_df(live_df)
        st.session_state["inventory_data"] = live_df
        st.toast("Edición en lote aplicada.")
        _trigger_rerun()
colA, colB = st.columns(2)
with colA:
    button_state = "success" if save_success else "idle"
    if action_button(
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
