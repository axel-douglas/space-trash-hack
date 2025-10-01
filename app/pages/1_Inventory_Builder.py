import _bootstrap  # noqa: F401

from datetime import datetime, timezone

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

    numeric_fields = {"mass_kg", "volume_l", "moisture_pct", "pct_mass", "pct_volume", "difficulty_factor"}

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
    id_col = _pick_col(source, ["id"])
    cat_col = _pick_col(source, ["category", "Category"])
    mat_col = _pick_col(source, ["material_family", "material", "Material"])
    mass_col = _pick_col(source, ["mass_kg", "kg", "Mass_kg"])
    vol_col = _pick_col(source, ["volume_l", "Volume_L"])
    flags_col = _pick_col(source, ["flags", "Flags"])

    editable = pd.DataFrame(
        {
            "id": source[id_col] if id_col else source.index.astype(str),
            "category": source[cat_col] if cat_col else "",
            "material_family": source[mat_col] if mat_col else "",
            "mass_kg": (
                pd.to_numeric(source[mass_col], errors="coerce").fillna(0.0)
                if mass_col
                else 0.0
            ),
            "volume_l": (
                pd.to_numeric(source[vol_col], errors="coerce").fillna(0.0)
                if vol_col
                else 0.0
            ),
            "flags": (source[flags_col].astype(str) if flags_col else ""),
        }
    )
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
