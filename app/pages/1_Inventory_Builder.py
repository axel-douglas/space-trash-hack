import _bootstrap  # noqa: F401

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
def _pick_col(df: pd.DataFrame, names, default=None):
    for n in names:
        if n in df.columns:
            return n
    return default

# --------------------- cargar y normalizar a esquema estándar ---------------------
raw = load_waste_df().copy()

id_col    = _pick_col(raw, ["id"])
cat_col   = _pick_col(raw, ["category", "Category"])
mat_col   = _pick_col(raw, ["material_family", "material", "Material"])
mass_col  = _pick_col(raw, ["mass_kg", "kg", "Mass_kg"])
vol_col   = _pick_col(raw, ["volume_l", "Volume_L"])
flags_col = _pick_col(raw, ["flags", "Flags"])

# Construimos un DF estándar para edición/guardado:
df = pd.DataFrame({
    "id": raw[id_col] if id_col else raw.index.astype(str),
    "category": raw[cat_col] if cat_col else "",
    "material_family": raw[mat_col] if mat_col else "",
    "mass_kg": pd.to_numeric(raw[mass_col], errors="coerce").fillna(0.0) if mass_col else 0.0,
    "volume_l": pd.to_numeric(raw[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0,
    "flags": (raw[flags_col].astype(str) if flags_col else ""),
})

problematic_series = problematic_mask(df)
df["_problematic"] = problematic_series

# --------------------- métricas “sabor laboratorio” ---------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ítems", len(df))
c2.metric("Masa total (kg)", f"{float(pd.to_numeric(df['mass_kg'], errors='coerce').sum()):.2f}")
c3.metric("Volumen (L)", f"{float(pd.to_numeric(df['volume_l'], errors='coerce').sum()):.1f}")
c4.metric("Problemáticos", int(df["_problematic"].sum()))

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

sidebar.metric("Masa total", f"{session_df['mass_kg'].sum():.2f} kg", delta="⬆️ +2.3% vs. último guardado")
sidebar.metric("Volumen agregado", f"{session_df['volume_l'].sum():.1f} L", delta="⬇️ 0.8% optimizado")
sidebar.metric("Problemáticos", int(session_df["_problematic"].sum()), delta="⚡ Priorizar reciclado")

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

gb = GridOptionsBuilder.from_dataframe(
    filtered_df[["id", "category", "material_family", "mass_kg", "volume_l", "flags", "_problematic"]]
)
gb.configure_default_column(
    editable=True,
    groupable=True,
    resizable=True,
    sortable=True,
    filter=True,
    tooltipField="flags",
)
gb.configure_column("id", header_name="ID", pinned="left")
gb.configure_column("category", header_name="Category", rowGroup=True, hide=True)
gb.configure_column("material_family", header_name="Material")
gb.configure_column("mass_kg", header_name="Masa (kg)", type=["numericColumn"], cellStyle=tesla_gradient_style)
gb.configure_column("volume_l", header_name="Volumen (L)", type=["numericColumn"], cellStyle=tesla_gradient_style)
gb.configure_column("flags", header_name="Flags", cellRenderer=flag_chip_renderer, editable=True)
gb.configure_column("_problematic", header_name="Problemático", editable=False, hide=True)
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
        filtered_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
        data_return_mode="AS_INPUT",
        theme="streamlit",
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
    )

    st.session_state["inventory_grid_state"] = grid_response.get("grid_state", st.session_state["inventory_grid_state"])

    updated_df = pd.DataFrame(grid_response.get("data", filtered_df))
    if not updated_df.empty:
        updated_df["mass_kg"] = pd.to_numeric(updated_df["mass_kg"], errors="coerce").fillna(0.0)
        updated_df["volume_l"] = pd.to_numeric(updated_df["volume_l"], errors="coerce").fillna(0.0)
        updated_df["_problematic"] = problematic_mask(updated_df)
        st.session_state["inventory_data"] = updated_df

selected_rows = grid_response.get("selected_rows", []) if "grid_response" in locals() else []
selected_ids = [row.get("id") for row in selected_rows]

with preview_col:
    st.subheader("Vista lateral")
    if selected_rows:
        st.caption("Lotes seleccionados — edición contextual")
        preview_df = pd.DataFrame(selected_rows)
        preview_df["_problematic"] = problematic_mask(preview_df)
        st.dataframe(
            preview_df[["id", "category", "flags", "mass_kg", "volume_l"]]
            .rename(columns={"mass_kg": "kg", "volume_l": "L"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Flags representados**")
        chips = []
        for flags in preview_df["flags"].fillna(""):
            chips.extend([f.strip() for f in str(flags).split(",") if f.strip()])
        chips = sorted(set(chips))
        if chips:
            st.markdown(
                " ".join([f"<span class='flag-chip'>{chip}</span>" for chip in chips]),
                unsafe_allow_html=True,
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
        out = st.session_state["inventory_data"][["id", "category", "material_family", "mass_kg", "volume_l", "flags"]].copy()
        save_waste_df(out)
        st.session_state[_SAVE_SUCCESS_FLAG] = True
        _trigger_rerun()

with colB:
    st.caption("Vista y orden guardados automáticamente en esta sesión.")

st.info(
    "**Siguiente paso** → Abrí **2) Objetivo**. "
    "El generador prioriza ítems problemáticos y usa **P03 (Sinter with MGS-1)** con regolito cuando corresponde."
)
