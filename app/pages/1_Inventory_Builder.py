# app/pages/1_Inventory_Builder.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from app.modules.io import load_waste_df, save_waste_df

st.set_page_config(page_title="Inventario", page_icon="🧱", layout="wide")

st.title("1) Inventario — Datos NASA")
st.caption("Usando directamente: `data/waste_inventory_sample.csv` (id, category, material_family, mass_kg, volume_l, flags).")

df = load_waste_df()

# --------- Panel de filtros (residuos problemáticos) ----------
with st.sidebar:
    st.header("🔎 Filtros")
    only_problem = st.toggle("Mostrar solo ‘residuos problemáticos’", value=False,
                             help="Pouches multicapa, térmicos, EVA/CTB, espuma ZOTEK F30, guantes nitrilo, aluminio/struts, etc.")
    text = st.text_input("Buscar por texto (category/material/flags)", "")

view = df.copy()
if only_problem:
    view = view[view["_problematic"]]

if text.strip():
    t = text.lower()
    view = view[
        view["material"].str.lower().str.contains(t) |
        view["notes"].str.lower().str.contains(t) |
        view["_source_category"].str.lower().str.contains(t) |
        view["_source_material_family"].str.lower().str.contains(t)
    ]

# --------- Métricas de cobertura “residuo problema” ----------
total_kg = df["kg"].sum()
prob_kg  = df.loc[df["_problematic"], "kg"].sum()
coverage = 0.0 if total_kg == 0 else 100.0 * prob_kg / total_kg

c1, c2, c3 = st.columns(3)
c1.metric("Kg totales", f"{total_kg:.2f}")
c2.metric("Kg problemáticos", f"{prob_kg:.2f}")
c3.metric("Cobertura problemática", f"{coverage:.1f}%")

st.markdown("—")

st.subheader("📦 Tabla editable (vista NASA → UI)")
st.caption("Columnas UI: **material** (category — material_family), **kg** (mass_kg), **notes** (flags). Se preserva la proveniencia en columnas ocultas.")

edited = st.data_editor(
    view,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "material": st.column_config.TextColumn("Material (category — family)"),
        "kg": st.column_config.NumberColumn("Masa (kg)", min_value=0.0, step=0.1),
        "notes": st.column_config.TextColumn("Flags/Notas"),
        "_source_id": st.column_config.TextColumn("ID NASA", help="id original del lote"),
        "_source_category": st.column_config.TextColumn("category"),
        "_source_material_family": st.column_config.TextColumn("material_family"),
        "_source_volume_l": st.column_config.NumberColumn("volume_l"),
        "_source_flags": st.column_config.TextColumn("flags"),
        "_problematic": st.column_config.CheckboxColumn("Problemático")
    },
    disabled=["_problematic"],  # derivado
)

# Importante: guardamos SIEMPRE el DF completo (no la vista filtrada)
if st.button("💾 Guardar inventario (formato NASA)", type="primary"):
    # Volvemos a mezclar los cambios en ‘view’ dentro del DF base por índice
    df.loc[edited.index, :] = edited
    save_waste_df(df)
    st.success("Inventario guardado en `data/waste_inventory_sample.csv` (esquema NASA).")

with st.expander("ℹ️ ¿Qué datos estamos usando y para qué sirven?"):
    st.markdown("""
- **`id`**: identifica el lote (trazabilidad en toda la demo).
- **`category`** y **`material_family`**: definen el **tipo de residuo** (pouches multicapa, textiles, espuma ZOTEK F30, EVA/CTB, guantes nitrilo, aluminio, etc.).
- **`mass_kg`** y **`volume_l`**: cuánto hay disponible (masa y volumen).
- **`flags`**: etiquetas críticas (ej. `multilayer`, `thermal`, `CTB`, `closed_cell`, `nitrile`, `struts`) que activan reglas del planificador de procesos y de seguridad.
    """)
