import streamlit as st
from app.modules.ui_blocks import inject_css, card, section
inject_css()
from pathlib import Path

st.set_page_config(
    page_title="Space Trash Hack — Demo",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("# 🛰️ Space Trash Hack — Demo")
st.markdown("**Objetivo:** convertir basura inorgánica en productos útiles, minimizando agua/energía/tiempo de tripulación y evitando PFAS/microplásticos/incineración.")

col1, col2 = st.columns([2,1], gap="large")

with col1:
    section("Flujo en 4 pasos")
card("1) Inventario",
     "Cargá/edita residuos disponibles (tabla NASA simplificada).")
card("2) Objetivo",
     "Elegí producto y prioridades (agua/energía/tiempo/seguridad).")
card("3) Generador",
     "Recetas (mezclas) + proceso sugerido con predicciones.")
card("4) Resultados",
     "Pareto, Sankey, checklist y métricas de impacto.")


with col2:
    st.subheader("Estado del sistema")
    data_ok = Path("../data/waste_inventory_sample.csv").exists()
    st.write("Datos de ejemplo:", "✅" if data_ok else "❌")
    st.caption("`data/waste_inventory_sample.csv` | `process_catalog.csv` | `targets_presets.json`")
    st.write("Modo:", "Demo local (modelos ligeros)")
    st.write("Restricciones:", "Sin incineración • Minimizar agua/energía • Evitar PFAS/microplásticos")
