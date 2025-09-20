import streamlit as st
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
    st.subheader("Flujo en 4 pasos")
    st.markdown("""
1. **Inventario** — cargar/editar residuos disponibles (NASA non-metabolic waste simplificado).  
2. **Objetivo** — elegir el producto (contenedor/utensilio/interior/herramienta) y prioridades (agua/energía/tiempo/seguridad).  
3. **Generador** — el motor propone **recetas** (mezcla de residuos) y **proceso** (pipeline).  
4. **Resultados** — métricas, Pareto, Sankey y checklist de fabricación.
""")
    st.info("Usá el menú **Pages** (a la izquierda) para avanzar por el wizard.")

with col2:
    st.subheader("Estado del sistema")
    data_ok = Path("../data/waste_inventory_sample.csv").exists()
    st.write("Datos de ejemplo:", "✅" if data_ok else "❌")
    st.caption("`data/waste_inventory_sample.csv` | `process_catalog.csv` | `targets_presets.json`")
    st.write("Modo:", "Demo local (modelos ligeros)")
    st.write("Restricciones:", "Sin incineración • Minimizar agua/energía • Evitar PFAS/microplásticos")
