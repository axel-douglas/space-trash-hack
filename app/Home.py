# --- path guard ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------

import streamlit as st

# ⚠️ PRIMER comando de Streamlit:
st.set_page_config(
    page_title="Space Trash Hack — Demo",
    page_icon="🛰️",
    layout="wide"
)

from pathlib import Path

# Encabezado minimalista y seguro (sin HTML crudo)
logo_svg = ROOT / "app" / "static" / "logo_rexai.svg"
cols = st.columns([0.15, 0.85])
with cols[0]:
    if logo_svg.exists():
        # Streamlit soporta SVG
        st.image(str(logo_svg), use_column_width=True)
with cols[1]:
    st.title("REX-AI Mars")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

st.markdown("**Objetivo:** convertir basura inorgánica en productos útiles, minimizando agua/energía/tiempo de tripulación y evitando PFAS/microplásticos/incineración.")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Flujo en 4 pasos")
    st.info("1) Inventario — Cargá/edita residuos disponibles (tabla NASA simplificada).")
    st.info("2) Objetivo — Elegí producto y prioridades (agua/energía/tiempo/seguridad).")
    st.info("3) Generador — Recetas (mezclas) + proceso sugerido con predicciones.")
    st.info("4) Resultados — Pareto, Sankey, checklist y métricas de impacto.")

with col2:
    st.subheader("Estado del sistema")
    data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
    st.write("Datos de ejemplo:", "✅" if data_ok else "❌")
    st.caption("Requeridos: `data/waste_inventory_sample.csv` · `process_catalog.csv` · `targets_presets.json`")
    st.write("Modo:", "Demo local (modelos ligeros)")
    st.write("Restricciones:", "Sin incineración • Minimizar agua/energía • Evitar PFAS/microplásticos")

st.markdown("---")
st.caption(
    "Ruta: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → "
    "5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator"
)
