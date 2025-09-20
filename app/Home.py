# --- path guard universal (funciona en Home.py y en pages/*) ---
import sys, pathlib
_here = pathlib.Path(__file__).resolve()
p = _here.parent
while p.name != "app" and p.parent != p:
    p = p.parent
repo_root = p.parent if p.name == "app" else _here.parent  # fallback
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# ----------------------------------------------------------------

import streamlit as st

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(
    page_title="Space Trash Hack — Demo",
    page_icon="🛰️",
    layout="wide"
)

from pathlib import Path
from app.modules.ui_blocks import inject_css, card, section

# Ahora sí, podemos inyectar CSS (esto usa st.markdown por dentro)
inject_css()

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
    # ruta robusta basada en el archivo actual
    data_dir = repo_root / "data"
    data_ok = (data_dir / "waste_inventory_sample.csv").exists()
    st.write("Datos de ejemplo:", "✅" if data_ok else "❌")
    st.caption("`data/waste_inventory_sample.csv` | `process_catalog.csv` | `targets_presets.json`")
    st.write("Modo:", "Demo local (modelos ligeros)")
    st.write("Restricciones:", "Sin incineración • Minimizar agua/energía • Evitar PFAS/microplásticos")

st.markdown("---")
st.caption("Ruta: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → 5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator")
