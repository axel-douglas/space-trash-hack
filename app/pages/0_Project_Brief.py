# app/pages/0_Project_Brief.py
import _bootstrap  # noqa: F401

import streamlit as st
from pathlib import Path

from app.modules.navigation import set_active_step
from app.modules.ui_blocks import load_theme
repo_root = Path(__file__).resolve().parents[2]

# ⚠️ PRIMER comando Streamlit:
st.set_page_config(page_title="REX-AI Mars — Brief", page_icon="🛰️", layout="wide")

_current_step = set_active_step("brief")

load_theme()
# ---------- Header ----------
logo_svg = repo_root / "app" / "static" / "logo_rexai.svg"
cols = st.columns([0.15, 0.85])
with cols[0]:
    if logo_svg.exists():
        st.image(str(logo_svg), use_column_width=True)
with cols[1]:
    st.title("REX-AI Mars — Brief")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

# ---------- Descripción ----------
st.markdown("""
<div class="hero">
  <div class="small">
    <b>Objetivo de misión</b>: reciclar basura inorgánica en el cráter Jezero transformándola en piezas útiles,
    minimizando <b>agua</b>, <b>energía</b> y <b>tiempo de tripulación</b>, evitando PFAS, microplásticos e incineración,
    e incorporando <b>regolito MGS-1</b> cuando el proceso lo admite.
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Restricciones clave")
    st.markdown(
        "- Sin incineración ni emisiones tóxicas\n"
        "- Minimizar agua y energía\n"
        "- Tiempo de crew limitado\n"
        "- MGS-1 como carga/mezcla (Jezero)"
    )
with c2:
    st.subheader("Estado de datos")
    inv_ok  = (repo_root / "data" / "waste_inventory_sample.csv").exists()
    proc_ok = (repo_root / "data" / "process_catalog.csv").exists()
    tgt_ok  = (repo_root / "data" / "targets_presets.json").exists()
    st.write("Inventario:", "✅" if inv_ok else "❌")
    st.write("Procesos:", "✅" if proc_ok else "❌")
    st.write("Targets:", "✅" if tgt_ok else "❌")
with c3:
    st.subheader("Navegación")
    st.markdown(
        "Usá la barra superior **Mission HUD** o las teclas `1-9` para saltar de paso.\n"
        "También podés abrir la barra lateral estándar de Streamlit para ver todas las páginas."
    )

st.divider()
st.info(
    "Usá la **barra lateral** o los botones de arriba para navegar. "
    "Los gráficos de predicciones y análisis viven en **5/6/7/8/9** según el flujo."
)
