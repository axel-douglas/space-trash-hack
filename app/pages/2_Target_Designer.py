import _bootstrap  # noqa: F401

import streamlit as st

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Objetivo", page_icon="🎯", layout="wide")

from app.modules.io import load_targets
from app.modules.luxe_components import target_configurator
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.ui_blocks import load_theme, section

current_step = set_active_step("target")

load_theme()

render_breadcrumbs(current_step)

st.title("2) Definir objetivo (TargetSpec)")

presets = load_targets()
if not presets:
    st.error("No se encontraron presets de objetivos. Verifica `data/targets_presets.json`.")
    st.stop()

scenario_options = (
    "Residence Renovations",
    "Cosmic Celebrations",
    "Daring Discoveries",
)

target = target_configurator(presets, scenario_options=scenario_options)

if target:
    st.session_state["target"] = target
    st.success("Objetivo listo. Abrí la página **3) Generador** para obtener recetas.")
