# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
from app.modules.scenarios import PLAYBOOKS

st.set_page_config(page_title="Scenario Playbooks", page_icon="📚", layout="wide")
st.title("7) Scenario Playbooks")

target = st.session_state.get("target", None)
if not target:
    st.warning("Definí primero el escenario en **2) Target Designer**.")
    st.stop()

scenario = target.get("scenario", "Residence Renovations")
pb = PLAYBOOKS.get(scenario)

st.subheader(f"{pb.name}")
st.write(pb.summary)

st.markdown("---")
st.subheader("Instrucciones detalladas")
for i, step in enumerate(pb.steps, start=1):
    st.markdown(f"**{i}. {step.title}**")
    st.write(step.detail)
