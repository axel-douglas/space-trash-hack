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

# ⚠️ Primero
st.set_page_config(page_title="Scenario Playbooks", page_icon="📚", layout="wide")

from app.modules.scenarios import PLAYBOOKS

st.title("7) Scenario Playbooks")

target = st.session_state.get("target", None)
if not target:
    st.warning("Definí primero el escenario en **2) Target Designer**.")
    st.stop()

scenario = target.get("scenario", "Residence Renovations")
pb = PLAYBOOKS.get(scenario)

if pb is None:
    st.error("No se encontró el playbook del escenario. Volvé a definir el objetivo.")
    st.stop()

st.subheader(f"{pb.name}")
st.write(pb.summary)

st.markdown("---")
st.subheader("Instrucciones detalladas")
for i, step in enumerate(pb.steps, start=1):
    st.markdown(f"**{i}. {step.title}**")
    st.write(step.detail)
