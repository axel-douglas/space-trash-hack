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
from app.modules.io import load_waste_df, save_waste_df

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Inventario", page_icon="🧱", layout="wide")

st.title("1) Inventario de residuos")
st.caption("Cargá/edita el inventario de basura inorgánica disponible.")

df = load_waste_df()
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
if st.button("💾 Guardar inventario", type="primary"):
    save_waste_df(edited)
    st.success("Inventario guardado.")

st.caption("Consejo: podés duplicar filas para simular nuevos lotes de residuos. Guardá antes de pasar al siguiente paso.")
