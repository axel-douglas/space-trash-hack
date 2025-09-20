# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import pandas as pd
from app.modules.io import load_waste_df, save_waste_df

st.set_page_config(page_title="Inventario", page_icon="🧱", layout="wide")

st.title("1) Inventario de residuos")
st.caption("Cargá/edita el inventario de basura inorgánica disponible.")

df = load_waste_df()
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
if st.button("💾 Guardar inventario", type="primary"):
    save_waste_df(edited)
    st.success("Inventario guardado.")
    
st.caption("Consejo: podés duplicar filas para simular nuevos lotes de residuos. Guardá antes de pasar al siguiente paso.")
