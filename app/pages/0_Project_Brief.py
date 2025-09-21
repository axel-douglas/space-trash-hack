# --- path guard universal ---
import sys, pathlib
_here = pathlib.Path(__file__).resolve()
p = _here.parent
while p.name != "app" and p.parent != p:
    p = p.parent
repo_root = p.parent if p.name == "app" else _here.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# --------------------------------

import streamlit as st
import pandas as pd

# ⚠️ PRIMER comando Streamlit:
st.set_page_config(page_title="REX-AI Mars — Brief", page_icon="🛰️", layout="wide")

# Header seguro (mismo patrón que en Home)
logo_svg = repo_root / "app" / "static" / "logo_rexai.svg"
cols = st.columns([0.15, 0.85])
with cols[0]:
    if logo_svg.exists():
        st.image(str(logo_svg), use_column_width=True)
with cols[1]:
    st.title("REX-AI Mars — Brief")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

st.subheader("Descripción")
st.write(
    "Reciclar basura inorgánica en Jezero Crater convirtiéndola en piezas útiles, "
    "minimizando agua/energía/tiempo de tripulación y evitando PFAS, microplásticos e incineración."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Restricciones clave")
    st.markdown("- Sin incineración ni emisiones tóxicas\n- Minimizar agua y energía\n- Tiempo de crew limitado\n- MGS-1 como carga/mezcla (Jezero)")
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
    colA, colB = st.columns(2)
    with colA:
        if st.button("🧱 1) Inventario"):
            st.switch_page("pages/1_Inventory_Builder.py")
        if st.button("⚙️ 3) Generador"):
            st.switch_page("pages/3_Generator.py" if (repo_root / "app" / "pages" / "3_Generator.py").exists() else "pages/3_Generator.py")
    with colB:
        if st.button("🎯 2) Objetivo"):
            st.switch_page("pages/2_Target_Designer.py")
        if st.button("📊 4) Resultados"):
            st.switch_page("pages/4_Results_and_Tradeoffs.py")

st.subheader("Predicciones de ensayo (demo)")
cands = st.session_state.get("candidates", [])
if cands:
    rows = []
    for i, c in enumerate(cands, 1):
        m = float(c.get("score", 40.0))
        d = 5 + (i % 3) * 5
        rows.append(dict(batch=f"B{i:02d}", mean=m, lo=m - d, hi=m + d))
    dfp = pd.DataFrame(rows)
else:
    dfp = pd.DataFrame({
        "batch": [f"B2E{i}" for i in range(1, 8)],
        "mean":  [48, 39, 42, 50, 35, 28, 37],
        "lo":    [30, 18, 25, 28, 15, 12, 20],
        "hi":    [62, 52, 58, 61, 45, 40, 49],
    })

# Gráfico simple y robusto (sin HTML/CSS custom)
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dfp["batch"], y=dfp["mean"], mode="lines+markers", name="mean"
))
fig.add_trace(go.Scatter(
    x=list(dfp["batch"]) + list(dfp["batch"][::-1]),
    y=list(dfp["hi"]) + list(dfp["lo"][::-1]),
    fill='toself', line=dict(width=0), name="CI"
))
fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=300)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.info("Usá la **barra lateral** o los botones de arriba para navegar las páginas.")
