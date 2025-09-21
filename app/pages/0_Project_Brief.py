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

# ⚠️ Primero
st.set_page_config(page_title="REX-AI Mars — Brief", page_icon="🛰️", layout="wide")

from app.modules.branding import inject_branding
from app.modules.charts import predictions_ci_chart

inject_branding()

st.markdown("### Brief del proyecto")
with st.container():
    c1, c2, c3 = st.columns([1.4,1.2,1])
    with c1:
        st.markdown(
            """
            <div class="rex-card">
              <b>Descripción</b><br/>
              Reciclar basura inorgánica en Jezero Crater convirtiéndola en
              piezas útiles (utensilios, contenedores, interior del hábitat) minimizando
              agua/energía/tiempo de tripulación y evitando PFAS, microplásticos e incineración.
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="rex-card">
              <b>Restricciones clave</b>
              <ul>
                <li>Sin incineración ni emisiones tóxicas</li>
                <li>Minimizar agua y energía</li>
                <li>Tiempo de crew limitado</li>
                <li>Jezero Crater: uso de MGS-1 como carga/mezcla</li>
              </ul>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        inv_ok  = (repo_root/'data/waste_inventory_sample.csv').exists()
        proc_ok = (repo_root/'data/process_catalog.csv').exists()
        tgt_ok  = (repo_root/'data/targets_presets.json').exists()
        st.markdown(
            f"""
            <div class="rex-card">
              <b>Estado</b><br/>
              <div class="rex-chip {'ok' if inv_ok else ''}">Inventario: {'✅' if inv_ok else '❌'}</div>
              <div class="rex-chip {'ok' if proc_ok else ''}">Procesos: {'✅' if proc_ok else '❌'}</div>
              <div class="rex-chip {'ok' if tgt_ok  else ''}">Targets: {'✅' if tgt_ok  else '❌'}</div>
            </div>
            """, unsafe_allow_html=True
        )

st.markdown("### Ingredientes / Literatura / Regulación")
with st.container():
    a,b,c = st.columns(3)
    with a:
        st.markdown(
            """
            <div class="rex-card">
              <b>Ingredientes (residuos)</b><br/>
              Tabla NASA simplificada con tejidos, empaques, EVA, aluminio, etc.
            </div>
            """, unsafe_allow_html=True
        )
        # Navegación robusta: si page_link falla, no rompemos la app
        try:
            # Nota: según la configuración, Streamlit puede esperar "pages/..." o "app/pages/..."
            # Probamos ambas rutas.
            try:
                st.page_link("pages/1_Inventory_Builder.py", label="Abrir Inventario →", icon="🧱")
            except Exception:
                st.page_link("app/pages/1_Inventory_Builder.py", label="Abrir Inventario →", icon="🧱")
        except Exception:
            st.caption("Abrí **1) Inventario** desde la barra lateral (menú de páginas).")

    with b:
        st.markdown(
            """
            <div class="rex-card">
              <b>Literatura</b><br/>
              Enlaces al dossier de recursos NASA/MGS-1 y opciones de reuso inmediato.
              Incluye tablas de equivalentes comerciales.
            </div>
            """, unsafe_allow_html=True
        )
    with c:
        st.markdown(
            """
            <div class="rex-card">
              <b>Regulatorio/Seguridad</b><br/>
              Checklists para evitar PFAS/microplásticos y compatibilidad con O₂/CO₂.
              Integrado al validador de seguridad de los procesos.
            </div>
            """, unsafe_allow_html=True
        )

st.markdown("### Predicciones de ensayo (demo)")
cands = st.session_state.get("candidates", [])
if cands:
    rows = []
    for i,c in enumerate(cands, start=1):
        m = float(c["score"]); d = 5 + (i % 3) * 5
        rows.append(dict(batch=f"B{i:02d}", mean=m, lo=m-d, hi=m+d))
    dfp = pd.DataFrame(rows)
else:
    dfp = pd.DataFrame({
        "batch": [f"B2E{i}" for i in range(1,8)],
        "mean":  [48,39,42,50,35,28,37],
        "lo":    [30,18,25,28,15,12,20],
        "hi":    [62,52,58,61,45,40,49],
    })

fig = predictions_ci_chart(dfp, title="Score predictions")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
colL, colR = st.columns([1,1])
with colL:
    try:
        try:
            st.page_link("pages/1_Inventory_Builder.py", label="Ir a 1) Inventario", icon="🧱")
        except Exception:
            st.page_link("app/pages/1_Inventory_Builder.py", label="Ir a 1) Inventario", icon="🧱")
    except Exception:
        st.caption("Abrí **1) Inventario** desde la barra lateral →")
with colR:
    try:
        try:
            st.page_link("pages/2_Target_Designer.py", label="Ir a 2) Objetivo", icon="🎯")
        except Exception:
            st.page_link("app/pages/2_Target_Designer.py", label="Ir a 2) Objetivo", icon="🎯")
    except Exception:
        st.caption("Abrí **2) Objetivo** desde la barra lateral →")
