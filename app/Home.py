# app/Home.py
# --- path guard ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------

import streamlit as st
from pathlib import Path

# ⚠️ PRIMER comando de Streamlit:
st.set_page_config(
    page_title="REX-AI Mars — Mission Hub",
    page_icon="🛰️",
    layout="wide"
)

# ---------- Estilos SpaceX-like (seguros) ----------
st.markdown("""
<style>
:root{
  --bd: rgba(130,140,160,.28);
}
.hero{border:1px solid var(--bd); border-radius:20px; padding:22px;
      background: radial-gradient(1000px 300px at 20% -10%, rgba(80,120,255,.10), transparent);}
.hero h1{margin:0 0 6px 0}
.grid{display:grid; grid-template-columns: 1fr 1fr; gap:16px}
.card{border:1px solid var(--bd); border-radius:16px; padding:16px; background:rgba(255,255,255,.02)}
.card h3{margin:.1rem 0 .25rem 0}
.kpi{border:1px solid var(--bd); border-radius:14px; padding:14px;}
.kpi h3{margin:0 0 6px 0; font-size:.95rem; opacity:.8}
.kpi .v{font-size:1.5rem; font-weight:800; letter-spacing:.2px}
.pill{display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.78rem;
      border:1px solid var(--bd); margin-right:6px}
.pill.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.pill.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.pill.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}
.small{font-size:.92rem; opacity:.9}
</style>
""", unsafe_allow_html=True)

# ---------- Encabezado ----------
logo_svg = ROOT / "app" / "static" / "logo_rexai.svg"
cols = st.columns([0.15, 0.85])
with cols[0]:
    if logo_svg.exists():
        st.image(str(logo_svg), use_column_width=True)
with cols[1]:
    st.title("REX-AI Mars — Mission Hub")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

st.markdown("""
<div class="hero">
  <h1>Centro de Operaciones</h1>
  <div class="small">
    Bienvenidos al punto de partida. Acá orquestás el ciclo <b>Inventario → Objetivo → Generador → Resultados</b>,
    y luego exploras <b>Comparar</b>, <b>Pareto & Export</b>, <b>Playbooks</b>, <b>Feedback & Impact</b> y <b>Capacity</b>.
  </div>
  <div style="margin-top:8px">
    <span class="pill info">Seguridad por diseño (sin incineración)</span>
    <span class="pill info">Uso de regolito MGS-1 cuando aplica</span>
    <span class="pill ok">Optimizaciones multi-objetivo</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Estado del sistema ----------
st.markdown("### Estado del sistema")
c1, c2, c3, c4 = st.columns(4)
data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
proc_ok = (ROOT / "data" / "process_catalog.csv").exists()
tgt_ok  = (ROOT / "data" / "targets_presets.json").exists()
with c1:
    st.markdown('<div class="kpi"><h3>Inventario</h3><div class="v">{}</div></div>'.format("✅" if data_ok else "❌"), unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi"><h3>Procesos</h3><div class="v">{}</div></div>'.format("✅" if proc_ok else "❌"), unsafe_allow_html=True)
with c3:
    st.markdown('<div class="kpi"><h3>Targets</h3><div class="v">{}</div></div>'.format("✅" if tgt_ok else "❌"), unsafe_allow_html=True)
with c4:
    st.markdown('<div class="kpi"><h3>Modo</h3><div class="v">Demo ligera</div></div>', unsafe_allow_html=True)

st.caption("Requeridos: `data/waste_inventory_sample.csv` · `process_catalog.csv` · `targets_presets.json`")

st.markdown("---")

# ---------- Flujo en 4 pasos (navegación guiada) ----------
st.markdown("### Flujo de misión (4 pasos principales)")

gL, gR = st.columns([2, 1], gap="large")
with gL:
    st.markdown("""
<div class="grid">
  <div class="card">
    <h3>1) Inventario</h3>
    <div class="small">Cargá/edita residuos (pouches, ZOTEK F30, EVA/CTB, nitrilo, Al). Señalamos problemáticos y trazabilidad.</div>
  </div>
  <div class="card">
    <h3>2) Objetivo</h3>
    <div class="small">Elegí el target (Container/Utensil/Interior/Tool), prioridades y límites (agua/energía/crew) y escenario.</div>
  </div>
  <div class="card">
    <h3>3) Generador</h3>
    <div class="small">Crea recetas priorizando “consumir el problema” y elige procesos coherentes (P02/P03/P04). Inyecta MGS-1 en P03.</div>
  </div>
  <div class="card">
    <h3>4) Resultados</h3>
    <div class="small">Desglose de score, Sankey de flujo, checklist de fabricación y recursos. Seguridad visual (badges).</div>
  </div>
</div>
""", unsafe_allow_html=True)

with gR:
    st.subheader("Ir ahora")
    cA, cB = st.columns(2)
    with cA:
        if st.button("🧱 Inventario"):
            st.switch_page("pages/1_Inventory_Builder.py")
        if st.button("⚙️ Generador"):
            st.switch_page("pages/3_Generator.py")
    with cB:
        if st.button("🎯 Objetivo"):
            st.switch_page("pages/2_Target_Designer.py")
        if st.button("📊 Resultados"):
            st.switch_page("pages/4_Results_and_Tradeoffs.py")

st.markdown("---")

# ---------- Qué es REX-AI (técnico + criollo) ----------
st.markdown("### ¿Qué es REX-AI? (para ingenieros y aprendices)")

e1, e2 = st.columns(2)
with e1:
    st.markdown("""
**Explicación en criollo**
- Es un **copiloto** que mira tu basura inorgánica y propone **mezclas + procesos** para fabricar algo útil.
- Aprende de las restricciones del hábitat (agua, kWh, tiempo de crew) y **evita rutas sucias** (PFAS, incineración).
- Cuando el proceso es **P03 (Sinter with MGS-1)**, mete regolito **MGS-1** como carga mineral (p. ej., 20%) para ganar rigidez.
- Te muestra los **trade-offs**: si apretás agua, quizás sube el tiempo de crew; si sumás MGS-1, crece rigidez pero puede bajar estanqueidad.
""")
with e2:
    st.markdown("""
**Bajo el capó (software)**
- **Módulos**: `io` (carga/guarda), `generator` (recetas), `process_planner` (filtrado por escenario/crew), `safety` (banderas), `explain` (scores), `analytics` (Pareto), `exporters` (JSON/CSV).
- **Scoring multiobjetivo** simple y **transparente** (legible por humanos): compatibilidad con target + penalización por recursos + bonus por “consumir problemáticos”.
- **Trazabilidad NASA**: cada candidato lleva `source_ids/categories/flags` para auditar de dónde viene cada kg.
- **Persistencia**: inventario y logs en `/data`; estado corto en `st.session_state` (target, candidates, selected).
""")

st.markdown("### Ciencia de materiales en dos frases")
st.markdown(
"- **Compatibilidades**: multilayer (PE-PET-Al) se presta a laminación térmica; espumas **ZOTEK F30** pueden densificarse/laminarse o sinterizar con **MGS-1**; CTB/EVA va a reuso/reconfiguración.\n"
"- **MGS-1**: como carga mineral aumenta rigidez y modula comportamiento térmico; hay que vigilar estanqueidad y parámetros de sinterizado."
)

st.markdown("---")
st.caption(
    "Ruta completa: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → "
    "5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator"
)
