# app/Home.py
# ───────────────────────── path guard ─────────────────────────
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ──────────────────────────────────────────────────────────────

import streamlit as st
from pathlib import Path

# ⚠️ Debe ser la PRIMERA llamada de Streamlit
st.set_page_config(
    page_title="REX-AI Mars — Mission Hub",
    page_icon="🛰️",
    layout="wide"
)

# ============ Estilos (livianos y seguros) ============
st.markdown("""
<style>
:root{
  --ink: #e8eefc;
  --bd: rgba(140,150,170,.28);
  --card: rgba(255,255,255,.03);
}
.block-container{padding-top: 1.8rem;}
.hero{
  border:1px solid var(--bd);
  border-radius:22px;
  padding:28px;
  margin-bottom:20px;
  background:
    radial-gradient(1400px 520px at 10% -10%, rgba(80,120,255,.12), transparent),
    linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.0));
}
.hero h1{margin:0 0 8px 0; font-size:2.05rem; letter-spacing:.2px;}
.tagline{font-size:1.1rem; opacity:.92; margin-bottom:14px}
.pills{margin-top:6px;}
.pill{
  display:inline-block; padding:4px 12px; border-radius:999px; font-weight:700; font-size:.78rem;
  border:1px solid var(--bd); margin-right:6px; background:var(--card)
}
.pill.ok{color:#136c3a; border-color:#b3e2c4; background:#e8f7ee}
.pill.info{color:#174ea6; border-color:#c6dcff; background:#e7f1ff}
.pill.warn{color:#8a6d1d; border-color:#ffe69b; background:#fff7da}
.grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap:18px; margin:16px 0;}
.card{border:1px solid var(--bd); border-radius:16px; padding:18px; background:var(--card);}
.card h3{margin:.1rem 0 .35rem 0; font-size:1.02rem}
.small{font-size:.92rem; opacity:.92}
.kpi{border:1px solid var(--bd); border-radius:14px; padding:14px; text-align:center; background:var(--card);}
.kpi h3{margin:0 0 6px 0; font-size:.95rem; opacity:.85}
.kpi .v{font-size:1.6rem; font-weight:800; letter-spacing:.2px}
.ghost{opacity:.66}
.section{margin-top:28px; margin-bottom:12px;}
.section h2{margin-bottom:6px;}
.cta-row{display:flex; gap:10px; flex-wrap:wrap}
hr.sep{border: none; height:1px; background:var(--bd); margin:6px 0 10px}
.badge{
  font-size:.72rem; font-weight:700; border:1px solid var(--bd); border-radius:999px; padding:2px 8px;
  color:#444; background:rgba(255,255,255,.5);
}
</style>
""", unsafe_allow_html=True)

# ============ Encabezado ============
logo_svg = ROOT / "app" / "static" / "logo_rexai.svg"
col_logo, col_title = st.columns([0.13, 0.87], vertical_alignment="center")
with col_logo:
    if logo_svg.exists():
        st.image(str(logo_svg), use_column_width=True)
with col_title:
    st.title("REX-AI Mars — Mission Hub")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

# ============ Hero ============
st.markdown("""
<div class="hero">
  <h1>Convertimos residuos en hardware útil para la misión</h1>
  <div class="tagline">
    REX-AI diseña recetas y procesos que maximizan utilidad con mínimos de agua, energía y tiempo de tripulación, 
    integrando <strong>MGS-1</strong> cuando corresponde.
  </div>
  <div class="pills">
    <span class="pill info">Reciclaje asistido por IA</span>
    <span class="pill ok">Optimización multi-objetivo</span>
    <span class="pill warn">Guardrails de seguridad</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============ Estado del sistema (KPI) ============
st.markdown("### Estado del sistema")
c1, c2, c3, c4 = st.columns(4)
data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
proc_ok = (ROOT / "data" / "process_catalog.csv").exists()
tgt_ok  = (ROOT / "data" / "targets_presets.json").exists()
with c1: st.markdown(f'<div class="kpi"><h3>Inventario</h3><div class="v">{"✅" if data_ok else "❌"}</div><div class="ghost small">waste_inventory_sample.csv</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi"><h3>Procesos</h3><div class="v">{"✅" if proc_ok else "❌"}</div><div class="ghost small">process_catalog.csv</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi"><h3>Targets</h3><div class="v">{"✅" if tgt_ok else "❌"}</div><div class="ghost small">targets_presets.json</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="kpi"><h3>Modo</h3><div class="v">Demo</div><div class="ghost small">modelos ligeros</div></div>', unsafe_allow_html=True)

# ============ Flujo principal ============
st.markdown("### Flujo de misión")
st.markdown("""
<div class="grid">
  <div class="card">
    <h3>1) Inventario</h3>
    <div class="small">Subí y editá residuos. Detectamos <strong>problemáticos</strong> (multicapa, espumas, EVA/CTB, nitrilo) y los marcamos para priorizar su valorización.</div>
  </div>
  <div class="card">
    <h3>2) Objetivo</h3>
    <div class="small">Elegí el producto (Container, Tool, Interior, Utensil) y definí límites: agua, energía y minutos de crew.</div>
  </div>
  <div class="card">
    <h3>3) Generador</h3>
    <div class="small">REX-AI arma recetas con trazabilidad (IDs/flags) y sugiere proceso (laminar, sinter + <strong>MGS-1</strong>, reuso CTB, etc.).</div>
  </div>
  <div class="card">
    <h3>4) Resultados</h3>
    <div class="small">Verás métricas, <em>trade-offs</em>, Sankey de materiales → proceso → producto y checklist de fabricación.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============ CTA navegación rápida ============
st.markdown('<div class="cta-row">', unsafe_allow_html=True)
colA, colB, colC, colD = st.columns(4)
with colA:
    if st.button("🧱 Abrir 1) Inventario", use_container_width=True):
        st.switch_page("pages/1_Inventory_Builder.py")
with colB:
    if st.button("🎯 Abrir 2) Objetivo", use_container_width=True):
        st.switch_page("pages/2_Target_Designer.py")
with colC:
    if st.button("⚙️ Abrir 3) Generador", use_container_width=True):
        st.switch_page("pages/3_Generator.py")
with colD:
    if st.button("📊 Abrir 4) Resultados", use_container_width=True):
        st.switch_page("pages/4_Results_and_Tradeoffs.py")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="sep" />', unsafe_allow_html=True)

# ============ Tabs: Modo Experto / Modo Criollo ============
tab_expert, tab_criollo = st.tabs(["👩‍🚀 Modo Experto", "🤠 Modo Criollo"])

with tab_expert:
    st.markdown("#### ¿Qué hace REX-AI (detalle técnico)?")
    st.markdown("""
- **Datos**: inventario (IDs, categorías, flags), catálogo de procesos (energía/agua/crew), targets (rigidez/estanqueidad y límites).
- **Generación**: selección de materiales ponderada por masa y “problematicidad”, heurísticas de proceso coherentes y **uso explícito de MGS-1** en P03.
- **Predicción ligera**: propiedades demo (rigidez, estanqueidad, masa final) + recursos derivados del proceso.
- **Score multi-objetivo**: cercanía al target, penalización por límites y **bono por consumir problemáticos**.
- **Trazabilidad**: cada candidato conserva `source_ids`, `source_categories`, `source_flags`, y `regolith_pct`.
- **Export**: JSON/CSV con todo el plan (para manufactura/validación).
""")
    st.info("Mensaje clave: hoy es una demo funcional con explicabilidad y guardrails. Los mismos hooks permiten escalar a modelos avanzados, MLOps y control de planta sin reescribir la UX.")

with tab_criollo:
    st.markdown("#### REX-AI explicado sin vueltas")
    st.markdown("""
**Imaginate** que el hábitat junta bolsas multicapa, espumas, bolsas EVA/CTB y guantes.  
REX-AI los ve como **ingredientes** y arma recetas para fabricar cosas útiles (un contenedor, una pieza interior, una herramienta).  
Te muestra **cuánta agua, energía y minutos de tripulación** vas a gastar y qué tan bien queda.  
Si el proceso es sinterizado, mete **regolito MGS-1** para darle cuerpo.  
Al final, te exporta la receta con todos los detalles para hacerla de verdad.
""")
    st.success("En criollo: es un recetario inteligente para transformar basura espacial en hardware útil, gastando lo menos posible.")

# ============ Sección “Cómo testear hoy” ============
st.markdown("### Cómo testear hoy (demo guide)")
g1, g2, g3 = st.columns(3)
with g1:
    st.markdown("**1) Cargá inventario**\n\nEditá la tabla y marcá flags (ej. `multilayer`, `CTB`, `ZOTEK`).")
with g2:
    st.markdown("**2) Definí el objetivo**\n\nElegí el producto y límites de agua/energía/minutos.")
with g3:
    st.markdown("**3) Generá y evaluá**\n\nCorré el generador, seleccioná una receta y mirá resultados/explicaciones.")

# ============ Ruta completa ============
st.markdown("---")
st.caption(
    "Ruta: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → "
    "5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator"
)
