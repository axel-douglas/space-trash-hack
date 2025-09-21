# app/Home.py
# ────────────────── Path guard (robusto para Streamlit Cloud) ──────────────────
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ───────────────────────────────────────────────────────────────────────────────

import streamlit as st
from pathlib import Path

# ⚠️ Debe ser la primera llamada de Streamlit
st.set_page_config(
    page_title="REX-AI Mars — Mission Hub",
    page_icon="🛰️",
    layout="wide"
)

# ─────────────────────────────── Estilos UI (SpaceX vibe) ─────────────────────
st.markdown("""
<style>
:root{
  --bd: rgba(130,140,160,.28);
  --ink: #e6eefc;
}
html, body, [data-testid="stAppViewContainer"] {background: radial-gradient(1400px 500px at 10% -10%, rgba(80,120,255,.08), transparent) !important;}
.hero{
  border:1px solid var(--bd); border-radius:24px; padding:28px; margin:2px 0 22px 0;
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01));
}
.hero h1{margin:0 0 6px 0; font-size:2.1rem; letter-spacing:.2px}
.hero .tag{display:inline-block; padding:5px 10px; border-radius:999px; border:1px solid var(--bd); margin:6px 6px 0 0; font-weight:700; font-size:.78rem;}
.hero .tag.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.hero .tag.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.hero .tag.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}

.grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:16px; margin:8px 0 6px 0;}
.card{border:1px solid var(--bd); border-radius:16px; padding:16px; background:rgba(255,255,255,.02)}
.card h3{margin:.2rem 0 .25rem 0; font-size:1.05rem}
.card p{margin:0; opacity:.9}

.kpi{border:1px solid var(--bd); border-radius:16px; padding:14px; text-align:center; background:rgba(255,255,255,.02)}
.kpi h4{margin:0 0 4px 0; font-size:.95rem; opacity:.8}
.kpi .v{font-size:1.6rem; font-weight:800; letter-spacing:.3px; margin-top:2px}

.nav-cta{display:flex; gap:10px; flex-wrap:wrap; margin-top:6px}
.small{font-size:.94rem; opacity:.95}
.section{margin-top:22px}
hr.sep{border:none; border-top:1px solid var(--bd); margin:16px 0}
.note{font-size:.88rem; opacity:.85}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────── Encabezado ───────────────────────────────────
logo_svg = ROOT / "app" / "static" / "logo_rexai.svg"
c_logo, c_head = st.columns([0.16, 0.84])
with c_logo:
    if logo_svg.exists():
        st.image(str(logo_svg), use_column_width=True)
with c_head:
    st.title("REX-AI Mars — Mission Hub")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

# ─────────────────────────────── Hero principal ────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Convierte basura espacial en ventaja estratégica</h1>
  <div class="small">
    REX-AI usa datos reales de inventario, procesos y límites de misión para generar <b>recetas</b> y <b>planes</b> que ahorran agua, energía y tiempo de tripulación. 
    Con trazabilidad de residuos y uso seguro de <b>MGS-1</b> como carga local.
  </div>
  <div>
    <span class="tag info">Reciclaje inteligente</span>
    <span class="tag ok">Optimización multi-objetivo</span>
    <span class="tag warn">Guardrails de seguridad</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────── KPIs del sistema ─────────────────────────────
c1, c2, c3, c4 = st.columns(4)
data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
proc_ok = (ROOT / "data" / "process_catalog.csv").exists()
tgt_ok  = (ROOT / "data" / "targets_presets.json").exists()
with c1: st.markdown(f'<div class="kpi"><h4>Inventario</h4><div class="v">{"✅" if data_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi"><h4>Procesos</h4><div class="v">{"✅" if proc_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi"><h4>Targets</h4><div class="v">{"✅" if tgt_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="kpi"><h4>Modo</h4><div class="v">Demo</div></div>', unsafe_allow_html=True)

# ───────────────────────────── CTA de navegación clara ────────────────────────
st.markdown("#### Empezar ahora")
c_nav1, c_nav2, c_nav3, c_nav4 = st.columns(4)
with c_nav1: st.page_link("pages/1_Inventory_Builder.py", label="🧱 1) Inventario", use_container_width=True)
with c_nav2: st.page_link("pages/2_Target_Designer.py", label="🎯 2) Objetivo", use_container_width=True)
with c_nav3: st.page_link("pages/3_Generator.py", label="⚙️ 3) Generador", use_container_width=True)
with c_nav4: st.page_link("pages/4_Results_and_Tradeoffs.py", label="📊 4) Resultados", use_container_width=True)

# ─────────────────────────────── Flujo de misión ──────────────────────────────
st.markdown("### Flujo de misión (de cero a decisión)")
st.markdown("""
<div class="grid">
  <div class="card">
    <h3>1) Inventario</h3>
    <p>Subí o edita residuos reales. REX-AI detecta <b>problemáticos</b> (pouches multicapa, espumas, EVA/CTB, nitrilo) y los prioriza.</p>
  </div>
  <div class="card">
    <h3>2) Objetivo</h3>
    <p>Elegí qué fabricar (Container, Tool, Interior, Utensil) y fija límites: agua, energía y minutos de crew.</p>
  </div>
  <div class="card">
    <h3>3) Generador</h3>
    <p>Calcula recetas plausibles, elige el proceso (laminado, sinter con <b>MGS-1</b>, reuso CTB) y predice propiedades.</p>
  </div>
  <div class="card">
    <h3>4) Resultados</h3>
    <p>Trade-offs, Sankey (residuo→proceso→producto), checklist de fabricación y seguridad.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────── Doble modo: Tech vs Criollo (toggle) ─────────────────
st.markdown("### ¿Cómo lo cuenta REX-AI?")
modo_criollo = st.toggle("🪄 Modo criollo (explicación para todos)", value=False)

if not modo_criollo:
    # ────────────────────────────── Versión técnica SpaceX ─────────────────────
    st.markdown("#### Para ingeniería (SpaceX-style)")
    st.markdown("""
- **Datos**: inventario NASA (IDs, categorías, flags), catálogo de procesos (`process_catalog.csv`), targets con límites.  
- **Motor**: generador con selección de materiales ponderada por masa y “problematic score” + heurísticas de proceso.  
- **MGS-1**: si el proceso es `P03`, se inyecta 10–30% de regolito y se refleja en pesos/propiedades.  
- **Predicción ligera**: rigidez/estanqueidad razonables según familia de material y proceso elegido; recursos escalados por kg.  
- **Score multi-objetivo**: cercanía a target + penalización (agua/kWh/crew) + bono por masa problemática consumida.  
- **Trazabilidad**: cada candidato guarda `source_ids`, `source_categories`, `source_flags` y `regolith_pct`.  
- **Guardrails**: sin incineración, evitar PFAS/microplásticos; checks de seguridad por proceso.
""")
else:
    # ─────────────────────────────── Versión criolla divertida ─────────────────
    st.markdown("#### En criollo (divertido y claro)")
    st.markdown("""
Pensalo como un **recetario de cocina espacial**:

1) Miramos qué hay en la “alacena” (tu basura inorgánica).  
2) Elegís qué plato querés (contenedor, herramienta).  
3) La app arma recetas con lo disponible y dice: **“con esto y este proceso te sale bien, gastás poco y es seguro”**.  
4) Te muestra si conviene mezclar con **MGS-1** (la harina local de Marte) y cuánto rinde la tanda.  

**¿Por qué importa?**  
Menos agua y energía gastada, menos minutos de astronauta ocupados y más piezas útiles cada semana.
""")

# ───────────────────────────── Por qué importa (con ejemplos) ─────────────────
st.markdown("### ¿Por qué es clave para la misión?")
st.markdown("""
- **Tiempo de crew** es oro: REX-AI empuja procesos rápidos de ejecutar.
- **Agua/energía** son limitadas: prioriza recetas frugales.
- **ISRU real**: con <b>MGS-1</b> convertimos “lo que sobra” en “lo que falta”.
- **De Marte a la Tierra**: lo que aprendemos sirve para reciclar multilayer/espumas que hoy son problema en ciudades.

**Ejemplos**  
- Pouches PE-PET-Al + laminar/press: láminas estancas para compartimentos.  
- ZOTEK F30 (espuma) + sinter con MGS-1: panel rígido para interiores de hábitat.  
""")

st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

# ───────────────────────────── Ruta extendida de trabajo ──────────────────────
st.markdown("#### Ruta completa de trabajo")
st.caption(
    "1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → "
    "5) Compare & Explain → 6) Pareto & Export → 7) Scenario Playbooks → 8) Feedback & Impact → 9) Capacity Simulator"
)

c_navA, c_navB, c_navC, c_navD, c_navE = st.columns(5)
with c_navA: st.page_link("pages/5_Compare_and_Explain.py", label="🧪 5) Compare & Explain", use_container_width=True)
with c_navB: st.page_link("pages/6_Pareto_and_Export.py", label="📤 6) Pareto & Export", use_container_width=True)
with c_navC: st.page_link("pages/7_Scenario_Playbooks.py", label="📚 7) Scenario Playbooks", use_container_width=True)
with c_navD: st.page_link("pages/8_Feedback_and_Impact.py", label="📝 8) Feedback & Impact", use_container_width=True)
with c_navE: st.page_link("pages/9_Capacity_Simulator.py", label="🧮 9) Capacity Simulator", use_container_width=True)

# ───────────────────────── Notas y garantías de diseño ────────────────────────
st.markdown('<hr class="sep"/>', unsafe_allow_html=True)
st.caption("Diseño pensado para jurado técnico y aprendices: fuerte narrativa, UI clara, sin dependencias JS externas, y navegación segura entre páginas.")
