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

# ============== Estilos SpaceX-like (seguros) ==============
st.markdown("""
<style>
:root{
  --bd: rgba(130,140,160,.28);
  --ink: rgba(230,240,255,.95);
}
.hero{
  border:1px solid var(--bd); border-radius:24px; padding:26px;
  background:
    radial-gradient(1000px 300px at 20% -10%, rgba(80,120,255,.10), transparent),
    linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0));
}
.hero h1{margin:.2rem 0 .3rem 0; letter-spacing:.2px}
.hero p{font-size:1.05rem; opacity:.92; margin:0}

.grid{display:grid; grid-template-columns: 1fr 1fr; gap:16px}
.grid3{display:grid; grid-template-columns: 1fr 1fr 1fr; gap:16px}

.card{
  border:1px solid var(--bd);
  border-radius:18px; padding:18px; background:rgba(255,255,255,.02)
}
.card h3{margin:.1rem 0 .25rem 0}
.card p{margin:.2rem 0 .4rem 0; opacity:.95}

.kpi{border:1px solid var(--bd); border-radius:16px; padding:16px;}
.kpi h4{margin:0 0 6px 0; font-size:.94rem; opacity:.8}
.kpi .v{font-size:1.5rem; font-weight:800; letter-spacing:.2px}

.pill{
  display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.78rem;
  border:1px solid var(--bd); margin-right:6px
}
.pill.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.pill.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.pill.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}

.lead{font-size:1.05rem; opacity:.96}
.small{font-size:.92rem; opacity:.9}
.center{text-align:center}
.bignum{font-size:2.2rem; font-weight:800; letter-spacing:.3px}
hr.s{
  border:0; border-top:1px solid var(--bd); margin:16px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ============== Encabezado ==============
logo_svg = ROOT / "app" / "static" / "logo_rexai.svg"
cols = st.columns([0.15, 0.85])
with cols[0]:
    if logo_svg.exists():
        st.image(str(logo_svg), use_column_width=True)
with cols[1]:
    st.title("REX-AI Mars — Mission Hub")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

# ============== HERO: ¿Qué va a probar el jurado? ==============
st.markdown("""
<div class="hero">
  <h1>Conocé a REX-AI, la tecnología que lo hace posible</h1>
  <p class="lead">
    Un “copiloto” de ingeniería que toma <b>residuos inorgánicos</b> y los transforma en <b>piezas útiles</b> dentro de un hábitat,
    minimizando <b>agua, energía y tiempo de tripulación</b>. Sin incineración. Con <b>MGS-1</b> cuando el proceso lo pide.
  </p>
  <div style="margin-top:10px">
    <span class="pill info">Optimización multi-objetivo</span>
    <span class="pill info">Procesos P02/P03/P04 (catálogo)</span>
    <span class="pill ok">Trazabilidad NASA end-to-end</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============== Estado del sistema (en vivo) ==============
st.markdown("### Estado de misión (en vivo)")
c1, c2, c3, c4 = st.columns(4)
data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
proc_ok = (ROOT / "data" / "process_catalog.csv").exists()
tgt_ok  = (ROOT / "data" / "targets_presets.json").exists()
with c1:
    st.markdown(f'<div class="kpi"><h4>Inventario</h4><div class="v">{"✅" if data_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><h4>Procesos</h4><div class="v">{"✅" if proc_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi"><h4>Targets</h4><div class="v">{"✅" if tgt_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c4:
    # KPIs rápidos desde session_state si existen
    cand_count = len(st.session_state.get("candidates", []))
    sel_ok = "✅" if st.session_state.get("selected") else "—"
    st.markdown(
        f'<div class="kpi"><h4>Run actual</h4>'
        f'<div class="v">{cand_count} cand.</div>'
        f'<div class="small">Seleccionado: {sel_ok}</div></div>',
        unsafe_allow_html=True
    )
st.caption("Requeridos: `data/waste_inventory_sample.csv` · `process_catalog.csv` · `targets_presets.json`")

st.markdown("---")

# ============== “Solo preguntale a REX” — storytelling + CTA ==============
left, right = st.columns([1.4, 1])
with left:
    st.subheader("“Solo preguntale a REX” — ¿Cómo arranca todo?")
    st.markdown("""
1) Cargás el **inventario real** (pouches multilayer, ZOTEK F30, EVA/CTB, nitrilo, Al).  
2) Definís un **objetivo** (p.ej., *Container*) con límites de **agua/kWh/crew** y escenario.  
3) El **Generador** propone **recetas + proceso** (P02/P03/P04), priorizando “consumir el problema”.  
4) En **Resultados**, ves **trade-offs**, Sankey, checklist de fabricación y seguridad.  
5) Luego comparás, haces Pareto, exportás planes, seguís playbooks, das feedback y simulas capacidad.
""")
with right:
    st.subheader("Ir al flujo")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🧱 1) Inventario"):
            st.switch_page("pages/1_Inventory_Builder.py")
        if st.button("⚙️ 3) Generador"):
            st.switch_page("pages/3_Generator.py")
    with colB:
        if st.button("🎯 2) Objetivo"):
            st.switch_page("pages/2_Target_Designer.py")
        if st.button("📊 4) Resultados"):
            st.switch_page("pages/4_Results_and_Tradeoffs.py")

# ============== Dos modos: Humano / Geek ==============
t1, t2 = st.tabs(["🧭 Dime cómo (modo humano)", "🧪 Dime cómo (modo geek)"])

with t1:
    st.markdown("""
- **REX-AI entiende tu basura**: señala lo problemático y te sugiere qué proceso conviene.  
- **No tira de la palanca fácil**: evita incineración y rutas “baratas” pero sucias.  
- **Si aparece P03 (Sinter)**: suma **MGS-1** para darle rigidez a la pieza.  
- **Te canta la posta** con un score simple de entender: cuánto cumple el objetivo, cuánto gasta y cuánto tiempo de crew pide.  
- **Aprende contigo**: lo que guardes en *Feedback & Impact* alimenta iteraciones más atinadas.
""")

with t2:
    st.markdown("""
**Arquitectura y lógica clave**
- **Módulos**  
  `io` (I/O de datos), `generator` (recetas y proceso), `process_planner` (filtro por escenario y crew),  
  `safety` (banderas), `explain` (score y desglose), `analytics` (Pareto), `exporters` (JSON/CSV).
- **Scoring multi-objetivo (legible)**  
  Similaridad con target (rigidez/estanqueidad) − penalizaciones (agua/kWh/crew) + bonus por masa “problemática” consumida.  
- **MGS-1 explícito**  
  Si `process_id == "P03"`, se inyecta `regolith_pct` (por defecto 0.2) y se ajustan propiedades (↑rigidez, cuidado con estanqueidad).  
- **Trazabilidad NASA**  
  Cada candidato guarda `source_ids/categories/flags` para auditoría; *no hay cajas negras*.
""")

st.markdown("---")

# ============== Qué vas a encontrar — “plataforma REX-AI” ==============
st.subheader("Qué vas a encontrar (plataforma REX-AI)")

st.markdown("""
<div class="grid3">
  <div class="card">
    <h3>Concept ↗︎</h3>
    <p>Planteá el objetivo del producto con límites operativos (agua, kWh, crew) y escenario.</p>
    <p class="small">Salida: un Target Spec verificable.</p>
  </div>
  <div class="card">
    <h3>Discovery ↗︎</h3>
    <p>Explorá combinaciones de residuos + procesos con heurísticas de compatibilidad y seguridad.</p>
    <p class="small">Salida: candidatos ordenados por score + trazabilidad.</p>
  </div>
  <div class="card">
    <h3>Elevate ↗︎</h3>
    <p>Compará, armá Pareto 3D, exportá un plan ejecutable, corré playbooks y medí impacto.</p>
    <p class="small">Salida: decisión explicada y lista para fabricar.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ============== KPIs rápidos de la sesión (si existen) ==============
st.markdown("### Telemetría de esta sesión")
cA, cB, cC = st.columns(3)
candidates = st.session_state.get("candidates", [])
selected = st.session_state.get("selected")
with cA:
    st.markdown(f'<div class="kpi"><h4>Candidatos generados</h4><div class="v">{len(candidates)}</div></div>', unsafe_allow_html=True)
with cB:
    sel_txt = "Sí" if selected else "No"
    st.markdown(f'<div class="kpi"><h4>Hay selección</h4><div class="v">{sel_txt}</div></div>', unsafe_allow_html=True)
with cC:
    # Target básico
    t = st.session_state.get("target", {})
    name = t.get("name", "—")
    st.markdown(f'<div class="kpi"><h4>Target actual</h4><div class="v">{name}</div></div>', unsafe_allow_html=True)

# ============== Playbook de prueba sugerido (guía al jurado) ==============
st.markdown("---")
st.subheader("Playbook de prueba sugerido para el jurado (5 minutos)")

st.markdown("""
**1)** Abrí **Inventario** y confirmá que hay pouches/foam/EVA/nitrilo/Al (los verás marcados como “problemáticos”).  
**2)** En **Objetivo**, elegí *Container* y fijá límites razonables (p.ej., agua ≤ 0.2 L, energía ≤ 1.2 kWh, crew ≤ 30 min).  
**3)** En **Generador**, pedí 6 opciones → seleccioná la que mejor balancee **score** y **seguridad**.  
**4)** En **Resultados**, mirá el **desglose del score**, el **Sankey** y el **checklist** de fabricación.  
**5)** En **Comparar** y **Pareto**, validá que la decisión esté respaldada por datos y exportá el plan.
""")

# ============== Cierres y navegación ==============
st.markdown("---")
st.caption(
    "Ruta completa: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → "
    "5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator"
)
