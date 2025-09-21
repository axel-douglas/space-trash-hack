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

# ---------- Estilos (seguros) ----------
st.markdown("""
<style>
:root{
  --bd: rgba(130,140,160,.25);
  --glass: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
}
.hero{border:1px solid var(--bd); border-radius:22px; padding:24px;
      background: radial-gradient(1200px 380px at 20% -10%, rgba(80,120,255,.10), transparent);}
.hero h1{margin:0 0 6px 0}
.hero .kicker{opacity:.8; letter-spacing:.2px; font-size:.95rem; margin-bottom:6px}

.grid{display:grid; grid-template-columns: 1fr 1fr; gap:16px}
.grid3{display:grid; grid-template-columns: 1fr 1fr 1fr; gap:16px}
.card{border:1px solid var(--bd); border-radius:18px; padding:18px; background:var(--glass)}
.card h3{margin:.1rem 0 .35rem 0}
.small{font-size:.95rem; opacity:.92}
.micro{font-size:.86rem; opacity:.85}
.hr{height:1px; background:var(--bd); margin:10px 0 14px 0}

.kpi{border:1px solid var(--bd); border-radius:16px; padding:14px; background:var(--glass)}
.kpi h4{margin:0 0 4px 0; font-size:.95rem; opacity:.8}
.kpi .v{font-size:1.45rem; font-weight:800; letter-spacing:.2px}

.step{border-left:4px solid rgba(80,120,255,.5); padding-left:12px; margin-bottom:8px}
.pill{display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.78rem;
      border:1px solid var(--bd); margin-right:6px}
.pill.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.pill.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.pill.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}
blockquote{margin:0; padding:8px 12px; border-left:3px solid rgba(130,140,160,.45); background:rgba(255,255,255,.03); border-radius:8px}
code.inline{background:rgba(130,140,160,.15); padding:1px 6px; border-radius:6px}
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

# ---------- HERO: Storytelling + promesa ----------
st.markdown("""
<div class="hero">
  <div class="kicker">CONOCE A <b>REX-AI</b>, LA TECNOLOGÍA QUE LO HACE POSIBLE</div>
  <h1>Basura espacial → Piezas útiles. En serio.</h1>
  <div class="small">
    Bolsas multicapa que nadie quiere reciclar, espumas ZOTEK F30 tercas, textiles EVA/CTB y guantes de nitrilo. 
    <b>REX-AI</b> las mira, entiende el objetivo (p. ej. un <i>Container</i> con buena rigidez y estanqueidad), 
    y propone <b>recetas+procesos</b> que minimizan <b>agua</b>, <b>energía</b> y <b>tiempo de crew</b>. 
    Si el proceso es <b>P03</b>, inyecta <b>MGS-1</b> (regolito de Jezero) como carga mineral. 
  </div>
  <div style="margin-top:10px">
    <span class="pill info">Optimización multi-objetivo</span>
    <span class="pill info">Trazabilidad de residuos (IDs, flags)</span>
    <span class="pill ok">Sin incineración / PFAS a raya</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ---------- Estado del sistema ----------
data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
proc_ok = (ROOT / "data" / "process_catalog.csv").exists()
tgt_ok  = (ROOT / "data" / "targets_presets.json").exists()
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="kpi"><h4>Inventario</h4><div class="v">{}</div><div class="micro">waste_inventory_sample.csv</div></div>'.format("✅" if data_ok else "❌"), unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi"><h4>Procesos</h4><div class="v">{}</div><div class="micro">process_catalog.csv</div></div>'.format("✅" if proc_ok else "❌"), unsafe_allow_html=True)
with c3:
    st.markdown('<div class="kpi"><h4>Targets</h4><div class="v">{}</div><div class="micro">targets_presets.json</div></div>'.format("✅" if tgt_ok else "❌"), unsafe_allow_html=True)
with c4:
    st.markdown('<div class="kpi"><h4>Modo</h4><div class="v">Demo ligera</div><div class="micro">Explicable por humanos</div></div>', unsafe_allow_html=True)

st.caption("Archivos requeridos en `/data`: `waste_inventory_sample.csv`, `process_catalog.csv`, `targets_presets.json`.")

st.markdown("---")

# ---------- “Solo pregúntale a REX-AI” (criollo vs geek) ----------
L, R = st.columns(2)
with L:
    st.subheader("Solo pregúntale a REX-AI (versión criolla)")
    st.markdown("""
<div class="step"><b>1) Contame qué querés fabricar</b>: elegís el objetivo y límites (agua, kWh, minutos de crew).</div>
<div class="step"><b>2) REX-AI mira tu inventario</b>: detecta “problemáticos” (pouches PE-PET-Al, espumas ZOTEK F30, EVA/CTB, nitrilo, etc.).</div>
<div class="step"><b>3) Propone recetas + proceso</b>: mezcla materiales y sugiere P02/P03/P04; si es P03, agrega <b>MGS-1</b>.</div>
<div class="step"><b>4) Te muestra trade-offs</b>: score explicable, Sankey, recursos, seguridad y checklist de fabricación.</div>
""", unsafe_allow_html=True)
    st.markdown('<blockquote>Como reconstruir una pieza con <i>Legos™</i>, pero los ladrillos son residuos + regolito.</blockquote>', unsafe_allow_html=True)

with R:
    st.subheader("Dime cómo, pero a la manera geek")
    st.markdown("""
- **Pipeline**: `io` → `generator` → `process_planner` → `safety` → `explain/analytics` → `exporters`.
- **Scoring**: compatibilidad con target + penalizaciones por recursos + <code class="inline">bonus</code> por “consumir problemáticos”.
- **P03 (Sinter + MGS-1)**: inyecta 10–30% de MGS-1 → ↑rigidez, posible ↓estanq.; parámetros de proceso del catálogo.
- **Trazabilidad**: cada candidato guarda `source_ids/categories/flags` para auditoría NASA.
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- Flujo (4 pasos) + CTAs ----------
st.subheader("Tu plan de vuelo (4 pasos)")
gL, gR = st.columns([2, 1], gap="large")
with gL:
    st.markdown("""
<div class="grid">
  <div class="card">
    <h3>1) Inventario</h3>
    <div class="small">Editá el laboratorio: masas, flags y categorías. Resaltamos problemáticos que el sistema prioriza.</div>
    <div class="hr"></div>
    <div class="micro">Tip: duplicar filas simula lotes nuevos. Guardar antes de seguir.</div>
  </div>
  <div class="card">
    <h3>2) Objetivo</h3>
    <div class="small">Ajustá el target (Container/Utensil/Interior/Tool), pesos de rigidez/estanqueidad y límites (L, kWh, min).</div>
    <div class="hr"></div>
    <div class="micro">El escenario (Residence/Celebrations/Discoveries) filtra procesos y penaliza tiempo de crew si así lo pedís.</div>
  </div>
  <div class="card">
    <h3>3) Generador</h3>
    <div class="small">Crea de 3 a 12 candidatos. Si hay espumas o pouches, preferirá P02/P03; CTB/EVA va hacia P04.</div>
    <div class="hr"></div>
    <div class="micro">En P03, entra MGS-1: verás cómo sube rigidez y cambian recursos.</div>
  </div>
  <div class="card">
    <h3>4) Resultados</h3>
    <div class="small">Score desglosado, Sankey residuos→proceso→producto, checklist y badges de seguridad.</div>
    <div class="hr"></div>
    <div class="micro">Listo para comparar, exportar y planificar ejecución.</div>
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

# ---------- “Aprende con feedback” ----------
st.subheader("Feedback: la clave del aprendizaje humano-en-el-loop")
F1, F2, F3 = st.columns(3)
with F1:
    st.markdown("""
<div class="card">
  <h3>Probar y anotar</h3>
  <div class="small">Después de fabricar, registra rigidez percibida, facilidad y problemas (bordes, olor, slip).</div>
  <div class="hr"></div>
  <div class="micro">Página <b>8) Feedback & Impact</b> consolida el historial y suma señales para iterar mejores pesos.</div>
</div>
""", unsafe_allow_html=True)
with F2:
    st.markdown("""
<div class="card">
  <h3>Métricas que importan</h3>
  <div class="small">kWh, L de agua, minutos de crew y kg valorizados. Con eso priorizás qué proceso escalar.</div>
  <div class="hr"></div>
  <div class="micro">También exportás JSON/CSV del plan ganador en <b>6) Pareto & Export</b>.</div>
</div>
""", unsafe_allow_html=True)
with F3:
    st.markdown("""
<div class="card">
  <h3>Capacidad de línea</h3>
  <div class="small">Simulá producción: lotes/turno, kg/lote y recursos. Visualizá el impacto por “sol” en <b>9) Capacity</b>.</div>
  <div class="hr"></div>
  <div class="micro">Ideal para planificar con restricciones reales del hábitat.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- Nota técnica final (por qué esto impacta) ----------
st.subheader("¿Por qué impacta a nivel ingeniería?")
st.markdown("""
- **Velocidad**: convierte inventarios caóticos en candidatos viables con trazabilidad y explicabilidad.
- **Rigor**: alinea objetivos (función/recursos/seguridad) con catálogos de proceso y <b>MGS-1</b> cuando aplica.
- **Decisión**: comparás, explicás trade-offs, exportás y ejecutás. El loop con feedback cierra la mejora continua.
""")

st.caption(
    "Ruta completa: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → "
    "5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator"
)
