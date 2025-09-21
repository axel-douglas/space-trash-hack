# app/Home.py
# ================== Path guard (siempre antes de importar módulos propios) ==================
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ===========================================================================================

# ⚠️ PRIMER comando de Streamlit:
import streamlit as st
st.set_page_config(page_title="REX-AI Mars — Mission Hub", page_icon="🛰️", layout="wide")

from pathlib import Path

# ================== Estilos (SpaceX-like, sobrios y legibles) ==================
st.markdown("""
<style>
:root{
  --bd: rgba(130,140,160,.28);
  --card: rgba(255,255,255,.02);
  --ink: rgba(230,238,255,.9);
}
html, body, [data-testid="stAppViewContainer"] {background: radial-gradient(1400px 600px at 5% -20%, rgba(80,120,255,.10), transparent);}
.hero{
  border:1px solid var(--bd); border-radius:28px; padding:28px; margin-bottom:22px;
  background: linear-gradient(135deg, rgba(30,36,48,.75), rgba(30,36,48,.55));
}
.hero h1{margin:0 0 8px 0; font-size:2.1rem; letter-spacing:.2px}
.hero .tag{display:inline-block; margin-top:10px}
.pill{display:inline-block; padding:5px 12px; border-radius:999px; font-weight:700; font-size:.78rem;
      border:1px solid var(--bd); margin-right:8px; background:var(--card)}
.pill.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.pill.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.pill.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}
.grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap:18px; margin:16px 0 6px 0;}
.card{border:1px solid var(--bd); border-radius:16px; padding:18px; background:var(--card);}
.card h3{margin:.1rem 0 .35rem 0;}
.card p{margin:0; opacity:.95}
.kpis{display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap:12px; margin-top:10px}
.kpi{border:1px solid var(--bd); border-radius:14px; padding:14px; text-align:center; background:var(--card)}
.kpi h4{margin:0 0 6px 0; font-size:.95rem; opacity:.8}
.kpi .v{font-size:1.5rem; font-weight:800; letter-spacing:.2px}
.section{margin-top:26px; margin-bottom:4px;}
.small{font-size:.92rem; opacity:.92}
.cta{display:flex; gap:8px; flex-wrap:wrap; margin-top:10px}
.cta a{padding:10px 14px; border-radius:12px; border:1px solid var(--bd); text-decoration:none; font-weight:700; font-size:.9rem}
.cta a.primary{background:#1b66ff1a; border-color:#6e9bff66}
.cta a:hover{background:rgba(255,255,255,.06)}
.callout{border:1px dashed var(--bd); border-radius:16px; padding:14px; margin-top:6px; background:rgba(255,255,255,.03)}
</style>
""", unsafe_allow_html=True)

# ================== Encabezado ==================
logo_svg = ROOT / "app" / "static" / "logo_rexai.svg"
c_logo, c_title = st.columns([0.13, 0.87])
with c_logo:
    if logo_svg.exists():
        st.image(str(logo_svg), use_column_width=True)
with c_title:
    st.markdown("### REX-AI Mars — Mission Hub")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

# ================== Hero ==================
st.markdown(f"""
<div class="hero">
  <h1>Convierte basura espacial en recursos de misión</h1>
  <div class="small">REX-AI prioriza residuos problemáticos (pouches multicapa, espumas ZOTEK, textiles EVA/CTB, guantes de nitrilo, etc.), sugiere procesos viables y
  optimiza agua, energía y tiempo de tripulación. Cuando corresponde, integra **MGS-1** (regolito de Jezero) como carga mineral.</div>
  <div class="tag">
    <span class="pill info">Reciclaje inteligente</span>
    <span class="pill ok">Optimización multi-objetivo</span>
    <span class="pill warn">Guardrails de seguridad</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ================== Toggle de narrativa: Ingeniero / Criollo ==================
mode = st.toggle("🧪 Modo Ingeniero (desactiva para Modo Criollo)", value=True)

# ================== Estado del sistema ==================
st.markdown("#### Estado del sistema")
data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
proc_ok = (ROOT / "data" / "process_catalog.csv").exists()
tgt_ok  = (ROOT / "data" / "targets_presets.json").exists()
colA, colB = st.columns([1.4, 1], vertical_alignment="center")

with colA:
    st.markdown("""
<div class="kpis">
  <div class="kpi"><h4>Inventario</h4><div class="v">""" + ("✅" if data_ok else "❌") + """</div></div>
  <div class="kpi"><h4>Procesos</h4><div class="v">""" + ("✅" if proc_ok else "❌") + """</div></div>
  <div class="kpi"><h4>Targets</h4><div class="v">""" + ("✅" if tgt_ok else "❌") + """</div></div>
  <div class="kpi"><h4>Modo</h4><div class="v">Demo</div></div>
</div>
""", unsafe_allow_html=True)

with colB:
    st.markdown("""
<div class="callout small">
<b>Checklist de datos</b><br>
• <code>data/waste_inventory_sample.csv</code><br>
• <code>data/process_catalog.csv</code><br>
• <code>data/targets_presets.json</code><br>
</div>
""", unsafe_allow_html=True)

# ================== Qué hace (según modo) ==================
st.markdown("#### ¿Qué hace REX-AI?")

if mode:
    # Modo Ingeniero
    st.markdown("""
- **Ingesta robusta** de inventario (normalización de columnas, detección de <em>problemáticos</em>).
- **Heurísticas de selección de proceso** (P02 laminar multicapa, P03 sinter + MGS-1, P04 reuso CTB).
- **Propiedades predichas** (rigidez, estanqueidad, masa final) y costos de recursos (kWh, L agua, crew-min).
- **Score multi-objetivo** transparente: compatibilidad con target + penalizaciones por recursos + bonus por “consumir problemáticos”.
- **Trazabilidad NASA**: cada candidato guarda <code>source_ids</code>, <code>source_categories</code>, <code>source_flags</code> y <code>regolith_pct</code>.
""")
else:
    # Modo Criollo
    st.markdown("""
- **Ordena la alacena** de basura espacial y marca lo que más molesta (⚠️ problemáticos).
- **Te sugiere la receta** y la forma de cocinarla (proceso), sin gastar de más agua ni energía.
- **Te dice cómo va a salir** (qué tan firme, qué tan estanco) y cuánto tiempo de crew te lleva.
- **Te muestra el porqué** con puntajes fáciles y una ruta clara para fabricar.
- **Te deja todo anotado** (qué usaste, de dónde salió, si metiste regolito MGS-1).
""")

# ================== Flujo de misión (con navegación) ==================
st.markdown("#### Flujo de misión (paso a paso)")
st.markdown("""
<div class="grid">
  <div class="card"><h3>1) Inventario</h3><p>Subí/edita residuos, detectamos <b>problemáticos</b> y masa/volumen.</p></div>
  <div class="card"><h3>2) Objetivo</h3><p>Elegí el producto (Container, Tool...) y límites de agua/energía/crew.</p></div>
  <div class="card"><h3>3) Generador</h3><p>REX-AI arma recetas que <i>consumen</i> problemáticos y elige procesos coherentes.</p></div>
  <div class="card"><h3>4) Resultados</h3><p>Trade-offs, Sankey, checklist y métricas clave para fabricar y evaluar.</p></div>
</div>
""", unsafe_allow_html=True)

colFlowA, colFlowB = st.columns([1.2, 1])
with colFlowA:
    st.markdown("###### Ir directo a…")
    st.page_link("pages/1_Inventory_Builder.py", label="🧱 1) Inventory Builder", icon="🧱")
    st.page_link("pages/2_Target_Designer.py", label="🎯 2) Target Designer", icon="🎯")
    st.page_link("pages/3_Generator.py", label="⚙️ 3) Generator", icon="⚙️")
    st.page_link("pages/4_Results_and_Tradeoffs.py", label="📊 4) Results & Trade-offs", icon="📊")
with colFlowB:
    st.markdown("###### O explorar herramientas avanzadas")
    st.page_link("pages/5_Compare_and_Explain.py", label="🧪 5) Compare & Explain", icon="🧪")
    st.page_link("pages/6_Pareto_and_Export.py", label="📤 6) Pareto & Export", icon="📤")
    st.page_link("pages/7_Scenario_Playbooks.py", label="📚 7) Scenario Playbooks", icon="📚")
    st.page_link("pages/8_Feedback_and_Impact.py", label="📝 8) Feedback & Impact", icon="📝")
    st.page_link("pages/9_Capacity_Simulator.py", label="🧮 9) Capacity Simulator", icon="🧮")

# ================== Por qué importa (con ejemplos rápidos) ==================
st.markdown("#### ¿Por qué importa?")
st.markdown("""
- **Crew-time**: cada minuto ahorrado vale millones (logística/seguridad).  
- **Recursos**: agua y energía son oro en Marte; el score penaliza excesos por diseño.  
- **Materiales**: convertir “lo que sobra” en “lo que falta” reduce dependencia de envíos desde la Tierra.  
- **Seguridad**: sin incineración, evitando PFAS/microplásticos, con guardrails de proceso visibles.

**Ejemplos express**  
• Pouches PE–PET–Al + Laminar (P02) → lámina con buena estanqueidad para compartimentos.  
• ZOTEK F30 (espuma) + Sinter MGS-1 (P03) → panel rígido/estructural ligero.  
• EVA/CTB + Reuse Kit (P04) → reconfig de hardware y particiones internas.
""")

# ================== Cierre: ruta de exploración ==================
st.markdown("---")
st.caption("Ruta sugerida: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → 5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator")

# ================== Extra: Mensaje de claridad técnica (confianza para el jurado) ==================
with st.expander("🔎 Transparencia técnica (para el jurado)"):
    st.markdown("""
**Qué está pasando bajo el capó**  
- Los módulos de datos normalizan columnas y etiquetan <em>problemáticos</em> con reglas claras.  
- El generador usa heurísticas determinísticas para elegir procesos y calcular recursos/props (demo).  
- Los candidatos guardan trazabilidad completa (IDs/categorías/flags), y cuando el proceso es P03 suman <b>regolito MGS-1</b>.  
- Los gráficos y tablas de todas las páginas **leen el estado real** (inventario, target, candidatos, seleccionado).  
- En un siguiente upgrade, se pueden reemplazar heurísticas por **surrogates ML** y optimizadores con incertidumbre, sin romper UX.
""")
