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

# ---------- Estilos SpaceX-like ----------
st.markdown("""
<style>
:root{ --bd: rgba(130,140,160,.28);}
.hero{
  border:1px solid var(--bd);
  border-radius:22px;
  padding:28px;
  margin-bottom:20px;
  background: radial-gradient(1200px 400px at 20% -20%, rgba(80,120,255,.12), transparent);
}
.hero h1{margin:0 0 6px 0; font-size:2rem}
.hero .tagline{font-size:1.1rem; opacity:.9; margin-bottom:12px}
.grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap:18px; margin:20px 0;}
.card{border:1px solid var(--bd); border-radius:16px; padding:18px; background:rgba(255,255,255,.02);}
.card h3{margin:.1rem 0 .25rem 0;}
.kpi{border:1px solid var(--bd); border-radius:14px; padding:14px; text-align:center;}
.kpi h3{margin:0 0 6px 0; font-size:.95rem; opacity:.8}
.kpi .v{font-size:1.6rem; font-weight:800; letter-spacing:.2px}
.pill{display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.78rem;
      border:1px solid var(--bd); margin-right:6px}
.pill.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.pill.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.pill.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}
.small{font-size:.92rem; opacity:.9}
.section{margin-top:30px; margin-bottom:18px;}
.section h2{margin-bottom:6px;}
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

# ---------- Hero narrative ----------
st.markdown("""
<div class="hero">
  <h1>Conoce a REX-AI</h1>
  <div class="tagline">
    Nuestra inteligencia artificial que convierte basura espacial en recursos para la misión.
  </div>
  <div>
    <span class="pill info">Reciclaje inteligente</span>
    <span class="pill ok">Optimización multi-objetivo</span>
    <span class="pill warn">Uso seguro de MGS-1</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Qué hace (versión narrativa + técnica) ----------
st.markdown("### ¿Qué hace REX-AI?")
st.markdown("""
**En criollo**:  
La nave genera **basura inorgánica** — pouches multicapa, espumas, textiles, guantes de nitrilo, piezas de aluminio.  
Lo que hace **REX-AI** es agarrar ese lío y decirte: *“Con esto podés armar un contenedor, un utensilio, una herramienta…”*  
Y no lo hace a ojo: combina datos reales de procesos, costos de crew, y hasta regula la mezcla con **regolito MGS-1** del cráter Jezero.

**Para ingenieros**:  
- Analiza inventarios reales (NASA Non-Metabolic Waste Categories).  
- Selecciona procesos viables (Shredder, Heat Lamination, Sinter con MGS-1, Reuso CTB).  
- Calcula propiedades predichas (rigidez, estanqueidad, masa final) con modelos ligeros.  
- Puntúa cada candidato con un **score multi-objetivo**: compatibilidad con target + penalización por recursos + bonus por “consumir problemáticos”.  
- Devuelve recetas con trazabilidad: cada ID, categoría y flag queda registrado.
""")

# ---------- Estado del sistema ----------
st.markdown("### Estado actual")
c1, c2, c3, c4 = st.columns(4)
data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
proc_ok = (ROOT / "data" / "process_catalog.csv").exists()
tgt_ok  = (ROOT / "data" / "targets_presets.json").exists()
with c1: st.markdown(f'<div class="kpi"><h3>Inventario</h3><div class="v">{"✅" if data_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi"><h3>Procesos</h3><div class="v">{"✅" if proc_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi"><h3>Targets</h3><div class="v">{"✅" if tgt_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="kpi"><h3>Modo</h3><div class="v">Demo</div></div>', unsafe_allow_html=True)

# ---------- Flujo ----------
st.markdown("### Flujo de misión")
st.markdown("""
<div class="grid">
  <div class="card"><h3>1) Inventario</h3><div class="small">Subí y editá los residuos. Detectamos los **problemáticos** y los marcamos.</div></div>
  <div class="card"><h3>2) Objetivo</h3><div class="small">Definí qué querés fabricar (ej: Container, Tool) y límites de recursos.</div></div>
  <div class="card"><h3>3) Generador</h3><div class="small">REX-AI arma recetas priorizando consumir problemáticos y elige procesos coherentes.</div></div>
  <div class="card"><h3>4) Resultados</h3><div class="small">Ves métricas, trade-offs, Sankey de flujo y checklist de fabricación.</div></div>
</div>
""", unsafe_allow_html=True)

# ---------- Explicación SpaceX-style ----------
st.markdown("### ¿Por qué importa?")
st.markdown("""
- **Nivel crew**: cada minuto de astronauta ahorrado vale millones. REX-AI aprende a minimizarlo.  
- **Nivel recursos**: el agua y energía en Marte son como oro; penalizamos cualquier exceso.  
- **Nivel materiales**: en lugar de mandar toneladas desde la Tierra, reusamos lo que ya está en el hábitat.  
- **Nivel misión**: más seguridad (sin incineración, sin PFAS), más resiliencia (aprovechar regolito local).  

**Ejemplo simple**:  
Si mezclamos espuma ZOTEK F30 con regolito MGS-1 → se logra un panel rígido que puede reforzar interiores.  
Si usamos pouches multicapa en laminar + presión → conseguimos láminas estancas para compartimentos.  
""")

# ---------- Ruta completa ----------
st.markdown("---")
st.caption(
    "Ruta: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → "
    "5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator"
)
