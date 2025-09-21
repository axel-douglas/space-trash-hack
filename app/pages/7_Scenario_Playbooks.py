# app/pages/7_Scenario_Playbooks.py
# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import pandas as pd

from app.modules.scenarios import PLAYBOOKS  # dict: {scenario: Playbook(name, summary, steps=[...])}

# ⚠️ Debe ser la primera llamada
st.set_page_config(page_title="Scenario Playbooks", page_icon="📚", layout="wide")

# ======== Estado compartido ========
target      = st.session_state.get("target", None)
state_sel   = st.session_state.get("selected", None)
candidato   = state_sel["data"] if state_sel else None
props       = candidato["props"] if candidato else None

# ======== Estilos SpaceX/NASA-like ========
st.markdown("""
<style>
:root{ --bd: rgba(140,140,160,.28); --ink: #0f172a; --muted: rgba(15,23,42,.65); }
.hero{border:1px solid var(--bd); border-radius:16px; padding:18px;
      background: radial-gradient(900px 260px at 20% -10%, rgba(80,120,255,.10), transparent);}
.pill{display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.78rem;
      border:1px solid var(--bd); margin-right:6px}
.pill.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.pill.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.pill.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}
.block{border:1px solid var(--bd); border-radius:16px; padding:16px;}
.step{border:1px dashed var(--bd); border-radius:14px; padding:14px; margin-bottom:12px;}
.step h4{margin:0 0 6px 0;}
.kpi{border:1px solid var(--bd); border-radius:14px; padding:14px; margin-bottom:10px;}
.kpi h3{margin:0 0 6px 0; font-size:.95rem; opacity:.8}
.kpi .v{font-size:1.6rem; font-weight:800; letter-spacing:.2px}
.legend{font-size:.9rem; opacity:.8}
.small{font-size:.92rem; opacity:.9}
.mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;}
</style>
""", unsafe_allow_html=True)

# ======== HERO ========
st.markdown("""
<div class="hero">
  <h1 style="margin:0 0 6px 0">📚 Scenario Playbooks</h1>
  <div class="small">
    Procedimientos guiados para ejecutar la receta seleccionada en contextos de misión
    (<b>Residence Renovations</b>, <b>Cosmic Celebrations</b>, <b>Daring Discoveries</b>).
    Cada playbook es una lista de pasos accionables, con material y recursos provenientes de tus datos reales.
  </div>
  <div class="legend" style="margin-top:8px">
    <span class="pill info">1) Elegí el playbook</span>
    <span class="pill info">2) Revisá pasos + recursos</span>
    <span class="pill ok">3) Ejecutá & registra feedback</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ======== Selector de Playbook ========
st.markdown("### 🎯 Escenario operativo")

if not target:
    st.info("Definí primero el escenario en **2) Target Designer**.")
    st.stop()

scenario_default = target.get("scenario", next(iter(PLAYBOOKS.keys())))
scenarios = list(PLAYBOOKS.keys())

col_sel, col_help = st.columns([1.5, 1.0])
with col_sel:
    query = st.text_input("Buscar escenario", value=scenario_default)
    # Simple “filtro por contiene”
    matches = [s for s in scenarios if query.lower() in s.lower()] or scenarios
    scenario = st.selectbox("Elegí un playbook", matches, index=0)

with col_help:
    st.markdown("""
<div class="block">
<b>¿Qué es un playbook?</b><br/>
Un <i>procedimiento de laboratorio</i> curado para un contexto concreto. Trae pasos,
checklist y pistas de seguridad. Ideal para entrenar aprendices y estandarizar corridas.
</div>
""", unsafe_allow_html=True)

pb = PLAYBOOKS.get(scenario)
if not pb:
    st.warning("No encontré el playbook seleccionado.")
    st.stop()

# ======== Brief del Playbook + Estado de datos ========
st.markdown("### 🧪 Brief del escenario")
bL, bR = st.columns([1.3, 1.0])

with bL:
    st.subheader(pb.name)
    st.markdown(pb.summary)

with bR:
    st.markdown("**Estado de datos**")
    st.markdown(
        f"""
- Target: **{target.get('name','-')}**
- Límites → Agua: **{target.get('max_water_l','-')} L** · Energía: **{target.get('max_energy_kwh','-')} kWh** · Crew: **{target.get('max_crew_min','-')} min**
- Candidato seleccionado: **{('Opción con ' + candidato['process_id'] + ' ' + candidato['process_name']) if candidato else '— (seleccioná en 3/6)'}**
        """
    )

# ======== Herramientas rápidas (conecta a datos reales si hay candidato) ========
st.markdown("### 🧰 Herramientas rápidas")

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown('<div class="kpi"><h3>Masa final</h3><div class="v">{}</div></div>'.format(
        f"{props.mass_final_kg:.2f} kg" if props else "—"
    ), unsafe_allow_html=True)
with t2:
    st.markdown('<div class="kpi"><h3>Energía por corrida</h3><div class="v">{}</div></div>'.format(
        f"{props.energy_kwh:.2f} kWh" if props else "—"
    ), unsafe_allow_html=True)
with t3:
    st.markdown('<div class="kpi"><h3>Agua por corrida</h3><div class="v">{}</div></div>'.format(
        f"{props.water_l:.2f} L" if props else "—"
    ), unsafe_allow_html=True)
with t4:
    st.markdown('<div class="kpi"><h3>Crew por corrida</h3><div class="v">{}</div></div>'.format(
        f"{props.crew_min:.0f} min" if props else "—"
    ), unsafe_allow_html=True)

st.caption("Estos valores vienen del **candidato activo** (seleccionalo en 3) Generador o en 6) Pareto).")

# ======== Paso a paso (tarjetas) ========
st.markdown("### 📋 Pasos del playbook")
for i, step in enumerate(pb.steps, start=1):
    c1, c2 = st.columns([0.08, 1.0], vertical_alignment="center")
    with c1:
        st.markdown(f"<div class='pill info'>Paso {i}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='step'><h4>{step.title}</h4><div class='small'>{step.detail}</div></div>", unsafe_allow_html=True)

# ======== Mini-estimador de turnos (para aprendices) ========
st.markdown("### ⏱️ Estimador de turnos (didáctico)")
st.caption("Para dimensionar rápidamente una operación. No reemplaza al simulador de capacidad (página 9).")

eL, eR = st.columns([1.2, 1.0])
with eL:
    batches = st.number_input("Corridas planificadas", 1, 200, 6, 1)
    if props:
        total_crew = batches * float(props.crew_min)
        total_kwh  = batches * float(props.energy_kwh)
        total_w    = batches * float(props.water_l)
        st.markdown(f"""
- **Crew total:** **{total_crew:.0f} min**  
- **Energía total:** **{total_kwh:.2f} kWh**  
- **Agua total:** **{total_w:.2f} L**
""")
    else:
        st.info("Seleccioná un candidato para ver números reales.")

with eR:
    st.markdown("""
<div class="block">
<b>En criollo:</b> si una corrida te consume 25 min de tripulación, y querés 6 corridas, vas a necesitar ~150 min.
Compará contra el límite del target y planificá los turnos.
</div>
""", unsafe_allow_html=True)

# ======== Checklist de ejecución (imprimible) ========
st.markdown("### ✅ Checklist de ejecución")
cl = st.text_area(
    "Lista imprimible (editable)",
    value=(
        f"- Verificar disponibilidad de materiales (incluyendo **MGS-1** si el proceso es sinterizado)\n"
        f"- Preparar equipo del proceso ({candidato['process_id']} {candidato['process_name']})\n"
        f"- Registrar lote, hora y operador\n"
        f"- Post-procesado y control visual del producto\n"
        f"- Cargar feedback en **8) Feedback & Impact**"
    ) if candidato else
    "- Seleccionar candidato en 3) Generador o 6) Pareto\n- Preparar equipo del proceso\n- Registrar lote, hora y operador\n- Cargar feedback en 8)"
)
st.download_button("⬇️ Descargar checklist (.txt)", data=cl.encode("utf-8"),
                   file_name="scenario_checklist.txt", mime="text/plain")

# ======== Glosario & valor (aprendices) ========
st.markdown("### ℹ️ Glosario rápido (para aprendices)")
gL, gR = st.columns(2)
with gL:
    st.markdown("""
- **Playbook:** guía paso a paso para que dos personas ejecuten igual en días distintos.
- **Crew (min):** minutos de tiempo humano necesario. En misiones, es oro.
- **MGS-1:** regolito simulado. Mejora rigidez cuando sinterizamos.
- **Frontera de Pareto:** conjunto de opciones que no se pueden mejorar en un eje sin empeorar otro.
""")
with gR:
    st.markdown("""
- **Target:** lo que querés fabricar + límites (energía/agua/crew).
- **Score:** cuán bien se ajusta al target, penalizando recursos.
- **Trazabilidad:** guardamos IDs y flags de los residuos para auditoría.
- **Feedback:** lo que nos decís tras probar; ajusta futuras decisiones.
""")

st.markdown("---")
st.caption("Tip: si un paso te resulta lento, registralo en **8) Feedback & Impact** — el sistema aprende de esos cuellos de botella.")
