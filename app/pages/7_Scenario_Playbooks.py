# app/pages/7_Scenario_Playbooks.py
import _bootstrap  # noqa: F401

import streamlit as st

from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.scenarios import PLAYBOOKS  # dict: {scenario: Playbook(name, summary, steps=[...])}
from app.modules.ui_blocks import load_theme

# ⚠️ Debe ser la primera llamada
st.set_page_config(page_title="Scenario Playbooks", page_icon="📚", layout="wide")

set_active_step("playbooks")

load_theme()

render_breadcrumbs("playbooks")

FEATURED_PLAYBOOKS = ("Residence Renovations", "Daring Discoveries")
GENERATOR_FILTER_PRESETS: dict[str, dict[str, bool]] = {
    "Residence Renovations": {
        "showroom_only_safe": True,
        "showroom_limit_energy": True,
        "showroom_limit_water": True,
        "showroom_limit_crew": True,
    },
    "Cosmic Celebrations": {
        "showroom_only_safe": True,
        "showroom_limit_energy": False,
        "showroom_limit_water": True,
        "showroom_limit_crew": False,
    },
    "Daring Discoveries": {
        "showroom_only_safe": False,
        "showroom_limit_energy": True,
        "showroom_limit_water": False,
        "showroom_limit_crew": True,
    },
}

# ======== Estado compartido ========
target      = st.session_state.get("target", None)
state_sel   = st.session_state.get("selected", None)
candidato   = state_sel["data"] if state_sel else None
props       = candidato["props"] if candidato else None

# ======== Selector de Playbook ========
if not target:
    st.info("Definí primero el escenario en **2) Target Designer**.")
    st.stop()

scenario_default = target.get("scenario", next(iter(PLAYBOOKS.keys())))
scenarios = list(PLAYBOOKS.keys())
ordered_scenarios = [s for s in FEATURED_PLAYBOOKS if s in scenarios]
ordered_scenarios.extend([s for s in scenarios if s not in ordered_scenarios])

if scenario_default not in ordered_scenarios:
    scenario_default = ordered_scenarios[0]

featured_indices = [
    idx + 1 for idx, name in enumerate(ordered_scenarios) if name in FEATURED_PLAYBOOKS
]

# ======== Estilos SpaceX/NASA-like ========
radio_feature_css = "".join(
    f"""
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"] > div:nth-child({index}) label {{
        border-color:rgba(250,204,21,0.65);
        box-shadow:0 16px 28px rgba(15,23,42,0.55), inset 0 0 0 1px rgba(250,204,21,0.35);
        background:linear-gradient(135deg, rgba(250,204,21,0.22), rgba(56,189,248,0.08));
        color:#f8fafc;
    }}
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"] > div:nth-child({index}) label:hover {{
        border-color:rgba(250,204,21,0.85);
        box-shadow:0 20px 34px rgba(15,23,42,0.65), inset 0 0 0 1px rgba(250,204,21,0.45);
    }}
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"] > div:nth-child({index}) label:has(input:checked) {{
        border-color:rgba(253,224,71,0.95);
        box-shadow:0 24px 38px rgba(15,23,42,0.7), inset 0 0 0 1px rgba(253,224,71,0.65);
        color:#0f172a;
        background:linear-gradient(135deg, rgba(253,224,71,0.65), rgba(56,189,248,0.32));
    }}
    """
    for index in featured_indices
)

st.markdown(
    f"""
    <style>
    .hero {{border-radius:16px; background: radial-gradient(900px 260px at 20% -10%, rgba(80,120,255,.10), transparent);}}
    .step{{border:1px dashed var(--bd); border-radius:14px; padding:14px; margin-bottom:12px;}}
    .step h4{{margin:0 0 6px 0;}}
    .mono{{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", monospace;}}
    .why-panel{{border:1px solid var(--bd); border-radius:16px; padding:16px; background:linear-gradient(135deg, rgba(59,130,246,0.12), rgba(13,148,136,0.10)); box-shadow:0 16px 28px rgba(15,23,42,0.22);}}
    .why-panel h4{{margin:0 0 8px 0; font-size:1.05rem;}}
    .why-panel p{{margin:0 0 10px 0; font-size:0.92rem; line-height:1.45;}}
    .why-panel ul{{margin:0; padding-left:18px; font-size:0.88rem; line-height:1.5;}}
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"]{{display:flex; gap:0.6rem; flex-wrap:wrap;}}
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"] label{{cursor:pointer; display:flex; align-items:center; gap:0.5rem; padding:0.55rem 1.35rem; border-radius:14px; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.62); color:#e2e8f0; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; box-shadow:inset 0 0 0 1px rgba(148,163,184,0.18); transition:all .2s ease; position:relative;}}
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"] label:hover{{border-color:rgba(148,163,184,0.75); box-shadow:0 12px 24px rgba(15,23,42,0.48), inset 0 0 0 1px rgba(148,163,184,0.32);}}
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"] label::after{{content:\"\"; position:absolute; inset:0; border-radius:14px; opacity:0; transition:opacity .2s ease; background:radial-gradient(circle at 30% 30%, rgba(148,197,255,0.32), transparent 70%); mix-blend-mode:screen;}}
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"] label:has(input:checked)::after{{opacity:1;}}
    div[data-testid=\"stRadio\"] > div[role=\"radiogroup\"] label:has(input:checked){{border:1px solid rgba(96,165,250,0.85); box-shadow:0 18px 32px rgba(15,23,42,0.62), inset 0 0 0 1px rgba(96,165,250,0.55); background:linear-gradient(135deg, rgba(59,130,246,0.45), rgba(56,189,248,0.25)); color:#0ea5e9;}}
    {radio_feature_css}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======== HERO ========
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.markdown("### 🎯 Escenario operativo")

col_sel, col_help = st.columns([1.5, 1.0])
with col_sel:
    st.caption("Seleccioná un playbook destacado o cambiá a otro escenario de misión.")
    scenario = st.radio(
        "Playbooks disponibles",
        options=ordered_scenarios,
        index=ordered_scenarios.index(scenario_default),
        key="playbook_selector",
        horizontal=True,
    )

pb = PLAYBOOKS.get(scenario)

with col_help:
    if pb:
        highlights = "".join(
            f"<li><strong>{step.title}</strong>: {step.detail}</li>"
            for step in pb.steps[:3]
        ) or ""
        st.markdown(
            f"""
<div class="block why-panel">
  <h4>🏆 Por qué gana</h4>
  <p>{pb.summary}</p>
  <ul>{highlights}</ul>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("""
<div class="block">
<b>¿Qué es un playbook?</b><br/>
Un <i>procedimiento de laboratorio</i> curado para un contexto concreto. Trae pasos,
checklist y pistas de seguridad. Ideal para entrenar aprendices y estandarizar corridas.
</div>
""", unsafe_allow_html=True)
if not pb:
    st.warning("No encontré el playbook seleccionado.")
    st.stop()

filters_payload = GENERATOR_FILTER_PRESETS.get(pb.name, {})

with col_sel:
    if st.button(
        f"⚡ Abrir generador con filtros de {pb.name}",
        use_container_width=True,
    ):
        st.session_state["_playbook_generator_filters"] = {
            "scenario": pb.name,
            "filters": filters_payload,
        }
        st.experimental_set_query_params(page="3_Generator")
        st.experimental_rerun()

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
