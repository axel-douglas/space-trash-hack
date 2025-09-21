# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from app.modules.explain import score_breakdown

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Resultados", page_icon="📊", layout="wide")

# ======== guardas de estado ========
state_sel = st.session_state.get("selected", None)
target = st.session_state.get("target", None)

if not state_sel or not target:
    st.warning("Seleccioná una receta en **3) Generador**.")
    st.stop()

sel   = state_sel["data"]
badge = state_sel["safety"]
p     = sel["props"]

# ======== estilo ligero ========
st.markdown("""
<style>
/* Cards KPI */
.kpi {border:1px solid rgba(128,128,128,0.25); border-radius:14px; padding:14px; margin-bottom:12px;}
.kpi h3 {margin:0 0 6px 0; font-size:0.95rem; opacity:0.8;}
.kpi .v {font-size:1.6rem; font-weight:700; letter-spacing:0.2px;}
.kpi .hint {font-size:0.85rem; opacity:0.75;}
/* Pills de estado */
.pill {display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.85rem;}
.pill.ok {background:#e8f7ee; color:#136c3a; border:1px solid #b3e2c4;}
.pill.warn {background:#fff3cd; color:#8a6d1d; border:1px solid #ffe69b;}
.pill.risk {background:#fdeceb; color:#8b1b13; border:1px solid #f5c2c0;}
/* Secciones */
h2 span.sub {font-size:0.95rem; font-weight:500; opacity:0.7; margin-left:8px;}
/* Tooltips helpers */
.small {font-size:0.9rem; opacity:0.85;}
</style>
""", unsafe_allow_html=True)

# ======== Cabecera tipo mission control ========
st.markdown("# 📊 Resultados & Trade-offs")
st.caption("Qué tan buena es la receta elegida, por qué, y cómo se reparte el esfuerzo de recursos. Todo en un vistazo, como en un MCC.")

topL, topR = st.columns([1.6, 1.0], vertical_alignment="center")

with topL:
    st.markdown(f"### {sel['process_id']} — {sel['process_name']}")
    st.markdown(
        f"**Materiales:** {', '.join(sel['materials'])}"
        + (f" &nbsp;&nbsp;·&nbsp;&nbsp;**MGS-1**: {int(sel.get('regolith_pct', 0)*100)}%" if sel.get("regolith_pct", 0) > 0 else "")
    )

with topR:
    # Estado de seguridad (pill + popover de explicación)
    level = badge.get("level","OK").lower()
    cls = "ok" if "ok" in level else ("risk" if "riesgo" in level or "risk" in level else "warn")
    st.markdown(f'<span class="pill {cls}">Seguridad: {badge.get("level","OK")}</span>', unsafe_allow_html=True)
    pop = st.popover("¿Qué chequeamos?")
    with pop:
        st.write(badge.get("detail", "Sin observaciones."))
        st.caption("Validaciones: PFAS/microplásticos evitados, sin incineración, flags NASA (EVA/CTB, multilayers, nitrilo).")

st.markdown("---")

# ======== KPIs con contexto ========
k1,k2,k3,k4,k5 = st.columns(5)
with k1:
    st.markdown('<div class="kpi"><h3>Score total</h3><div class="v">{:.2f}</div><div class="hint">Función + Recursos + Bono problemáticos</div></div>'.format(sel["score"]), unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi"><h3>Rigidez</h3><div class="v">{:.2f}</div><div class="hint">Objetivo: {:.2f}</div></div>'.format(p.rigidity, float(target["rigidity"])), unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi"><h3>Estanqueidad</h3><div class="v">{:.2f}</div><div class="hint">Objetivo: {:.2f}</div></div>'.format(p.tightness, float(target["tightness"])), unsafe_allow_html=True)
with k4:
    st.markdown('<div class="kpi"><h3>Energía (kWh)</h3><div class="v">{:.2f}</div><div class="hint">Máx: {:.2f}</div></div>'.format(p.energy_kwh, float(target["max_energy_kwh"])), unsafe_allow_html=True)
with k5:
    st.markdown('<div class="kpi"><h3>Agua (L)</h3><div class="v">{:.2f}</div><div class="hint">Máx: {:.2f}</div></div>'.format(p.water_l, float(target["max_water_l"])), unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.markdown('<div class="kpi"><h3>Crew-time (min)</h3><div class="v">{:.0f}</div><div class="hint">Máx: {:.0f}</div></div>'.format(p.crew_min, float(target["max_crew_min"])), unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi"><h3>Masa final (kg)</h3><div class="v">{:.2f}</div><div class="hint">Post-proceso / mermas</div></div>'.format(p.mass_final_kg), unsafe_allow_html=True)
with c3:
    # Mini ayuda para crew-time-low
    if target.get("crew_time_low", False):
        st.markdown('<div class="kpi"><h3>Modo</h3><div class="v">Crew-time Low</div><div class="hint">Más peso al tiempo de tripulación</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="kpi"><h3>Modo</h3><div class="v">Balanceado</div><div class="hint">Trade-off estándar</div></div>', unsafe_allow_html=True)

# ======== Tabs principales: (1) Score anatomy (2) Flujo Sankey (3) Checklist (4) Trazabilidad ========
tab1, tab2, tab3, tab4 = st.tabs(["🧩 Anatomía del Score", "🔀 Flujo del proceso (Sankey)", "🛠️ Checklist & Próximos pasos", "🛰️ Trazabilidad NASA"])

# --- TAB 1: Anatomía del Score ---
with tab1:
    st.markdown("## 🧩 Anatomía del Score <span class='sub'>(explicabilidad)</span>", unsafe_allow_html=True)
    parts = score_breakdown(p, target, crew_time_low=target.get("crew_time_low", False))
    # Asegurar índice correcto
    if isinstance(parts, pd.DataFrame) and "component" in parts.columns and "contribution" in parts.columns:
        fig_bar = go.Figure(
            data=[go.Bar(
                x=parts["component"],
                y=parts["contribution"],
                text=[f"{v:.2f}" for v in parts["contribution"]],
                textposition="outside"
            )]
        )
        fig_bar.update_layout(
            margin=dict(l=10,r=10,t=10,b=10),
            yaxis_title="Aporte al Score",
            xaxis_title="Componente",
            height=360
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No se pudo construir el desglose. Verifica `score_breakdown`.")

    # Popover didáctico
    pop1 = st.popover("¿Qué estoy viendo?")
    with pop1:
        st.markdown("""
- Cada barra es una **pieza del puntaje**:
  - **Función**: qué tan cerca está tu receta de la *rigidez* y *estanqueidad* objetivo.
  - **Recursos**: te premia por **bajo consumo** de energía/agua y **poco tiempo** de tripulación.
  - **Seguridad base**: piso de seguridad (las banderas duras se validan aparte).
- Si activaste *Crew-time Low*, la barra de **tiempo** pesa más.
""")

# --- TAB 2: Flujo Sankey ---
with tab2:
    st.markdown("## 🔀 Flujo de materiales → proceso → producto", unsafe_allow_html=True)

    labels = sel["materials"] + [sel["process_name"], "Producto"]
    src = list(range(len(sel["materials"])))
    tgt = [len(sel["materials"])] * len(sel["materials"])
    # Si vienen pesos con regolito + redondeos, normalizamos para que la suma sea 1
    weights = sel.get("weights", [])
    if weights and abs(sum(weights) - 1.0) > 1e-6:
        s = sum(weights)
        weights = [w/s for w in weights]
    vals = [round(w*100, 1) for w in weights] if weights else [100/len(src)]*len(src)

    # proceso -> producto
    src += [len(sel["materials"])]
    tgt += [len(sel["materials"]) + 1]
    vals += [100.0]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=20, thickness=18),
        link=dict(source=src, target=tgt, value=vals)
    )])
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

    pop2 = st.popover("¿Cómo leerlo?")
    with pop2:
        st.markdown("""
- **Izquierda**: residuos seleccionados (con sus **pesos relativos**).
- **Centro**: el **proceso** (p.ej., Laminar o *Sinter with MGS-1* si hay regolito).
- **Derecha**: el **producto** final.
- Si ves **MGS-1_regolith** en materiales, significa **ISRU**: aprovechamos regolito como carga mineral.
""")

# --- TAB 3: Checklist & Próximos pasos ---
with tab3:
    st.markdown("## 🛠️ Checklist de fabricación")
    st.markdown(f"""
1. **Preparar/Triturar**: acondicionar materiales (**{', '.join(sel['materials'])}**).
2. **Ejecutar proceso**: **{sel['process_id']} {sel['process_name']}** con parámetros estándar del hábitat.
3. **Enfriar & post-proceso**: verificar bordes, ajuste y *fit*.
4. **Registrar feedback**: rigidez percibida, facilidad de uso, y problemas (bordes, olor, slip, etc.).
    """)

    st.markdown("### ⏱️ Recursos estimados")
    cA,cB,cC = st.columns(3)
    cA.metric("Energía", f"{p.energy_kwh:.2f} kWh")
    cB.metric("Agua", f"{p.water_l:.2f} L")
    cC.metric("Crew-time", f"{p.crew_min:.0f} min")

    pop3 = st.popover("¿Por qué importa?")
    with pop3:
        st.markdown("""
- En Marte **no hay camión de la basura**: cada minuto de tripulación y cada litro de agua cuenta.
- Minimizar recursos mantiene la **operación sostenible** y deja margen para otras tareas científicas.
""")

# --- TAB 4: Trazabilidad NASA ---
with tab4:
    st.markdown("## 🛰️ Trazabilidad NASA (inputs → plan)")
    # IDs / categorías / flags para auditar que usamos lo problemático
    st.markdown("**IDs usados:** " + ", ".join(sel.get("source_ids", []) or ["—"]))
    st.markdown("**Categorías:** " + ", ".join(map(str, sel.get("source_categories", []) or ["—"])))
    st.markdown("**Flags:** " + ", ".join(map(str, sel.get("source_flags", []) or ["—"])))
    st.caption("Esto permite demostrar que estamos atacando pouches multilayer, espumas ZOTEK, EVA/CTB, nitrilo, etc.")

# ======== Bloque final de educación rápida ========
st.markdown("---")
edu = st.popover("ℹ️ Entender estos trade-offs (explicación simple)")
with edu:
    st.markdown("""
- **Score**: es como el *promedio ponderado* de todo lo que te importa (función + recursos + seguridad).
- **Rigidez/Estanqueidad**: si tu contenedor se **deforma** o **pierde** porosidad, no sirve; por eso están arriba del todo.
- **Energía/Agua/Crew**: si te pasás de los límites objetivo, **penaliza**. Marte **no perdona** derroches.
- **Sankey**: te muestra **qué entra**, **cómo se procesa**, **qué sale**. Ayuda a “ver” si el plan es coherente.
- **MGS-1**: si aparece, es **ISRU** (usar lo que hay en Marte). Menos dependencia de la Tierra, más puntos por sostenibilidad.
""")
