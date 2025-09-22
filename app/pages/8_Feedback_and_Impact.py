# app/pages/8_Feedback_and_Impact.py
# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

# ⚠️ Debe ser la PRIMERA llamada Streamlit:
import streamlit as st
st.set_page_config(page_title="Feedback & Impact", page_icon="📝", layout="wide")

import pandas as pd
import altair as alt
from datetime import datetime
from io import StringIO

from app.modules.impact import (
    ImpactEntry, FeedbackEntry, append_impact, append_feedback,
    load_impact_df, load_feedback_df, summarize_impact
)
from app.modules.learning import (
    prepare_learning_bundle,
    pareto_shift_data,
)

# ========= estilos SpaceX/NASA-like =========
st.markdown("""
<style>
:root{ --bd: rgba(140,140,160,.28); --ink:#0f172a; --muted: rgba(15,23,42,.65); }
.hero{border:1px solid var(--bd); border-radius:18px; padding:18px;
      background: radial-gradient(900px 260px at 25% -10%, rgba(80,120,255,.10), transparent);}
.pill{display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.78rem;
      border:1px solid var(--bd); margin-right:6px}
.pill.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.pill.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.pill.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}
.block{border:1px solid var(--bd); border-radius:16px; padding:16px;}
.kpi{border:1px solid var(--bd); border-radius:14px; padding:14px; margin-bottom:10px;}
.kpi h3{margin:0 0 6px 0; font-size:.95rem; opacity:.8}
.kpi .v{font-size:1.6rem; font-weight:800; letter-spacing:.2px}
.small{font-size:.92rem; opacity:.9}
.legend{font-size:.9rem; opacity:.8}
hr{border:none;height:1px;background:var(--bd); margin:8px 0 16px 0}
table{font-size:0.95rem}
</style>
""", unsafe_allow_html=True)

# ========= estado compartido =========
target      = st.session_state.get("target", None)
state_sel   = st.session_state.get("selected", None)
candidato   = state_sel["data"] if state_sel else None
props       = candidato["props"] if candidato else None
regolith_pct = (candidato.get("regolith_pct", 0.0) if candidato else 0.0)
option_idx_sel = state_sel.get("option_idx", 0) if state_sel else 0

# ========= HERO =========
st.markdown("""
<div class="hero">
  <h1 style="margin:0 0 6px 0">📝 Feedback & Impact (HIL — Human-in-the-Loop)</h1>
  <div class="small">
    Esta consola registra el <b>impacto real</b> de cada corrida y el <b>feedback técnico</b> de materiales
    (rigidez, porosidad, superficie, unión, fallas), para que Rex-AI aprenda día a día.
    Todo conecta con tus datos: target, candidato y proceso seleccionados.
  </div>
  <div class="legend" style="margin-top:8px">
    <span class="pill info">1) Registra impacto</span>
    <span class="pill info">2) Envia feedback de materiales</span>
    <span class="pill ok">3) Visualiza métricas acumuladas</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ========= Panel A: Registrar IMPACTO de la corrida =========
st.markdown("### A) Registrar impacto de la corrida")

colA, colB = st.columns([1.2, 1.0])
with colA:
    if not candidato or not target:
        st.info("Seleccioná un candidato en **3) Generador** / **6) Pareto** para habilitar registro con datos reales.")
    else:
        st.markdown("**Contexto**")
        st.write(f"- Escenario: **{target.get('scenario','-')}**")
        st.write(f"- Target: **{target.get('name','-')}**")
        st.write(f"- Proceso: **{candidato['process_id']} {candidato['process_name']}**")
        st.write(f"- Materiales: **{', '.join(candidato['materials'])}**")
        if regolith_pct > 0:
            st.write(f"- MGS-1 (regolito): **{regolith_pct*100:.0f}%** de la mezcla")

with colB:
    if props:
        st.markdown("**Recursos/outputs de la corrida (predicción base)**")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Masa (kg)", f"{props.mass_final_kg:.2f}")
        c2.metric("kWh", f"{props.energy_kwh:.2f}")
        c3.metric("Agua (L)", f"{props.water_l:.2f}")
        c4.metric("Crew (min)", f"{props.crew_min:.0f}")
        c5.metric("Score", f"{candidato['score']:.2f}")

st.markdown("")
colBtn1, colBtn2 = st.columns([1,2])
with colBtn1:
    if candidato and target and st.button("💾 Guardar impacto de esta corrida", type="primary"):
        p = props
        entry = ImpactEntry(
            ts_iso=datetime.utcnow().isoformat(),
            scenario=target.get("scenario","-"),
            target_name=target.get("name","-"),
            option_idx=int(option_idx_sel or 0),
            materials="|".join(candidato.get("materials", [])),
            weights="|".join(map(str, candidato.get("weights", []))),
            process_id=candidato.get("process_id","-"),
            process_name=candidato.get("process_name","-"),
            mass_final_kg=float(p.mass_final_kg),
            energy_kwh=float(p.energy_kwh),
            water_l=float(p.water_l),
            crew_min=float(p.crew_min),
            score=float(candidato.get("score", 0.0)),
            pred_rigidity=float(p.rigidity),
            pred_tightness=float(p.tightness),
            regolith_pct=float(regolith_pct)
        )
        append_impact(entry)
        st.success("Impacto registrado en el log.")

with colBtn2:
    st.caption("Tip: registra impacto tras cada corrida para construir una curva de aprendizaje de materiales (y detectar desvíos).")

st.markdown("---")

# ========= Panel B: Feedback TÉCNICO de MATERIALES =========
st.markdown("### B) Feedback técnico (materiales) — nivel laboratorio")

with st.form("feedback_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        astronaut = st.text_input("Operador / Astronauta", "")
        option_idx = st.number_input("Opción elegida #", min_value=1, step=1, value=int(option_idx_sel or 1))
        overall = st.slider("Satisfacción global", 0, 10, 8, help="0=Malísimo, 10=Excelente")
    with col2:
        rigid_ok = st.slider("Rigidez percibida", 0, 10, 8)
        porosity = st.slider("Porosidad / compactación", 0, 10, 3, help="0=baja porosidad (mejor), 10=muy poroso")
        surface  = st.slider("Calidad de superficie", 0, 10, 7)
    with col3:
        bonding  = st.slider("Unión entre capas / partículas", 0, 10, 7)
        failure  = st.selectbox("Modo de falla observado", ["-", "Fragil", "Dúctil", "Delaminación", "Agarre insuficiente", "Fatiga"])
        ease_ok  = st.slider("Facilidad de proceso (ejecución)", 0, 10, 8)

    issues = st.text_area("Problemas específicos (bordes, olor, slip, etc.)", "")
    notes  = st.text_area("Notas libres / setup / parámetros", "")

    submitted = st.form_submit_button("Enviar feedback")
    if submitted:
        entry = FeedbackEntry(
            ts_iso=datetime.utcnow().isoformat(),
            astronaut=astronaut or "anon",
            scenario=target.get("scenario","-") if target else "-",
            target_name=target.get("name","-") if target else "-",
            option_idx=int(option_idx),
            rigidity_ok=bool(rigid_ok >= 6),
            ease_ok=bool(ease_ok >= 6),
            issues=issues,
            notes=notes,
            overall=float(overall),
            porosity=float(porosity),
            surface=float(surface),
            bonding=float(bonding),
            failure_mode=str(failure)
        )
        append_feedback(entry)
        st.success("Feedback guardado. Rex-AI utilizará estas señales para ajustar pesos/penalizaciones y recomendaciones.")

st.markdown("---")

# ========= Panel C: Impacto ACUMULADO y ANALÍTICA =========
st.markdown("### C) Panel de impacto acumulado (conecta a tus logs)")

idf = load_impact_df()
fdf = load_feedback_df()
sumy = summarize_impact(idf) if idf is not None and len(idf) else {"runs":0,"kg":0,"kwh":0,"water_l":0,"crew_min":0}

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Corridas", int(sumy["runs"]))
k2.metric("Kg valorizados", f"{sumy['kg']:.2f} kg")
k3.metric("Energía total", f"{sumy['kwh']:.2f} kWh")
k4.metric("Agua total", f"{sumy['water_l']:.2f} L")
k5.metric("Crew total", f"{sumy['crew_min']:.0f} min")

if idf is not None and len(idf):
    # Normalizamos timestamp a día para tendencias
    tmp = idf.copy()
    tmp["date"] = pd.to_datetime(tmp["ts_iso"]).dt.date

    cL, cR = st.columns([1.1, 1.0])

    with cL:
        st.markdown("**Tendencia diaria (kg / kWh / L / crew)**")
        trend = tmp.groupby("date").agg({
            "mass_final_kg":"sum",
            "energy_kwh":"sum",
            "water_l":"sum",
            "crew_min":"sum"
        }).reset_index().rename(columns={
            "mass_final_kg":"Kg",
            "energy_kwh":"kWh",
            "water_l":"Agua (L)",
            "crew_min":"Crew (min)"
        })
        st.line_chart(trend.set_index("date"))

    with cR:
        st.markdown("**Distribución por proceso (n corridas)**")
        dist = tmp.groupby(["process_id","process_name"]).size().reset_index(name="runs").sort_values("runs", ascending=False)
        st.bar_chart(dist.set_index("process_name")["runs"])
        st.caption("¿Dónde estamos invirtiendo tiempo? ¿Vale la pena mover corridas a procesos más eficientes?")

    st.markdown("**Detalle de corridas (impact log)**")
    st.dataframe(idf, use_container_width=True, hide_index=True)

    # Export
    csv_buf = StringIO(); idf.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Descargar impacto (CSV)", data=csv_buf.getvalue().encode("utf-8"),
                       file_name="impact_log.csv", mime="text/csv")
else:
    st.info("Aún no hay corridas registradas en el log de impacto.")

# ========= Panel D: Feedback → desplazamiento de la frontera de Pareto =========
bundle = prepare_learning_bundle(idf, fdf)
shift = pareto_shift_data(bundle.merged)
st.markdown("---")
st.markdown("### D) Feedback → desplazamiento de la frontera de Pareto")
if shift:
    scatter = shift["scatter"]
    base = alt.Chart(scatter).mark_circle(size=90, opacity=0.65).encode(
        x="Energía (kWh)",
        y="Score",
        color=alt.Color("accepted", title="Resultado"),
        tooltip=["Energía (kWh)", "Agua (L)", "Crew (min)", "Score", "accepted"],
    )
    chart = base
    for label, front_df in shift["fronts"].items():
        color = "#ff9f1c" if label == "Aceptado" else "#2ec4b6"
        layer = (
            alt.Chart(front_df)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=85))
            .encode(
                x="Energía (kWh)",
                y="Score",
                color=alt.value(color),
                tooltip=["Energía (kWh)", "Agua (L)", "Crew (min)", "Score"],
            )
        )
        chart = chart + layer
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "La curva naranja es la Pareto usando sólo ensayos aceptados; si se mueve a la izquierda/arriba respecto al histórico "
        "(turquesa), el feedback está guiando hacia recetas con menos consumo y mejor Score."
    )

    if not bundle.dataset.empty and not bundle.merged[bundle.merged["accepted"] == 1].empty:
        aceptadas = bundle.merged[bundle.merged["accepted"] == 1]
        media = aceptadas[["score", "energy_kwh", "water_l", "crew_min"]].mean()
        st.markdown(
            f"**Resumen HIL** → Score medio aceptado: {media['score']:.2f} | kWh: {media['energy_kwh']:.2f} | "
            f"Agua: {media['water_l']:.2f} L | Crew: {media['crew_min']:.1f} min"
        )
else:
    st.info("Registrá feedback y corridas para analizar el desplazamiento de la Pareto.")

st.markdown("---")
st.markdown("### E) Lectura rápida para aprendices (¿por qué esto importa?)")
g1, g2 = st.columns(2)
with g1:
    st.markdown("""
- **Impacto = realidad**: qué tanto residuo convertimos en producto y a qué costo (energía/agua/crew).
- **Feedback ≠ opinión suelta**: capturamos señales de materiales (rigidez, porosidad, unión) que Rex-AI usa para ajustar decisiones.
- **Efecto MGS-1**: cuando el proceso es sinterizado, verás el % de regolito registrado automáticamente; si percibís más porosidad, anotarlo aquí ayuda.
""")
with g2:
    st.markdown("""
- **Cómo usarlo**: tras cada corrida, registra impacto (1 click) y cuelga feedback (2 min).
- **Cómo leerlo**: mirá la tendencia diaria; si los kWh suben por pieza, hay algo en el setup.
- **Qué permite**: cerrar el loop *plan → ejecutar → medir → aprender* como en un banco de pruebas de SpaceX/NASA.
""")

st.caption("¿Ideas para nuevas métricas? Podemos agregar dureza Shore, módulo aparente, densidad aparente, etc., si pasan a formar parte de la captura del laboratorio.")
