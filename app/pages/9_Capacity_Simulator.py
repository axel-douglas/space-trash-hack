# app/pages/9_Capacity_Simulator.py
import _bootstrap  # noqa: F401

# ⚠️ Debe ser la PRIMERA llamada de Streamlit
import streamlit as st

st.set_page_config(page_title="Capacity Simulator", page_icon="🧮", layout="wide")

import math
import numpy as np
import pandas as pd
import plotly.express as px

# Usamos tu módulo existente para mantener compatibilidad
from app.modules.capacity import LineConfig, simulate

# ============== Estilos SpaceX/NASA-like ==============
st.markdown(
    """
    <style>
    .callout{border-left:4px solid #8ab4ff; padding:10px 12px; background:rgba(138,180,255,.08); border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============== Estado compartido con el resto de la app ==============
target    = st.session_state.get("target", None)
selected  = st.session_state.get("selected", None)
cand      = selected["data"] if selected else None
props     = cand["props"] if cand else None

# Defaults inteligentes (si hay candidato activo, usamos sus recursos por lote)
default_kg_per_batch      = round(float(props.mass_final_kg), 2) if props else 0.95
default_kwh_per_batch     = round(float(props.energy_kwh), 2)    if props else 1.20
default_water_per_batch   = round(float(props.water_l), 2)       if props else 0.10
default_crew_min_per_batch= round(float(props.crew_min), 1)      if props else 25.0

# ============== HERO ==============
st.markdown("""
<div class="hero">
  <h1 style="margin:0 0 6px 0">🧮 Capacity Simulator — Mission Ops</h1>
  <div class="small">
    Planifica cuánta <b>producción</b> (kg) podés lograr en un horizonte de <b>soles</b> con
    tus <b>recursos</b> (kWh, agua, minutos de crew) y tu línea de proceso. 
    Trae por defecto los parámetros del candidato seleccionado (si lo hay) para cerrar el loop: <i>Generador → Resultados → Pareto → Capacidad</i>.
  </div>
  <div class="legend" style="margin-top:8px">
    <span class="pill info">1) Define tus turnos y lotes</span>
    <span class="pill info">2) Ajusta recursos por lote</span>
    <span class="pill info">3) Aplica límites/downtime</span>
    <span class="pill ok">4) Simula y compara contra la meta</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============== Panel de Configuración (Inputs) ==============
st.markdown("### 1) Configuración de la línea (como en el piso del hábitat)")

cA, cB, cC = st.columns([1.2, 1.1, 1.0])

with cA:
    st.markdown("**Turnos y horizonte**")
    shifts_per_sol = st.slider("Turnos por sol marciano", 1, 6, 2, help="Cantidad de turnos operativos por sol.")
    num_sols       = st.slider("Soles simulados", 1, 180, 30, help="Ventana de planificación (soles).")
    downtime_pct   = st.slider("Probabilidad de downtime (%)", 0, 30, 5, help="Tiempo improductivo por mantenimiento, setup, fallas.")
    efficiency     = st.slider("Eficiencia de línea (×)", 0.70, 1.10, 0.95, 0.01,
                               help="Factor que impacta kg por lote (p.ej., 0.9 = 90% de lo nominal).")

with cB:
    st.markdown("**Lotes y recursos por lote**")
    batches_per_shift = st.number_input("Lotes por turno", 1, 200, 3, 1,
                                        help="Cuántos ciclos completos de proceso haces en un turno.")
    kg_min, kg_max = 0.05, 200.0
    kg_default = min(kg_max, max(kg_min, float(default_kg_per_batch)))
    kg_per_batch      = st.number_input("Kg por lote", kg_min, kg_max, kg_default, 0.05)

    kwh_min, kwh_max = 0.0, 50.0
    kwh_default = min(kwh_max, max(kwh_min, float(default_kwh_per_batch)))
    energy_kwh_per_batch = st.number_input("kWh por lote", kwh_min, kwh_max, kwh_default, 0.01)

    water_min, water_max = 0.0, 50.0
    water_default = min(water_max, max(water_min, float(default_water_per_batch)))
    water_l_per_batch = st.number_input("Agua (L) por lote", water_min, water_max, water_default, 0.01)

    crew_min, crew_max = 0.0, 600.0
    crew_default = min(crew_max, max(crew_min, float(default_crew_min_per_batch)))
    crew_min_per_batch= st.number_input("Crew (min) por lote", crew_min, crew_max, crew_default, 1.0)

with cC:
    st.markdown("**Límites por sol (capas de seguridad)**")
    limit_kwh  = st.number_input("Límite kWh/sol", 0.0, 10_000.0, 250.0, 1.0,
                                 help="Tope de energía disponible por sol.")
    limit_water= st.number_input("Límite Agua (L)/sol", 0.0, 10_000.0, 30.0, 0.5,
                                 help="Tope de agua utilizable por sol.")
    limit_crew = st.number_input("Límite Crew (min)/sol", 0.0, 10_000.0, 600.0, 10.0,
                                 help="Tope de minutos de tripulación por sol.")
    goal_kg    = st.number_input("Meta de producción (kg)", 0.0, 100_000.0, 250.0, 1.0,
                                 help="Objetivo total para el horizonte.")

st.markdown("---")

# ============== Simulación (conectada a tus datos) ==============
st.markdown("### 2) Simular")

if st.button("▶️ Ejecutar simulación", type="primary"):
    # 2.1: simulación base con tu módulo
    cfg = LineConfig(
        batches_per_shift=int(batches_per_shift),
        kg_per_batch=float(kg_per_batch) * float(efficiency),
        energy_kwh_per_batch=float(energy_kwh_per_batch),
        water_l_per_batch=float(water_l_per_batch),
        crew_min_per_batch=float(crew_min_per_batch)
    )
    base = simulate(cfg, shifts_per_sol=int(shifts_per_sol), num_sols=int(num_sols))
    # base = {"batches","kg","kwh","water_l","crew_min"}

    # 2.2: aplicar downtime y límites por sol (distribución diaria)
    days = np.arange(1, int(num_sols)+1)

    # producción teórica por sol (antes de límites/downtime)
    batches_per_day = int(shifts_per_sol) * int(batches_per_shift)
    kg_day_nom   = cfg.kg_per_batch * batches_per_day
    kwh_day_nom  = cfg.energy_kwh_per_batch * batches_per_day
    water_day_nom= cfg.water_l_per_batch * batches_per_day
    crew_day_nom = cfg.crew_min_per_batch * batches_per_day

    # downtime & jitter ligero para hacerlo realista
    rng = np.random.default_rng(42)
    downtime_mask = rng.random(size=len(days)) < (float(downtime_pct)/100.0)
    jitter = rng.normal(1.0, 0.03, size=len(days))

    kg_day   = kg_day_nom   * jitter * (~downtime_mask)
    kwh_day  = kwh_day_nom  * jitter * (~downtime_mask)
    water_day= water_day_nom* jitter * (~downtime_mask)
    crew_day = crew_day_nom * jitter * (~downtime_mask)

    # aplicar límites por sol
    def apply_limits(k_kg, k_kwh, k_water, k_crew):
        # verificamos el “recurso cuello de botella” y recortamos proporcionalmente
        ratios = []
        if limit_kwh  > 0: ratios.append(limit_kwh  / max(1e-6, k_kwh))
        if limit_water> 0: ratios.append(limit_water/ max(1e-6, k_water))
        if limit_crew > 0: ratios.append(limit_crew / max(1e-6, k_crew))
        factor = min(1.0, *ratios) if ratios else 1.0
        return (k_kg*factor, k_kwh*factor, k_water*factor, k_crew*factor, factor)

    rows = []
    for d, a,b,c,dcrew in zip(days, kg_day, kwh_day, water_day, crew_day):
        out_kg, out_kwh, out_water, out_crew, util = apply_limits(a,b,c,dcrew)
        rows.append({
            "sol": int(d),
            "kg": float(out_kg),
            "kWh": float(out_kwh),
            "Agua (L)": float(out_water),
            "Crew (min)": float(out_crew),
            "utilización vs. límites": float(util),
            "downtime": bool(downtime_mask[d-1])
        })
    daily = pd.DataFrame(rows)
    daily["kg_cum"] = daily["kg"].cumsum()

    # 2.3: KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Kg totales", f"{daily['kg'].sum():.2f}")
    col2.metric("kWh totales", f"{daily['kWh'].sum():.2f}")
    col3.metric("Agua total (L)", f"{daily['Agua (L)'].sum():.2f}")
    col4.metric("Crew total (min)", f"{daily['Crew (min)'].sum():.0f}")
    # Día en que se alcanza la meta (si se alcanza)
    hit_goal_day = np.argmax(daily["kg_cum"].values >= goal_kg) + 1 if (daily["kg_cum"] >= goal_kg).any() else None
    if hit_goal_day:
        col5.metric("Meta alcanzada en sol", f"{hit_goal_day}")
    else:
        col5.metric("Meta alcanzada", "No", delta=f"Faltan {max(0.0, goal_kg - daily['kg'].sum()):.1f} kg")

    # 2.4: Visualizaciones principales
    a1, a2 = st.columns([1.3, 1.0])

    with a1:
        st.markdown("**Producción acumulada (kg) vs Meta**")
        fig = px.line(daily, x="sol", y="kg_cum", markers=True)
        fig.add_hline(y=goal_kg, line_dash="dash", line_color="#888", annotation_text="Meta (kg)")
        fig.update_layout(height=360, xaxis_title="Sol", yaxis_title="Kg acumulados")
        st.plotly_chart(fig, use_container_width=True)

    with a2:
        st.markdown("**Consumo por sol (kWh / Agua / Crew)**")
        fig2 = px.line(
            daily.melt(id_vars=["sol"], value_vars=["kWh","Agua (L)","Crew (min)"], var_name="Recurso", value_name="Valor"),
            x="sol", y="Valor", color="Recurso", markers=True
        )
        fig2.update_layout(height=360, xaxis_title="Sol")
        st.plotly_chart(fig2, use_container_width=True)

    b1, b2 = st.columns([1.0, 1.0])
    with b1:
        st.markdown("**Utilización vs. límites (1 = al límite)**")
        fig3 = px.area(daily, x="sol", y="utilización vs. límites")
        fig3.add_hline(y=1.0, line_dash="dot", line_color="#999")
        fig3.update_layout(height=300, yaxis_range=[0, 1.15], xaxis_title="Sol", yaxis_title="Utilización")
        st.plotly_chart(fig3, use_container_width=True)

    with b2:
        st.markdown("**Downtime por sol**")
        dt = daily[["sol","downtime"]].copy()
        dt["valor"] = dt["downtime"].astype(int)
        fig4 = px.bar(dt, x="sol", y="valor", color="downtime", color_discrete_map={True:"#f59e0b", False:"#93c5fd"})
        fig4.update_layout(height=300, xaxis_title="Sol", yaxis_title="Downtime (0/1)", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # 2.5: Tabla diaria y export
    st.markdown("**Tabla diaria (post-límites y downtime)**")
    st.dataframe(daily, use_container_width=True, hide_index=True)

    csv = daily.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar simulación (CSV)", data=csv, file_name="capacity_daily.csv", mime="text/csv")

    # 2.6: Bloque de interpretación (criollo + técnico)
    st.markdown("### 3) ¿Cómo leer esto? (criollo & técnico)")
    cL, cR = st.columns(2)
    with cL:
        st.markdown("""
**En criollo**
- Pensá cada lote como una “hornada”. Si metés 3 lotes por turno y 2 turnos por sol, son 6 hornadas/sol.
- La línea no es perfecta: podés perder tiempo por ajustes/mantenimiento (**downtime**).
- Además, hay límites duros (kWh/sol, agua/sol, minutos de crew). Si te pasás, el simulador recorta la producción de ese día.
- Buscá que la curva **Kg acumulados** cruce la **meta** antes de tu deadline.
""")
    with cR:
        st.markdown("""
**Técnico (rápido)**
- **kg_día** = `kg_lote × lotes_turno × turnos_sol × eficiencia × (1 - downtime)`  
  Luego se aplica un factor de recorte por el recurso **cuello de botella** del día.
- **Cuello de botella** = argmin { `límite_recurso / consumo_día_recurso` }  
  Ese recurso define la **utilización** (si =1, estás al límite).
- Si tu **utilización** está cerca de 1 en muchos días → aumentar límites o bajar consumo por lote.
""")

    # 2.7: Sugerencias automáticas (pequeño optimizador heurístico)
    st.markdown("### 4) Sugerencias automáticas (heurística)")
    tips = []
    if daily["utilización vs. límites"].mean() > 0.95:
        tips.append("Estás pegando contra límites casi todos los días. Considerá subir kWh/sol, Agua/sol o Crew/sol, o bajar recursos por lote.")
    if not hit_goal_day:
        tips.append("No alcanzás la meta: incrementá `lotes por turno` o `turnos por sol`, o subí `kg por lote` (si la calidad lo permite).")
    if (downtime_pct > 10):
        tips.append("Downtime alto. Evalúa mantenimiento preventivo o buffers de WIP para desacoplar equipos.")
    if efficiency < 0.9:
        tips.append("La eficiencia es baja. Revisa setup, temperatura/tiempos del proceso y entrenamiento de operadores.")
    if not tips:
        tips.append("Setup sólido. Probá un pequeño DoE: ±5% en kg por lote y lotes por turno para hallar un ‘sweet spot’.")
    st.markdown("<div class='callout'>" + "<br>• ".join(["**Recomendaciones:**"] + tips) + "</div>", unsafe_allow_html=True)

else:
    # Estado inicial con guía
    st.info("Configura los parámetros arriba y pulsa **▶️ Ejecutar simulación**. "
            "Si ya elegiste un candidato en Generador/Pareto, pre-cargamos los recursos por lote para que simules sobre datos reales.")

# ============== Apoyo didáctico para aprendices ==============
st.markdown("---")
st.markdown("### 5) Lectura rápida para aprendices")
g1, g2 = st.columns(2)
with g1:
    st.markdown("""
- **¿Para qué sirve?** Para saber cuántos kg podés producir en X soles sin romper tus límites (kWh, agua, crew).
- **¿Qué toco primero?** Probá subir `lotes por turno` y `turnos por sol`. Luego ajustá `kg por lote`.
- **¿Por qué hay ‘downtime’?** Porque en la vida real se hacen ajustes, hay esperas y pequeñas fallas.
""")
with g2:
    st.markdown("""
- **¿Qué miro para decidir?**  
  1) que la curva **Kg acumulados** cruce tu meta,  
  2) que la **utilización** no sea 1 todos los días,  
  3) que no te pases de los límites de recursos.
- **¿Qué exporto?** La tabla diaria (CSV) para presentar al equipo y justificar recursos o turnos.
""")
