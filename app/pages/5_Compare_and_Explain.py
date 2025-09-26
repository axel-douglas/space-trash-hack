import app  # noqa: F401

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.modules.explain import compare_table, score_breakdown
from app.modules.ui_blocks import load_theme

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Comparar & Explicar", page_icon="🧪", layout="wide")

load_theme()

# ======== estado requerido ========
cands  = st.session_state.get("candidates", [])
target = st.session_state.get("target", None)
if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

# ======== estilo visual ========
st.markdown(
    """
    <style>
    .kpi {border-color: rgba(128,128,128,0.25); margin-bottom:12px;}
    .kpi .v {font-weight:700;}
    .pill {font-weight:600; font-size:0.85rem;}
    h2 span.sub {font-size:0.95rem; font-weight:500; opacity:0.7; margin-left:8px;}
    .small {font-size:0.9rem; opacity:0.85;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# 🧪 Compare & Explain")
st.caption("Compará candidatos como en un *design review*: qué rinde más, dónde gasta menos, y por qué elige la IA esa receta.")

# ======== tabla comparativa base ========
df_base = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False))
# Aseguramos columnas esperadas y nombres amigables
expected_cols = ["Opción","Score","Proceso","Materiales","Energía (kWh)","Agua (L)","Crew (min)","Masa (kg)"]
for col in expected_cols:
    if col not in df_base.columns:
        # intentamos mapear por nombres aproximados si hiciera falta
        pass

# KPIs generales
colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown(f'<div class="kpi"><h3>Opciones generadas</h3><div class="v">{len(cands)}</div><div class="hint">Muestra suficiente para comparar</div></div>', unsafe_allow_html=True)
with colB:
    st.markdown(f'<div class="kpi"><h3>Mejor Score</h3><div class="v">{df_base["Score"].max():.2f}</div><div class="hint">Top actual</div></div>', unsafe_allow_html=True)
with colC:
    st.markdown(f'<div class="kpi"><h3>Consumo mínimo de agua</h3><div class="v">{df_base["Agua (L)"].min():.2f} L</div><div class="hint">Entre todas las opciones</div></div>', unsafe_allow_html=True)
with colD:
    st.markdown(f'<div class="kpi"><h3>Energía mínima</h3><div class="v">{df_base["Energía (kWh)"].min():.2f} kWh</div><div class="hint">Entre todas las opciones</div></div>', unsafe_allow_html=True)

st.markdown("### Tabla consolidada")
st.dataframe(df_base, use_container_width=True, hide_index=True)

# ======== gráficos de vista general ========
st.markdown("### Vistas rápidas")
g1, g2 = st.columns(2)

with g1:
    # Bubble 2D: Energía vs Agua, tamaño Crew, color Score
    fig_sc = px.scatter(
        df_base,
        x="Energía (kWh)",
        y="Agua (L)",
        size="Crew (min)",
        color="Score",
        hover_data=["Opción","Proceso","Materiales"],
        title="Trade-off rápido: Energía vs Agua (tamaño = Crew)"
    )
    fig_sc.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

with g2:
    # Ranking de Score
    df_rank = df_base.sort_values("Score", ascending=False)
    fig_bar = go.Figure(data=[go.Bar(
        x=df_rank["Opción"],
        y=df_rank["Score"],
        text=[f"{v:.2f}" for v in df_rank["Score"]],
        textposition="outside"
    )])
    fig_bar.update_layout(
        title="Ranking por Score",
        margin=dict(l=10,r=10,t=40,b=10),
        yaxis_title="Score",
        xaxis_title="Opción",
        height=420
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ======== Tabs de análisis profundo ========
tab1, tab2, tab3 = st.tabs(["🔍 Inspector de candidato", "⚔️ Duelo (A vs B)", "📖 Explicación didáctica"])

# --- TAB 1: Inspector de candidato ---
with tab1:
    st.markdown("## 🔍 Inspector de candidato <span class='sub'>(ver detalle y desglose del score)</span>", unsafe_allow_html=True)
    pick = st.number_input("Elegí la Opción #", min_value=1, max_value=len(cands), value=1, step=1)
    c = cands[int(pick)-1]

    # Resumen del candidato
    cL, cR = st.columns([1.2, 1.0])
    with cL:
        st.markdown(f"**Proceso:** {c['process_id']} — {c['process_name']}")
        st.markdown(f"**Materiales:** {', '.join(c['materials'])}")
        st.markdown(f"**Pesos:** {c['weights']}")
        st.markdown(f"**Score:** {c['score']:.2f}")
        p = c["props"]
        st.markdown(f"**Predicción** → Rigidez {p.rigidity:.2f} | Estanqueidad {p.tightness:.2f} | Masa {p.mass_final_kg:.2f} kg")
        st.markdown(f"**Recursos** → Energía {p.energy_kwh:.2f} kWh | Agua {p.water_l:.2f} L | Crew {p.crew_min:.0f} min")
        # Pills de ajuste a límites
        def _pill(val, limit, reverse=False, label="ok"):
            ok = (val <= limit) if not reverse else (val >= limit)
            cls = "ok" if ok else "bad"
            return f'<span class="pill {cls}">{label}: {"OK" if ok else "Fuera de límite"}</span>'
        st.markdown(
            _pill(p.energy_kwh, float(target["max_energy_kwh"]), label="Energía") + " " +
            _pill(p.water_l,   float(target["max_water_l"]),   label="Agua") + " " +
            _pill(p.crew_min,  float(target["max_crew_min"]),  label="Crew"),
            unsafe_allow_html=True
        )

        # Traza NASA (si está disponible)
        ids = c.get("source_ids", [])
        if ids:
            st.caption("**Trazabilidad NASA** — IDs: " + ", ".join(ids))
            st.caption("Categorías: " + ", ".join(map(str, c.get("source_categories", []))))
            st.caption("Flags: " + ", ".join(map(str, c.get("source_flags", []))))
        if c.get("regolith_pct", 0) > 0:
            st.caption(f"**ISRU**: incluye **MGS-1_regolith** ({int(c['regolith_pct']*100)}%).")

    with cR:
        parts = score_breakdown(c["props"], target, crew_time_low=target.get("crew_time_low", False))
        if isinstance(parts, pd.DataFrame) and "component" in parts.columns and "contribution" in parts.columns:
            fig_d = go.Figure(
                data=[go.Bar(
                    x=parts["component"],
                    y=parts["contribution"],
                    text=[f"{v:.2f}" for v in parts["contribution"]],
                    textposition="outside"
                )]
            )
            fig_d.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Aporte", height=360)
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No se pudo construir el desglose. Verifica `score_breakdown`.")

    # Ayuda didáctica
    pop = st.popover("¿Cómo interpretar este panel?")
    with pop:
        st.markdown("""
- **Objetivo**: ver *qué te da* este candidato y *cuánto cuesta* (energía/agua/crew).
- **Pills**: si alguna dice “Fuera de límite”, sabés dónde ajustar (bajar kWh, usar P03 con más MGS-1, etc.).
- **Desglose**: te muestra qué pesa en el *score*: función vs. recursos. Si activaste *Crew-time Low*, el tiempo pesa más.
""")

# --- TAB 2: Duelo (A vs B) ---
with tab2:
    st.markdown("## ⚔️ Duelo de candidatos (A vs B)")
    left, right = st.columns(2)
    with left:
        a_idx = st.number_input("Candidato A (#)", min_value=1, max_value=len(cands), value=1, step=1, key="duel_a")
        A = cands[a_idx-1]
    with right:
        b_idx = st.number_input("Candidato B (#)", min_value=1, max_value=len(cands), value=min(2, len(cands)), step=1, key="duel_b")
        B = cands[b_idx-1]

    # Cuadro comparativo simple
    def _row(k, fa, fb, suffix=""):
        return f"| **{k}** | {fa}{suffix} | {fb}{suffix} |"

    Ap, Bp = A["props"], B["props"]
    table_md = "\n".join([
        "| Métrica | A | B |",
        "|---|---:|---:|",
        _row("Score", A["score"], B["score"]),
        _row("Energía (kWh)", f"{Ap.energy_kwh:.2f}", f"{Bp.energy_kwh:.2f}"),
        _row("Agua (L)",      f"{Ap.water_l:.2f}",    f"{Bp.water_l:.2f}"),
        _row("Crew (min)",    f"{Ap.crew_min:.0f}",   f"{Bp.crew_min:.0f}"),
        _row("Rigidez",       f"{Ap.rigidity:.2f}",   f"{Bp.rigidity:.2f}"),
        _row("Estanqueidad",  f"{Ap.tightness:.2f}",  f"{Bp.tightness:.2f}"),
        _row("Masa final (kg)", f"{Ap.mass_final_kg:.2f}", f"{Bp.mass_final_kg:.2f}"),
    ])
    st.markdown(table_md)

    # Visual complementaria: barras apiladas “recursos”
    res_df = pd.DataFrame([
        {"Candidato":"A","Energía (kWh)":Ap.energy_kwh,"Agua (L)":Ap.water_l,"Crew (min)":Ap.crew_min},
        {"Candidato":"B","Energía (kWh)":Bp.energy_kwh,"Agua (L)":Bp.water_l,"Crew (min)":Bp.crew_min},
    ])
    fig_res = go.Figure(data=[
        go.Bar(name="Energía (kWh)", x=res_df["Candidato"], y=res_df["Energía (kWh)"]),
        go.Bar(name="Agua (L)",      x=res_df["Candidato"], y=res_df["Agua (L)"]),
        go.Bar(name="Crew (min)",    x=res_df["Candidato"], y=res_df["Crew (min)"]),
    ])
    fig_res.update_layout(barmode="group", margin=dict(l=10,r=10,t=10,b=10), height=360)
    st.plotly_chart(fig_res, use_container_width=True)

    # Conclusión en criollo (reglas simples)
    concl = []
    if A["score"] > B["score"]:
        concl.append("🟢 **A** gana en Score global.")
    elif B["score"] > A["score"]:
        concl.append("🟢 **B** gana en Score global.")
    else:
        concl.append("⚖️ Empate en Score.")

    if Ap.energy_kwh < Bp.energy_kwh: concl.append("⚡ A usa **menos energía**.")
    elif Ap.energy_kwh > Bp.energy_kwh: concl.append("⚡ B usa **menos energía**.")
    if Ap.water_l < Bp.water_l: concl.append("💧 A usa **menos agua**.")
    elif Ap.water_l > Bp.water_l: concl.append("💧 B usa **menos agua**.")
    if Ap.crew_min < Bp.crew_min: concl.append("👩‍🚀 A consume **menos crew-time**.")
    elif Ap.crew_min > Bp.crew_min: concl.append("👩‍🚀 B consume **menos crew-time**.")
    if Ap.rigidity > Bp.rigidity: concl.append("🧱 A ofrece **más rigidez**.")
    elif Ap.rigidity < Bp.rigidity: concl.append("🧱 B ofrece **más rigidez**.")
    if Ap.tightness > Bp.tightness: concl.append("🧴 A ofrece **más estanqueidad**.")
    elif Ap.tightness < Bp.tightness: concl.append("🧴 B ofrece **más estanqueidad**.")

    st.markdown("**Conclusión rápida:**")
    st.markdown("- " + "\n- ".join(concl))

    pop_duel = st.popover("¿Cómo usar este duelo?")
    with pop_duel:
        st.markdown("""
- Elegí **A** y **B** para un *cara a cara* objetivo.
- Mirá primero el **Score**, luego validá restricciones: si B gana en Score pero excede **agua**, quizá **A** sea más viable operativamente.
- Si tu modo es *Crew-time Low*, priorizá el que **baje minutos** de tripulación, aunque requiera algo más de energía.
""")

# --- TAB 3: Explicación didáctica ---
with tab3:
    st.markdown("## 📖 ¿Qué significa comparar acá?")
    st.markdown("""
- **Tabla consolidada**: es tu **panel de vuelo**. Vuela directo al *trade-off* que importa.
- **Scatter**: te da el “mapa” de terreno (agua vs. energía) y el tamaño de burbuja te recuerda el **crew-time**.
- **Ranking**: si tenés poco tiempo, mirá el top de Score y usalo como *shortlist*.
- **Inspector**: abrí uno y entendé *por qué* suma ese Score (barras del desglose).
- **Duelo**: si hay discusión en el equipo, el cara a cara lo vuelve incuestionable.
""")
    pop_help = st.popover("Tips expertos")
    with pop_help:
        st.markdown("""
- Si aparece **MGS-1_regolith** en un candidato, estás haciendo **ISRU**: buena señal en misiones largas.
- Si el límite es **agua**, buscá burbujas **abajo**; si es **energía**, buscá **izquierda**.
- Si te falta rigidez, favorecé mezclas con **Al** o procesos **P02/P03**; si te falta estanqueidad, **pouches multilayer** ayudan tras laminado.
""")
