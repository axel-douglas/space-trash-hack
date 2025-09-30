import _bootstrap  # noqa: F401

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from app.modules.ui_blocks import load_theme, layout_block
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.luxe_components import (
    GlassCard,
    GlassStack,
    MetricGalaxy,
    MetricItem,
    TeslaHero,
    ChipRow,
)

from app.modules.data_sources import (
    load_regolith_granulometry,
    load_regolith_spectral_curves,
    load_regolith_thermal_profiles,
)
from app.modules.explain import score_breakdown

st.set_page_config(page_title="Rex-AI • Resultados", page_icon="📊", layout="wide")

set_active_step("results")

load_theme()

render_breadcrumbs("results")

selected = st.session_state.get("selected")
target = st.session_state.get("target")
if not selected or not target:
    st.warning("Seleccioná una receta en **3 · Generador**.")
    st.stop()

cand = selected["data"]
props = cand["props"]
heur = cand.get("heuristic_props", props)
ci = cand.get("confidence_interval") or {}
uncertainty = cand.get("uncertainty") or {}
comparisons = cand.get("model_variants") or {}
importance = cand.get("feature_importance") or []
metadata = cand.get("ml_prediction", {}).get("metadata", {})
latent = cand.get("latent_vector", [])
regolith_pct = cand.get("regolith_pct", 0.0)
materials = cand.get("materials", [])
score = cand.get("score", 0.0)
safety = selected.get("safety", {"level": "—", "detail": ""})


@st.cache_data(show_spinner=False)
def _load_regolith_context():
    return {
        "granulometry": load_regolith_granulometry(),
        "spectra": load_regolith_spectral_curves(),
        "thermal": load_regolith_thermal_profiles(),
    }

st.markdown(
    f"""
    <section class="hero-res layer-glow fade-in">
      <h1>📊 Resultado seleccionado · Score {score:.3f}</h1>
      <p>Proceso <strong>{cand['process_id']} · {cand['process_name']}</strong>. La IA Rex-AI proporciona predicciones con trazabilidad NASA, bandas de confianza y comparación contra heurísticas originales.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

cards = []
TeslaHero(
    title=f"Resultado seleccionado · Score {score:.3f}",
    subtitle=(
        f"Proceso {cand['process_id']} · {cand['process_name']}. "
        "La IA Rex-AI proporciona predicciones con trazabilidad NASA, bandas de confianza y comparación contra heurísticas originales."
    ),
    chips=[
        {"label": f"Target: {target.get('name', '—')}", "tone": "accent"},
        {"label": f"Crew priority: {'baja' if target.get('crew_time_low') else 'balance'}", "tone": "info"},
    ],
    icon="📊",
    gradient="linear-gradient(135deg, rgba(20,184,166,0.22), rgba(14,165,233,0.08))",
    glow="rgba(45,212,191,0.42)",
    density="cozy",
    parallax_icons=[
        {"icon": "🧪", "top": "18%", "left": "78%", "size": "3.6rem", "speed": "21s"},
        {"icon": "🛰️", "top": "60%", "left": "84%", "size": "4.2rem", "speed": "27s"},
    ],
    variant="minimal",
).render()

icon_map = {
    "Rigidez": "🧱",
    "Estanqueidad": "💧",
    "Energía (kWh)": "⚡",
    "Agua (L)": "🚰",
    "Crew (min)": "🧑‍🚀",
}
metric_items: list[MetricItem] = []
labels = [
    ("Rigidez", props.rigidity, heur.rigidity, ci.get("rigidez")),
    ("Estanqueidad", props.tightness, heur.tightness, ci.get("estanqueidad")),
    ("Energía (kWh)", props.energy_kwh, heur.energy_kwh, ci.get("energy_kwh")),
    ("Agua (L)", props.water_l, heur.water_l, ci.get("water_l")),
    ("Crew (min)", props.crew_min, heur.crew_min, ci.get("crew_min")),
]
for label, val_ml, val_h, interval in labels:
    ci_html = ""
    if interval:
        ci_html = f"<div class='delta'>CI 95%: [{interval[0]:.3f}, {interval[1]:.3f}]</div>"

    cards.append(
        f"<div class='stat-card layer-shadow'>"
        f"<span>{label}</span>"
        f"<strong>{val_ml:.3f}</strong>"
        f"<div class='delta'>Heurística: {val_h:.3f} · Δ {val_ml - val_h:+.3f}</div>"
        f"{ci_html}"
        "</div>"
    )

    delta_value = val_ml - val_h
    caption_bits = [f"Heurística: {val_h:.3f}"]
    if interval:
        try:
            caption_bits.append(f"CI 95% [{interval[0]:.3f}, {interval[1]:.3f}]")
        except (TypeError, ValueError, IndexError):
            pass

    metric_items.append(
        MetricItem(
            label=label,
            value=f"{val_ml:.3f}",
            delta=f"Δ {delta_value:+.3f}",
            caption=" · ".join(caption_bits),
            icon=icon_map.get(label),
        )
    )

metrics_html = "<div class='metric-grid fade-in'>" + "".join(cards) + "</div>"
st.markdown(metrics_html, unsafe_allow_html=True)

MetricGalaxy(metrics=metric_items, density="compact").render()
if uncertainty:
    st.caption("Desviaciones modelo: " + ", ".join(f"{k} {v:.3f}" for k, v in uncertainty.items()))

with st.container():
    st.markdown("### 🧬 Contribuciones de features (RandomForest)")
    if importance:
        df_imp = pd.DataFrame(importance, columns=["feature", "value"])
        chart = alt.Chart(df_imp).mark_bar(color="#34d399").encode(
            x=alt.X("value", title="Contribución"),
            y=alt.Y("feature", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip("value", format=".3f")],
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Sin metadata de importancia disponible para este modelo.")

with st.container():
    st.markdown("### 🧾 Comparativa heurística vs IA")
    df_compare = pd.DataFrame(
        {
            "Métrica": ["Rigidez", "Estanqueidad", "Energía", "Agua", "Crew"],
            "Heurística": [heur.rigidity, heur.tightness, heur.energy_kwh, heur.water_l, heur.crew_min],
            "IA Rex-AI": [props.rigidity, props.tightness, props.energy_kwh, props.water_l, props.crew_min],
        }
    )
    st.dataframe(df_compare.style.format({"Heurística": "{:.3f}", "IA Rex-AI": "{:.3f}"}), use_container_width=True)
    if comparisons:
        st.caption("Modelos secundarios (XGBoost / TabTransformer):")
        st.dataframe(pd.DataFrame(comparisons).T.style.format("{:.3f}"), use_container_width=True)

st.markdown("### 🎯 Score anatomy")
parts = score_breakdown(props, target, crew_time_low=target.get("crew_time_low", False))
chart_parts = alt.Chart(parts).mark_bar(color="#60a5fa").encode(
    x=alt.X("component", sort=None, title="Componente"),
    y=alt.Y("contribution", title="Aporte"),
    tooltip=["component", alt.Tooltip("contribution", format=".3f")],
).properties(height=280)
st.altair_chart(chart_parts, use_container_width=True)

with st.container():
    st.markdown("### 🛰️ Contexto y trazabilidad")
    context_data = _load_regolith_context()
    chips_html = ChipRow(
        [
            {
                "label": f"Seguridad: {safety['level']} · {safety['detail']}",
                "tone": "info",
            },
            {
                "label": f"Regolito MGS-1: {int(regolith_pct * 100)}%",
                "tone": "accent",
            },
            {
                "label": f"Entrenado: {metadata.get('trained_at', '—')}",
                "tone": "info",
            },
            {
                "label": f"Muestras: {metadata.get('n_samples', '—')}",
                "tone": "info",
            },
        ],
        render=False,
    )
    materials_text = ", ".join(materials) if materials else "—"
    ids_text = ", ".join(cand.get("source_ids", [])) or "—"
    latent_text = (
        ", ".join(f"{v:.2f}" for v in latent[:8]) if latent else "—"
    )
    GlassStack(
        cards=[
            GlassCard(
                title="Trazabilidad Rex-AI",
                body=(
                    f"{chips_html}"
                    f"<p style='margin-top:12px;'>Materiales: {materials_text}</p>"
                    f"<p>Fuente IDs NASA: {ids_text}</p>"
                    f"<p>Latent vector (autoencoder): {latent_text}</p>"
                ),
                icon="🛰️",
            )
        ],
        columns_min="22rem",
        density="cozy",
    ).render()
    src = getattr(props, "source", "heuristic")
    if src.startswith("rexai"):
        trained_at = metadata.get("trained_at", "?")
        st.caption(f"Predicciones por modelo Rex-AI (**{src}**, entrenado {trained_at}).")
    else:
        st.caption("Predicciones heurísticas basadas en reglas NASA.")

    st.markdown("#### 🔬 Señales de laboratorio (NASA ISRU)")
    gran_df = context_data["granulometry"]
    spectra_df = context_data["spectra"]
    thermal_bundle = context_data["thermal"]

    with layout_block("layout-grid layout-grid--dual layout-grid--flow", parent=None) as lab_grid:
        with layout_block("depth-stack layer-shadow", parent=lab_grid) as gran_panel:
            gran_panel.markdown("**Granulometría acumulada MGS-1**")
            if not gran_df.empty:
                gran_chart = (
                    alt.Chart(gran_df)
                    .mark_line(color="#34d399", interpolate="monotone", size=2)
                    .encode(
                        x=alt.X(
                            "diameter_microns:Q",
                            title="Tamaño de partícula (µm)",
                            scale=alt.Scale(reverse=True),
                        ),
                        y=alt.Y("cumulative_retained:Q", title="% acumulado retenido"),
                        tooltip=[
                            alt.Tooltip("diameter_microns:Q", title="Tamiz", format=".0f"),
                            alt.Tooltip("pct_retained:Q", title="% canal", format=".2f"),
                            alt.Tooltip("cumulative_retained:Q", title="% acumulado", format=".1f"),
                            alt.Tooltip("pct_passing:Q", title="% pasa", format=".1f"),
                        ],
                    )
                )
                gran_points = gran_chart.mark_circle(size=60, color="#22d3ee")
                gran_panel.altair_chart(gran_chart + gran_points, use_container_width=True)
                gran_panel.caption(
                    "Tooltip → granulometría fina = mayor estanqueidad; un tamiz grueso aporta rigidez estructural."
                )
            else:
                gran_panel.info("Sin datos granulométricos disponibles.")

        with layout_block("depth-stack layer-shadow", parent=lab_grid) as spectra_panel:
            spectra_panel.markdown("**Curva espectral VNIR**")
            if not spectra_df.empty:
                focus_label = "MGS-1 Prototype"
                focus_curve = spectra_df[spectra_df["sample"] == focus_label]
                others_curve = spectra_df[spectra_df["sample"] != focus_label]
                layers = []
                if not others_curve.empty:
                    base_chart = (
                        alt.Chart(others_curve)
                        .mark_line(color="#64748b", opacity=0.35)
                        .encode(
                            x=alt.X("wavelength_nm:Q", title="Longitud de onda (nm)"),
                            y=alt.Y("reflectance:Q", title="Reflectancia"),
                            tooltip=[
                                alt.Tooltip("sample:N", title="Muestra"),
                                alt.Tooltip("wavelength_nm:Q", title="λ", format=".0f"),
                                alt.Tooltip("reflectance_pct:Q", title="% reflectancia", format=".1f"),
                            ],
                        )
                    )
                    layers.append(base_chart)
                highlight = (
                    alt.Chart(focus_curve)
                    .mark_line(color="#38bdf8", size=2.4)
                    .encode(
                        x="wavelength_nm:Q",
                        y="reflectance:Q",
                        tooltip=[
                            alt.Tooltip("wavelength_nm:Q", title="λ", format=".0f"),
                            alt.Tooltip("reflectance_pct:Q", title="MGS-1 %", format=".1f"),
                        ],
                    )
                )
                layers.append(highlight)
                spectra_panel.altair_chart(alt.layer(*layers), use_container_width=True)
                spectra_panel.caption(
                    "La firma espectral azul resalta olivinos/hematitas: ayuda a anticipar rigidez y compuestos refractarios."
                )
            else:
                spectra_panel.info("Sin espectros VNIR cargados.")

    with layout_block("layout-grid layout-grid--dual layout-grid--flow", parent=None) as thermal_grid:
        with layout_block("depth-stack layer-shadow", parent=thermal_grid) as tg_panel:
            tg_panel.markdown("**TG · masa residual vs temperatura**")
            tg_df = thermal_bundle.tg_curve if thermal_bundle else pd.DataFrame()
            if isinstance(tg_df, pd.DataFrame) and not tg_df.empty:
                tg_chart = (
                    alt.Chart(tg_df)
                    .mark_line(color="#f97316", interpolate="monotone")
                    .encode(
                        x=alt.X("temperature_c:Q", title="Temperatura (°C)"),
                        y=alt.Y("mass_pct:Q", title="Masa residual (%)"),
                        tooltip=[
                            alt.Tooltip("temperature_c:Q", title="Temp", format=".0f"),
                            alt.Tooltip("mass_pct:Q", title="Masa %", format=".2f"),
                            alt.Tooltip("mass_loss_pct:Q", title="Pérdida %", format=".2f"),
                        ],
                    )
                    .properties(height=260)
                )
                tg_panel.altair_chart(tg_chart, use_container_width=True)
                tg_panel.caption("Picos de pérdida → secado y desgasificación; anticipa tiempos de horno y consumo energético.")
            else:
                tg_panel.info("Sin perfil TG de NASA disponible.")

        with layout_block("depth-stack layer-shadow", parent=thermal_grid) as ega_panel:
            ega_panel.markdown("**EGA · gases liberados**")
            ega_df = thermal_bundle.ega_long if thermal_bundle else pd.DataFrame()
            if isinstance(ega_df, pd.DataFrame) and not ega_df.empty:
                ega_chart = (
                    alt.Chart(ega_df)
                    .mark_line()
                    .encode(
                        x=alt.X("temperature_c:Q", title="Temperatura (°C)"),
                        y=alt.Y("signal_ppb:Q", title="Señal relativa (ppb eq.)"),
                        color=alt.Color("species_label:N", title="Gas"),
                        tooltip=[
                            alt.Tooltip("species_label:N", title="Gas"),
                            alt.Tooltip("temperature_c:Q", title="Temp", format=".0f"),
                            alt.Tooltip("signal_ppb:Q", title="Intensidad", format=".2f"),
                        ],
                    )
                    .properties(height=260)
                )
                ega_panel.altair_chart(ega_chart, use_container_width=True)
                ega_panel.caption("H₂O/CO₂ altos → vigilar porosidad y estanqueidad; SO₂ indica impurezas que afectan sellado.")
            else:
                ega_panel.info("Sin gases EGA registrados.")

    if regolith_pct > 0:
        st.success(
            f"Tu mezcla incorpora {regolith_pct*100:.0f}% de MGS-1: granos finos sellan juntas, pero vigila las liberaciones "
            "de H₂O/CO₂ para no perder estanqueidad."
        )
    else:
        st.caption(
            "Aunque tu candidato no usa regolito, estas curvas sirven como referencia ISRU para futuras iteraciones."
        )

st.markdown("### 📥 Export quick facts")
with layout_block("layout-grid layout-grid--dual layout-grid--flow", parent=None) as export_grid:
    with layout_block("depth-stack layer-shadow", parent=export_grid) as export_panel:
        export_panel.json(
            {
                "process": {"id": cand["process_id"], "name": cand["process_name"]},
                "materials": cand["materials"],
                "weights": cand.get("weights", []),
                "predictions": {
                    "rigidez": props.rigidity,
                    "estanqueidad": props.tightness,
                    "energy_kwh": props.energy_kwh,
                    "water_l": props.water_l,
                    "crew_min": props.crew_min,
                },
                "confidence_interval": ci,
                "uncertainty": uncertainty,
                "model_metadata": metadata,
                "score": score,
            },
        )
    with layout_block("side-panel layer-shadow", parent=export_grid) as safety_panel:
        badge = safety
        level = badge.get("level", "OK").lower()
        cls = "ok" if "ok" in level else ("risk" if "riesgo" in level or "risk" in level else "warn")
        safety_panel.markdown(
            f'<span class="pill {cls}">Seguridad: {badge.get("level", "OK")}</span>',
            unsafe_allow_html=True,
        )
        pop = safety_panel.popover("¿Qué chequeamos?")
        with pop:
            st.write(badge.get("detail", "Sin observaciones."))
            st.caption(
                "Validaciones: PFAS/microplásticos evitados, sin incineración, flags NASA (EVA/CTB, multilayers, nitrilo)."
            )

st.markdown("---")

# ======== KPIs con contexto ========
kpi_primary = [
    ("Score total", f"{score:.2f}", "Función + Recursos + Bono problemáticos"),
    ("Rigidez", f"{props.rigidity:.2f}", f"Objetivo: {float(target['rigidity']):.2f}"),
    ("Estanqueidad", f"{props.tightness:.2f}", f"Objetivo: {float(target['tightness']):.2f}"),
    ("Energía (kWh)", f"{props.energy_kwh:.2f}", f"Máx: {float(target['max_energy_kwh']):.2f}"),
    ("Agua (L)", f"{props.water_l:.2f}", f"Máx: {float(target['max_water_l']):.2f}"),
]
primary_html = "<div class='metric-grid fade-in'>" + "".join(
    f"<div class='kpi layer-shadow'><h3>{title}</h3><div class='v'>{value}</div><div class='hint'>{hint}</div></div>"
    for title, value, hint in kpi_primary
) + "</div>"
st.markdown(primary_html, unsafe_allow_html=True)

kpi_secondary = [
    ("Crew-time (min)", f"{props.crew_min:.0f}", f"Máx: {float(target['max_crew_min']):.0f}"),
    ("Masa final (kg)", f"{props.mass_final_kg:.2f}", "Post-proceso / mermas"),
    (
        "Modo",
        "Crew-time Low" if target.get("crew_time_low", False) else "Balanceado",
        "Más peso al tiempo de tripulación" if target.get("crew_time_low", False) else "Trade-off estándar",
    ),
]
secondary_html = "<div class='metric-grid fade-in'>" + "".join(
    f"<div class='kpi layer-shadow'><h3>{title}</h3><div class='v'>{value}</div><div class='hint'>{hint}</div></div>"
    for title, value, hint in kpi_secondary
) + "</div>"
st.markdown(secondary_html, unsafe_allow_html=True)

# ======== Tabs principales: (1) Score anatomy (2) Flujo Sankey (3) Checklist (4) Trazabilidad ========
tab1, tab2, tab3, tab4 = st.tabs(["🧩 Anatomía del Score", "🔀 Flujo del proceso (Sankey)", "🛠️ Checklist & Próximos pasos", "🛰️ Trazabilidad NASA"])

# --- TAB 1: Anatomía del Score ---
with tab1:
    st.markdown("## 🧩 Anatomía del Score <span class='sub'>(explicabilidad)</span>", unsafe_allow_html=True)
    parts = score_breakdown(props, target, crew_time_low=target.get("crew_time_low", False))
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

    labels = materials + [cand["process_name"], "Producto"]
    src = list(range(len(materials)))
    tgt = [len(materials)] * len(materials)
    # Si vienen pesos con regolito + redondeos, normalizamos para que la suma sea 1
    weights = cand.get("weights", [])
    if weights and abs(sum(weights) - 1.0) > 1e-6:
        s = sum(weights)
        weights = [w/s for w in weights]
    vals = [round(w*100, 1) for w in weights] if weights else [100/len(src)]*len(src)

    # proceso -> producto
    src += [len(materials)]
    tgt += [len(materials) + 1]
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
1. **Preparar/Triturar**: acondicionar materiales (**{', '.join(materials)}**).
2. **Ejecutar proceso**: **{cand['process_id']} {cand['process_name']}** con parámetros estándar del hábitat.
3. **Enfriar & post-proceso**: verificar bordes, ajuste y *fit*.
4. **Registrar feedback**: rigidez percibida, facilidad de uso, y problemas (bordes, olor, slip, etc.).
    """)

    st.markdown("### ⏱️ Recursos estimados")
    resource_html = "<div class='metric-grid fade-in'>" + "".join([
        f"<div class='kpi layer-shadow'><h3>Energía</h3><div class='v'>{props.energy_kwh:.2f} kWh</div></div>",
        f"<div class='kpi layer-shadow'><h3>Agua</h3><div class='v'>{props.water_l:.2f} L</div></div>",
        f"<div class='kpi layer-shadow'><h3>Crew-time</h3><div class='v'>{props.crew_min:.0f} min</div></div>",
    ]) + "</div>"
    st.markdown(resource_html, unsafe_allow_html=True)

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
    st.markdown("**IDs usados:** " + ", ".join(cand.get("source_ids", []) or ["—"]))
    st.markdown("**Categorías:** " + ", ".join(map(str, cand.get("source_categories", []) or ["—"])))
    st.markdown("**Flags:** " + ", ".join(map(str, cand.get("source_flags", []) or ["—"])))
    st.caption("Esto permite demostrar que estamos atacando pouches multilayer, espumas ZOTEK, EVA/CTB, nitrilo, etc.")
    feat = cand.get("features", {})
    if feat:
        feat_view = {
            "Masa total (kg)": feat.get("total_mass_kg"),
            "Densidad (kg/m³)": feat.get("density_kg_m3"),
            "Humedad": feat.get("moisture_frac"),
            "Dificultad": feat.get("difficulty_index"),
            "Recupero gas": feat.get("gas_recovery_index"),
            "Reuso logístico": feat.get("logistics_reuse_index"),
            "SiO₂ (regolito)": feat.get("oxide_sio2"),
            "FeOT (regolito)": feat.get("oxide_feot"),
        }
        st.markdown("**Features NASA/ML**")
        st.dataframe(pd.DataFrame([feat_view]), hide_index=True, use_container_width=True)
    latent_view = latent or feat.get("latent_vector")
    if latent_view:
        st.info(
            f"Vector latente Rex-AI de {len(latent_view)} dimensiones listo para clustering o búsqueda de recetas similares."
        )

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
