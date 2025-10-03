from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_sortables import sort_items

from app.modules.explain import compare_table, score_breakdown
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.ui_blocks import initialise_frontend, load_theme, pill
from app.modules.io import (
    MissingDatasetError,
    format_missing_dataset_message,
    load_waste_df,
)


def _generate_storytelling(
    df: pd.DataFrame, target_payload: dict, duel_annotations: list[dict[str, str]]
) -> list[str]:
    """Construye insights en lenguaje natural usando reglas heurísticas."""
    insights: list[str] = []
    if df.empty:
        return insights

    top = df.sort_values("Score", ascending=False).iloc[0]
    insights.append(
        f"🥇 **#{int(top['Opción'])}** domina el score con {top['Score']:.2f}, impulsado por {top['Proceso']}."
    )

    for metric in ["Agua (L)", "Energía (kWh)", "Crew (min)"]:
        if metric in df.columns:
            best_idx = df[metric].astype(float).idxmin()
            best_row = df.loc[best_idx]
            insights.append(
                f"🔎 En {metric.lower()}, la opción #{int(best_row['Opción'])} consume {best_row[metric]:.2f}, el menor del set."
            )

    if "Rigidez" in df.columns:
        rigidity_idx = df["Rigidez"].astype(float).idxmax()
        rigid_row = df.loc[rigidity_idx]
        insights.append(
            f"🧱 Si la misión prioriza rigidez, la opción #{int(rigid_row['Opción'])} alcanza {rigid_row['Rigidez']:.2f}."
        )

    if target_payload:
        limites = {
            "Energía (kWh)": float(target_payload.get("max_energy_kwh", 0)),
            "Agua (L)": float(target_payload.get("max_water_l", 0)),
            "Crew (min)": float(target_payload.get("max_crew_min", 0)),
        }
        for metric, limit in limites.items():
            if limit and metric in df.columns:
                peor = df[metric].astype(float).max()
                if peor > limit:
                    insights.append(
                        f"⚠️ Algunas opciones superan el límite de {metric} ({limit:.1f}). Ajustá recetas para bajar a {limit:.1f}."
                    )
                    break

    if duel_annotations:
        best = duel_annotations[0]
        insights.append(
            f"⚔️ En el duelo, destaca {best['metric']} con ventaja para {best['advantage']} ({best['diff_text']})."
        )

    return insights

POLYMER_NUMERIC_COLUMNS = (
    "pc_density_density_g_per_cm3",
    "pc_mechanics_tensile_strength_mpa",
    "pc_mechanics_modulus_gpa",
    "pc_thermal_glass_transition_c",
    "pc_ignition_ignition_temperature_c",
    "pc_ignition_burn_time_min",
)

POLYMER_LABEL_COLUMNS = (
    "pc_density_sample_label",
    "pc_mechanics_sample_label",
    "pc_thermal_sample_label",
    "pc_ignition_sample_label",
)

ALUMINIUM_NUMERIC_COLUMNS = (
    "aluminium_tensile_strength_mpa",
    "aluminium_yield_strength_mpa",
    "aluminium_elongation_pct",
)

ALUMINIUM_LABEL_COLUMNS = (
    "aluminium_processing_route",
    "aluminium_class_id",
)

POLYMER_LABEL_MAP = {
    "density_g_cm3": "ρ ref (g/cm³)",
    "tensile_mpa": "σₜ ref (MPa)",
    "modulus_gpa": "E ref (GPa)",
    "glass_c": "Tg (°C)",
    "ignition_c": "Ignición (°C)",
    "burn_min": "Burn (min)",
}

ALUMINIUM_LABEL_MAP = {
    "tensile_mpa": "σₜ Al (MPa)",
    "yield_mpa": "σᵧ Al (MPa)",
    "elongation_pct": "ε Al (%)",
}


def _collect_external_profiles(candidate: dict, inventory: pd.DataFrame) -> dict[str, dict[str, float]]:
    if not isinstance(candidate, dict) or inventory.empty:
        return {}

    ids = {str(value).strip() for value in candidate.get("source_ids", []) if str(value).strip()}
    if not ids:
        return {}

    mask = pd.Series(False, index=inventory.index)
    if "id" in inventory.columns:
        mask |= inventory["id"].astype(str).isin(ids)
    if "_source_id" in inventory.columns:
        mask |= inventory["_source_id"].astype(str).isin(ids)

    subset = inventory.loc[mask]
    if subset.empty:
        return {}

    payload: dict[str, dict[str, float]] = {}

    def _section(numeric_columns: tuple[str, ...]) -> dict[str, float]:
        relevant = [column for column in numeric_columns if column in subset.columns]
        metrics: dict[str, float] = {}
        for column in relevant:
            series = pd.to_numeric(subset[column], errors="coerce")
            if not series.notna().any():
                continue
            if column == "pc_density_density_g_per_cm3":
                metrics.setdefault("density_g_cm3", float(series.mean()))
            elif column == "pc_mechanics_tensile_strength_mpa":
                metrics.setdefault("tensile_mpa", float(series.mean()))
            elif column == "pc_mechanics_modulus_gpa":
                metrics.setdefault("modulus_gpa", float(series.mean()))
            elif column == "pc_thermal_glass_transition_c":
                metrics.setdefault("glass_c", float(series.mean()))
            elif column == "pc_ignition_ignition_temperature_c":
                metrics.setdefault("ignition_c", float(series.mean()))
            elif column == "pc_ignition_burn_time_min":
                metrics.setdefault("burn_min", float(series.mean()))
            elif column == "aluminium_tensile_strength_mpa":
                metrics.setdefault("tensile_mpa", float(series.mean()))
            elif column == "aluminium_yield_strength_mpa":
                metrics.setdefault("yield_mpa", float(series.mean()))
            elif column == "aluminium_elongation_pct":
                metrics.setdefault("elongation_pct", float(series.mean()))
        return metrics

    polymer_metrics = _section(POLYMER_NUMERIC_COLUMNS)
    if polymer_metrics:
        payload["polymer"] = polymer_metrics

    aluminium_metrics = _section(ALUMINIUM_NUMERIC_COLUMNS)
    if aluminium_metrics:
        payload["aluminium"] = aluminium_metrics

    return payload


def _format_reference_value(key: str, value: float) -> str:
    if key in {"density_g_cm3", "modulus_gpa"}:
        return f"{value:.2f}"
    if key == "burn_min":
        return f"{value:.1f}"
    return f"{value:.0f}"


# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Comparar & Explicar", page_icon="🧪", layout="wide")
initialise_frontend()
current_step = set_active_step("compare")

load_theme()

render_breadcrumbs(current_step)
# ======== estado requerido ========
cands  = st.session_state.get("candidates", [])
target = st.session_state.get("target", None)
if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

try:
    inventory_df = load_waste_df()
except MissingDatasetError as error:
    st.error(format_missing_dataset_message(error))
    st.stop()

st.title("🧪 Compare & Explain")
st.caption(
    "Compará candidatos como en un *design review*: qué rinde más, dónde gasta menos, y por qué elige la IA esa receta."
)

# ======== tabla comparativa base ========
df_base = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False))
# Aseguramos columnas esperadas y nombres amigables
expected_cols = ["Opción","Score","Proceso","Materiales","Energía (kWh)","Agua (L)","Crew (min)","Masa (kg)"]
for col in expected_cols:
    if col not in df_base.columns:
        # intentamos mapear por nombres aproximados si hiciera falta
        pass

df_base["ρ ref (g/cm³)"] = np.nan
df_base["σₜ ref (MPa)"] = np.nan
df_base["σₜ Al (MPa)"] = np.nan
df_base["σᵧ Al (MPa)"] = np.nan

reference_rows: list[dict[str, float]] = []
for idx, candidate in enumerate(cands, start=1):
    metrics = _collect_external_profiles(candidate, inventory_df)
    polymer_metrics = metrics.get("polymer", {})
    aluminium_metrics = metrics.get("aluminium", {})
    mask = df_base["Opción"] == idx
    if polymer_metrics:
        if "density_g_cm3" in polymer_metrics:
            df_base.loc[mask, "ρ ref (g/cm³)"] = polymer_metrics["density_g_cm3"]
        if "tensile_mpa" in polymer_metrics:
            df_base.loc[mask, "σₜ ref (MPa)"] = polymer_metrics["tensile_mpa"]
    if aluminium_metrics:
        if "tensile_mpa" in aluminium_metrics:
            df_base.loc[mask, "σₜ Al (MPa)"] = aluminium_metrics["tensile_mpa"]
        if "yield_mpa" in aluminium_metrics:
            df_base.loc[mask, "σᵧ Al (MPa)"] = aluminium_metrics["yield_mpa"]
    if polymer_metrics or aluminium_metrics:
        reference_rows.append({
            "Opción": idx,
            **{POLYMER_LABEL_MAP.get(key, key): value for key, value in polymer_metrics.items()},
            **{ALUMINIUM_LABEL_MAP.get(key, key): value for key, value in aluminium_metrics.items()},
        })

# ======== tabla comparativa base ========
st.subheader("📊 Tabla comparativa de candidatos")
st.caption("Visualizá el score junto a recursos y propiedades clave.")
st.dataframe(df_base.set_index("Opción"), use_container_width=True)

# Sección de métricas externas
if reference_rows:
    st.markdown("### 🔬 Métricas externas por candidato")
    reference_df = pd.DataFrame(reference_rows).set_index("Opción")
    st.dataframe(reference_df, use_container_width=True)

    scatter_poly = df_base.dropna(subset=["σₜ ref (MPa)", "Score"])
    if not scatter_poly.empty:
        fig_poly = px.scatter(
            scatter_poly,
            x="σₜ ref (MPa)",
            y="Score",
            size="ρ ref (g/cm³)",
            color="Proceso",
            hover_data=["Materiales"],
            title="Score vs. resistencia de polímeros",
        )
        st.plotly_chart(fig_poly, use_container_width=True)

    scatter_alu = df_base.dropna(subset=["σₜ Al (MPa)", "Score"])
    if not scatter_alu.empty:
        fig_alu = px.scatter(
            scatter_alu,
            x="σₜ Al (MPa)",
            y="Score",
            color="Proceso",
            hover_data=["Materiales"],
            title="Score vs. resistencia aluminio",
        )
        st.plotly_chart(fig_alu, use_container_width=True)

# KPIs generales
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Opciones generadas", len(cands))
    st.caption("Muestra suficiente para comparar")
with kpi_cols[1]:
    st.metric("Mejor Score", f"{df_base['Score'].max():.2f}")
    st.caption("Top actual")
with kpi_cols[2]:
    st.metric("Consumo mínimo de agua", f"{df_base['Agua (L)'].min():.2f} L")
    st.caption("Entre todas las opciones")
with kpi_cols[3]:
    st.metric("Energía mínima", f"{df_base['Energía (kWh)'].min():.2f} kWh")
    st.caption("Entre todas las opciones")

# ======== Panel Comparómetro interactivo ========
st.markdown("## 🧭 Comparómetro side-by-side")
st.caption("Arrastrá para priorizar candidatos y obtener visualizaciones con sombreado adaptativo.")

candidate_labels = [
    f"#{row.Opción} · {row.Proceso} · Score {row.Score:.2f}"
    for _, row in df_base.iterrows()
]
label_to_index = {label: i for i, label in enumerate(candidate_labels)}
sorted_labels = sort_items(
    candidate_labels,
    header="Arrastrá para elegir prioridad",
    direction="vertical",
    key="comparometer_sort_order",
)
top_labels = sorted_labels[:2] if sorted_labels else candidate_labels[:2]
selected_indices = [label_to_index.get(lbl, 0) for lbl in top_labels]

metric_config = [
    ("Score", "Score", True, "Mayor score = mejor balance global"),
    ("Rigidez", "Rigidez", True, "Más rigidez significa estructura robusta"),
    ("Estanqueidad", "Estanqueidad", True, "Más estanqueidad protege atmósfera interna"),
    ("Energía (kWh)", "Energía (kWh)", False, "Menos kWh libera capacidad energética"),
    ("Agua (L)", "Agua (L)", False, "Menos agua consumida facilita logística"),
    ("Crew (min)", "Crew (min)", False, "Menos minutos libera crew-time"),
    ("Masa (kg)", "Masa (kg)", False, "Menor masa reduce penalización de lanzamiento"),
]

metric_cols = [m[1] for m in metric_config if m[1] in df_base.columns]
if metric_cols:
    df_metrics = df_base[["Opción"] + metric_cols].copy()
else:
    df_metrics = pd.DataFrame({"Opción": df_base["Opción"]})

if metric_cols:
    norm_matrix = []
    for name, col, higher_is_better, _ in metric_config:
        if col not in df_metrics.columns:
            continue
        series = df_metrics[col].astype(float)
        if series.nunique(dropna=True) <= 1:
            norm = np.ones_like(series, dtype=float)
        else:
            min_v = series.min()
            max_v = series.max()
            span = max(max_v - min_v, 1e-6)
            norm = (series - min_v) / span
            if not higher_is_better:
                norm = 1 - norm
        norm_matrix.append(norm)

    norm_matrix = np.vstack(norm_matrix) if norm_matrix else np.empty((0, len(df_metrics)))
    heatmap_values = df_metrics[metric_cols].astype(float).values
    hover_text = []
    for row in df_metrics.itertuples():
        texts = []
        for name, col, higher_is_better, desc in metric_config:
            if col not in df_metrics.columns:
                continue
            attr_name = col.replace(" ", "_").replace("(", "").replace(")", "")
            value = getattr(row, attr_name, np.nan)
            base_series = df_metrics[col].astype(float)
            if higher_is_better:
                delta = value - base_series.mean()
                direction = "por encima" if delta >= 0 else "por debajo"
            else:
                delta = base_series.mean() - value
                direction = "ahorra" if delta >= 0 else "consume más"
            delta_text = f"{abs(delta):.2f}" if np.isfinite(delta) else "N/A"
            narrative = f"{desc}. Se ubica {direction} vs. promedio ({delta_text})."
            texts.append(
                f"<b>Opción {row.Opción}</b><br>{name}: {value:.2f}<br>{narrative}"
            )
        hover_text.append(texts)

    z_values = norm_matrix.T if norm_matrix.size else np.zeros_like(heatmap_values, dtype=float)

    fig_matrix = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=[cfg[0] for cfg in metric_config if cfg[1] in df_metrics.columns],
            y=[f"#{row.Opción}" for row in df_metrics.itertuples()],
            customdata=heatmap_values,
            text=np.round(heatmap_values, 2),
            texttemplate="%{text}",
            hoverinfo="text",
            hovertext=hover_text,
            colorscale="RdYlGn",
            reversescale=True,
            showscale=True,
            zmin=0,
            zmax=1,
        )
    )
    fig_matrix.update_layout(
        title="Matrix heatmap de desempeño",
        height=max(360, 160 + 28 * len(df_metrics)),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.caption("La escala aplica shading condicional: verde = desempeño competitivo, rojo = zona de riesgo.")
else:
    st.info("No se encontraron métricas cuantitativas para renderizar la heatmap.")

if selected_indices:
    comp_cols = st.columns(2)
    for slot, idx in enumerate(selected_indices[:2]):
        cand = cands[idx]
        col_slot = comp_cols[slot]
        with col_slot:
            st.markdown(f"### Candidato {'A' if slot == 0 else 'B'} — #{idx+1}")
            st.markdown(f"**Proceso:** {cand['process_id']} · {cand['process_name']}")
            st.markdown(f"**Score:** {cand['score']:.2f}")
            props = cand["props"]
            metric_vals = []
            heat_vals = []
            hover_vals = []
            labels = []
            for name, col, higher_is_better, desc in metric_config:
                if col not in df_metrics.columns:
                    continue
                labels.append(name)
                if col == "Score":
                    val = cand["score"]
                elif col == "Rigidez":
                    val = props.rigidity
                elif col == "Estanqueidad":
                    val = props.tightness
                elif col == "Energía (kWh)":
                    val = props.energy_kwh
                elif col == "Agua (L)":
                    val = props.water_l
                elif col == "Crew (min)":
                    val = props.crew_min
                elif col == "Masa (kg)":
                    val = getattr(props, "mass_final_kg", np.nan)
                else:
                    val = np.nan
                metric_vals.append(val)
                series = df_metrics[col].astype(float)
                min_v, max_v = series.min(), series.max()
                span = max(max_v - min_v, 1e-6)
                normalized = (val - min_v) / span
                if not higher_is_better:
                    normalized = 1 - normalized
                heat_vals.append(normalized)
                hover_vals.append(
                    f"{name}: {val:.2f} — {'mejor' if normalized >= 0.66 else ('alerta' if normalized <= 0.33 else 'estable')}"
                )

            fig_card = go.Figure(
                data=go.Heatmap(
                    z=[heat_vals],
                    x=labels,
                    y=["Desempeño"],
                    text=np.round(metric_vals, 2),
                    texttemplate="%{text}",
                    hoverinfo="text",
                    hovertext=[hover_vals],
                    colorscale=[[0, "#ff5f6d"], [0.5, "#f5f7fa"], [1, "#35d07f"]],
                    showscale=False,
                    zmin=0,
                    zmax=1,
                )
            )
            fig_card.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=180,
                title="Mapa de performance",
            )
            st.plotly_chart(fig_card, use_container_width=True)

            if heat_vals:
                st.progress(
                    min(max(float(np.nanmean(heat_vals)), 0.0), 1.0),
                    text="Índice holográfico de salud",
                )


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
duel_annotations: list[dict[str, str]] = []
tab1, tab2, tab3 = st.tabs(["🔍 Inspector de candidato", "⚔️ Duelo (A vs B)", "📖 Explicación didáctica"])

# --- TAB 1: Inspector de candidato ---
with tab1:
    st.subheader("🔍 Inspector de candidato")
    st.caption("Ver detalle y desglose del score para cada opción.")
    pick = st.number_input("Elegí la Opción #", min_value=1, max_value=len(cands), value=1, step=1)
    c = cands[int(pick)-1]

    # Resumen del candidato
    cL, cR = st.columns([1.2, 1.0])
    with cL:
        st.markdown(f"#### Candidato #{int(pick)} — {c['process_name']}")
        st.caption(f"Proceso {c['process_id']} · Materiales: {', '.join(c['materials'])}")

        p = c["props"]
        primary_metrics = st.columns(3)
        with primary_metrics[0]:
            st.metric("Score", f"{c['score']:.2f}")
        with primary_metrics[1]:
            st.metric("Rigidez", f"{p.rigidity:.2f}")
        with primary_metrics[2]:
            st.metric("Estanqueidad", f"{p.tightness:.2f}")

        resource_metrics = st.columns(3)
        with resource_metrics[0]:
            st.metric("Energía (kWh)", f"{p.energy_kwh:.2f}")
        with resource_metrics[1]:
            st.metric("Agua (L)", f"{p.water_l:.2f}")
        with resource_metrics[2]:
            st.metric("Crew (min)", f"{p.crew_min:.0f}")

        st.metric("Masa final (kg)", f"{p.mass_final_kg:.2f}")

        weights_series = pd.Series(c["weights"], name="Peso")
        weights_df = (
            weights_series.rename_axis("Factor").reset_index()
            if not weights_series.empty
            else pd.DataFrame({"Factor": [], "Peso": []})
        )
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

        # Pills de ajuste a límites
        pill_markup: list[str] = []
        limits = [
            ("Energía", p.energy_kwh, float(target.get("max_energy_kwh", 0) or 0), "kWh"),
            ("Agua", p.water_l, float(target.get("max_water_l", 0) or 0), "L"),
            ("Crew", p.crew_min, float(target.get("max_crew_min", 0) or 0), "min"),
        ]
        for label, value, limit, unit in limits:
            if limit <= 0:
                continue
            within_limit = value <= limit
            kind = "ok" if within_limit else "risk"
            pill_markup.append(
                pill(
                    f"{label}: {value:.2f}/{limit:.2f} {unit}",
                    kind=kind,
                    render=False,
                )
            )
        if pill_markup:
            st.markdown(" ".join(pill_markup), unsafe_allow_html=True)

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

    Ap, Bp = A["props"], B["props"]

    duel_metrics = [
        ("Score", A["score"], B["score"], True, "⭐ Equilibrio global"),
        ("Energía (kWh)", Ap.energy_kwh, Bp.energy_kwh, False, "⚡ Demanda energética"),
        ("Agua (L)", Ap.water_l, Bp.water_l, False, "💧 Consumo hídrico"),
        ("Crew (min)", Ap.crew_min, Bp.crew_min, False, "👩‍🚀 Minutos tripulación"),
        ("Rigidez", Ap.rigidity, Bp.rigidity, True, "🧱 Resistencia estructural"),
        ("Estanqueidad", Ap.tightness, Bp.tightness, True, "🧴 Sellado"),
        ("Masa (kg)", getattr(Ap, "mass_final_kg", np.nan), getattr(Bp, "mass_final_kg", np.nan), False, "🚀 Impacto de masa"),
    ]

    duel_df = pd.DataFrame([
        {
            "Métrica": name,
            "A": val_a,
            "B": val_b,
            "Mayor_es_mejor": higher,
            "Narrativa": narrative,
        }
        for name, val_a, val_b, higher, narrative in duel_metrics
        if np.isfinite(val_a) and np.isfinite(val_b)
    ])

    holographic_palette = ["#08f7fe", "#fe53bb"]

    frames = []
    annotations = []
    for row in duel_df.itertuples():
        if row.A == 0 and row.B == 0:
            pct_diff = 0.0
        else:
            baseline = row.A if row.A != 0 else 1e-6
            pct_diff = ((row.B - row.A) / baseline) * 100
        advantage = "B" if ((row.B > row.A) == row.Mayor_es_mejor) else "A"
        diff_text = f"{pct_diff:+.1f}% vs A" if advantage == "B" else f"{(-pct_diff):+.1f}% vs B"
        annotations.append({
            "metric": row.Métrica,
            "advantage": advantage,
            "diff_text": diff_text,
            "narrative": row.Narrativa,
        })
        frames.append(
            go.Frame(
                name=row.Métrica,
                data=[
                    go.Bar(
                        x=["A", "B"],
                        y=[row.A, row.B],
                        marker=dict(color=holographic_palette, line=dict(color="#0c0c2d", width=1.5)),
                        text=[f"{row.A:.2f}", f"{row.B:.2f}"],
                        textposition="outside",
                        hovertemplate=(
                            "%{x} → %{y:.2f}<br>" + row.Narrativa + "<extra>" + row.Métrica + "</extra>"
                        ),
                    )
                ],
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=0.5,
                            y=max(row.A, row.B) * 1.15 if max(row.A, row.B) != 0 else 1,
                            xref="paper",
                            yref="y",
                            text=f"{row.Métrica}: ventaja {advantage} ({diff_text})",
                            showarrow=False,
                            font=dict(color="#d4f1f4", size=16, family="Space Mono"),
                        )
                    ]
                ),
            )
        )

    initial_frame = frames[0] if frames else None
    base_data = initial_frame.data if initial_frame else []

    duel_annotations = annotations

    duel_fig = go.Figure(
        data=base_data,
        frames=frames,
        layout=go.Layout(
            template="plotly_dark",
            title="Duelo holográfico — animación métrica a métrica",
            xaxis=dict(title="Candidato", showgrid=False),
            yaxis=dict(title="Valor", showgrid=True, gridcolor="#23395d"),
            paper_bgcolor="#060613",
            plot_bgcolor="rgba(6,6,19,0.95)",
            margin=dict(l=40, r=20, t=80, b=40),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="▶️ Reproducir", method="animate", args=[[frame.name for frame in frames], {"frame": {"duration": 900, "redraw": True}, "fromcurrent": True}]),
                        dict(label="⏸ Pausa", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
                    ],
                    x=0.02,
                    y=1.2,
                    bgcolor="#111636",
                    bordercolor="#08f7fe",
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.05,
                    y=1.08,
                    len=0.9,
                    currentvalue=dict(prefix="Métrica: ", font=dict(color="#08f7fe", size=14)),
                    steps=[
                        dict(
                            label=frame.name,
                            method="animate",
                            args=[[frame.name], {"frame": {"duration": 600, "redraw": True}, "mode": "immediate"}],
                        )
                        for frame in frames
                    ],
                )
            ],
        ),
    )

    st.plotly_chart(duel_fig, use_container_width=True)

    if annotations:
        best = annotations[0]
        st.success(
            f"**{best['metric']}**: ventaja para **{best['advantage']}** ({best['diff_text']}). {best['narrative']}."
        )

    st.dataframe(
        duel_df[["Métrica", "A", "B"]].assign(
            Diferencia=lambda d: d["B"] - d["A"],
            Diferencia_pct=lambda d: np.where(d["A"] == 0, np.nan, ((d["B"] - d["A"]) / d["A"]) * 100),
        ),
        use_container_width=True,
        hide_index=True,
    )

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

# ======== Módulo de storytelling ========
st.markdown("## 🧠 Storytelling asistido por IA (beta)")
story_toggle = st.toggle("Activar narrativas automáticas", value=True)
if story_toggle:
    insights = _generate_storytelling(df_base, target, duel_annotations)
    if insights:
        st.markdown("\n".join(f"- {text}" for text in insights))
    else:
        st.info("No se encontraron insights adicionales para narrar.")
else:
    st.caption("Activalo para recibir insights automatizados y atajos de decisión.")
