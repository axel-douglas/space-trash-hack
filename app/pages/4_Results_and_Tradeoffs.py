import sys
from pathlib import Path

if not __package__:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.modules.data_sources import (
    load_regolith_granulometry,
    load_regolith_spectral_curves,
    load_regolith_thermal_profiles,
)
from app.modules.explain import score_breakdown
from app.modules.io import (
    MissingDatasetError,
    format_missing_dataset_message,
    load_waste_df,
)
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.page_data import build_candidate_metric_table, build_resource_table
from app.modules.schema import (
    ALUMINIUM_LABEL_COLUMNS,
    ALUMINIUM_LABEL_MAP,
    ALUMINIUM_NUMERIC_COLUMNS,
    POLYMER_LABEL_COLUMNS,
    POLYMER_LABEL_MAP,
    POLYMER_METRIC_COLUMNS,
    numeric_series,
)
from app.modules.ui_blocks import (
    configure_page,
    initialise_frontend,
    layout_block,
    render_brand_header,
)

configure_page(page_title="Rex-AI • Resultados", page_icon="📊")
initialise_frontend()

current_step = set_active_step("results")

render_brand_header()

render_breadcrumbs(current_step)

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
process_id = cand.get("process_id", "—")
process_name = cand.get("process_name", "Proceso")
try:
    inventory_df = load_waste_df()
except MissingDatasetError as error:
    st.error(format_missing_dataset_message(error))
    st.stop()
polymer_density_distribution = numeric_series(
    inventory_df, "pc_density_density_g_per_cm3"
)
polymer_tensile_distribution = numeric_series(
    inventory_df, "pc_mechanics_tensile_strength_mpa"
)
aluminium_tensile_distribution = numeric_series(
    inventory_df, "aluminium_tensile_strength_mpa"
)
aluminium_yield_distribution = numeric_series(
    inventory_df, "aluminium_yield_strength_mpa"
)


def _get_value(source, attr, default=0.0):
    if source is None:
        return default
    if hasattr(source, attr):
        return getattr(source, attr)
    if isinstance(source, dict):
        return source.get(attr, default)
    return default
def _collect_external_profiles(candidate: dict, inventory: pd.DataFrame) -> dict[str, dict[str, object]]:
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

    payload: dict[str, dict[str, object]] = {}

    def _section(numeric_columns: tuple[str, ...], label_columns: tuple[str, ...]) -> dict[str, object] | None:
        relevant = [column for column in numeric_columns if column in subset.columns]
        if not relevant:
            return None
        numeric_df = subset[relevant].apply(pd.to_numeric, errors="coerce")
        mask_numeric = numeric_df.notna().any(axis=1)
        if not mask_numeric.any():
            return None
        rows = subset.loc[mask_numeric]
        metrics: dict[str, float] = {}
        for column in relevant:
            series = pd.to_numeric(rows[column], errors="coerce")
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

        if not metrics:
            return None

        labels: list[str] = []
        for column in label_columns:
            if column not in rows.columns:
                continue
            series = rows[column].dropna().astype(str).str.strip().replace("", pd.NA).dropna()
            labels.extend(series.tolist())

        return {"metrics": metrics, "labels": sorted(dict.fromkeys(labels))}

    polymer = _section(POLYMER_METRIC_COLUMNS, POLYMER_LABEL_COLUMNS)
    if polymer:
        payload["polymer"] = polymer

    aluminium = _section(ALUMINIUM_NUMERIC_COLUMNS, ALUMINIUM_LABEL_COLUMNS)
    if aluminium:
        payload["aluminium"] = aluminium

    return payload


def _format_reference_value(key: str, value: float) -> str:
    if key in {"density_g_cm3", "modulus_gpa"}:
        return f"{value:.2f}"
    if key == "burn_min":
        return f"{value:.1f}"
    return f"{value:.0f}"


@st.cache_data(show_spinner=False)
def _load_regolith_context():
    return {
        "granulometry": load_regolith_granulometry(),
        "spectra": load_regolith_spectral_curves(),
        "thermal": load_regolith_thermal_profiles(),
    }

header_chips = [
    {"label": f"Score {score:.3f}", "tone": "accent"},
    {"label": f"Target: {target.get('name', '—')}", "tone": "info"},
]
if target.get("crew_time_low"):
    header_chips.append({"label": "Crew-time low", "tone": "caution"})
if ci:
    header_chips.append({"label": "CI 95% activo", "tone": "success"})

st.title(f"📊 {process_id} · {process_name}")
st.caption(
    "Predicciones Rex-AI con trazabilidad NASA y contraste frente a la heurística de referencia."
)

if header_chips:
    chip_columns = st.columns(len(header_chips))
    for column, chip in zip(chip_columns, header_chips, strict=False):
        column.markdown(f"**{chip['label']}**")

metrics_df = build_candidate_metric_table(props, heur, score, ci, uncertainty)
st.subheader("Métricas clave")
metrics_style = metrics_df.style.format(
    {
        "IA Rex-AI": "{:.3f}",
        "Heurística": "{:.3f}",
        "Δ (IA - Heurística)": "{:+.3f}",
        "σ": "{:.3f}",
    }
)
try:
    metrics_style = metrics_style.hide(axis="index")
except AttributeError:
    metrics_style = metrics_style.hide_index()
st.dataframe(metrics_style, use_container_width=True)

comparison_df = metrics_df[metrics_df["Indicador"] != "Score total"].copy()
comparison_df = comparison_df.melt(
    id_vars=["Indicador"],
    value_vars=["IA Rex-AI", "Heurística"],
    var_name="Modelo",
    value_name="Valor",
).dropna(subset=["Valor"])
if not comparison_df.empty:
    chart = alt.Chart(comparison_df).mark_bar().encode(
        x=alt.X("Valor:Q", title="Valor"),
        y=alt.Y("Indicador:N", sort="-x"),
        color=alt.Color("Modelo:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=["Indicador", "Modelo", alt.Tooltip("Valor", format=".3f")],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

resource_df = build_resource_table(props, target)
st.subheader("Consumos frente a límites")
resource_style = resource_df.style.format(
    {
        "Uso": "{:.3f}",
        "Límite": "{:.3f}",
        "Utilización (%)": "{:.1f}",
    }
)
try:
    resource_style = resource_style.hide(axis="index")
except AttributeError:
    resource_style = resource_style.hide_index()
st.dataframe(resource_style, use_container_width=True)

util_plot = resource_df.dropna(subset=["Utilización (%)"])
if not util_plot.empty:
    domain_max = float(util_plot["Utilización (%)"].max()) + 10.0
    domain_max = max(120.0, domain_max)
    utilisation_chart = alt.Chart(util_plot).mark_bar(color="#0ea5e9").encode(
        x=alt.X(
            "Utilización (%):Q",
            title="Utilización (%)",
            scale=alt.Scale(domain=(0, domain_max)),
        ),
        y=alt.Y("Recurso:N", sort="-x"),
        tooltip=[
            "Recurso",
            alt.Tooltip("Utilización (%)", format=".1f"),
            alt.Tooltip("Uso", format=".3f"),
            alt.Tooltip("Límite", format=".3f"),
        ],
    ).properties(height=200)
    st.altair_chart(utilisation_chart, use_container_width=True)

external_profiles = _collect_external_profiles(cand, inventory_df)
if external_profiles:
    st.markdown("### 🧪 Propiedades externas de referencia")

    polymer_section = external_profiles.get("polymer")
    if polymer_section:
        polymer_labels = polymer_section.get("labels") or []
        if polymer_labels:
            st.caption("Polímeros fuente: " + ", ".join(polymer_labels))

        polymer_metrics = polymer_section.get("metrics", {})
        if polymer_metrics:
            metric_columns = st.columns(len(polymer_metrics))
            for column, (metric_key, metric_value) in zip(
                metric_columns, polymer_metrics.items(), strict=False
            ):
                label = POLYMER_LABEL_MAP.get(metric_key, metric_key)
                column.metric(label, _format_reference_value(metric_key, float(metric_value)))

            density_value = polymer_metrics.get("density_g_cm3")
            tensile_value = polymer_metrics.get("tensile_mpa")
            chart_cols = st.columns(2)
            density_area, tensile_area = chart_cols
            if polymer_density_distribution.empty:
                density_area.info(
                    "No hay densidades de polímeros en el inventario actual para comparar."
                )
            elif density_value:
                base = alt.Chart(
                    pd.DataFrame({"density": polymer_density_distribution})
                ).mark_bar(color="#22d3ee", opacity=0.55).encode(
                    x=alt.X("density:Q", bin=alt.Bin(maxbins=18), title="Densidad inventario (g/cm³)"),
                    y=alt.Y("count()", title="Ítems"),
                )
                rule = alt.Chart(pd.DataFrame({"density": [density_value]})).mark_rule(
                    color="#f97316", size=3
                ).encode(x="density:Q")
                density_area.altair_chart(base + rule, use_container_width=True)

            if polymer_tensile_distribution.empty:
                tensile_area.info(
                    "No hay datos de resistencia a tracción de polímeros en el inventario actual."
                )
            elif tensile_value:
                base = alt.Chart(
                    pd.DataFrame({"tensile": polymer_tensile_distribution})
                ).mark_bar(color="#f472b6", opacity=0.55).encode(
                    x=alt.X("tensile:Q", bin=alt.Bin(maxbins=18), title="σₜ inventario (MPa)"),
                    y=alt.Y("count()", title="Ítems"),
                )
                rule = alt.Chart(pd.DataFrame({"tensile": [tensile_value]})).mark_rule(
                    color="#f97316", size=3
                ).encode(x="tensile:Q")
                tensile_area.altair_chart(base + rule, use_container_width=True)

    aluminium_section = external_profiles.get("aluminium")
    if aluminium_section:
        aluminium_labels = aluminium_section.get("labels") or []
        if aluminium_labels:
            st.caption("Aleaciones/Procesos: " + ", ".join(aluminium_labels))

        aluminium_metrics = aluminium_section.get("metrics", {})
        if aluminium_metrics:
            metric_columns = st.columns(len(aluminium_metrics))
            for column, (metric_key, metric_value) in zip(
                metric_columns, aluminium_metrics.items(), strict=False
            ):
                label = ALUMINIUM_LABEL_MAP.get(metric_key, metric_key)
                column.metric(label, _format_reference_value(metric_key, float(metric_value)))

            tensile_value = aluminium_metrics.get("tensile_mpa")
            yield_value = aluminium_metrics.get("yield_mpa")
            chart_cols = st.columns(2)
            tensile_area, yield_area = chart_cols
            if aluminium_tensile_distribution.empty:
                tensile_area.info(
                    "No hay datos de tracción de aluminio en el inventario actual para comparar."
                )
            elif tensile_value:
                base = alt.Chart(
                    pd.DataFrame({"tensile": aluminium_tensile_distribution})
                ).mark_bar(color="#f97316", opacity=0.55).encode(
                    x=alt.X("tensile:Q", bin=alt.Bin(maxbins=18), title="σₜ inventario (MPa)"),
                    y=alt.Y("count()", title="Ítems"),
                )
                rule = alt.Chart(pd.DataFrame({"tensile": [tensile_value]})).mark_rule(
                    color="#22d3ee", size=3
                ).encode(x="tensile:Q")
                tensile_area.altair_chart(base + rule, use_container_width=True)

            if aluminium_yield_distribution.empty:
                yield_area.info(
                    "No hay datos de límite de fluencia de aluminio en el inventario actual."
                )
            elif yield_value:
                base = alt.Chart(
                    pd.DataFrame({"yield_strength": aluminium_yield_distribution})
                ).mark_bar(color="#fb923c", opacity=0.55).encode(
                    x=alt.X("yield_strength:Q", bin=alt.Bin(maxbins=18), title="σᵧ inventario (MPa)"),
                    y=alt.Y("count()", title="Ítems"),
                )
                rule = alt.Chart(pd.DataFrame({"yield_strength": [yield_value]})).mark_rule(
                    color="#22d3ee", size=3
                ).encode(x="yield_strength:Q")
                yield_area.altair_chart(base + rule, use_container_width=True)

    st.caption(
        "Comparativa con laboratorios/industria (`polymer_composite_*`, `aluminium_alloys.csv`)."
    )

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
    compare_rows = [
        ("Rigidez", "rigidity"),
        ("Estanqueidad", "tightness"),
        ("Energía", "energy_kwh"),
        ("Agua", "water_l"),
        ("Crew", "crew_min"),
    ]
    df_compare = pd.DataFrame(
        {
            "Métrica": [label for label, _ in compare_rows],
            "Heurística": [_get_value(heur, attr, float("nan")) for _, attr in compare_rows],
            "IA Rex-AI": [_get_value(props, attr, float("nan")) for _, attr in compare_rows],
        }
    )
    st.dataframe(df_compare.style.format({"Heurística": "{:.3f}", "IA Rex-AI": "{:.3f}"}), use_container_width=True)
    if comparisons:
        st.caption("Modelos secundarios (XGBoost / TabTransformer):")
        st.dataframe(pd.DataFrame(comparisons).T.style.format("{:.3f}"), use_container_width=True)

with st.expander("🛰️ Contexto y trazabilidad", expanded=True):
    context_data = _load_regolith_context()
    materials_text = ", ".join(materials) if materials else "—"
    ids_text = ", ".join(cand.get("source_ids", [])) or "—"
    latent_text = ", ".join(f"{v:.2f}" for v in latent[:8]) if latent else "—"

    st.markdown("**Resumen operativo**")
    regolith_label = "<abbr title='Martian Global Simulant 1; referencia granulométrica y química para ISRU'>MGS-1</abbr>"
    autoencoder_label = "<abbr title='Red autoencoder: compresión no supervisada que captura la firma multivariable de la receta'>autoencoder</abbr>"
    summary_lines = [
        (
            f"- **Seguridad operacional**: nivel {safety['level']} · {safety['detail']}. "
            "Úsalo para priorizar mitigaciones y definir ventanas de trabajo."
        ),
        (
            f"- **Fracción de regolito {regolith_label}**: {int(regolith_pct * 100)}% de la mezcla base. "
            "Ajusta con esta métrica los ensayos de validación y la logística de tamizado."
        ),
        (
            f"- **Entrenamiento del modelo Rex-AI**: {metadata.get('trained_at', '—')} (vigencia del dataset). "
            "Consulta esta fecha antes de certificar cambios de proceso."
        ),
        (
            f"- **Cobertura de muestras**: {metadata.get('n_samples', '—')} registros analizados. "
            "Úsalo para estimar la robustez estadística y planificar nuevos muestreos."
        ),
    ]
    st.markdown("\n".join(summary_lines), unsafe_allow_html=True)
    st.markdown(
        f"**Materiales mezclados:** {materials_text} · Empléalos como check-list para requisición y trazabilidad interna."
    )
    st.markdown(
        f"**Fuente IDs NASA:** {ids_text} · Referencia cruzada para auditorías y revisiones de configuración."
    )
    st.markdown(
        f"**Vector latente ({autoencoder_label})**: {latent_text} · Interpreta la proximidad a casos previos antes de aprobar extrapolaciones.",
        unsafe_allow_html=True,
    )
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

    st.caption(
        "Nota operativa: confronta estas curvas con la recepción de lotes (granulometría, espectros y perfiles térmicos)"
        " para validar parámetros críticos antes de ajustar recetas o escalado."
    )

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

with st.expander("📥 Export quick facts"):
    with layout_block("layout-grid layout-grid--dual layout-grid--flow", parent=None) as export_grid:
        with layout_block("depth-stack layer-shadow", parent=export_grid) as export_panel:
            export_panel.json(
                {
                    "process": {
                        "id": cand.get("process_id", "—"),
                        "name": cand.get("process_name", "—"),
                    },
                    "materials": materials,
                    "weights": cand.get("weights", []),
                    "predictions": {
                        "rigidez": _get_value(props, "rigidity", None),
                        "estanqueidad": _get_value(props, "tightness", None),
                        "energy_kwh": _get_value(props, "energy_kwh", None),
                        "water_l": _get_value(props, "water_l", None),
                        "crew_min": _get_value(props, "crew_min", None),
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

with st.expander("🎯 Objetivos y límites de misión"):
    st.markdown(
        """
        - **Rigidez objetivo:** {rig_obj:.2f}
        - **Estanqueidad objetivo:** {tight_obj:.2f}
        - **Límites recursos:** energía ≤ {energy_max:.2f} kWh · agua ≤ {water_max:.2f} L · crew ≤ {crew_max:.0f} min
        - **Modo:** {mode_label}
        """.format(
            rig_obj=float(target["rigidity"]),
            tight_obj=float(target["tightness"]),
            energy_max=float(target["max_energy_kwh"]),
            water_max=float(target["max_water_l"]),
            crew_max=float(target["max_crew_min"]),
            mode_label="Crew-time Low" if target.get("crew_time_low", False) else "Balanceado",
        )
    )

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

    scenario_label = str(target.get("scenario") or "").strip()
    product_label = str(target.get("name") or "").strip() or "Producto"
    product_node_label = f"{scenario_label} · {product_label}" if scenario_label else product_label

    material_labels = [str(item) for item in materials]
    labels = material_labels + [process_name, product_node_label]
    src = list(range(len(material_labels)))
    tgt = [len(material_labels)] * len(material_labels)

    weights = (cand.get("weights") or [])[: len(material_labels)]
    if len(weights) < len(material_labels):
        weights = [*weights, *([0.0] * (len(material_labels) - len(weights)))]
    if weights:
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1 / len(material_labels)] * len(material_labels) if material_labels else []
        material_vals = [round(w * 100, 1) for w in weights]
    else:
        material_vals = [round(100 / len(material_labels), 1)] * len(material_labels) if material_labels else []

    vals = material_vals.copy()

    src.append(len(material_labels))
    tgt.append(len(material_labels) + 1)
    vals.append(100.0)

    highlight_styles = {
        "zotek": {
            "color": "#f97316",
            "hover": "♻️ ZOTEK F30 compactada: libera volumen útil y mantiene aislamiento térmico.",
        },
        "carbon": {
            "color": "#6366f1",
            "hover": "🧪 Carbono recuperado: refuerza juntas y captura compuestos volátiles.",
        },
    }


    def _material_kind(label: str) -> str | None:
        text = label.casefold()
        if "zotek" in text:
            return "zotek"
        if "carbon" in text or "carbón" in text or "carbono" in text:
            return "carbon"
        return None


    def _with_alpha(color: str, alpha: float) -> str:
        if color.startswith("rgba"):
            return color
        value = color.lstrip("#")
        if len(value) != 6:
            return color
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"


    scenario_key = scenario_label.casefold()
    scenario_hover_map = {
        "residence renovations": "🏠 Panel modular: maximizamos volumen habitable con espumas laminadas.",
        "daring discoveries": "🔬 Junta conductiva: el carbono recuperado asegura sellos para experimentos.",
        "cosmic celebrations": "🎉 Decor utilitaria: textiles seguros mantienen la moral de la tripulación.",
    }
    scenario_value_hover = scenario_hover_map.get(
        scenario_key,
        ("Valorizamos los residuos críticos en el producto final" if scenario_label else "Producto final valorizado"),
    )

    node_colors: list[str] = []
    node_hovertemplates: list[str] = []
    link_colors: list[str] = []
    link_customdata: list[str] = []
    link_hovertemplates: list[str] = []

    default_material_color = "#94a3b8"
    for label, value in zip(material_labels, material_vals):
        kind = _material_kind(label)
        if kind:
            style = highlight_styles[kind]
            color = style["color"]
            detail = f"{style['hover']}<br>Peso: {value:.1f}% del mix."
        else:
            color = default_material_color
            detail = f"Peso: {value:.1f}% del mix de residuos."
        node_colors.append(color)
        node_hovertemplates.append(f"{label}<br>{detail}<extra></extra>")
        link_colors.append(_with_alpha(color, 0.6))
        link_message = f"{label} → {process_name}<br>{detail}"
        link_customdata.append(link_message)
        link_hovertemplates.append("%{customdata}<extra></extra>")

    process_color = "#2dd4bf"
    node_colors.append(process_color)
    node_hovertemplates.append(
        f"{process_name}<br>Proceso priorizado por score Rex-AI.<extra></extra>"
    )

    product_color = "#4ade80"
    node_colors.append(product_color)
    node_hovertemplates.append(f"{product_node_label}<br>{scenario_value_hover}<extra></extra>")

    link_colors.append(_with_alpha(product_color, 0.55))
    link_customdata.append(
        f"{process_name} → {product_node_label}<br>{scenario_value_hover}"
    )
    link_hovertemplates.append("%{customdata}<extra></extra>")

    scenario_caption_map = {
        "residence renovations": (
            "Escenario Residence Renovations: comprimimos ZOTEK y films para paneles que liberan espacio vital."
        ),
        "daring discoveries": (
            "Escenario Daring Discoveries: el carbono recuperado alimenta juntas conductivas para experimentos."
        ),
        "cosmic celebrations": (
            "Escenario Cosmic Celebrations: textiles y films se transforman en utilería segura para la tripulación."
        ),
    }
    scenario_caption = scenario_caption_map.get(
        scenario_key,
        (
            f"Escenario {scenario_label}: transformamos los residuos en {product_label}."
            if scenario_label
            else f"El flujo muestra cómo obtenemos {product_label} a partir de los residuos seleccionados."
        ),
    )

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=20,
                    thickness=18,
                    color=node_colors,
                    hovertemplate=node_hovertemplates,
                ),
                link=dict(
                    source=src,
                    target=tgt,
                    value=vals,
                    color=link_colors,
                    customdata=link_customdata,
                    hovertemplate=link_hovertemplates,
                ),
            )
        ]
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(scenario_caption)

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
    materials_display = ", ".join(materials) if materials else "—"
    st.markdown(
        f"""
1. **Preparar/Triturar**: acondicionar materiales (**{materials_display}**).
2. **Ejecutar proceso**: **{process_id} {process_name}** con parámetros estándar del hábitat.
3. **Enfriar & post-proceso**: verificar bordes, ajuste y *fit*.
4. **Registrar feedback**: rigidez percibida, facilidad de uso, y problemas (bordes, olor, slip, etc.).
        """
    )

    st.markdown("### ⏱️ Recursos estimados")
    resource_entries = [
        ("⚡ Energía (kWh)", _get_value(props, "energy_kwh", float("nan")), "{:.2f}"),
        ("🚰 Agua (L)", _get_value(props, "water_l", float("nan")), "{:.2f}"),
        ("🧑‍🚀 Crew-time (min)", _get_value(props, "crew_min", float("nan")), "{:.0f}"),
    ]
    resource_cols = st.columns(len(resource_entries))
    for column, (label, value, fmt) in zip(resource_cols, resource_entries, strict=False):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            column.metric(label, "—")
        else:
            column.metric(label, fmt.format(value))

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
