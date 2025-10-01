import _bootstrap  # noqa: F401

import altair as alt
import pandas as pd
import streamlit as st

from app.modules.candidate_showroom import render_candidate_showroom
from app.modules.generator import generate_candidates
from app.modules.io import load_waste_df, load_process_df  # si tu IO usa load_process_catalog, cámbialo aquí
from app.modules.ml_models import get_model_registry
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.process_planner import choose_process
from app.modules.safety import check_safety, safety_badge
from app.modules.ui_blocks import (
    badge_group,
    layout_block,
    layout_stack,
    load_theme,
    micro_divider,
    minimal_button,
)
from app.modules.luxe_components import ChipRow, MetricSpec, RankingCockpit
from app.modules.visualizations import ConvergenceScene

st.set_page_config(page_title="Rex-AI • Generador", page_icon="🤖", layout="wide")

set_active_step("generator")

load_theme()

render_breadcrumbs("generator")

# ----------------------------- Helpers -----------------------------
TARGET_DISPLAY = {
    "rigidez": "Rigidez",
    "estanqueidad": "Estanqueidad",
    "energy_kwh": "Energía (kWh)",
    "water_l": "Agua (L)",
    "crew_min": "Crew (min)",
}


def _format_label_summary(summary: dict[str, dict[str, float]] | None) -> str:
    if not summary:
        return ""

    parts: list[str] = []
    for source, stats in summary.items():
        if not isinstance(stats, dict):
            parts.append(str(source))
            continue

        label = str(source)
        count = stats.get("count")
        mean_weight = stats.get("mean_weight")

        fragment = label
        try:
            if count is not None:
                fragment = f"{label}×{int(count)}"
        except (TypeError, ValueError):
            fragment = label

        try:
            if mean_weight is not None:
                fragment = f"{fragment} (w≈{float(mean_weight):.2f})"
        except (TypeError, ValueError):
            pass

        parts.append(fragment)

    return " · ".join(parts)

# ----------------------------- Encabezado -----------------------------
st.header("Generador asistido por IA")
badge_group(
    (
        "RandomForest + XGBoost (alternativo)",
        "Confianza 95%",
        "Comparación heurística vs IA",
        "Trazabilidad NASA + MGS-1",
    )
)
st.caption(
    "Rex-AI combina residuos NASA, ejecuta Ax + BoTorch y muestra trazabilidad con métricas"
    " explicables en cada lote."
)

# ----------------------------- Pre-condición: target -----------------------------
target = st.session_state.get("target")
if not target:
    st.warning("Configura primero el objetivo en **2 · Target Designer** para habilitar el generador.")
    st.stop()

# ----------------------------- Datos base -----------------------------
waste_df = load_waste_df()
proc_df = load_process_df()
proc_filtered = choose_process(
    target["name"], proc_df,
    scenario=target.get("scenario"),
    crew_time_low=target.get("crew_time_low", False)
)
if proc_filtered is None or proc_filtered.empty:
    proc_filtered = proc_df.copy()

generator_state_key = "generator_button_state"
generator_trigger_key = "generator_button_trigger"
generator_error_key = "generator_button_error"
button_state = st.session_state.get(generator_state_key, "idle")
button_error = st.session_state.get(generator_error_key)

# ----------------------------- Panel de control + IA -----------------------------
with layout_block("layout-grid layout-grid--dual layout-grid--flow", parent=None) as grid:
    with layout_stack(parent=grid) as left_column:
        with layout_block("side-panel layer-shadow fade-in", parent=left_column) as control:
            control.markdown("### 🎛️ Configuración")
            stored_mode = st.session_state.get("prediction_mode", "Modo Rex-AI (ML)")
            mode = control.radio(
                "Motor de predicción",
                ("Modo Rex-AI (ML)", "Modo heurístico"),
                index=0 if stored_mode == "Modo Rex-AI (ML)" else 1,
                help="Usá Rex-AI para predicciones ML o quedate con la estimación heurística reproducible.",
            )
            st.session_state["prediction_mode"] = mode
            use_ml = mode == "Modo Rex-AI (ML)"

            control.markdown("#### Parámetros principales")
            col_iters, col_recipes = control.columns(2)
            opt_evals = col_iters.slider(
                "Iteraciones (Ax/BoTorch)",
                0,
                60,
                18,
                help="Cantidad de pasos bayesianos para refinar el lote.",
            )
            n_candidates = col_recipes.slider(
                "Recetas a explorar",
                3,
                12,
                6,
                help="Cantidad de combinaciones candidatas por lote.",
            )

            with control.expander("Opciones avanzadas") as advanced:
                seed_default = st.session_state.get("generator_seed_input", "")
                seed_input = advanced.text_input(
                    "Semilla (opcional)",
                    value=seed_default,
                    help="Ingresá un entero para repetir exactamente el mismo lote.",
                )
                st.session_state["generator_seed_input"] = seed_input

            crew_low = target.get("crew_time_low", False)
            control.caption(
                "Los resultados privilegian %s"
                % ("tiempo de tripulación" if crew_low else "un balance general")
            )

            with control:
                run = minimal_button(
                    "Generar lote",
                    key="generator_run_button",
                    state=button_state,
                    width="full",
                    help_text="Ejecuta Ax + BoTorch con los parámetros seleccionados.",
                    loading_label="Generando lote…",
                    success_label="Lote listo",
                    error_label="Reintentar",
                    status_hints={
                        "idle": "",
                        "loading": "Ejecutando optimizador",
                        "success": "Resultados actualizados",
                        "error": "Revisá la configuración",
                    },
                )

            if run and button_state != "loading":
                st.session_state[generator_state_key] = "loading"
                st.session_state[generator_trigger_key] = True
                st.session_state.pop(generator_error_key, None)
                st.experimental_rerun()

            button_state_now = st.session_state.get(generator_state_key)
            if button_error and button_state_now == "error":
                control.error(button_error)
            elif button_state_now == "success":
                control.caption("✅ Última corrida lista abajo. Volvé a ejecutar si cambias parámetros.")
            if not use_ml:
                control.info("Modo heurístico activo: las métricas se basan en reglas físicas y no en ML.")

            if isinstance(proc_filtered, pd.DataFrame) and not proc_filtered.empty:
                preview_map = [
                    ("process_id", "ID"),
                    ("name", "Proceso"),
                    ("match_score", "Score"),
                    ("crew_min_per_batch", "Crew (min)"),
                    ("match_reason", "Por qué"),
                ]
                cols_present = [col for col, _ in preview_map if col in proc_filtered.columns]
                if cols_present:
                    control.markdown("#### Procesos sugeridos")
                    control.caption("Filtrado según residuo/flags y escenario seleccionado.")
                    preview_df = proc_filtered[cols_present].head(5).rename(columns=dict(preview_map))
                    control.dataframe(preview_df, hide_index=True, use_container_width=True)
    with layout_stack(parent=grid) as right_column:
        with layout_block("side-panel layer-shadow fade-in", parent=right_column) as target_card:
            target_card.markdown("### 🎯 Objetivo activo")
            target_card.markdown(f"**{target.get('name', '—')}**")
            scenario_label = target.get("scenario") or "Escenario general"
            target_card.caption(f"Escenario: {scenario_label}")
            if target.get("crew_time_low"):
                badge_group(["⏱️ Prioriza crew-time"], parent=target_card)
            limits = [
                ("Rigidez objetivo", target.get("rigidity")),
                ("Estanqueidad objetivo", target.get("tightness")),
                ("Máx. energía (kWh)", target.get("max_energy_kwh")),
                ("Máx. agua (L)", target.get("max_water_l")),
                ("Máx. crew (min)", target.get("max_crew_min")),
            ]
            summary_rows = []
            for label, value in limits:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    summary_rows.append({"Variable": label, "Valor": "—"})
                else:
                    summary_rows.append({"Variable": label, "Valor": f"{numeric_value:.2f}"})
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                target_card.dataframe(summary_df, hide_index=True, use_container_width=True)

        with layout_block("depth-stack layer-shadow fade-in-delayed", parent=right_column) as ai_panel:
            ai_panel.markdown("### 🧠 Modelo Rex-AI")
            model_registry = get_model_registry()
            trained_at = model_registry.metadata.get("trained_at", "—")
            n_samples = model_registry.metadata.get("n_samples", "—")
            top_features = model_registry.feature_importance_avg[:5]
            if top_features:
                df_feat = pd.DataFrame(top_features, columns=["feature", "weight"])
                chart = alt.Chart(df_feat).mark_bar(color="#60a5fa").encode(
                    x=alt.X("weight", title="Importancia promedio"),
                    y=alt.Y("feature", sort="-x", title="Feature"),
                    tooltip=["feature", alt.Tooltip("weight", format=".3f")],
                ).properties(height=180)
                ai_panel.altair_chart(chart, use_container_width=True)
            ai_panel.caption(
                f"Entrenado: {trained_at} · Muestras: {n_samples} · Features: {len(model_registry.feature_names)}"
            )
            if model_registry.metadata.get("random_forest", {}).get("metrics", {}).get("overall"):
                overall = model_registry.metadata["random_forest"]["metrics"]["overall"]
                try:
                    ai_panel.caption(
                        f"MAE promedio: {overall.get('mae', float('nan')):.3f} · RMSE: {overall.get('rmse', float('nan')):.3f} · R²: {overall.get('r2', float('nan')):.3f}"
                    )
                except Exception:
                    pass
            label_summary_text = model_registry.label_distribution_label()
            if label_summary_text and label_summary_text != "—":
                ai_panel.caption(f"Fuentes de labels: {label_summary_text}")
# ----------------------------- Generación -----------------------------
if st.session_state.get("generator_button_trigger"):
    seed_value: int | None = None
    seed_raw = st.session_state.get("generator_seed_input", "").strip()
    if seed_raw:
        try:
            seed_value = int(seed_raw, 0)
        except ValueError:
            st.session_state["generator_button_state"] = "error"
            st.session_state["generator_button_error"] = "Ingresá un entero válido para la semilla (por ejemplo 42 o 0x2A)."
            st.session_state["generator_button_trigger"] = False
            st.stop()
    try:
        result = generate_candidates(
            waste_df,
            proc_filtered,
            target,
            n=n_candidates,
            crew_time_low=target.get("crew_time_low", False),
            optimizer_evals=opt_evals,
            use_ml=use_ml,
            seed=seed_value,
        )
        if isinstance(result, tuple):
            cands, history = result
        else:
            cands, history = result, pd.DataFrame()
        if isinstance(cands, pd.DataFrame):
            cands = cands.to_dict("records")
        elif isinstance(cands, dict):
            cands = [cands]
        else:
            cands = [dict(c) for c in cands]
        history_df = history if isinstance(history, pd.DataFrame) else pd.DataFrame()
        st.session_state["candidates"] = cands
        st.session_state["optimizer_history"] = history_df
        st.session_state["generator_button_state"] = "success"
        st.session_state["generator_button_trigger"] = False
        st.session_state.pop("generator_button_error", None)
    except Exception as exc:  # noqa: BLE001
        st.session_state["generator_button_state"] = "error"
        st.session_state["generator_button_error"] = f"Error generando candidatos: {exc}"
        st.session_state["generator_button_trigger"] = False

# ----------------------------- Si no hay candidatos aún -----------------------------
st.divider()
cands = st.session_state.get("candidates", [])
history_df = st.session_state.get("optimizer_history", pd.DataFrame())

if not cands:
    st.info(
        "Todavía no hay candidatos. Configurá los controles y presioná **Generar lote**. "
        "Asegurate de que el inventario tenga pouches, espumas, EVA/CTB, textiles o nitrilo; "
        "y que el catálogo incluya P02/P03/P04."
    )
    with st.expander("¿Qué hace el generador (en criollo)?", expanded=False):
        st.markdown(
            "- **Mira tus residuos** (enfocado en los problemáticos de NASA).\n"
            "- **Elige un proceso** coherente (laminar, sinter con regolito, reconfigurar CTB, etc.).\n"
            "- **Predice** propiedades y recursos de la receta.\n"
            "- **Puntúa** balanceando objetivos y costos.\n"
            "- **Muestra trazabilidad** para ver qué basura se valorizó."
        )
    st.stop()

# ----------------------------- Historial del optimizador -----------------------------
if isinstance(history_df, pd.DataFrame) and not history_df.empty:
    scene = ConvergenceScene(
        history_df,
        subtitle=(
            "Pistas visuales sobre cómo evoluciona el frente Pareto tras cada iteración "
            "(pasá el cursor para leer hipervolumen, dominancia y scores)."
        ),
    )
    scene.render(st)

# ----------------------------- Resumen de ranking -----------------------------
summary_rows: list[dict[str, object]] = []
for idx, cand in enumerate(cands, start=1):
    props = cand.get("props")
    if props is None:
        continue
    aux = cand.get("auxiliary") or {}
    summary_rows.append(
        {
            "Rank": idx,
            "Score": cand.get("score"),
            "Proceso": f"{cand.get('process_id', '')} · {cand.get('process_name', '')}",
            "Rigidez": getattr(props, "rigidity", float("nan")),
            "Estanqueidad": getattr(props, "tightness", float("nan")),
            "Energía (kWh)": getattr(props, "energy_kwh", float("nan")),
            "Agua (L)": getattr(props, "water_l", float("nan")),
            "Crew (min)": getattr(props, "crew_min", float("nan")),
            "Seal": "✅" if aux.get("passes_seal", True) else "⚠️",
            "Riesgo": aux.get("process_risk_label", "—"),
        }
    )

if summary_rows:
    st.subheader("Ranking multiobjetivo")
    st.caption("Ordenado por score total, con sellado y riesgo resumidos.")
    cockpit = RankingCockpit(
        entries=summary_rows,
        metric_specs=[
            MetricSpec("Rigidez", "Rigidez", "{:.2f}"),
            MetricSpec("Estanqueidad", "Estanqueidad", "{:.2f}"),
            MetricSpec("Energía (kWh)", "Energía", "{:.2f}", unit="kWh", higher_is_better=False),
            MetricSpec("Agua (L)", "Agua", "{:.1f}", unit="L", higher_is_better=False),
            MetricSpec("Crew (min)", "Crew", "{:.1f}", unit="min", higher_is_better=False),
        ],
        key="generator_ranking",
        score_label="Score",
        selection_label="📌 Candidato destacado",
    )
    selected_summary = cockpit.render()
    if selected_summary is not None:
        st.session_state["generator_ranking_focus"] = selected_summary

# ----------------------------- Showroom de candidatos -----------------------------
st.subheader("Resultados del generador")
st.caption(
    "Explorá cada receta como card 3D con tabs de propiedades, recursos y trazabilidad. "
    "Ajustá el timeline lateral para priorizar rigidez o agua y filtrar rápidamente."
)

filtered_cands = render_candidate_showroom(cands, target)
for i, c in enumerate(cands):
    p = c["props"]
    header = f"Opción {i+1} — Score {c['score']} — Proceso {c['process_id']} {c['process_name']}"
    with st.expander(header, expanded=(i == 0)):
        # Badges
        badges = []
        if c.get("regolith_pct", 0) > 0:
            badges.append("⛰️ ISRU: +MGS-1")
        src_cats = " ".join(map(str, c.get("source_categories", []))).lower()
        src_flags = " ".join(map(str, c.get("source_flags", []))).lower()
        problem_present = any([
            "pouches" in src_cats, "multilayer" in src_flags,
            "foam" in src_cats, "ctb" in src_flags, "eva" in src_cats,
            "nitrile" in src_cats, "wipe" in src_flags
        ])
        if problem_present:
            badges.append("♻️ Valorización de problemáticos")
        aux = c.get("auxiliary") or {}
        if aux:
            if aux.get("passes_seal") is False:
                badges.append("⚠️ Revisar estanqueidad")
            elif aux.get("passes_seal"):
                badges.append("✅ Sellado OK")
            risk_label = aux.get("process_risk_label")
            if risk_label:
                badges.append(f"🏷️ Riesgo {risk_label}")
        if badges:
            badge_group(badges)
            ChipRow([{ "label": badge } for badge in badges], tone="accent")

        pred_error = c.get("prediction_error")
        if pred_error:
            st.error(f"Predicción ML no disponible: {pred_error}")

        # Resumen técnico
        with layout_block("layout-grid layout-grid--dual layout-grid--flow", parent=None) as detail_grid:
            with layout_block("depth-stack layer-shadow", parent=detail_grid) as left_panel:
                left_panel.markdown("**🧪 Materiales**")
                left_panel.write(", ".join(c["materials"]))
                left_panel.markdown("**⚖️ Pesos en mezcla**")
                left_panel.write(c["weights"])

                left_panel.markdown("**🔬 Predicción**" if not pred_error else "**🔬 Estimación heurística**")
                if pred_error:
                    metrics_html = (
                        f"<div class='metric-grid'>"
                        f"<div class='stat-card'><span>Rigidez</span><strong>{p.rigidity:.2f}</strong></div>"
                        f"<div class='stat-card'><span>Estanqueidad</span><strong>{p.tightness:.2f}</strong></div>"
                        f"<div class='stat-card'><span>Masa final</span><strong>{p.mass_final_kg:.2f} kg</strong></div>"
                        f"</div>"
                    )
                    left_panel.markdown(metrics_html, unsafe_allow_html=True)
                else:
                    metrics_html = (
                        f"<div class='metric-grid fade-in'>"
                        f"<div class='stat-card layer-glow'><span>Rigidez</span><strong>{p.rigidity:.2f}</strong></div>"
                        f"<div class='stat-card layer-glow'><span>Estanqueidad</span><strong>{p.tightness:.2f}</strong></div>"
                        f"<div class='stat-card layer-glow'><span>Masa final</span><strong>{p.mass_final_kg:.2f} kg</strong></div>"
                        f"</div>"
                    )
                    left_panel.markdown(metrics_html, unsafe_allow_html=True)
                src = c.get("prediction_source", "heuristic")
                meta_payload = {}
                if isinstance(c.get("ml_prediction"), dict):
                    meta_payload = c["ml_prediction"].get("metadata", {}) or {}
                if pred_error:
                    left_panel.caption("Fallback heurístico mostrado por indisponibilidad del modelo.")
                elif str(src).startswith("rexai"):
                    t_at = meta_payload.get("trained_at", "?")
                    latent = c.get("latent_vector", [])
                    latent_note = "" if not latent else f" · Vector latente {len(latent)}D"
                    left_panel.caption(f"Predicción por modelo ML (**{src}**, entrenado {t_at}){latent_note}.")
                    summary_text = _format_label_summary(
                        meta_payload.get("label_summary") or model_registry.label_summary
                    )
                    if summary_text:
                        left_panel.caption(f"Dataset Rex-AI: {summary_text}")
                else:
                    left_panel.caption("Predicción heurística basada en reglas.")

                ci = c.get("confidence_interval") or {}
                unc = c.get("uncertainty") or {}
                if ci:
                    rows: list[dict[str, float | str]] = []
                    for key, bounds in ci.items():
                        label = TARGET_DISPLAY.get(key, key)
                        try:
                            lo_val, hi_val = float(bounds[0]), float(bounds[1])
                        except (TypeError, ValueError, IndexError):
                            lo_val, hi_val = float("nan"), float("nan")
                        row: dict[str, float | str] = {
                            "Variable": label,
                            "Lo": lo_val,
                            "Hi": hi_val,
                        }
                        sigma_val = unc.get(key)
                        if sigma_val is not None:
                            try:
                                row["σ (std)"] = float(sigma_val)
                            except (TypeError, ValueError):
                                row["σ (std)"] = float("nan")
                        rows.append(row)
                    if rows and not pred_error:
                        left_panel.markdown("**📉 Intervalos de confianza (95%)**")
                        ci_df = pd.DataFrame(rows)
                        left_panel.dataframe(ci_df, hide_index=True, use_container_width=True)

                feature_imp = c.get("feature_importance") or []
                if feature_imp and not pred_error:
                    left_panel.markdown("**🪄 Features que más influyen**")
                    fi_df = pd.DataFrame(feature_imp, columns=["feature", "impact"])
                    chart = alt.Chart(fi_df).mark_bar(color="#60a5fa").encode(
                        x=alt.X("impact", title="Impacto relativo"),
                        y=alt.Y("feature", sort="-x", title="Feature"),
                        tooltip=["feature", alt.Tooltip("impact", format=".3f")],
                    ).properties(height=180)
                    left_panel.altair_chart(chart, use_container_width=True)

            with layout_block("depth-stack layer-shadow", parent=detail_grid) as right_panel:
                right_panel.markdown("**🔧 Proceso**")
                right_panel.write(f"{c['process_id']} — {c['process_name']}")
                right_panel.markdown("**📉 Recursos estimados**")
                resources = [
                    ("Energía (kWh)", p.energy_kwh, target["max_energy_kwh"], f"{p.energy_kwh:.2f} / {target['max_energy_kwh']}"),
                    ("Agua (L)", p.water_l, target["max_water_l"], f"{p.water_l:.2f} / {target['max_water_l']}"),
                    ("Crew (min)", p.crew_min, target["max_crew_min"], f"{p.crew_min:.0f} / {target['max_crew_min']}"),
                ]
                with layout_block("resource-grid", parent=right_panel) as res_grid:
                    for label, value, limit, caption in resources:
                        with layout_block("resource-card layer-shadow", parent=res_grid) as card:
                            card.markdown(f"**{label}**")
                            card.progress(_res_bar(value, limit))
                            card.caption(caption)

        micro_divider()

        # Trazabilidad NASA
        st.markdown("**🛰️ Trazabilidad NASA**")
        st.write("IDs usados:", ", ".join(c.get("source_ids", [])) or "—")
        st.write("Categorías:", ", ".join(map(str, c.get("source_categories", []))) or "—")
        st.write("Flags:", ", ".join(map(str, c.get("source_flags", []))) or "—")
        if c.get("regolith_pct", 0) > 0:
            st.write(f"**MGS-1 agregado:** {c['regolith_pct']*100:.0f}%")

        # Features resumen (si los hay)
        feat = c.get("features", {})
        if feat:
            feat_summary = {
                "Masa total (kg)": feat.get("total_mass_kg"),
                "Densidad (kg/m³)": feat.get("density_kg_m3"),
                "Humedad": feat.get("moisture_frac"),
                "Dificultad": feat.get("difficulty_index"),
                "Recupero gas": feat.get("gas_recovery_index"),
                "Reuso logístico": feat.get("logistics_reuse_index"),
            }
            feat_df = pd.DataFrame([feat_summary])
            st.markdown("**Features NASA/ML (alimentan la IA)**")
            st.dataframe(feat_df, hide_index=True, use_container_width=True)
            if feat.get("latent_vector"):
                st.caption("Latente Rex-AI incluido para análisis generativo.")

        breakdown = c.get("score_breakdown") or {}
        contribs = breakdown.get("contributions") or {}
        penalties = breakdown.get("penalties") or {}
        if contribs or penalties:
            st.markdown("**⚖️ Desglose del score**")
            with layout_block("layout-grid layout-grid--balanced", parent=None) as score_grid:
                if contribs:
                    contrib_df = pd.DataFrame([
                        {"Factor": k, "+": float(v)} for k, v in contribs.items()
                    ]).sort_values("+", ascending=False)
                    with layout_block("depth-stack layer-shadow", parent=score_grid) as positive_card:
                        positive_card.markdown("**Contribuciones**")
                        positive_card.dataframe(contrib_df, hide_index=True, use_container_width=True)
                if penalties:
                    pen_df = pd.DataFrame([
                        {"Penalización": k, "-": float(v)} for k, v in penalties.items()
                    ]).sort_values("-", ascending=False)
                    with layout_block("depth-stack layer-shadow", parent=score_grid) as penalty_card:
                        penalty_card.markdown("**Penalizaciones**")
                        penalty_card.dataframe(pen_df, hide_index=True, use_container_width=True)

        # Seguridad
        flags = check_safety(c["materials"], c["process_name"], c["process_id"])
        badge = safety_badge(flags)
        st.info(f"Seguridad: {badge['level']} · {badge['detail']}")

        # Botón de selección
        if st.button(f"✅ Seleccionar Opción {i+1}", key=f"pick_{i}"):
            st.session_state["selected"] = {"data": c, "safety": badge}
            st.success("Opción seleccionada. Abrí **4) Resultados**, **5) Comparar & Explicar** o **6) Pareto & Export**.")

# ----------------------------- Explicación en criollo (popover global) -----------------------------
top = filtered_cands[0] if filtered_cands else (cands[0] if cands else None)
pop = st.popover("🧠 ¿Por qué estas recetas pintan bien? (explicación en criollo)")
with pop:
    bullets = []
    bullets.append("• Sumamos puntos si **rigidez/estanqueidad** se acercan a lo que pediste.")
    bullets.append("• Restamos si se pasa en **agua/energía/tiempo** de la tripulación.")
    if top:
        cats = " ".join(map(str, top.get("source_categories", []))).lower()
        flg = " ".join(map(str, top.get("source_flags", []))).lower()
        if any(k in cats or k in flg for k in ["pouches", "multilayer", "foam", "eva", "ctb", "nitrile", "wipe"]):
            bullets.append("• Bonus porque priorizamos **basura problemática** (la que más molesta en la base).")
        if top.get("regolith_pct", 0) > 0:
            bullets.append("• Usa **MGS-1** (regolito) como carga mineral → ISRU: menos dependencia de la Tierra.")
    st.markdown("\n".join(bullets))

# ----------------------------- Glosario -----------------------------
micro_divider()
with st.expander("📚 Glosario ultra rápido", expanded=False):
    st.markdown(
        "- **ISRU**: *In-Situ Resource Utilization*. Usar recursos del lugar (en Marte, el **regolito** MGS-1).\n"
        "- **P02 – Press & Heat Lamination**: “plancha” y “fusiona” multicapa para dar forma.\n"
        "- **P03 – Sinter with MGS-1**: mezcla con regolito y sinteriza → piezas rígidas.\n"
        "- **P04 – CTB Reconfig**: reusar/transformar bolsas EVA/CTB con herrajes.\n"
        "- **Score**: qué tanto ‘cierra’ la opción según objetivo y límites de recursos/tiempo."
    )
st.info("Tip: generá varias opciones y pasá a **4) Resultados**, **5) Comparar** y **6) Pareto & Export** para cerrar tu plan.")
