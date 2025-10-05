import sys
from pathlib import Path

if not __package__:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

import math
from contextlib import contextmanager
from typing import Any, Generator, Mapping

import altair as alt
import pandas as pd
import streamlit as st

from app.modules.candidate_showroom import render_candidate_showroom
from app.modules.io import (  # si tu IO usa load_process_catalog, cámbialo aquí
    MissingDatasetError,
    format_missing_dataset_message,
    load_process_df,
    load_waste_df,
)
from app.modules.ml_models import get_model_registry
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.page_data import build_ranking_table
from app.modules.process_planner import choose_process
from app.modules.safety import check_safety, safety_badge, build_safety_compliance
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
    action_button,
    configure_page,
    chipline,
    initialise_frontend,
    layout_block,
    layout_stack,
    micro_divider,
    pill,
    render_brand_header,
    render_dataset_badge,
)
from app.modules.visualizations import ConvergenceScene
from app.modules.utils import (
    format_label_summary,
    format_number,
    format_resource_text,
    physical_dataset_tooltip,
    safe_float,
    uses_physical_dataset,
)
from app.pages.generator_view_model import GeneratorViewModel


TARGET_DISPLAY = {
    "rigidez": "Rigidez",
    "estanqueidad": "Estanqueidad",
    "energy_kwh": "Energía (kWh)",
    "water_l": "Agua (L)",
    "crew_min": "Crew (min)",
}

PREDICTION_ATTR_MAP = {
    "rigidez": "rigidity",
    "estanqueidad": "tightness",
    "energy_kwh": "energy_kwh",
    "water_l": "water_l",
    "crew_min": "crew_min",
}


@contextmanager
def _optional_container_expander(
    container: Any,
    label: str,
    *,
    warning_message: str | None = None,
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Use ``container.expander`` when possible, otherwise yield ``None`` and warn."""

    expander_fn = getattr(container, "expander", None)
    if callable(expander_fn):
        with expander_fn(label, **kwargs) as maybe_inner:
            if maybe_inner is None and warning_message:
                st.warning(warning_message)
            yield maybe_inner
        return

    if warning_message:
        st.warning(warning_message)
    yield None


def _container_text_input(container: Any, *args: Any, **kwargs: Any) -> str:
    """Render ``st.text_input`` within ``container`` if available."""

    target = getattr(container, "text_input", None) if container is not None else None
    if callable(target):
        return target(*args, **kwargs)

    return st.text_input(*args, **kwargs)


configure_page(page_title="Rex-AI • Generador", page_icon="🤖")
initialise_frontend()

current_step = set_active_step("generator")

render_brand_header()

render_breadcrumbs(current_step)

view_model = GeneratorViewModel.from_streamlit()
_playbook_prefilters, _playbook_prefill_label = view_model.pop_playbook_prefill()

st.header("Generador asistido por IA")

target = view_model.get_target()


def _collect_external_profiles(candidate: Mapping[str, Any], inventory: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(candidate, Mapping) or not isinstance(inventory, pd.DataFrame) or inventory.empty:
        return {}

    raw_ids = candidate.get("source_ids") or []
    ids = {str(value).strip() for value in raw_ids if str(value).strip()}
    if not ids:
        return {}

    mask = pd.Series(False, index=inventory.index)
    if "id" in inventory.columns:
        mask |= inventory["id"].astype(str).isin(ids)
    if "_source_id" in inventory.columns:
        mask |= inventory["_source_id"].astype(str).isin(ids)

    subset = inventory.loc[mask].copy()
    if subset.empty:
        return {}

    payload: dict[str, Any] = {}

    def _build_section(
        numeric_columns: tuple[str, ...],
        label_columns: tuple[str, ...],
    ) -> dict[str, Any] | None:
        relevant_numeric = [column for column in numeric_columns if column in subset.columns]
        if not relevant_numeric:
            return None

        numeric_df = subset[relevant_numeric].apply(pd.to_numeric, errors="coerce")
        mask_numeric = numeric_df.notna().any(axis=1)
        if not mask_numeric.any():
            return None

        rows = subset.loc[mask_numeric]
        metrics: dict[str, float] = {}

        for column in relevant_numeric:
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
            label_series = rows[column].dropna().astype(str).str.strip().replace("", pd.NA).dropna()
            labels.extend(label_series.tolist())

        unique_labels = sorted(dict.fromkeys(labels))
        return {"metrics": metrics, "labels": unique_labels}

    polymer_section = _build_section(POLYMER_METRIC_COLUMNS, POLYMER_LABEL_COLUMNS)
    if polymer_section:
        payload["polymer"] = polymer_section

    aluminium_section = _build_section(ALUMINIUM_NUMERIC_COLUMNS, ALUMINIUM_LABEL_COLUMNS)
    if aluminium_section:
        payload["aluminium"] = aluminium_section

    return payload


def _render_reference_distribution(
    series: pd.Series,
    reference_value: object,
    *,
    field: str,
    axis_label: str,
    histogram_color: str,
    reference_color: str,
    empty_message: str,
    opacity: float = 0.85,
) -> None:
    if series.empty:
        st.info(empty_message)
        return

    numeric_reference = safe_float(reference_value)
    if numeric_reference is None:
        return

    base = (
        alt.Chart(pd.DataFrame({field: series}))
        .mark_bar(color=histogram_color, opacity=opacity)
        .encode(
            x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=18), title=axis_label),
            y=alt.Y("count()", title="Ítems"),
            tooltip=["count()"],
        )
    )
    reference = (
        alt.Chart(pd.DataFrame({field: [numeric_reference]}))
        .mark_rule(color=reference_color, size=3)
        .encode(x=f"{field}:Q")
    )
    st.altair_chart(base + reference, use_container_width=True)


def _format_reference_value(key: str, value: float) -> str:
    if key in {"density_g_cm3", "modulus_gpa"}:
        return f"{value:.2f}"
    if key == "burn_min":
        return f"{value:.1f}"
    return f"{value:.0f}"


def _collect_target_badges(target: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(target, dict):
        return []

    badges: list[tuple[str, str]] = []

    water_limit = safe_float(target.get("max_water_l"))
    if water_limit is not None:
        badges.append((f"💧 Agua ≤ {water_limit:.0f} L", "warn"))

    energy_limit = safe_float(target.get("max_energy_kwh"))
    if energy_limit is not None:
        badges.append((f"⚡ Energía ≤ {energy_limit:.0f} kWh", "warn"))

    if target.get("crew_time_low"):
        badges.append(("⏱️ Crew-time priorizado", "warn"))

    target_name = str(target.get("name") or "").strip().casefold()
    if target_name == "residence renovations":
        badges.append(("🏠 Prioridad: volumen habitable", "ok"))
    elif target_name == "daring discoveries":
        badges.append(("🛰️ Prioridad: rigidez estructural", "ok"))

    return badges


def render_safety_indicator(
    candidate: dict[str, Any],
    target_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Renderiza la indicación de seguridad para un candidato y devuelve el badge."""
    materials_raw = candidate.get("materials") or []
    if isinstance(materials_raw, (list, tuple, set)):
        materials = [str(item) for item in materials_raw]
    elif materials_raw:
        materials = [str(materials_raw)]
    else:
        materials = []

    process_name = str(candidate.get("process_name") or "")
    process_id = str(candidate.get("process_id") or "")

    flags = check_safety(materials, process_name, process_id)
    badge = safety_badge(flags)
    compliance_payload = build_safety_compliance(candidate, target_data, flags)
    badge["pfas"] = bool(flags.pfas)
    badge["microplastics"] = bool(flags.microplastics)
    badge["incineration"] = bool(flags.incineration)
    badge.update(compliance_payload)

    level = badge.get("level", "OK")
    detail = badge.get("detail", "")
    kind = "risk" if level == "Riesgo" else "ok"
    icon = "⚠️" if level == "Riesgo" else "🛡️"

    pill(f"{icon} Seguridad · {level}", kind=kind)
    if level == "Riesgo" and detail:
        st.warning(detail)

    with st.container():
        st.markdown("**🚥 Semáforo PFAS & microplásticos**")
        signal_chips = [
            {
                "label": "PFAS controlados" if not badge.get("pfas") else "PFAS en vigilancia",
                "icon": "🧪",
                "tone": "positive" if not badge.get("pfas") else "danger",
            },
            {
                "label": "Microplásticos mitigados"
                if not badge.get("microplastics")
                else "Microplásticos en riesgo",
                "icon": "🧴",
                "tone": "positive" if not badge.get("microplastics") else "warning",
            },
        ]
        chipline(signal_chips, render=True)

        mitigation_actions: list[dict[str, str]] = []
        if badge.get("pfas"):
            mitigation_actions.append(
                {
                    "label": "Aislar compuestos fluorados y monitorear filtrado",
                    "icon": "🛡️",
                    "tone": "warning",
                }
            )
        else:
            mitigation_actions.append(
                {
                    "label": "Receta sin PFAS identificados",
                    "icon": "✅",
                    "tone": "positive",
                }
            )
        if badge.get("microplastics"):
            mitigation_actions.append(
                {
                    "label": "Encapsular shredder y sellar descarga",
                    "icon": "🧯",
                    "tone": "warning",
                }
            )
        else:
            mitigation_actions.append(
                {
                    "label": "Liberación de microplásticos contenida",
                    "icon": "🧬",
                    "tone": "positive",
                }
            )
        chipline(mitigation_actions, render=True)

        detail_note = str(detail).strip()
        microcopy = "Esta receta evita PFAS y reusa calor residual."
        if detail_note:
            microcopy = f"{microcopy} {detail_note}"
        st.caption(microcopy)

    compliance_rows = badge.get("compliance", [])
    resource_rows = badge.get("resource_compliance", [])

    if compliance_rows or resource_rows:
        with st.container():
            if compliance_rows and resource_rows:
                env_col, resource_col = st.columns(2)
            else:
                env_col = st.container()
                resource_col = None

            if compliance_rows:
                with env_col:
                    st.markdown("**🧾 Checklist de cumplimiento**")
                    for row in compliance_rows:
                        icon = row.get("icon", "•")
                        label = row.get("label", "")
                        message = row.get("message", "")
                        st.markdown(f"- {icon} **{label}** · {message}")

            if resource_rows:
                target_col = resource_col if resource_col is not None else env_col
                with target_col:
                    st.markdown("**🔋 Recursos vs misión**")
                    for row in resource_rows:
                        icon = row.get("icon", "•")
                        label = row.get("label", "")
                        message = row.get("message", "")
                        st.markdown(f"- {icon} **{label}** · {message}")

    return badge


def render_candidate_card(
    candidate: dict[str, Any],
    idx: int,
    target_data: dict[str, Any],
    model_registry: Any,
    view_model: GeneratorViewModel,
) -> None:
    props = candidate.get("props")
    if props is None:
        return

    score_text = format_number(candidate.get("score"))
    process_id = str(candidate.get("process_id") or "—")
    process_name = str(candidate.get("process_name") or "")
    process_label = " ".join(part for part in [process_id, process_name] if part).strip()
    header = f"Opción {idx} — Score {score_text} — Proceso {process_label}"

    with st.expander(header, expanded=(idx == 1)):
        card = st.container()

        badges: list[str] = []
        regolith_badge_pct = safe_float(candidate.get("regolith_pct"))
        if regolith_badge_pct and regolith_badge_pct > 0:
            badges.append("⛰️ ISRU: +MGS-1")
        src_cats = " ".join(map(str, candidate.get("source_categories", []))).lower()
        src_flags = " ".join(map(str, candidate.get("source_flags", []))).lower()
        problem_present = any(
            key in src_cats or key in src_flags
            for key in ["pouches", "multilayer", "foam", "ctb", "eva", "nitrile", "wipe"]
        )
        if problem_present:
            badges.append("♻️ Valorización de problemáticos")
        aux = candidate.get("auxiliary") or {}
        if aux.get("passes_seal") is False:
            badges.append("⚠️ Revisar estanqueidad")
        elif aux.get("passes_seal"):
            badges.append("✅ Sellado OK")
        risk_label = aux.get("process_risk_label")
        if risk_label:
            badges.append(f"🏷️ Riesgo {risk_label}")
        if badges:
            card.markdown(" ".join(badges))

        pred_error = candidate.get("prediction_error")
        if pred_error:
            card.error(f"Predicción ML no disponible: {pred_error}")

        info_col, process_col = card.columns([1.7, 1.3])

        with info_col:
            info_col.markdown("**🧪 Materiales y mezcla**")

            materials_raw = candidate.get("materials") or []
            if isinstance(materials_raw, (list, tuple, set)):
                material_labels = [str(item) for item in materials_raw]
            elif materials_raw:
                material_labels = [str(materials_raw)]
            else:
                material_labels = []

            weights_raw = candidate.get("weights")
            if isinstance(weights_raw, Mapping):
                weights_list = [
                    safe_float(weights_raw.get(label)) for label in material_labels
                ]
            elif isinstance(weights_raw, (list, tuple, set)):
                weights_list = [safe_float(value) for value in weights_raw]
            elif weights_raw is None:
                weights_list = []
            else:
                try:
                    weights_list = [safe_float(weights_raw)]
                except Exception:  # noqa: BLE001
                    weights_list = []

            row_count = max(len(material_labels), len(weights_list))
            if row_count == 0:
                info_col.write("—")
            else:
                if len(material_labels) < row_count:
                    material_labels.extend(
                        [f"Componente {idx+1}" for idx in range(len(material_labels), row_count)]
                    )
                if len(weights_list) < row_count:
                    weights_list.extend([None] * (row_count - len(weights_list)))

                numeric_weights: list[float | None] = []
                for value in weights_list:
                    number = safe_float(value)
                    numeric_weights.append(number)

                valid_weights = [w for w in numeric_weights if w is not None and math.isfinite(w)]
                total_weight = sum(valid_weights) if valid_weights else 0.0

                features = candidate.get("features") or {}
                total_mass = safe_float(features.get("total_mass_kg"))

                fractions: list[float | None] = []
                masses: list[float | None] = []
                for weight in numeric_weights:
                    if weight is not None and math.isfinite(weight) and total_weight > 0:
                        frac = weight / total_weight
                    elif total_weight > 0:
                        frac = None
                    else:
                        frac = None
                    fractions.append(frac)

                    if total_mass is not None and frac is not None:
                        masses.append(total_mass * frac)
                    elif total_mass is None and weight is not None and math.isfinite(weight):
                        masses.append(weight)
                    else:
                        masses.append(None)

                mix_df = pd.DataFrame(
                    {
                        "Material": material_labels,
                        "Masa (kg)": masses,
                        "Fracción (%)": [
                            frac * 100 if frac is not None else float("nan") for frac in fractions
                        ],
                    }
                )

                def _highlight_critical(row: pd.Series) -> list[str]:
                    material = str(row.get("Material", "")).casefold()
                    critical_tokens = ("mgs-1", "mgs1", "regolith")
                    if any(token in material for token in critical_tokens):
                        return ["background-color: #fef3c7"] * len(row)
                    return [""] * len(row)

                styled_mix = mix_df.style.format(
                    {
                        "Masa (kg)": lambda v: "—" if pd.isna(v) else f"{float(v):.2f} kg",
                        "Fracción (%)": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}%",
                    }
                ).apply(_highlight_critical, axis=1)

                try:
                    styled_mix = styled_mix.hide(axis="index")
                except AttributeError:
                    styled_mix = styled_mix.hide_index()

                info_col.dataframe(styled_mix, use_container_width=True)

            info_col.markdown("**🔬 Predicción**" if not pred_error else "**🔬 Estimación heurística**")
            metrics = [
                ("Rigidez", getattr(props, "rigidity", None), 2),
                ("Estanqueidad", getattr(props, "tightness", None), 2),
                ("Masa final (kg)", getattr(props, "mass_final_kg", None), 2),
            ]
            metric_cols = info_col.columns(len(metrics))
            for col, (label, value, precision) in zip(metric_cols, metrics):
                col.metric(label, format_number(value, precision=precision))

            src = candidate.get("prediction_source", "heuristic")
            meta_payload = {}
            if isinstance(candidate.get("ml_prediction"), dict):
                meta_payload = candidate["ml_prediction"].get("metadata", {}) or {}

            summary_source = meta_payload.get("label_summary") or model_registry.label_summary
            summary_text = format_label_summary(summary_source)
            uses_physical = uses_physical_dataset(src) and not pred_error
            dataset_tooltip = (
                physical_dataset_tooltip(
                    summary=summary_text or None,
                    trained_at=meta_payload.get("trained_at"),
                )
                if uses_physical
                else None
            )

            render_dataset_badge(
                container=info_col,
                uses_physical_dataset=uses_physical,
                tooltip=dataset_tooltip,
            )

            if pred_error:
                info_col.caption("Fallback heurístico mostrado por indisponibilidad del modelo.")
            elif uses_physical:
                trained_at = meta_payload.get("trained_at", "?")
                latent = candidate.get("latent_vector", [])
                latent_note = "" if not latent else f" · Vector latente {len(latent)}D"
                info_col.caption(
                    f"Predicción por modelo ML (**{src}**, entrenado {trained_at}){latent_note}."
                )
                if summary_text:
                    info_col.caption(f"Dataset Rex-AI: {summary_text}")
            else:
                info_col.caption("Predicción heurística basada en reglas.")

            ci = candidate.get("confidence_interval") or {}
            unc = candidate.get("uncertainty") or {}
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
                    attr_name = PREDICTION_ATTR_MAP.get(key)
                    center_value = None
                    if attr_name and hasattr(props, attr_name):
                        center_value = getattr(props, attr_name)
                    try:
                        if center_value is not None:
                            row["Predicción"] = float(center_value)
                    except (TypeError, ValueError):
                        row["Predicción"] = float("nan")
                    sigma_val = unc.get(key)
                    if sigma_val is not None:
                        try:
                            row["σ (std)"] = float(sigma_val)
                        except (TypeError, ValueError):
                            row["σ (std)"] = float("nan")
                    rows.append(row)
                if rows and not pred_error:
                    info_col.markdown("**📉 Intervalos de confianza (95%)**")
                    ci_df = pd.DataFrame(rows)
                    chart_df = ci_df.copy()
                    for column in ["Lo", "Hi", "Predicción", "σ (std)"]:
                        if column in chart_df.columns:
                            chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
                    chart_df = chart_df.dropna(subset=["Lo", "Hi"])
                    if not chart_df.empty:
                        if "Predicción" not in chart_df.columns:
                            chart_df["Predicción"] = (chart_df["Lo"] + chart_df["Hi"]) / 2
                        else:
                            chart_df["Predicción"] = chart_df["Predicción"].fillna(
                                (chart_df["Lo"] + chart_df["Hi"]) / 2
                            )
                        tooltips = [
                            alt.Tooltip("Variable:N", title="Indicador"),
                            alt.Tooltip("Predicción:Q", title="Predicción", format=".3f"),
                            alt.Tooltip("Lo:Q", title="Lo 95%", format=".3f"),
                            alt.Tooltip("Hi:Q", title="Hi 95%", format=".3f"),
                        ]
                        if "σ (std)" in chart_df.columns:
                            tooltips.append(alt.Tooltip("σ (std):Q", title="σ", format=".3f"))
                        base = alt.Chart(chart_df)
                        error_bars = base.mark_errorbar(color="#94a3b8").encode(
                            x=alt.X("Variable:N", title="Indicador"),
                            y=alt.Y("Lo:Q", title="Valor", scale=alt.Scale(zero=False)),
                            y2="Hi:Q",
                        )
                        points = base.mark_point(filled=True, size=140, color="#38bdf8").encode(
                            x=alt.X("Variable:N", title="Indicador"),
                            y=alt.Y("Predicción:Q", title="Valor", scale=alt.Scale(zero=False)),
                            tooltip=tooltips,
                        )
                        info_col.altair_chart(error_bars + points, use_container_width=True)

                    info_col.dataframe(ci_df, hide_index=True, use_container_width=True)

            feature_imp = candidate.get("feature_importance") or []
            if feature_imp and not pred_error:
                info_col.markdown("**🪄 Features que más influyen**")
                fi_df = pd.DataFrame(feature_imp, columns=["feature", "impact"])
                chart = alt.Chart(fi_df).mark_bar(color="#60a5fa").encode(
                    x=alt.X("impact", title="Impacto relativo"),
                    y=alt.Y("feature", sort="-x", title="Feature"),
                    tooltip=["feature", alt.Tooltip("impact", format=".3f")],
                ).properties(height=180)
                info_col.altair_chart(chart, use_container_width=True)

        with process_col:
            process_col.markdown("**🔧 Proceso**")
            process_col.write(process_label or "—")

            process_col.markdown("**📉 Recursos estimados**")
            resources = [
                ("Energía (kWh)", getattr(props, "energy_kwh", None), target_data.get("max_energy_kwh"), 2),
                ("Agua (L)", getattr(props, "water_l", None), target_data.get("max_water_l"), 1),
                ("Crew (min)", getattr(props, "crew_min", None), target_data.get("max_crew_min"), 0),
            ]
            for label, value, limit, precision in resources:
                resource_text = format_resource_text(value, limit, precision=precision)
                process_col.markdown(f"- **{label}:** {resource_text}")

        st.divider()

        st.markdown("**🛰️ Trazabilidad NASA**")
        st.write("IDs usados:", ", ".join(candidate.get("source_ids", [])) or "—")
        st.write(
            "Categorías:",
            ", ".join(map(str, candidate.get("source_categories", []))) or "—",
        )
        st.write("Flags:", ", ".join(map(str, candidate.get("source_flags", []))) or "—")
        regolith_pct = safe_float(candidate.get("regolith_pct"))
        if regolith_pct and regolith_pct > 0:
            st.write(f"**MGS-1 agregado:** {regolith_pct * 100:.0f}%")

        feat = candidate.get("features", {})
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
            highlight_badges: list[tuple[str, str]] = []
            logistics_reuse = feat.get("logistics_reuse_index")
            if logistics_reuse is not None:
                highlight_badges.append(
                    (
                        f"🚚 Reuso logístico: {format_number(logistics_reuse)}",
                        "ok",
                    )
                )
            gas_recovery = feat.get("gas_recovery_index")
            if gas_recovery is not None:
                highlight_badges.append(
                    (
                        f"🧪 Recupero gas: {format_number(gas_recovery)}",
                        "ok",
                    )
                )
            if regolith_pct and regolith_pct > 0:
                highlight_badges.append(
                    (f"⛰️ Mezcla MGS-1: {regolith_pct * 100:.0f}%", "warn")
                )
            if highlight_badges:
                badge_columns = st.columns(len(highlight_badges))
                for column, (label, kind) in zip(badge_columns, highlight_badges):
                    with column:
                        pill(label, kind=kind)
            logistics_metrics: list[dict[str, object]] = []
            if logistics_reuse is not None:
                logistics_metrics.append(
                    {
                        "Indicador": "Reuso logístico",
                        "Valor": float(logistics_reuse),
                        "Contexto": "Logistics-to-Living",
                    }
                )
            if gas_recovery is not None:
                logistics_metrics.append(
                    {
                        "Indicador": "Recupero gas",
                        "Valor": float(gas_recovery),
                        "Contexto": "Logistics-to-Living",
                    }
                )
            if logistics_metrics:
                st.markdown("**Logistics-to-Living · métricas clave**")
                logistics_df = pd.DataFrame(logistics_metrics)
                logistics_style = logistics_df.style.format({"Valor": "{:.2f}"})
                try:
                    logistics_style = logistics_style.hide(axis="index")
                except AttributeError:
                    logistics_style = logistics_style.hide_index()
                st.dataframe(logistics_style, use_container_width=True)
                chart = alt.Chart(logistics_df).mark_bar(color="#14b8a6").encode(
                    x=alt.X("Valor:Q", title="Valor"),
                    y=alt.Y("Indicador:N", sort="-x"),
                    tooltip=["Indicador", alt.Tooltip("Valor", format=".2f")],
                )
                st.altair_chart(chart, use_container_width=True)

        external_profiles = _collect_external_profiles(candidate, waste_df)
        polymer_section = external_profiles.get("polymer") or {}
        aluminium_section = external_profiles.get("aluminium") or {}
        should_render_reference = bool(polymer_section or aluminium_section)
        if (
            polymer_density_distribution.empty
            or polymer_tensile_distribution.empty
            or aluminium_tensile_distribution.empty
            or aluminium_yield_distribution.empty
        ):
            should_render_reference = True

        if should_render_reference:
            st.markdown("**Propiedades externas (NASA/industria)**")

            polymer_labels = polymer_section.get("labels") or []
            if polymer_labels:
                label_columns = st.columns(len(polymer_labels))
                for column, label in zip(label_columns, polymer_labels, strict=False):
                    with column:
                        pill(f"Polímero · {label}", kind="info")

            polymer_metrics = polymer_section.get("metrics", {}) or {}
            if polymer_metrics:
                metric_columns = st.columns(len(polymer_metrics))
                for column, (metric_key, metric_value) in zip(
                    metric_columns, polymer_metrics.items(), strict=False
                ):
                    label = POLYMER_LABEL_MAP.get(metric_key, metric_key)
                    column.metric(label, _format_reference_value(metric_key, float(metric_value)))

            density_value = polymer_metrics.get("density_g_cm3")
            tensile_value = polymer_metrics.get("tensile_mpa")
            _render_reference_distribution(
                polymer_density_distribution,
                density_value,
                field="density",
                axis_label="Densidad inventario (g/cm³)",
                histogram_color="#22d3ee",
                reference_color="#f97316",
                empty_message=(
                    "No hay densidades de polímeros en el inventario actual para comparar."
                ),
            )

            _render_reference_distribution(
                polymer_tensile_distribution,
                tensile_value,
                field="tensile",
                axis_label="σₜ inventario (MPa)",
                histogram_color="#f472b6",
                reference_color="#f97316",
                empty_message=(
                    "No hay datos de resistencia a tracción de polímeros en el inventario actual."
                ),
            )

            aluminium_labels = aluminium_section.get("labels") or []
            if aluminium_labels:
                label_columns = st.columns(len(aluminium_labels))
                for column, label in zip(label_columns, aluminium_labels, strict=False):
                    with column:
                        pill(f"Aluminio · {label}", kind="accent")

            aluminium_metrics = aluminium_section.get("metrics", {}) or {}
            if aluminium_metrics:
                metric_columns = st.columns(len(aluminium_metrics))
                for column, (metric_key, metric_value) in zip(
                    metric_columns, aluminium_metrics.items(), strict=False
                ):
                    label = ALUMINIUM_LABEL_MAP.get(metric_key, metric_key)
                    column.metric(label, _format_reference_value(metric_key, float(metric_value)))

            tensile_value = aluminium_metrics.get("tensile_mpa")
            yield_value = aluminium_metrics.get("yield_mpa")
            _render_reference_distribution(
                aluminium_tensile_distribution,
                tensile_value,
                field="tensile",
                axis_label="σₜ inventario (MPa)",
                histogram_color="#f97316",
                reference_color="#22d3ee",
                empty_message=(
                    "No hay datos de tracción de aluminio en el inventario actual para comparar."
                ),
            )

            _render_reference_distribution(
                aluminium_yield_distribution,
                yield_value,
                field="yield_strength",
                axis_label="σᵧ inventario (MPa)",
                histogram_color="#fb923c",
                reference_color="#22d3ee",
                empty_message=(
                    "No hay datos de límite de fluencia de aluminio en el inventario actual."
                ),
            )

            st.caption(
                "Comparativa contra distribuciones del inventario (`polymer_composite_*`, `aluminium_alloys.csv`)."
            )

            scenario_label = str((target_data or {}).get("scenario") or "").strip()
            scenario_casefold = scenario_label.casefold()
            if scenario_casefold == "daring discoveries":
                st.info(
                    "🧯 Escenario Daring Discoveries: conectar con el playbook de filtros de carbono"
                    " para capturar VOCs y regenerar cartuchos en paralelo al proceso." 
                )
            elif scenario_casefold == "residence renovations":
                st.info(
                    "🏠 Escenario Residence: priorizar valorización de volumen útil en hábitat y"
                    " logística compacta." 
                )
            if feat.get("latent_vector"):
                st.caption("Latente Rex-AI incluido para análisis generativo.")

        breakdown = candidate.get("score_breakdown") or {}
        contribs = breakdown.get("contributions") or {}
        penalties = breakdown.get("penalties") or {}
        if contribs or penalties:
            st.markdown("**⚖️ Desglose del score**")
            contrib_col, penalty_col = st.columns(2)
            if contribs:
                contrib_df = pd.DataFrame(
                    [{"Factor": k, "+": float(v)} for k, v in contribs.items()]
                ).sort_values("+", ascending=False)
                with contrib_col:
                    st.markdown("**Contribuciones**")
                    st.dataframe(contrib_df, hide_index=True, use_container_width=True)
            if penalties:
                pen_df = pd.DataFrame(
                    [{"Penalización": k, "-": float(v)} for k, v in penalties.items()]
                ).sort_values("-", ascending=False)
                with penalty_col:
                    st.markdown("**Penalizaciones**")
                    st.dataframe(pen_df, hide_index=True, use_container_width=True)

        badge = render_safety_indicator(candidate, target_data=target_data)

        if st.button(f"✅ Seleccionar Opción {idx}", key=f"pick_{idx}"):
            view_model.set_selected(candidate, badge)
            st.success(
                "Opción seleccionada. Avanzá a **4 · Resultados**, **5 · Compare & Explain** "
                "o **6 · Pareto & Export** para revisar métricas, justificar la decisión y "
                "descargar entregables."
            )

playbook_filters_applied = False
if _playbook_prefilters is not None and isinstance(target, dict):
    playbook_filters_applied = view_model.apply_playbook_prefilters(
        _playbook_prefilters, target
    )

# ----------------------------- Encabezado -----------------------------
st.header("🧪 Generador asistido por IA")
header_badges = _collect_target_badges(target)
if header_badges:
    badge_columns = st.columns(len(header_badges))
    for column, (label, kind) in zip(badge_columns, header_badges):
        with column:
            pill(label, kind=kind)
st.caption(
    "Construí propuestas que cumplan el objetivo definido, comparando consumo"
    " de recursos, riesgo operativo y señales de laboratorio en una sola vista."
)

st.info(
    "**Flujo recomendado**\n"
    "1. Configurá el objetivo → elegí el escenario y los límites técnicos clave.\n"
    "2. Ajustá filtros → concentrá la búsqueda en residuos, procesos y penalizaciones"
    " relevantes.\n"
    "3. Generá candidatos → ejecutá la IA y revisá cómo cada receta equilibra"
    " rigidez, estanqueidad y recursos.\n"
    "4. Analizá resultados → usá las páginas siguientes para validar riesgos,"
    " explicar la decisión y preparar entregables."
)

if playbook_filters_applied:
    if _playbook_prefill_label:
        st.success(
            f"Filtros recomendados para **{_playbook_prefill_label}** activados."
            " Explorá el showroom para ver cómo afectan al ranking."
        )
    else:
        st.success(
            "Filtros recomendados activados. Revisá el showroom para entender la"
            " priorización propuesta."
        )

# ----------------------------- Pre-condición: target -----------------------------
if not target:
    st.warning(
        "Definí primero las metas de rigidez, estanqueidad y recursos en **2 · Target"
        " Designer** para habilitar el generador."
    )
    st.stop()

# ----------------------------- Datos base -----------------------------
try:
    waste_df = load_waste_df()
    proc_df = load_process_df()
except MissingDatasetError as error:
    st.error(format_missing_dataset_message(error))
    st.stop()
polymer_density_distribution = numeric_series(
    waste_df, "pc_density_density_g_per_cm3"
)
polymer_tensile_distribution = numeric_series(
    waste_df, "pc_mechanics_tensile_strength_mpa"
)
aluminium_tensile_distribution = numeric_series(
    waste_df, "aluminium_tensile_strength_mpa"
)
aluminium_yield_distribution = numeric_series(
    waste_df, "aluminium_yield_strength_mpa"
)
proc_filtered = choose_process(
    target["name"], proc_df,
    scenario=target.get("scenario"),
    crew_time_low=target.get("crew_time_low", False)
)
if proc_filtered is None or proc_filtered.empty:
    proc_filtered = proc_df.copy()

view_model.ensure_defaults()
button_state = view_model.button_state
button_error = view_model.button_error

model_registry = get_model_registry()

# ----------------------------- Panel de control + IA -----------------------------
with layout_block("layout-grid layout-grid--dual layout-grid--flow", parent=None) as grid:
    with layout_stack(parent=grid) as left_column:
        with layout_block("side-panel layer-shadow fade-in", parent=left_column) as control:
            control.markdown("### 🎛️ Configurar lote")
            stored_mode = view_model.get_prediction_mode("Modo Rex-AI (ML)")
            mode = control.radio(
                "Motor de predicción",
                ("Modo Rex-AI (ML)", "Modo heurístico"),
                index=0 if stored_mode == "Modo Rex-AI (ML)" else 1,
                help="Elegí entre la IA entrenada o la estimación basada en reglas según el tipo de respuesta que necesitás comparar.",
            )
            view_model.set_prediction_mode(mode)
            use_ml = mode == "Modo Rex-AI (ML)"

            control.markdown("#### Ajustar parámetros")
            col_iters, col_recipes = control.columns(2)
            opt_evals = col_iters.slider(
                "Iteraciones",
                0,
                60,
                18,
                help="Cantidad de propuestas que la IA evaluará antes de mostrarte el lote final.",
            )
            n_candidates = col_recipes.slider(
                "Recetas a explorar",
                3,
                12,
                6,
                help="Cantidad de recetas distintas que se compararán en cada lote generado.",
            )

            advanced_warning = (
                "No es posible mostrar las opciones avanzadas en modo expandido; "
                "se utiliza la vista básica."
            )
            with _optional_container_expander(
                control,
                "Opciones avanzadas",
                warning_message=advanced_warning,
            ) as advanced:
                seed_default = view_model.get_seed_input()
                seed_input = _container_text_input(
                    advanced,
                    "Semilla (opcional)",
                    value=seed_default,
                    help="Ingresá un entero fijo para reproducir exactamente el mismo lote cuando vuelvas a generar.",
                )
                view_model.set_seed_input(seed_input)

            crew_low = target.get("crew_time_low", False)
            control.caption(
                "Los resultados priorizan %s"
                % ("tiempo de tripulación" if crew_low else "un balance general")
            )

            with control:
                run = action_button(
                    "⚙️ Generar lote",
                    key="generator_run_button",
                    state=button_state,
                    width="full",
                    help_text="Ejecuta Ax + BoTorch con los parámetros seleccionados.",
                    state_labels={
                        "loading": "Generando lote…",
                        "success": "Lote listo",
                        "error": "Reintentar",
                    },
                    state_messages={
                        "loading": "Ejecutando optimizador",
                        "success": "Resultados actualizados",
                        "error": "Revisá la configuración",
                    },
                )

            result: object | None = None
            if run and button_state != "loading":
                view_model.trigger_generation()
                st.rerun()

            button_state_now = view_model.button_state
            if button_error and button_state_now == "error":
                control.error(button_error)
            elif button_state_now == "success":
                control.caption("✅ Última corrida disponible abajo. Volvé a ejecutar si cambiás parámetros.")
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
            target_card.markdown("### 🎯 Objetivo")
            target_card.markdown(f"**{target.get('name', '—')}**")
            scenario_label = target.get("scenario") or "Escenario general"
            target_card.caption(f"Escenario: {scenario_label}")
            target_badges = _collect_target_badges(target)
            for label, kind in target_badges:
                pill(label, kind=kind)
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
            ai_panel.markdown("### 🧠 Modelo IA")
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
if view_model.should_generate:
    seed_value, seed_error = view_model.parse_seed()
    if seed_error:
        view_model.set_error(seed_error)
        st.stop()

    try:
        result = view_model.generate_candidates(
            waste_df,
            proc_filtered,
            target,
            n_candidates=n_candidates,
            crew_time_low=target.get("crew_time_low", False),
            optimizer_evals=opt_evals,
            use_ml=use_ml,
            seed=seed_value,
        )
    except Exception as exc:  # noqa: BLE001
        view_model.set_error(f"Error generando candidatos: {exc}")
    else:
        candidates_raw: object | None = None
        history_raw: object | None = None
        if isinstance(result, tuple):
            if len(result) >= 2:
                candidates_raw, history_raw = result[0], result[1]
            elif len(result) == 1:
                candidates_raw = result[0]
        elif isinstance(result, list):
            candidates_raw = result
        elif result is not None:
            candidates_raw = result

        if candidates_raw is None:
            processed_candidates: list[dict[str, object]] = []
        elif isinstance(candidates_raw, pd.DataFrame):
            processed_candidates = candidates_raw.to_dict("records")
        elif isinstance(candidates_raw, dict):
            processed_candidates = [candidates_raw]
        elif isinstance(candidates_raw, list):
            processed_candidates = []
            for cand in candidates_raw:
                if isinstance(cand, dict):
                    processed_candidates.append(cand)
                else:
                    try:
                        processed_candidates.append(dict(cand))
                    except (TypeError, ValueError):
                        try:
                            processed_candidates.append(vars(cand))
                        except TypeError:
                            processed_candidates.append({})
        else:
            try:
                processed_candidates = [dict(candidates_raw)]
            except (TypeError, ValueError):
                try:
                    processed_candidates = list(candidates_raw)
                except TypeError:
                    processed_candidates = [candidates_raw]

        if isinstance(history_raw, pd.DataFrame):
            history_df = history_raw
        elif history_raw is None:
            history_df = pd.DataFrame()
        else:
            history_df = pd.DataFrame(history_raw)

        view_model.store_results(processed_candidates, history_df)

    button_state = view_model.button_state
    button_error = view_model.button_error

# ----------------------------- Si no hay candidatos aún -----------------------------
st.divider()
cands = view_model.candidates
history_df = view_model.optimizer_history

if not cands:
    st.info(
        "Todavía no hay candidatos. Configurá los controles y presioná **Generar lote**. "
        "Verificá que el inventario incluya pouches, espumas, EVA/CTB, textiles o nitrilo y "
        "que el catálogo contenga P02, P03 o P04."
    )
    with st.expander("¿Cómo funciona el generador?", expanded=False):
        st.markdown(
            "- **Revisa residuos** con foco en los problemáticos de NASA.\n"
            "- **Elige un proceso** consistente (laminar, sinter con regolito, reconfigurar CTB).\n"
            "- **Predice** propiedades y recursos de cada receta.\n"
            "- **Puntúa** balanceando objetivos y costos.\n"
            "- **Muestra trazabilidad** para ver qué residuos se valorizaron."
        )
    st.stop()

# ----------------------------- Historial del optimizador -----------------------------
if isinstance(history_df, pd.DataFrame) and not history_df.empty:
    scene = ConvergenceScene(
        history_df,
        subtitle=(
            "Visualizá cómo evoluciona el frente Pareto tras cada iteración. Pasá el cursor "
            "para ver hipervolumen, dominancia y scores."
        ),
    )
    scene.render(st)

# ----------------------------- Resumen de ranking -----------------------------
summary_df = build_ranking_table(cands)

if not summary_df.empty:
    st.subheader("Ranking de candidatos")
    st.caption("Ordenado por score total con sellado y riesgo resumidos.")
    summary_style = summary_df.style.format(
        {
            "Score": "{:.3f}",
            "Rigidez": "{:.3f}",
            "Estanqueidad": "{:.3f}",
            "Energía (kWh)": "{:.3f}",
            "Agua (L)": "{:.3f}",
            "Crew (min)": "{:.1f}",
        }
    )
    try:
        summary_style = summary_style.hide(axis="index")
    except AttributeError:
        summary_style = summary_style.hide_index()
    st.dataframe(summary_style, use_container_width=True)

    score_chart = alt.Chart(summary_df).mark_bar(color="#2563eb").encode(
        x=alt.X("Score:Q", title="Score"),
        y=alt.Y("Proceso:N", sort="-x"),
        tooltip=["Proceso", alt.Tooltip("Score", format=".3f")],
    ).properties(height=220)
    st.altair_chart(score_chart, use_container_width=True)

    option_labels = {
        int(row["Rank"]): f"Opción {int(row['Rank'])} · {row['Proceso']}"
        for _, row in summary_df.iterrows()
    }
    select_options = list(option_labels.keys())
    selected_rank = st.selectbox(
        "📌 Destacar candidato",
        options=select_options,
        format_func=lambda value: option_labels.get(int(value), f"Opción {value}"),
    )
    if selected_rank is not None:
        focused = summary_df[summary_df["Rank"] == selected_rank].head(1)
        if not focused.empty:
            view_model.set_ranking_focus(focused.iloc[0].to_dict())

# ----------------------------- Showroom de candidatos -----------------------------
st.subheader("Resultados del generador")
st.caption(
    "Explorá cada receta con tabs de propiedades, recursos y trazabilidad. "
    "Ajustá el timeline lateral para priorizar rigidez o agua y filtrar rápido."
)

filtered_cands = render_candidate_showroom(cands, target)
for idx, candidate in enumerate(cands, start=1):
    render_candidate_card(candidate, idx, target, model_registry, view_model)

# ----------------------------- Explicación rápida (popover global) -----------------------------
top = filtered_cands[0] if filtered_cands else (cands[0] if cands else None)
pop = st.popover("🧠 ¿Por qué destacamos estas recetas?")
with pop:
    bullets = []
    bullets.append("• Sumamos puntos cuando **rigidez** y **estanqueidad** se acercan al objetivo.")
    bullets.append("• Restamos si supera límites de **agua**, **energía** o **tiempo de crew**.")
    if top:
        cats = " ".join(map(str, top.get("source_categories", []))).lower()
        flg = " ".join(map(str, top.get("source_flags", []))).lower()
        if any(k in cats or k in flg for k in ["pouches", "multilayer", "foam", "eva", "ctb", "nitrile", "wipe"]):
            bullets.append("• Priorizamos residuos problemáticos (pouches, EVA, multilayer, nitrilo, wipes).")
        if top.get("regolith_pct", 0) > 0:
            bullets.append("• Valoramos **MGS-1** como carga mineral para ISRU y menos dependencia de la Tierra.")
    st.markdown("\n".join(bullets))

# ----------------------------- Glosario -----------------------------
micro_divider()
with st.expander("📚 Glosario rápido", expanded=False):
    st.markdown(
        "- **ISRU**: *In-Situ Resource Utilization*. Usar recursos del lugar (en Marte, el **regolito** MGS-1).\n"
        "- **P02 – Press & Heat Lamination**: “plancha” y “fusiona” multicapa para dar forma.\n"
        "- **P03 – Sinter with MGS-1**: mezcla con regolito y sinteriza → piezas rígidas.\n"
        "- **P04 – CTB Reconfig**: reusar/transformar bolsas EVA/CTB con herrajes.\n"
        "- **Score**: qué tanto ‘cierra’ la opción según objetivo y límites de recursos/tiempo."
    )
st.info("Generá varias opciones y pasá a **4) Resultados**, **5) Comparar** y **6) Pareto & Export** para cerrar tu plan.")
