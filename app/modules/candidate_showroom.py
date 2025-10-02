"""Interactive 3D candidate showroom component for the generator page."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import streamlit as st

from app.modules.safety import check_safety, safety_badge
from app.modules.ui_blocks import action_button


_CSS_KEY = "__candidate_showroom_css__"
_SUCCESS_KEY = "__candidate_showroom_success__"
_MODAL_KEY = "showroom_modal"


def _normalize_success(payload: object) -> dict[str, Any]:
    """Return a normalized success payload with message and candidate index."""

    if isinstance(payload, dict):
        message = str(payload.get("message") or "")
        candidate_key = payload.get("candidate_key")
        if candidate_key in (None, ""):
            candidate_idx = payload.get("candidate_idx")
            try:
                candidate_key = str(int(candidate_idx))
            except (TypeError, ValueError):
                candidate_key = None
        else:
            candidate_key = str(candidate_key)
        return {"message": message, "candidate_key": candidate_key}

    if isinstance(payload, str) and payload.strip():
        return {"message": payload, "candidate_key": None}

    return {"message": "", "candidate_key": None}


def render_candidate_showroom(
    candidates: Sequence[dict],
    target: dict,
) -> list[dict]:
    """Render the CandidateShowroom component and return the filtered candidates."""

    if not candidates:
        st.info("Todavía no hay recetas para mostrar en el showroom.")
        return []

    _inject_css()

    success_data = _normalize_success(st.session_state.get(_SUCCESS_KEY))

    score_values = [float(_safe_number(c.get("score"))) for c in candidates]
    score_min, score_max = min(score_values), max(score_values)
    if score_min == score_max:
        score_min -= 0.01
        score_max += 0.01

    score_threshold = st.slider(
        "Score mínimo a visualizar",
        min_value=float(score_min),
        max_value=float(score_max),
        value=float(score_min),
        step=0.01,
        key="showroom_score_threshold",
    )

    only_safe = st.checkbox(
        "Sólo candidatos seguros",
        value=False,
        key="showroom_only_safe",
        help="Oculta recetas con bandera de riesgo en seguridad.",
    )

    resource_limits: dict[str, float] = {}
    resource_labels: list[str] = []

    energy_target = _extract_target_limit(target, "max_energy_kwh")
    water_target = _extract_target_limit(target, "max_water_l")
    crew_target = _extract_target_limit(target, "max_crew_min")

    available_resources: list[tuple[str, float | None]] = [
        ("energy", energy_target),
        ("water", water_target),
        ("crew", crew_target),
    ]

    resource_keys = [res for res, default in available_resources if default is not None]
    if resource_keys:
        st.caption("Límites de recursos")
        cols = st.columns(len(resource_keys), gap="small")
        col_idx = 0

        energy_values = [
            _safe_number(getattr((cand.get("props") or {}), "energy_kwh", None))
            for cand in candidates
        ]
        water_values = [
            _safe_number(getattr((cand.get("props") or {}), "water_l", None))
            for cand in candidates
        ]
        crew_values = [
            _safe_number(getattr((cand.get("props") or {}), "crew_min", None))
            for cand in candidates
        ]

        for resource in resource_keys:
            col = cols[col_idx]
            col_idx += 1

            if resource == "energy" and energy_target is not None:
                with col:
                    if st.checkbox(
                        "Respetar energía",
                        value=True,
                        key="showroom_limit_energy",
                        help="Descarta recetas que superen el objetivo de energía.",
                    ):
                        limit = _render_float_limit_slider(
                            "Energía máxima (kWh)",
                            default=energy_target,
                            values=energy_values,
                            step=0.05,
                            key="showroom_energy_limit_value",
                        )
                        resource_limits["energy"] = limit
                        resource_labels.append(f"Energía ≤ {limit:.2f} kWh")
                    else:
                        st.session_state.pop("showroom_energy_limit_value", None)

            if resource == "water" and water_target is not None:
                with col:
                    if st.checkbox(
                        "Respetar agua",
                        value=True,
                        key="showroom_limit_water",
                        help="Descarta recetas que usen más agua que la meta.",
                    ):
                        limit = _render_float_limit_slider(
                            "Agua máxima (L)",
                            default=water_target,
                            values=water_values,
                            step=0.01,
                            key="showroom_water_limit_value",
                        )
                        resource_limits["water"] = limit
                        resource_labels.append(f"Agua ≤ {limit:.2f} L")
                    else:
                        st.session_state.pop("showroom_water_limit_value", None)

            if resource == "crew" and crew_target is not None:
                with col:
                    if st.checkbox(
                        "Respetar crew",
                        value=True,
                        key="showroom_limit_crew",
                        help="Descarta recetas que requieran más crew que la meta.",
                    ):
                        limit = _render_int_limit_slider(
                            "Crew máximo",
                            default=crew_target,
                            values=crew_values,
                            key="showroom_crew_limit_value",
                        )
                        resource_limits["crew"] = float(limit)
                        resource_labels.append(f"Crew ≤ {int(limit)}")
                    else:
                        st.session_state.pop("showroom_crew_limit_value", None)

    threshold_active = score_threshold > score_min + 1e-6

    rows = _prepare_rows(
        candidates,
        score_threshold=score_threshold,
        only_safe=only_safe,
        threshold_active=threshold_active,
        resource_limits=resource_limits,
    )

    if not rows:
        st.warning("No hay candidatos que cumplan con los filtros seleccionados.")
        return []

    scenario_label = str(target.get("scenario") or "").strip()

    _render_candidate_table(
        rows,
        success_data,
        score_threshold,
        only_safe,
        threshold_active,
        resource_labels,
        scenario=scenario_label,
    )

    return [row["candidate"] for row in rows]


def _prepare_rows(
    candidates: Sequence[dict],
    *,
    score_threshold: float,
    only_safe: bool,
    threshold_active: bool,
    resource_limits: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resource_limits = resource_limits or {}
    energy_limit = resource_limits.get("energy")
    water_limit = resource_limits.get("water")
    crew_limit = resource_limits.get("crew")

    for idx, cand in enumerate(candidates):
        props = cand.get("props") or {}
        aux = cand.get("auxiliary") or {}
        materials = cand.get("materials", [])

        rigidity = _safe_number(_get_prop_value(props, "rigidity"))
        water = _safe_number(_get_prop_value(props, "water_l"))
        energy = _safe_number(_get_prop_value(props, "energy_kwh"))
        crew = _safe_number(_get_prop_value(props, "crew_min"))
        score = _safe_number(cand.get("score"), default=0.0)

        safety_flags = check_safety(materials, cand.get("process_name", ""), cand.get("process_id", ""))
        badge = safety_badge(safety_flags)
        level_text = str(badge.get("level", "")).strip().lower()
        is_safe = level_text in {"ok", "seguro", "safe"}

        if score < score_threshold or (only_safe and not is_safe):
            continue
        if energy_limit is not None and energy > energy_limit + 1e-9:
            continue
        if water_limit is not None and water > water_limit + 1e-9:
            continue
        if crew_limit is not None and crew > crew_limit + 1e-9:
            continue

        badge_sources: list[str] = []
        badge_sources.extend(str(b) for b in cand.get("timeline_badges", []))
        badge_sources.extend(_collect_badges(cand, aux))
        if only_safe and is_safe:
            badge_sources.append("🛡️ Filtro: seguros")
        if threshold_active:
            badge_sources.append(f"🎯 Score ≥ {score_threshold:.2f}")
        if energy_limit is not None:
            badge_sources.append("⚡ Dentro de límite de energía")
        if water_limit is not None:
            badge_sources.append("💧 Dentro de límite de agua")
        if crew_limit is not None:
            badge_sources.append("👥 Dentro de límite de crew")

        seen: set[str] = set()
        unique_badges: list[str] = []
        for badge_text in badge_sources:
            if badge_text and badge_text not in seen:
                unique_badges.append(badge_text)
                seen.add(badge_text)

        rows.append(
            {
                "candidate": cand,
                "score": score,
                "rigidity": rigidity,
                "water": water,
                "energy": energy,
                "crew": crew,
                "safety": badge,
                "is_safe": is_safe,
                "badges": unique_badges,
                "key": str(idx),
                "process_id": str(cand.get("process_id") or "—"),
                "process_name": str(cand.get("process_name") or "Proceso"),
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def _render_candidate_table(
    rows: Sequence[dict[str, Any]],
    success_data: dict[str, Any],
    score_threshold: float,
    only_safe: bool,
    threshold_active: bool,
    resource_labels: Sequence[str],
    *,
    scenario: str | None = None,
) -> None:
    st.markdown("#### Ranking de candidatos por score")

    active_filters: list[str] = []
    if only_safe:
        active_filters.append("Sólo seguros")
    if threshold_active and rows:
        active_filters.append(f"Score ≥ {score_threshold:.2f}")
    active_filters.extend(resource_labels)

    if active_filters:
        st.caption("Filtros activos: " + ", ".join(active_filters))

    st.markdown("<div class='candidate-table'>", unsafe_allow_html=True)
    column_weights = [0.24, 0.1, 0.1, 0.12, 0.12, 0.12, 0.1, 0.1]
    header_cols = st.columns(column_weights, gap="small")
    header_cols[0].markdown("**Proceso**")
    header_cols[1].markdown("**Score**")
    header_cols[2].markdown("**Rigidez**")
    header_cols[3].markdown("**Agua (L)**")
    header_cols[4].markdown("**Energía (kWh)**")
    header_cols[5].markdown("**Crew (min)**")
    header_cols[6].markdown("**Seguridad**")
    header_cols[7].markdown("**Acción**")
    st.markdown("<hr class='candidate-table__divider' />", unsafe_allow_html=True)

    for rank, row in enumerate(rows, start=1):
        _render_candidate_row(
            rank,
            row,
            success_data,
            column_weights,
            scenario=scenario,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_candidate_row(
    rank: int,
    row: dict[str, Any],
    success_data: dict[str, Any],
    column_weights: Sequence[float],
    *,
    scenario: str | None = None,
) -> None:
    (
        process_col,
        score_col,
        rigidity_col,
        water_col,
        energy_col,
        crew_col,
        safety_col,
        action_col,
    ) = st.columns(column_weights, gap="small")

    process_name = row["process_name"]
    process_id = row["process_id"]
    badges_html = "".join(
        f"<span class='showroom-badge'>{badge}</span>" for badge in row.get("badges", [])
    )
    process_col.markdown(
        f"<div class='candidate-proc'><strong>#{rank:02d} · {process_name}</strong><span>ID {process_id}</span></div>",
        unsafe_allow_html=True,
    )
    if badges_html:
        process_col.markdown(
            f"<div class='showroom-badges'>{badges_html}</div>",
            unsafe_allow_html=True,
        )

    score_col.markdown(f"<div class='metric-cell'>{row['score']:.3f}</div>", unsafe_allow_html=True)
    rigidity_col.markdown(
        f"<div class='metric-cell'>{row['rigidity']:.2f}</div>",
        unsafe_allow_html=True,
    )
    water_col.markdown(
        f"<div class='metric-cell'>{row['water']:.2f}</div>",
        unsafe_allow_html=True,
    )
    energy_col.markdown(
        f"<div class='metric-cell'>{row['energy']:.2f}</div>",
        unsafe_allow_html=True,
    )
    crew_col.markdown(
        f"<div class='metric-cell'>{row['crew']:.1f}</div>",
        unsafe_allow_html=True,
    )

    badge = row["safety"]
    level = str(badge.get("level", "—"))
    detail = str(badge.get("detail", ""))
    level_class = "ok" if row["is_safe"] else "risk"

    safety_col.markdown(
        "<div class='safety-pill safety-pill--"
        f"{level_class}'><strong>{level}</strong><span>{detail}</span></div>",
        unsafe_allow_html=True,
    )

    candidate_key = row["key"]
    modal_key = st.session_state.get(_MODAL_KEY)
    success_key = success_data.get("candidate_key")

    if modal_key == candidate_key and success_key != candidate_key:
        btn_state = "loading"
    elif success_key == candidate_key:
        btn_state = "success"
    else:
        btn_state = "idle"

    with action_col:
        if action_button(
            "✨ Seleccionar",
            key=f"showroom_select_{candidate_key}",
            state=btn_state,
            width="full",
            loading_label="Abriendo holograma…",
            success_label="Receta seleccionada",
            help_text="Confirmá la receta desde la ventana emergente.",
            status_hints={
                "idle": "",
                "loading": "Mostrando holograma",
                "success": "Receta lista para confirmar",
                "error": "Reintentá la selección",
            },
        ):
            st.session_state[_MODAL_KEY] = candidate_key
            current = _normalize_success(st.session_state.get(_SUCCESS_KEY))
            if current.get("candidate_key") != candidate_key:
                st.session_state.pop(_SUCCESS_KEY, None)

    if st.session_state.get(_MODAL_KEY) == candidate_key:
        with st.modal("Confirmación holográfica", key=f"modal_{candidate_key}"):
            st.markdown(
                _modal_html(row["candidate"], badge, scenario=scenario),
                unsafe_allow_html=True,
            )
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                confirm_label = _scenario_result_cta(scenario)
                if action_button(
                    f"✅ {confirm_label}",
                    key=f"confirm_{candidate_key}",
                    state="idle",
                    width="full",
                    loading_label="Sincronizando…",
                    success_label="Receta confirmada",
                    status_hints={
                        "idle": "",
                        "loading": "Sincronizando selección",
                        "success": "Receta confirmada",
                        "error": "No se pudo confirmar",
                    },
                    button_type="primary",
                ):
                    st.session_state["selected"] = {
                        "data": row["candidate"],
                        "safety": badge,
                    }
                    st.session_state[_SUCCESS_KEY] = {
                        "message": (
                            f"{process_name} confirmado. Revisá **4) Resultados**, "
                            "**5) Comparar** o **6) Pareto**."
                        ),
                        "candidate_key": candidate_key,
                    }
                    st.session_state.pop(_MODAL_KEY, None)
            with col_cancel:
                if action_button(
                    "Cancelar",
                    key=f"cancel_{candidate_key}",
                    state="idle",
                    width="full",
                ):
                    st.session_state.pop(_MODAL_KEY, None)

    if success_key == candidate_key and success_data.get("message"):
        action_col.markdown(
            f"<div class='inline-success'>✅ {success_data['message']}</div>",
            unsafe_allow_html=True,
        )


def _scenario_result_cta(scenario: str | None) -> str:
    scenario_key = (scenario or "").strip().casefold()
    label_map = {
        "residence renovations": "Enviar a Resultados Residence",
        "daring discoveries": "Enviar a Resultados Daring",
        "cosmic celebrations": "Enviar a Resultados Cosmic",
    }
    default_label = "Enviar a Resultados"
    return label_map.get(scenario_key, default_label if not scenario_key else f"Enviar a Resultados {scenario.strip()}")


def _scenario_steps(scenario: str | None) -> list[str]:
    scenario_key = (scenario or "").strip().casefold()
    base_steps = [
        "Revisá resultados detallados en la pestaña 4.",
        "Compará alternativas en la pestaña 5.",
        "Exportá plan u órdenes en la pestaña 6.",
    ]
    if scenario_key == "residence renovations":
        return [
            "Verificá paneles laminados antes de sellar la cabina Residence.",
            base_steps[0],
            base_steps[1],
        ]
    if scenario_key == "daring discoveries":
        return [
            "Prepará junta conductiva para el ensamblaje Daring.",
            base_steps[0],
            base_steps[2],
        ]
    if scenario_key == "cosmic celebrations":
        return [
            "Coordina logística ceremonial con la tripulación Cosmic.",
            base_steps[0],
            base_steps[1],
        ]
    return base_steps


def _safety_reminders(badge: dict) -> list[str]:
    detail = str(badge.get("detail", ""))
    lowered = detail.casefold()
    reminders: list[str] = []
    if "pfas" in lowered or "fluor" in lowered:
        reminders.append("⚠️ Aislar compuestos con indicios PFAS antes de continuar con el plan.")
    if "micropl" in lowered:
        reminders.append("⚠️ Capturá microplásticos generados durante el procesamiento y regístralos.")
    return reminders


def _modal_html(cand: dict, badge: dict, *, scenario: str | None = None) -> str:
    process = f"{cand.get('process_id', '')} · {cand.get('process_name', '')}".strip()
    score = _safe_number(cand.get("score"))
    steps_html = "".join(f"<li>{step}</li>" for step in _scenario_steps(scenario))
    reminders = _safety_reminders(badge)
    reminders_html = "".join(f"<p class='modal-reminder'>{text}</p>" for text in reminders)
    return f"""
    <div class='modal-holo'>
      <h2>Confirmar receta seleccionada</h2>
      <p>Proceso <strong>{process or 'Proceso'}</strong> con score <strong>{score:.3f}</strong>.</p>
      <ol>
        {steps_html}
      </ol>
      <div class='modal-badge'>Seguridad {badge['level']} · {badge['detail']}</div>
      {reminders_html}
    </div>
    """


def _collect_badges(cand: dict, aux: dict) -> list[str]:
    badges: list[str] = []
    if _safe_number(cand.get("regolith_pct"), default=0.0) > 0:
        badges.append("⛰️ ISRU MGS-1")
    src_cats = " ".join(map(str, cand.get("source_categories", []))).lower()
    src_flags = " ".join(map(str, cand.get("source_flags", []))).lower()
    if any(
        key in src_cats or key in src_flags
        for key in ["pouches", "multilayer", "foam", "eva", "ctb", "nitrile", "wipe"]
    ):
        badges.append("♻️ Valorización problemáticos")
    if aux.get("passes_seal", True):
        badges.append("🛡️ Seal ready")
    return badges


def _safe_number(value, default: float | None = None) -> float:
    try:
        if value is None:
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return float(default or 0.0)


def _get_prop_value(props: object, key: str) -> Any:
    if isinstance(props, Mapping):
        return props.get(key)
    return getattr(props, key, None)


def _inject_css() -> None:
    if st.session_state.get(_CSS_KEY):
        return

    st.markdown(
        """
        <style>
        .candidate-table {
            position: relative;
            background: linear-gradient(155deg, rgba(15,23,42,0.92), rgba(30,41,59,0.88));
            border-radius: 26px;
            padding: 26px 30px;
            border: 1px solid rgba(148,163,184,0.2);
            box-shadow: 12px 18px 40px rgba(2,6,23,0.5);
            overflow-x: auto;
        }
        .candidate-table [data-testid="stHorizontalBlock"] {
            min-width: 780px;
        }
        .candidate-table [data-testid="column"] {
            min-width: 110px;
        }
        .candidate-table [data-testid="column"]:first-child {
            min-width: 220px;
        }
        .candidate-table__divider {
            border: none;
            border-top: 1px solid rgba(148,163,184,0.2);
            margin: 12px 0 24px;
        }
        .candidate-proc {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .candidate-proc strong {
            font-size: 1rem;
            letter-spacing: 0.04em;
        }
        .candidate-proc span {
            font-size: 0.78rem;
            opacity: 0.7;
        }
        .showroom-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .showroom-badge {
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.78rem;
            background: rgba(59,130,246,0.18);
            border: 1px solid rgba(148,163,184,0.26);
            backdrop-filter: blur(12px);
        }
        .metric-cell {
            font-size: 0.95rem;
            font-variant-numeric: tabular-nums;
            padding: 8px 0;
            white-space: nowrap;
        }
        .metric-cell {
            text-align: center;
        }
        .metric-cell::after {
            content: "";
        }
        .safety-pill {
            border-radius: 16px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 2px;
            font-size: 0.8rem;
            border: 1px solid rgba(148,163,184,0.22);
        }
        .safety-pill strong {
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-size: 0.75rem;
        }
        .safety-pill span {
            font-size: 0.74rem;
            opacity: 0.8;
        }
        .safety-pill--ok {
            background: rgba(45,212,191,0.16);
            border-color: rgba(16,185,129,0.45);
            color: #ccfbf1;
        }
        .safety-pill--risk {
            background: rgba(248,113,113,0.14);
            border-color: rgba(248,113,113,0.35);
            color: #fecaca;
        }
        .inline-success {
            margin-top: 10px;
            font-size: 0.78rem;
            color: #bbf7d0;
        }
        .modal-holo {
            background: radial-gradient(circle at top, rgba(59,130,246,0.3), rgba(15,23,42,0.92));
            border:1px solid rgba(148,163,184,0.3);
            border-radius:24px;
            padding:24px 28px;
            box-shadow: 0 0 50px rgba(59,130,246,0.45);
        }
        .modal-holo h2 {margin-top:0;}
        .modal-holo ol {padding-left:20px;}
        .modal-badge {
            margin-top:18px;
            padding:8px 12px;
            border-radius:12px;
            background:rgba(45,212,191,0.16);
            border:1px solid rgba(94,234,212,0.35);
            display:inline-block;
        }
        @media (max-width: 960px) {
            .candidate-table {
                padding: 24px 20px 28px;
            }
            .candidate-table [data-testid="stHorizontalBlock"] {
                min-width: 720px;
            }
            .candidate-table [data-testid="column"]:first-child {
                min-width: 200px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_CSS_KEY] = True



def _extract_target_limit(target: dict, key: str) -> float | None:
    try:
        value = target.get(key)
    except AttributeError:
        return None
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_float_limit_slider(
    label: str,
    *,
    default: float,
    values: Sequence[float],
    step: float,
    key: str,
) -> float:
    numbers = [default]
    numbers.extend(v for v in values if v is not None)
    slider_max = max(numbers)
    if slider_max <= default:
        slider_max = default + max(step, abs(default) * 0.1 or step)
    if slider_max <= 0:
        slider_max = max(default + step, step)
    return float(
        st.slider(
            label,
            min_value=0.0,
            max_value=float(slider_max),
            value=float(default),
            step=step,
            key=key,
        )
    )


def _render_int_limit_slider(
    label: str,
    *,
    default: float,
    values: Sequence[float],
    key: str,
) -> int:
    default_int = int(round(default))
    numbers = [default_int]
    numbers.extend(int(round(v)) for v in values if v is not None)
    slider_max = max(numbers)
    if slider_max <= default_int:
        slider_max = default_int + 1
    if slider_max <= 0:
        slider_max = max(default_int + 1, 1)
    return int(
        st.slider(
            label,
            min_value=0,
            max_value=int(slider_max),
            value=int(default_int),
            step=1,
            key=key,
        )
    )
