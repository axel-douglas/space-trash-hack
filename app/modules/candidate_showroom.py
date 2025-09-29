"""Interactive 3D candidate showroom component for the generator page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import streamlit as st

from app.modules.safety import check_safety, safety_badge


_CSS_KEY = "__candidate_showroom_css__"
_SUCCESS_KEY = "__candidate_showroom_success__"


@dataclass
class _PropSnapshot:
    label: str
    value: float
    target: float | None
    heuristic: float | None
    ci: Sequence[float] | None


def render_candidate_showroom(
    candidates: Sequence[dict],
    target: dict,
) -> list[dict]:
    """Render the CandidateShowroom component and return the filtered candidates."""

    if not candidates:
        st.info("Todavía no hay recetas para mostrar en el showroom.")
        return []

    _inject_css()

    success_payload = st.session_state.pop(_SUCCESS_KEY, None)
    if success_payload:
        st.success(success_payload)

    priority = st.slider(
        "Prioridad: Rigidez ↔ Agua",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=(
            "Arrastrá para ponderar rigidez (1.0) frente al consumo de agua (0.0). "
            "Afecta el orden sugerido en la timeline lateral."
        ),
        key="showroom_priority_slider",
    )

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

    filtered = [
        cand for cand in candidates if float(_safe_number(cand.get("score"))) >= score_threshold
    ]
    if not filtered:
        filtered = list(candidates)

    timeline_sorted = _sort_for_timeline(filtered, priority, target)

    col_timeline, col_cards = st.columns([0.32, 0.68], gap="large")

    with col_timeline:
        st.markdown("#### 🛰️ Timeline comparativa")
        st.caption(
            "Orden sugerido según la ponderación elegida. Cada punto resume score, rigidez y agua."
        )
        st.markdown(_build_timeline_html(timeline_sorted, target), unsafe_allow_html=True)

    with col_cards:
        for idx, cand in enumerate(filtered):
            _render_candidate_card(cand, idx, target)

    return filtered


def _render_candidate_card(cand: dict, idx: int, target: dict) -> None:
    props = cand.get("props")
    heur = cand.get("heuristic_props", props)
    ci = cand.get("confidence_interval") or {}
    uncertainty = cand.get("uncertainty") or {}
    aux = cand.get("auxiliary") or {}

    rigidity = _safe_number(getattr(props, "rigidity", None))
    tightness = _safe_number(getattr(props, "tightness", None))
    energy = _safe_number(getattr(props, "energy_kwh", None))
    water = _safe_number(getattr(props, "water_l", None))
    crew = _safe_number(getattr(props, "crew_min", None))

    heur_rig = getattr(heur, "rigidity", None)
    heur_rig = _safe_number(heur_rig) if heur_rig is not None else None
    heur_tight = getattr(heur, "tightness", None)
    heur_tight = _safe_number(heur_tight) if heur_tight is not None else None

    score = _safe_number(cand.get("score"), default=0.0)
    process_label = f"{cand.get('process_id', '—')} · {cand.get('process_name', '—')}"

    badges = _collect_badges(cand, aux)

    st.markdown("<div class='showroom-card'>", unsafe_allow_html=True)
    st.markdown(
        _card_header_html(idx, score, process_label, badges, aux),
        unsafe_allow_html=True,
    )

    tab_props, tab_resources, tab_trace = st.tabs([
        "Propiedades",
        "Recursos",
        "Trazabilidad",
    ])

    with tab_props:
        snapshots: list[_PropSnapshot] = [
            _PropSnapshot(
                label="Rigidez",
                value=rigidity,
                target=_safe_number(target.get("rigidity")),
                heuristic=heur_rig,
                ci=ci.get("rigidez"),
            ),
            _PropSnapshot(
                label="Estanqueidad",
                value=tightness,
                target=_safe_number(target.get("tightness")),
                heuristic=heur_tight,
                ci=ci.get("estanqueidad"),
            ),
        ]
        st.markdown(_radial_group_html(snapshots), unsafe_allow_html=True)

        if uncertainty:
            unc_html = "<div class='uncertainty-grid'>" + "".join(
                f"<span><strong>{k}</strong> ±{_safe_number(v, default=0.0):.3f}</span>"
                for k, v in uncertainty.items()
            ) + "</div>"
            st.markdown(unc_html, unsafe_allow_html=True)

    with tab_resources:
        max_energy = _safe_number(target.get("max_energy_kwh")) or None
        max_water = _safe_number(target.get("max_water_l")) or None
        max_crew = _safe_number(target.get("max_crew_min")) or None

        st.markdown(
            _neo_bar_group_html(
                [
                    ("Energía (kWh)", energy, max_energy),
                    ("Agua (L)", water, max_water),
                    ("Crew (min)", crew, max_crew),
                ]
            ),
            unsafe_allow_html=True,
        )

    with tab_trace:
        st.markdown(_traceability_html(cand), unsafe_allow_html=True)

    # Seguridad y selección
    materials = cand.get("materials", [])
    safety_flags = check_safety(materials, cand.get("process_name", ""), cand.get("process_id", ""))
    badge = safety_badge(safety_flags)
    st.markdown(
        f"<div class='safety-line'>🔒 Seguridad: <span>{badge['level']} · {badge['detail']}</span></div>",
        unsafe_allow_html=True,
    )

    btn_key = f"showroom_select_{idx}"
    if st.button("Seleccionar esta receta", key=btn_key, type="primary"):
        st.session_state["showroom_modal"] = idx

    if st.session_state.get("showroom_modal") == idx:
        with st.modal("Confirmación holográfica", key=f"modal_{idx}"):
            st.markdown(_modal_html(cand, badge), unsafe_allow_html=True)
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("Confirmar selección", key=f"confirm_{idx}", type="primary"):
                    st.session_state["selected"] = {"data": cand, "safety": badge}
                    st.session_state[_SUCCESS_KEY] = (
                        f"Opción {idx + 1} lista. Revisá **4) Resultados**, **5) Comparar** o **6) Pareto**."
                    )
                    st.session_state.pop("showroom_modal", None)
            with col_cancel:
                if st.button("Cancelar", key=f"cancel_{idx}"):
                    st.session_state.pop("showroom_modal", None)

    st.markdown("</div>", unsafe_allow_html=True)


def _build_timeline_html(candidates: Sequence[dict], target: dict) -> str:
    items = []
    for idx, cand in enumerate(candidates, start=1):
        props = cand.get("props")
        rigidity = _safe_number(getattr(props, "rigidity", None))
        water = _safe_number(getattr(props, "water_l", None))
        water_max = _safe_number(target.get("max_water_l"), default=1.0) or 1.0
        water_ratio = min(max(water / water_max if water_max else 0.0, 0.0), 1.0)
        score = _safe_number(cand.get("score"))
        process = f"{cand.get('process_id', '')} · {cand.get('process_name', '')}".strip()
        items.append(
            f"""
            <li class='timeline-item'>
              <div class='timeline-index'>#{idx:02d}</div>
              <div class='timeline-body'>
                <strong>{score:.3f}</strong>
                <span>{process or 'Proceso'}</span>
                <div class='timeline-bars'>
                  <div class='timeline-pill'>Rigidez {rigidity:.2f}</div>
                  <div class='timeline-pill pill-water'>Agua {water:.2f} ({water_ratio*100:.0f}% máx)</div>
                </div>
              </div>
            </li>
            """
        )

    return "<ul class='timeline'>" + "".join(items) + "</ul>"


def _sort_for_timeline(candidates: Iterable[dict], priority: float, target: dict) -> list[dict]:
    weight_r = priority
    weight_w = 1.0 - priority
    max_water = _safe_number(target.get("max_water_l"), default=1.0) or 1.0

    def _score(cand: dict) -> float:
        props = cand.get("props")
        rigidity = _safe_number(getattr(props, "rigidity", None))
        water = _safe_number(getattr(props, "water_l", None))
        water_ratio = water / max_water if max_water else 0.0
        return (weight_r * rigidity) - (weight_w * water_ratio)

    return sorted(candidates, key=_score, reverse=True)


def _card_header_html(
    idx: int,
    score: float,
    process_label: str,
    badges: Sequence[str],
    aux: dict,
) -> str:
    seal = "✅" if aux.get("passes_seal", True) else "⚠️"
    risk = aux.get("process_risk_label", "—")
    badges_html = "".join(f"<span class='showroom-badge'>{b}</span>" for b in badges)
    return f"""
    <div class='showroom-card__head'>
      <div>
        <span class='showroom-rank'>Opción {idx + 1:02d}</span>
        <h3>{process_label}</h3>
        <div class='showroom-meta'>Score {score:.3f} · Seal {seal} · Riesgo {risk}</div>
        <div class='showroom-badges'>{badges_html}</div>
      </div>
    </div>
    """


def _radial_group_html(snapshots: Sequence[_PropSnapshot]) -> str:
    blocks: list[str] = []
    for snap in snapshots:
        if snap.target in (None, 0):
            pct = min(max(snap.value, 0.0), 1.0) * 100
        else:
            pct = min(max(snap.value / snap.target, 0.0), 1.0) * 100
        heur_text = (
            f"<span class='micro'>Heurístico {snap.heuristic:.2f}</span>" if snap.heuristic is not None else ""
        )
        ci_text = ""
        if snap.ci:
            try:
                lo, hi = snap.ci
                ci_text = f"<span class='micro'>CI95% [{float(lo):.2f}, {float(hi):.2f}]</span>"
            except (TypeError, ValueError):
                ci_text = ""
        blocks.append(
            f"""
            <div class='radial' style='--percent:{pct:.1f};'>
              <div class='radial-core'>
                <div class='value'>{snap.value:.2f}</div>
                <span>{snap.label}</span>
                {heur_text}
                {ci_text}
              </div>
            </div>
            """
        )
    return "<div class='radial-wrap'>" + "".join(blocks) + "</div>"


def _neo_bar_group_html(rows: Sequence[tuple[str, float, float | None]]) -> str:
    blocks: list[str] = []
    for label, value, limit in rows:
        limit = limit or 0.0
        pct = 0.0
        if limit:
            pct = min(max(value / limit, 0.0), 1.0) * 100
        blocks.append(
            f"""
            <div class='neo-bar' style='--fill:{pct:.1f};'>
              <span><strong>{label}</strong><em>{value:.2f}{' / ' + str(limit) if limit else ''}</em></span>
              <div class='meter'></div>
            </div>
            """
        )
    return "<div class='neo-group'>" + "".join(blocks) + "</div>"


def _traceability_html(cand: dict) -> str:
    source_ids = ", ".join(cand.get("source_ids", [])) or "—"
    categories = ", ".join(map(str, cand.get("source_categories", []))) or "—"
    flags = ", ".join(map(str, cand.get("source_flags", []))) or "—"
    materials = ", ".join(cand.get("materials", [])) or "—"
    regolith_pct = _safe_number(cand.get("regolith_pct"), default=0.0) * 100
    feature_imp = cand.get("feature_importance") or []
    feature_rows = "".join(
        f"<tr><td>{name}</td><td>{_safe_number(value, default=0.0):.3f}</td></tr>"
        for name, value in feature_imp[:6]
    )
    feature_table = (
        f"<table class='feat-table'><tbody>{feature_rows}</tbody></table>" if feature_rows else ""
    )
    return f"""
    <div class='trace-grid'>
      <div><span class='label'>IDs NASA</span><p>{source_ids}</p></div>
      <div><span class='label'>Categorías</span><p>{categories}</p></div>
      <div><span class='label'>Flags</span><p>{flags}</p></div>
      <div><span class='label'>Materiales</span><p>{materials}</p></div>
      <div><span class='label'>MGS-1</span><p>{regolith_pct:.0f}%</p></div>
    </div>
    {feature_table}
    """


def _modal_html(cand: dict, badge: dict) -> str:
    process = f"{cand.get('process_id', '')} · {cand.get('process_name', '')}".strip()
    score = _safe_number(cand.get("score"))
    return f"""
    <div class='modal-holo'>
      <h2>Confirmar receta seleccionada</h2>
      <p>Proceso <strong>{process or 'Proceso'}</strong> con score <strong>{score:.3f}</strong>.</p>
      <ol>
        <li>Revisá resultados detallados en la pestaña 4.</li>
        <li>Compará alternativas en la pestaña 5.</li>
        <li>Exportá plan u órdenes en la pestaña 6.</li>
      </ol>
      <div class='modal-badge'>Seguridad {badge['level']} · {badge['detail']}</div>
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


def _inject_css() -> None:
    if st.session_state.get(_CSS_KEY):
        return

    st.markdown(
        """
        <style>
        .showroom-card {
            position: relative;
            background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(30,41,59,0.88));
            border-radius: 26px;
            padding: 28px 30px 24px;
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 12px 18px 40px rgba(2,6,23,0.55), -6px -10px 30px rgba(59,130,246,0.08);
            margin-bottom: 30px;
            transition: transform 0.6s ease, box-shadow 0.6s ease;
            overflow: hidden;
        }
        .showroom-card::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 26px;
            background: radial-gradient(circle at top left, rgba(59,130,246,0.28), transparent 45%);
            opacity: 0;
            transition: opacity 0.6s ease;
            pointer-events: none;
        }
        .showroom-card:hover {
            transform: translateY(-6px) rotateX(1.5deg) rotateY(-1.5deg);
            box-shadow: 18px 26px 60px rgba(2,6,23,0.65);
        }
        .showroom-card:hover::after {opacity:1;}
        .showroom-card__head h3 {
            margin: 4px 0 6px;
        }
        .showroom-rank {
            font-size: 0.9rem;
            letter-spacing: 0.08em;
            opacity: 0.72;
            text-transform: uppercase;
        }
        .showroom-meta {
            font-size: 0.85rem;
            opacity: 0.75;
        }
        .showroom-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }
        .showroom-badge {
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.78rem;
            background: rgba(59,130,246,0.18);
            border: 1px solid rgba(148,163,184,0.26);
            backdrop-filter: blur(12px);
        }
        .radial-wrap {
            display:flex;
            gap:18px;
            flex-wrap:wrap;
            margin: 18px 0 6px;
        }
        .radial {
            width: 150px;
            aspect-ratio:1;
            border-radius:32px;
            background: conic-gradient(#38bdf8 calc(var(--percent) * 3.6deg), rgba(30,41,59,0.55) 0deg);
            display:flex;
            align-items:center;
            justify-content:center;
            position:relative;
            transition: background 0.4s ease;
        }
        .radial::before {
            content:"";
            position:absolute;
            inset:12px;
            border-radius:26px;
            background: rgba(15,23,42,0.95);
            box-shadow: inset 0 0 30px rgba(15,23,42,0.9);
        }
        .radial-core {
            position:relative;
            text-align:center;
            color:#e2e8f0;
        }
        .radial .value {
            font-size:1.4rem;
            font-weight:600;
        }
        .radial span {
            display:block;
            font-size:0.82rem;
            opacity:0.8;
        }
        .radial .micro {
            font-size:0.72rem;
            opacity:0.7;
        }
        .neo-group {
            display:flex;
            flex-direction:column;
            gap:14px;
            margin-top:16px;
        }
        .neo-bar {
            padding:14px 16px;
            border-radius:18px;
            background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(30,41,59,0.88));
            box-shadow: 10px 10px 22px rgba(2,6,23,0.55), -8px -8px 18px rgba(59,130,246,0.12);
            position:relative;
        }
        .neo-bar span {
            display:flex;
            justify-content:space-between;
            font-size:0.78rem;
            text-transform:uppercase;
            letter-spacing:0.06em;
            opacity:0.75;
        }
        .neo-bar strong {
            color:#bae6fd;
            font-weight:600;
        }
        .neo-bar em {
            font-style:normal;
            color:#cbd5f5;
        }
        .neo-bar .meter {
            margin-top:12px;
            width:100%;
            height:12px;
            border-radius:20px;
            background:rgba(148,163,184,0.18);
            overflow:hidden;
            position:relative;
        }
        .neo-bar .meter::after {
            content:"";
            position:absolute;
            inset:0;
            width:calc(var(--fill) * 1%);
            background:linear-gradient(90deg, rgba(59,130,246,0.9), rgba(45,212,191,0.9));
            box-shadow:0 0 16px rgba(45,212,191,0.45);
            transition:width 0.6s ease;
        }
        .trace-grid {
            display:grid;
            gap:12px;
            grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
            margin-top:12px;
        }
        .trace-grid .label {
            font-size:0.72rem;
            letter-spacing:0.08em;
            text-transform:uppercase;
            opacity:0.6;
        }
        .trace-grid p {
            margin:4px 0 0;
        }
        .feat-table {
            width:100%;
            margin-top:16px;
            border-collapse:collapse;
            font-size:0.82rem;
        }
        .feat-table td {
            padding:6px 8px;
            border-bottom:1px solid rgba(148,163,184,0.16);
        }
        .uncertainty-grid {
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            margin-top:10px;
            font-size:0.76rem;
            opacity:0.7;
        }
        .timeline {
            list-style:none;
            padding:0 0 0 20px;
            margin:22px 0 0;
            border-left:2px solid rgba(148,163,184,0.25);
        }
        .timeline-item {
            position:relative;
            margin-bottom:18px;
        }
        .timeline-item::before {
            content:"";
            position:absolute;
            left:-27px;
            top:6px;
            width:12px;
            height:12px;
            border-radius:50%;
            background:linear-gradient(135deg, rgba(59,130,246,0.9), rgba(45,212,191,0.9));
            box-shadow:0 0 12px rgba(56,189,248,0.6);
        }
        .timeline-index {
            font-size:0.75rem;
            opacity:0.65;
            letter-spacing:0.1em;
        }
        .timeline-body {
            padding-left:6px;
        }
        .timeline-body strong {
            display:block;
            font-size:1.05rem;
        }
        .timeline-body span {
            display:block;
            font-size:0.82rem;
            opacity:0.75;
        }
        .timeline-bars {
            display:flex;
            gap:8px;
            flex-wrap:wrap;
            margin-top:8px;
        }
        .timeline-pill {
            padding:4px 10px;
            border-radius:999px;
            font-size:0.72rem;
            background:rgba(59,130,246,0.15);
            border:1px solid rgba(148,163,184,0.24);
        }
        .timeline-pill.pill-water {
            background:rgba(14,165,233,0.15);
        }
        .safety-line {
            margin:22px 0 12px;
            font-size:0.85rem;
            opacity:0.85;
        }
        .safety-line span {
            font-weight:600;
            color:#a5f3fc;
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
        div[data-baseweb="tab-list"] {
            gap:8px;
        }
        div[data-baseweb="tab"] {
            border-radius:999px !important;
            background:rgba(30,41,59,0.6);
            border:1px solid rgba(148,163,184,0.18);
            padding:10px 20px;
            transition:all 0.3s ease;
            color:#e2e8f0;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background:rgba(59,130,246,0.28);
            border-color:rgba(56,189,248,0.55);
            box-shadow:0 0 18px rgba(59,130,246,0.45);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_CSS_KEY] = True


