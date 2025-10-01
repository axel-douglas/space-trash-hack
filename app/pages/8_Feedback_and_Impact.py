# app/pages/8_Feedback_and_Impact.py
import _bootstrap  # noqa: F401

# ⚠️ Debe ser la PRIMERA llamada Streamlit:
import streamlit as st

st.set_page_config(page_title="Feedback & Impact", page_icon="📝", layout="wide")

from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.ui_blocks import load_theme

set_active_step("feedback")

load_theme()

render_breadcrumbs("feedback")

import json
import pandas as pd
from datetime import datetime
from io import StringIO
from typing import Any

from app.modules.impact import (
    ImpactEntry, FeedbackEntry, append_impact, append_feedback,
    load_impact_df, load_feedback_df, summarize_impact
)
from app.modules.data_sources import load_regolith_thermal_profiles


def _parse_extra_blob(blob: Any) -> dict:
    """Convierte el campo `extra` a un dict manejando texto plano o JSON."""
    if isinstance(blob, dict):
        return blob
    if not isinstance(blob, str):
        return {}
    text = blob.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    except json.JSONDecodeError:
        pass

    data = {}
    leftovers: list[str] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            data[key.strip()] = value.strip()
        else:
            leftovers.append(chunk)
    if leftovers and "raw" not in data:
        data["raw"] = "; ".join(leftovers)
    return data


def _with_extra_columns(df: pd.DataFrame, rename_map: dict[str, str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    work = df.copy()
    rename_map = rename_map or {}
    rename_existing = {col: rename_map[col] for col in rename_map if col in work.columns}
    if rename_existing:
        work = work.rename(columns=rename_existing)
    if "extra" not in work.columns:
        work["extra"] = [{} for _ in range(len(work))]
        return work
    meta_df = pd.DataFrame([_parse_extra_blob(val) for val in work["extra"]])
    if meta_df.empty:
        return work
    if rename_map:
        meta_df = meta_df.rename(columns={col: rename_map.get(col, col) for col in meta_df.columns})
    for column in meta_df.columns:
        if column in work.columns:
            work[column] = work[column].fillna(meta_df[column])
        else:
            work[column] = meta_df[column]
    return work

# ========= estilos SpaceX/NASA-like =========
# (Se cargan desde app/static/theme.css vía el tema global)

# ========= estado compartido =========
target       = st.session_state.get("target", None)
state_sel    = st.session_state.get("selected", None)
candidato    = state_sel["data"] if state_sel else None
props        = candidato["props"] if candidato else None
regolith_pct = (candidato.get("regolith_pct", 0.0) if candidato else 0.0)
scenario_label = ""
if target:
    scenario_label = str(target.get("scenario") or "").strip()
scenario_key = scenario_label.casefold()
thermo_summary = _regolith_thermal_summary() if regolith_pct > 0 else None
regolith_observations = _regolith_observation_lines(regolith_pct, thermo_summary)


@st.cache_data(show_spinner=False)
def _regolith_thermal_summary():
    bundle = load_regolith_thermal_profiles()
    peaks = bundle.gas_peaks.to_dict("records") if isinstance(bundle.gas_peaks, pd.DataFrame) else []
    events = bundle.mass_events.to_dict("records") if isinstance(bundle.mass_events, pd.DataFrame) else []
    return {"peaks": peaks, "events": events}


def _regolith_observation_lines(regolith_pct: float, thermo: dict[str, Any] | None) -> list[str]:
    if regolith_pct <= 0 or not thermo:
        return []

    lines: list[str] = []
    lines.append(
        f"{regolith_pct * 100:.0f}% de MGS-1: monitorear densificación, ventilación y sellos al liberar volátiles."
    )

    peaks = thermo.get("peaks", []) if isinstance(thermo, dict) else []
    for peak in peaks[:2]:
        temperature = peak.get("temperature_c")
        species = peak.get("species_label") or peak.get("species") or "Volátiles"
        signal = peak.get("signal_ppb")
        temp_txt = f"{temperature:.0f} °C" if isinstance(temperature, (int, float)) else "pico térmico"
        signal_txt = f" (~{signal:.2f} ppb eq.)" if isinstance(signal, (int, float)) else ""
        lines.append(f"TG/EGA: {species} con liberación cerca de {temp_txt}{signal_txt}.")

    events = thermo.get("events", []) if isinstance(thermo, dict) else []
    for event in events[:2]:
        label = (event.get("event") or "").replace("_", " ").strip().capitalize()
        mass_pct = event.get("mass_pct")
        temperature = event.get("temperature_c")
        mass_txt = f"{mass_pct:.1f}%" if isinstance(mass_pct, (int, float)) else "variación"
        temp_txt = f"{temperature:.0f} °C" if isinstance(temperature, (int, float)) else "el perfil térmico"
        lines.append(f"TG: {label or 'Evento'} → {mass_txt} alrededor de {temp_txt}.")

    return lines


def _scenario_side_hints(scenario_key: str) -> list[str]:
    base_map: dict[str, list[str]] = {
        "residence renovations": [
            "Priorizar paneles compactados que recuperen volumen habitable.",
            "Registrar rigidez en marcos reforzados y sellado de bordes laminados.",
            "Comparar tiempo de prensado vs. minutos de crew para futuras corridas.",
        ],
        "daring discoveries": [
            "Validar unión del carbono recuperado con la matriz polimérica.",
            "Inspeccionar fijación de mallas/filtros tras compactación o sinterizado.",
            "Documentar cualquier cambio en conductividad o manejo de polvo fino.",
        ],
    }
    return base_map.get(scenario_key, [])


_FEEDBACK_FIELD_KEYS = {
    "overall": "feedback_overall",
    "rigidity": "feedback_rigidity",
    "porosity": "feedback_porosity",
    "surface": "feedback_surface",
    "bonding": "feedback_bonding",
    "ease": "feedback_ease",
    "failure": "feedback_failure",
    "issues": "feedback_issues",
    "notes": "feedback_notes",
}


_PRESET_BASES: dict[str, dict[str, Any]] = {
    "Residence checklist": {
        "fields": {
            "overall": 8,
            "rigidity": 8,
            "porosity": 4,
            "surface": 7,
            "bonding": 8,
            "ease": 7,
            "failure": "Delaminación",
        },
        "issues": [
            "Verificar microcanales en paneles laminados post-prensado.",
            "Confirmar rigidez de bordes reforzados con regolito y CTB.",
        ],
        "notes": [
            "Checklist: documentar tiempo/temperatura de prensado y uso de refuerzos CTB.",
            "Ventilar el horno al escalar temperatura para evitar degradación de espumas.",
        ],
    },
    "Daring checklist": {
        "fields": {
            "overall": 7,
            "rigidity": 7,
            "porosity": 5,
            "surface": 6,
            "bonding": 8,
            "ease": 6,
            "failure": "Agarre insuficiente",
        },
        "issues": [
            "Revisar zonas con falta de unión por exceso de carbono en la mezcla.",
            "Inspeccionar fijación de mallas/filtros conductivos tras compactación.",
        ],
        "notes": [
            "Checklist: registrar % de carbono añadido y presión/tiempo de compactación.",
            "Medir continuidad eléctrica o sellos anti-polvo según aplique.",
        ],
    },
}


def _build_feedback_preset(name: str, *, regolith_lines: list[str]) -> dict[str, Any]:
    preset = _PRESET_BASES.get(name)
    if not preset:
        return {}

    payload: dict[str, Any] = dict(preset.get("fields", {}))
    issues_lines = list(preset.get("issues", []))
    notes_lines = list(preset.get("notes", []))

    if regolith_lines:
        rego_issue = [f"Revisá: {line}" for line in regolith_lines[:2]]
        rego_notes = [f"TG/EGA: {line}" for line in regolith_lines]
        issues_lines.extend(rego_issue)
        notes_lines.extend(rego_notes)

    payload["issues"] = "\n".join(f"- {line}" for line in issues_lines)
    payload["notes"] = "\n".join(f"- {line}" for line in notes_lines)
    return payload

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
        st.write(f"- Escenario: **{scenario_label or '-'}**")
        st.write(f"- Target: **{target.get('name','-')}**")
        st.write(f"- Proceso: **{candidato['process_id']} {candidato['process_name']}**")
        st.write(f"- Materiales: **{', '.join(candidato['materials'])}**")
        if regolith_pct > 0:
            st.write(f"- MGS-1 (regolito): **{regolith_pct*100:.0f}%** de la mezcla")
            thermo = thermo_summary or {}
            peak_lines: list[str] = []
            for peak in thermo.get("peaks", []):
                species = peak.get("species_label") or peak.get("species") or "Pico"
                temperature = peak.get("temperature_c")
                signal = peak.get("signal_ppb")
                temp_txt = f"{temperature:.0f} °C" if isinstance(temperature, (int, float)) else "temperatura clave"
                signal_txt = f" (~{signal:.2f} ppb eq.)" if isinstance(signal, (int, float)) else ""
                peak_lines.append(f"    - {species}: pico a {temp_txt}{signal_txt}")
            event_lines: list[str] = []
            for event in thermo.get("events", []):
                label = event.get("event", "")
                temperature = event.get("temperature_c")
                temp_txt = f"{temperature:.0f} °C" if isinstance(temperature, (int, float)) else "el perfil térmico"
                mass_pct = event.get("mass_pct")
                if isinstance(mass_pct, (int, float)) and label.startswith("mass_"):
                    event_lines.append(f"    - Masa ≤ {mass_pct:.1f}% cerca de {temp_txt}")
                elif label == "max_mass_loss_rate":
                    event_lines.append(f"    - Mayor tasa de desgasificación cerca de {temp_txt}")
            if peak_lines or event_lines:
                st.markdown("**Guía térmica NASA (TG/EGA):**")
                if peak_lines:
                    st.markdown("- Pico gases:\n" + "\n".join(peak_lines))
                if event_lines:
                    st.markdown("- Eventos TG:\n" + "\n".join(event_lines))
                st.caption(
                    "Utilizá estos picos para saber cuándo ventilar el horno y revisar porosidad/estanqueidad en la pieza."
                )

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
            materials="|".join(candidato.get("materials", [])),
            weights="|".join(map(str, candidato.get("weights", []))),
            process_id=candidato.get("process_id","-"),
            process_name=candidato.get("process_name","-"),
            mass_final_kg=float(p.mass_final_kg),
            energy_kwh=float(p.energy_kwh),
            water_l=float(p.water_l),
            crew_min=float(p.crew_min),
            score=float(candidato.get("score", 0.0)),
            extra={"regolith_pct": round(float(regolith_pct), 4)}
        )
        append_impact(entry)
        st.success("Impacto registrado en el log.")

with colBtn2:
    st.caption("Tip: registra impacto tras cada corrida para construir una curva de aprendizaje de materiales (y detectar desvíos).")

st.markdown("---")

# ========= Panel B: Feedback TÉCNICO de MATERIALES =========
st.markdown("### B) Feedback técnico (materiales) — nivel laboratorio")

col_feedback, col_reco = st.columns([1.45, 0.85])

with col_feedback:
    default_state = {
        "feedback_astronaut": "",
        "feedback_option_idx": 1,
        "feedback_overall": 8,
        "feedback_rigidity": 8,
        "feedback_porosity": 3,
        "feedback_surface": 7,
        "feedback_bonding": 7,
        "feedback_ease": 8,
        "feedback_failure": "-",
        "feedback_issues": "",
        "feedback_notes": "",
        "feedback_preset_last_applied": None,
        "feedback_last_scenario": None,
    }
    for key, default in default_state.items():
        st.session_state.setdefault(key, default)

    preset_options = ["Manual", "Residence checklist", "Daring checklist"]
    preset_key = "feedback_preset_select"
    if preset_key not in st.session_state:
        st.session_state[preset_key] = "Manual"

    desired_preset = "Manual"
    if scenario_key == "residence renovations":
        desired_preset = "Residence checklist"
    elif scenario_key == "daring discoveries":
        desired_preset = "Daring checklist"

    if st.session_state.get("feedback_last_scenario") != scenario_key:
        st.session_state["feedback_last_scenario"] = scenario_key
        st.session_state[preset_key] = desired_preset
        st.session_state["feedback_preset_last_applied"] = None

    selected_preset = st.selectbox(
        "📋 Checklist sugerida",
        preset_options,
        key=preset_key,
        help="Precarga sliders y notas con recomendaciones según el escenario detectado.",
    )

    last_applied = st.session_state.get("feedback_preset_last_applied")
    if selected_preset != "Manual" and selected_preset != last_applied:
        preset_payload = _build_feedback_preset(selected_preset, regolith_lines=regolith_observations)
        if preset_payload:
            for field, value in preset_payload.items():
                state_key = _FEEDBACK_FIELD_KEYS.get(field)
                if state_key is not None:
                    st.session_state[state_key] = value
        st.session_state["feedback_preset_last_applied"] = selected_preset
    elif selected_preset == "Manual" and last_applied not in (None, "Manual"):
        st.session_state["feedback_preset_last_applied"] = "Manual"

    failure_options = ["-", "Fragil", "Dúctil", "Delaminación", "Agarre insuficiente", "Fatiga"]
    if st.session_state.get("feedback_failure") not in failure_options:
        st.session_state["feedback_failure"] = failure_options[0]

    with st.form("feedback_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            astronaut = st.text_input("Operador / Astronauta", key="feedback_astronaut")
            option_idx = st.number_input(
                "Opción elegida #",
                min_value=1,
                step=1,
                value=int(st.session_state.get("feedback_option_idx", 1)),
                key="feedback_option_idx",
            )
            overall = st.slider(
                "Satisfacción global",
                0,
                10,
                st.session_state.get("feedback_overall", 8),
                help="0=Malísimo, 10=Excelente",
                key="feedback_overall",
            )
        with col2:
            rigidity_score = st.slider(
                "Rigidez percibida",
                0,
                10,
                st.session_state.get("feedback_rigidity", 8),
                key="feedback_rigidity",
            )
            porosity = st.slider(
                "Porosidad / compactación",
                0,
                10,
                st.session_state.get("feedback_porosity", 3),
                help="0=baja porosidad (mejor), 10=muy poroso",
                key="feedback_porosity",
            )
            surface = st.slider(
                "Calidad de superficie",
                0,
                10,
                st.session_state.get("feedback_surface", 7),
                key="feedback_surface",
            )
        with col3:
            bonding = st.slider(
                "Unión entre capas / partículas",
                0,
                10,
                st.session_state.get("feedback_bonding", 7),
                key="feedback_bonding",
            )
            failure = st.selectbox(
                "Modo de falla observado",
                failure_options,
                key="feedback_failure",
            )
            ease_score = st.slider(
                "Facilidad de proceso (ejecución)",
                0,
                10,
                st.session_state.get("feedback_ease", 8),
                key="feedback_ease",
            )

        issues = st.text_area(
            "Problemas específicos (bordes, olor, slip, etc.)",
            key="feedback_issues",
        )
        notes = st.text_area(
            "Notas libres / setup / parámetros",
            key="feedback_notes",
        )

        submitted = st.form_submit_button("Enviar feedback")
        if submitted:
            entry = FeedbackEntry(
                ts_iso=datetime.utcnow().isoformat(),
                astronaut=astronaut or "anon",
                scenario=target.get("scenario","-") if target else "-",
                target_name=target.get("name","-") if target else "-",
                option_idx=int(option_idx),
                rigidity_ok=bool(rigidity_score >= 6),
                ease_ok=bool(ease_score >= 6),
                issues=issues,
                notes=notes,
                # campos extendidos en `.extra` (si tu dataclass no los tiene, se guardan como texto)
                extra={
                    "overall": overall,
                    "porosity": porosity,
                    "surface": surface,
                    "bonding": bonding,
                    "failure": failure,
                }
            )
            append_feedback(entry)
            st.success("Feedback guardado. Rex-AI utilizará estas señales para ajustar pesos/penalizaciones y recomendaciones.")

with col_reco:
    st.markdown("#### Observaciones sugeridas")
    if scenario_label:
        st.caption(f"Escenario detectado: {scenario_label}")
    if regolith_pct > 0:
        st.caption(f"Mezcla con MGS-1: {regolith_pct * 100:.0f}%")

    scenario_hints = _scenario_side_hints(scenario_key)
    combined: list[str] = []
    seen: set[str] = set()
    for hint in scenario_hints + regolith_observations:
        if hint and hint not in seen:
            combined.append(hint)
            seen.add(hint)

    if combined:
        st.markdown("\n".join(f"- {hint}" for hint in combined))
    else:
        st.info("Seleccioná un target con escenario para ver recomendaciones automáticas.")

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
    impact_display = _with_extra_columns(idf, {
        "regolith_pct": "Regolith (%)",
        "extra_regolith_pct": "Regolith (%)"
    })
    if "Regolith (%)" in impact_display.columns:
        impact_display["Regolith (%)"] = impact_display["Regolith (%)"].apply(
            lambda v: f"{float(v) * 100:.0f}%" if isinstance(v, str) and v.replace('.', '', 1).isdigit() else v
        )
    st.dataframe(impact_display, use_container_width=True, hide_index=True)

    # Export
    csv_buf = StringIO(); idf.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Descargar impacto (CSV)", data=csv_buf.getvalue().encode("utf-8"),
                       file_name="impact_log.csv", mime="text/csv")
else:
    st.info("Aún no hay corridas registradas en el log de impacto.")

st.markdown("---")

st.markdown("### C.1) Feedback capturado (metadatos completos)")
if fdf is not None and len(fdf):
    feedback_display = _with_extra_columns(fdf, {
        "overall": "Satisfacción", "porosity": "Porosidad",
        "surface": "Superficie", "bonding": "Unión",
        "failure": "Falla observada",
        "extra_overall": "Satisfacción", "extra_porosity": "Porosidad",
        "extra_surface": "Superficie", "extra_bonding": "Unión",
        "extra_failure": "Falla observada"
    })
    # Para registros antiguos sin `extra`, mostramos '-'
    for col in ["Satisfacción", "Porosidad", "Superficie", "Unión", "Falla observada"]:
        if col in feedback_display.columns:
            feedback_display[col] = feedback_display[col].replace({"": "-"}).fillna("-")
    st.dataframe(feedback_display, use_container_width=True, hide_index=True)
    csv_buf_fb = StringIO(); fdf.to_csv(csv_buf_fb, index=False)
    st.download_button("⬇️ Descargar feedback (CSV)", data=csv_buf_fb.getvalue().encode("utf-8"),
                       file_name="feedback_log.csv", mime="text/csv")
else:
    st.info("Aún no hay feedback registrado.")

st.markdown("---")
st.markdown("### D) Lectura rápida para aprendices (¿por qué esto importa?)")
g1, g2 = st.columns(2)
with g1:
    st.markdown("""
- **Impacto = realidad**: qué tanto residuo convertimos en producto y a qué costo (energía/agua/crew).
- **Feedback ≠ opinión suelta**: capturamos señales de materiales (rigidez, porosidad, unión) que Rex-AI usa para ajustar decisiones.
- **Efecto MGS-1**: cuando el proceso es sinterizado, verás en el log `extra=regolith_pct=XX`. El regolito sube rigidez pero puede abrir microcanales al liberar H₂O/CO₂: anotá si aumenta porosidad o si hubo que ventilar más el horno.
""")
with g2:
    st.markdown("""
- **Cómo usarlo**: tras cada corrida, registra impacto (1 click) y cuelga feedback (2 min).  
- **Cómo leerlo**: mirá la tendencia diaria; si los kWh suben por pieza, hay algo en el setup.  
- **Qué permite**: cerrar el loop *plan → ejecutar → medir → aprender* como en un banco de pruebas de SpaceX/NASA.
""")

st.caption("¿Ideas para nuevas métricas? Podemos agregar dureza Shore, módulo aparente, densidad aparente, etc., si pasan a formar parte de la captura del laboratorio.")
