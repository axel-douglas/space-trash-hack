# app/pages/8_Feedback_and_Impact.py
import app  # noqa: F401

# ⚠️ Debe ser la PRIMERA llamada Streamlit:
import streamlit as st
from app.modules.ui_blocks import load_theme

st.set_page_config(page_title="Feedback & Impact", page_icon="📝", layout="wide")

load_theme()

import json
import pandas as pd
from datetime import datetime
from io import StringIO
from typing import Any

from app.modules.impact import (
    ImpactEntry, FeedbackEntry, append_impact, append_feedback,
    load_impact_df, load_feedback_df, summarize_impact
)


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
# (Se cargan desde app/static/theme.css vía load_theme)

# ========= estado compartido =========
target      = st.session_state.get("target", None)
state_sel   = st.session_state.get("selected", None)
candidato   = state_sel["data"] if state_sel else None
props       = candidato["props"] if candidato else None
regolith_pct = (candidato.get("regolith_pct", 0.0) if candidato else 0.0)

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
        st.write(f"- Escenario: **{target.get('scenario','-')}**")
        st.write(f"- Target: **{target.get('name','-')}**")
        st.write(f"- Proceso: **{candidato['process_id']} {candidato['process_name']}**")
        st.write(f"- Materiales: **{', '.join(candidato['materials'])}**")
        if regolith_pct > 0:
            st.write(f"- MGS-1 (regolito): **{regolith_pct*100:.0f}%** de la mezcla")

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

with st.form("feedback_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        astronaut = st.text_input("Operador / Astronauta", "")
        option_idx = st.number_input("Opción elegida #", min_value=1, step=1, value=1)
        overall = st.slider("Satisfacción global", 0, 10, 8, help="0=Malísimo, 10=Excelente")
    with col2:
        rigid_ok = st.slider("Rigidez percibida", 0, 10, 8)
        porosity = st.slider("Porosidad / compactación", 0, 10, 3, help="0=baja porosidad (mejor), 10=muy poroso")
        surface  = st.slider("Calidad de superficie", 0, 10, 7)
    with col3:
        bonding  = st.slider("Unión entre capas / partículas", 0, 10, 7)
        failure  = st.selectbox("Modo de falla observado", ["-", "Fragil", "Dúctil", "Delaminación", "Agarre insuficiente", "Fatiga"])
        ease_ok  = st.slider("Facilidad de proceso (ejecución)", 0, 10, 8)

    issues = st.text_area("Problemas específicos (bordes, olor, slip, etc.)", "")
    notes  = st.text_area("Notas libres / setup / parámetros", "")

    submitted = st.form_submit_button("Enviar feedback")
    if submitted:
        entry = FeedbackEntry(
            ts_iso=datetime.utcnow().isoformat(),
            astronaut=astronaut or "anon",
            scenario=target.get("scenario","-") if target else "-",
            target_name=target.get("name","-") if target else "-",
            option_idx=int(option_idx),
            rigidity_ok=bool(rigid_ok >= 6),
            ease_ok=bool(ease_ok >= 6),
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
- **Efecto MGS-1**: cuando el proceso es sinterizado, verás en el log `extra=regolith_pct=XX`. El regolito tiende a subir rigidez y bajar estanqueidad; si percibís más porosidad, anotarlo aquí ayuda.
""")
with g2:
    st.markdown("""
- **Cómo usarlo**: tras cada corrida, registra impacto (1 click) y cuelga feedback (2 min).  
- **Cómo leerlo**: mirá la tendencia diaria; si los kWh suben por pieza, hay algo en el setup.  
- **Qué permite**: cerrar el loop *plan → ejecutar → medir → aprender* como en un banco de pruebas de SpaceX/NASA.
""")

st.caption("¿Ideas para nuevas métricas? Podemos agregar dureza Shore, módulo aparente, densidad aparente, etc., si pasan a formar parte de la captura del laboratorio.")
