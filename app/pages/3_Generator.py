# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import pandas as pd
from app.modules.ui_blocks import inject_css, pill, section, card
from app.modules.io import load_waste_df, load_process_df
from app.modules.process_planner import choose_process
from app.modules.generator import generate_candidates
from app.modules.safety import check_safety, safety_badge

# 1) st.set_page_config DEBE ir primero
st.set_page_config(page_title="Generador", page_icon="⚙️", layout="wide")

# 2) Inyectamos CSS global (incluye tipografías y pequeños estilos de cards/pills)
inject_css()

# ------ CSS local (aspecto “mission console”) ------
st.markdown("""
<style>
/* Hero barra */
.hero {
  padding: 18px 20px;
  border-radius: 16px;
  background: radial-gradient(1200px 600px at 15% -10%, rgba(0,200,255,0.12), transparent),
              linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(6px);
}
.hero h1 {margin:0; font-size: 1.6rem;}
.hero .sub {opacity:.8; font-size:.95rem; margin-top:6px;}

.console {
  border: 1px dashed rgba(255,255,255,0.15);
  border-radius: 14px; padding: 14px;
  margin: 6px 0 18px 0;
}

/* KPI chips */
.kpi {
  display:flex; gap:10px; align-items:center;
  padding:10px 12px; border-radius:12px;
  border:1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.03);
}
.kpi .v {font-weight:700; font-variant-numeric: tabular-nums;}
.kpi .t {opacity:.7; font-size:.9rem;}

.legend {
  display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 0 0;
}
.legend span {
  font-size:.78rem; padding:4px 8px; border-radius:999px;
  border:1px solid rgba(255,255,255,0.1); opacity:.9;
}
.badge {
  display:inline-flex; align-items:center; gap:6px;
  padding:6px 10px; border-radius:999px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  font-size:.85rem; margin-right:6px; margin-bottom:4px;
}
.hr-micro {height:1px; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent); margin:10px 0;}
.small {font-size:.9rem; opacity:.85;}
.hint {opacity:.9; font-size:.92rem;}
.note {
  border-left: 3px solid #1f9d55; padding: 8px 12px; background: rgba(31,157,85,0.1);
  border-radius: 6px; margin: 8px 0;
}
.warn {
  border-left: 3px solid #d9534f; padding: 8px 12px; background: rgba(217,83,79,0.08);
  border-radius: 6px; margin: 8px 0;
}
.rec {
  border-left: 3px solid #37a1f2; padding: 8px 12px; background: rgba(55,161,242,0.08);
  border-radius: 6px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Encabezado “Mission Console” --------------------
st.markdown("""
<div class="hero">
  <h1>⚙️ Generador • Mission Console</h1>
  <div class="sub">Acá mezclamos <b>basura inorgánica</b> con <b>procesos del hábitat</b> para proponer recetas
  que maximizan utilidad (rigidez/estanqueidad), minimizan recursos (agua/energía/tiempo de tripulación) y,
  cuando corresponde, aprovechan <b>regolito MGS-1</b> (ISRU).</div>
</div>
""", unsafe_allow_html=True)

# -------------------- Precondición: target definido --------------------
target = st.session_state.get("target", None)
if not target:
    st.warning("Definí primero el objetivo en **2) Target Designer**.")
    st.stop()

# -------------------- Estado rápido del inventario y procesos --------------------
waste_df = load_waste_df()
proc_df  = load_process_df()
proc_filtered = choose_process(
    target["name"], proc_df,
    scenario=target.get("scenario"),
    crew_time_low=target.get("crew_time_low", False)
)
if proc_filtered is None or proc_filtered.empty:
    proc_filtered = proc_df.copy()

# KPI bar
total_items = len(waste_df) if waste_df is not None else 0
mass_col = "mass_kg" if "mass_kg" in waste_df.columns else ("kg" if "kg" in waste_df.columns else None)
total_mass = float(waste_df[mass_col].sum()) if (total_items>0 and mass_col) else 0.0

# estimación de “problemáticos” con reglas (en vivo, sin mutar el DF)
def _is_problematic_row_view(row: pd.Series) -> bool:
    cat = str(row.get("category", "")).lower()
    fam = str(row.get("material_family", "")).lower()
    flg = str(row.get("flags", "")).lower()
    rules = [
        "pouches" in cat or "multilayer" in flg or "pe-pet-al" in fam,
        "foam" in cat or "zotek" in fam or "closed_cell" in flg,
        "eva" in cat or "ctb" in flg or "nomex" in fam or "nylon" in fam or "polyester" in fam,
        "glove" in cat or "nitrile" in fam,
        "wipe" in flg or "textile" in cat,
    ]
    return any(rules)

prob_count = 0
if total_items:
    try:
        prob_count = int(waste_df.apply(_is_problematic_row_view, axis=1).sum())
    except Exception:
        prob_count = 0

c1,c2,c3,c4 = st.columns([1.2,1.2,1.2,1.2])
with c1:
    st.markdown(f"""<div class="kpi">🧱 <div><div class="v">{total_items}</div><div class="t">ítems en inventario</div></div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi">⚖️ <div><div class="v">{total_mass:.2f} kg</div><div class="t">masa total</div></div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi">⚠️ <div><div class="v">{prob_count}</div><div class="t">problemáticos detectados</div></div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi">🧪 <div><div class="v">{len(proc_filtered)}</div><div class="t">procesos habilitados</div></div></div>""", unsafe_allow_html=True)

# Leyenda rápida de cómo puntúa el generador
st.markdown("""
<div class="legend">
  <span>🎯 Se ajusta al objetivo (rigidez/estanqueidad)</span>
  <span>💧/⚡️/👩‍🚀 Penaliza agua/energía/tiempo</span>
  <span>⛰️ ISRU con MGS-1 cuando aplica</span>
  <span>♻️ Bono por “basura problemática” valorizada</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)

# -------------------- Panel de control --------------------
left, right = st.columns([1.2, 1.1])
with left:
    section("Control del generador", "Elegí cuántas opciones querés probar.")
    n = st.slider("Número de candidatos a generar", 3, 12, 6)
    opt_evals = st.slider(
        "Evaluaciones del optimizador (BO/MILP)",
        0, 80, 24,
        help="Cantidad de iteraciones adicionales para refinar el frente Pareto."
    )

with right:
    section("Tips de uso", "Para quienes no son expertos:")
    st.markdown("""
- **“Generar opciones”**: el sistema mezcla residuos reales del inventario con procesos disponibles y propone recetas.
- **¿Qué es “ISRU”?** Uso de recursos in situ (en Marte) → agregamos **MGS-1** cuando el proceso lo permite.
- **¿Cómo puntúa?** Suma por encaje con el objetivo, resta por consumir más de lo permitido (agua/energía/tiempo), y **bonifica** si usamos basura difícil.
""")

# CTA principal
if st.button("🚀 Generar opciones", type="primary", use_container_width=True):
    result = generate_candidates(
        waste_df, proc_filtered, target, n=n,
        crew_time_low=target.get("crew_time_low", False),
        optimizer_evals=opt_evals
    )
    if isinstance(result, tuple):
        cands, history = result
    else:
        cands, history = result, pd.DataFrame()
    st.session_state["candidates"] = cands
    st.session_state["optimizer_history"] = history

st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)

# -------------------- Si no hay candidatos aún --------------------
cands = st.session_state.get("candidates", [])
history_df = st.session_state.get("optimizer_history", pd.DataFrame())
if not cands:
    st.info("Todavía no hay candidatos. Configurá el número y presioná **Generar opciones**. "
            "Recomendación: asegurate de que tu inventario tenga pouches, espumas, EVA/CTB, textiles o nitrilo; "
            "y que el catálogo incluya P02/P03/P04.")
    with st.expander("¿Qué hace el generador (en criollo)?", expanded=False):
        st.markdown("""
- **Mira tus residuos** (con foco en los problemáticos de NASA).
- **Elige un proceso** coherente (laminar, sinter con regolito, reconfigurar CTB, etc.).
- **Predice** propiedades y recursos de la receta.
- **Puntúa** balanceando objetivos y costos.
- **Muestra trazabilidad** para que se vea qué basura se valorizó.
""")
    st.stop()

# -------------------- Render de candidatos con UX explicativa --------------------
def _res_bar(current: float, limit: float) -> float:
    if limit is None or float(limit) <= 0:
        return 0.0
    return max(0.0, min(1.0, current/float(limit)))

if isinstance(history_df, pd.DataFrame) and not history_df.empty:
    st.subheader("Convergencia del optimizador")
    st.caption("Seguimiento rápido de hipervolumen y porcentaje de soluciones dominadas.")
    valid_hist = history_df.dropna(subset=["hypervolume"])
    if not valid_hist.empty:
        last = valid_hist.iloc[-1]
        m1, m2, m3 = st.columns([1, 1, 1])
        m1.metric("Hipervolumen", f"{last['hypervolume']:.3f}")
        m2.metric("Dominancia", f"{last['dominance_ratio']*100:.1f}%")
        m3.metric("Tamaño Pareto", f"{int(last['pareto_size'])}")
        chart_data = valid_hist.set_index("iteration")["hypervolume"].to_frame()
        chart_data["dominancia"] = valid_hist.set_index("iteration")["dominance_ratio"]
        st.line_chart(chart_data)

st.subheader("Resultados del generador")
st.caption("Cada ‘Opción’ es una combinación concreta de residuos + proceso, con predicción de propiedades y consumo de recursos. "
           "Usá los expanders para ver detalles y trazabilidad NASA.")

for i, c in enumerate(cands):
    p = c["props"]
    header = f"Opción {i+1} — Score {c['score']} — Proceso {c['process_id']} {c['process_name']}"
    with st.expander(header, expanded=(i == 0)):
        # Línea de badges
        badges = []
        if c.get("regolith_pct", 0) > 0:
            badges.append("⛰️ ISRU: +MGS-1")
        # Heurística “problemático presente”
        src_cats = " ".join(map(str, c.get("source_categories", []))).lower()
        src_flags = " ".join(map(str, c.get("source_flags", []))).lower()
        problem_present = any([
            "pouches" in src_cats, "multilayer" in src_flags,
            "foam" in src_cats, "ctb" in src_flags, "eva" in src_cats,
            "nitrile" in src_cats, "wipe" in src_flags
        ])
        if problem_present:
            badges.append("♻️ Valorización de problemáticos")
        if badges:
            st.markdown(" ".join([f'<span class="badge">{b}</span>' for b in badges]), unsafe_allow_html=True)

        # Resumen técnico
        colA, colB = st.columns([1.1, 1])
        with colA:
            st.markdown("**🧪 Materiales**")
            st.write(", ".join(c["materials"]))
            st.markdown("**⚖️ Pesos en mezcla**")
            st.write(c["weights"])

            st.markdown("**🔬 Predicción (demo)**")
            colA1, colA2, colA3 = st.columns(3)
            colA1.metric("Rigidez", f"{p.rigidity:.2f}")
            colA2.metric("Estanqueidad", f"{p.tightness:.2f}")
            colA3.metric("Masa final", f"{p.mass_final_kg:.2f} kg")

        with colB:
            st.markdown("**🔧 Proceso**")
            st.write(f"{c['process_id']} — {c['process_name']}")
            st.markdown("**📉 Recursos estimados**")
            colB1, colB2, colB3 = st.columns([1,1,1])
            colB1.write("Energía (kWh)")
            colB1.progress(_res_bar(p.energy_kwh, target["max_energy_kwh"]))
            colB1.caption(f"{p.energy_kwh:.2f} / {target['max_energy_kwh']}")

            colB2.write("Agua (L)")
            colB2.progress(_res_bar(p.water_l, target["max_water_l"]))
            colB2.caption(f"{p.water_l:.2f} / {target['max_water_l']}")

            colB3.write("Crew (min)")
            colB3.progress(_res_bar(p.crew_min, target["max_crew_min"]))
            colB3.caption(f"{p.crew_min:.0f} / {target['max_crew_min']}")

        st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)

        # Trazabilidad NASA
        st.markdown("**🛰️ Trazabilidad NASA**")
        st.write("IDs usados:", ", ".join(c.get("source_ids", [])) or "—")
        st.write("Categorías:", ", ".join(map(str, c.get("source_categories", []))) or "—")
        st.write("Flags:", ", ".join(map(str, c.get("source_flags", []))) or "—")
        if c.get("regolith_pct", 0) > 0:
            st.write(f"**MGS-1 agregado:** {c['regolith_pct']*100:.0f}%")

        # Seguridad (badges)
        st.markdown("**🛡️ Seguridad**")
        flags = check_safety(c["materials"], c["process_name"], c["process_id"])
        badge = safety_badge(flags)
        if badge["level"] == "Riesgo":
            pill("Riesgo", "risk"); st.warning(badge["detail"])
        else:
            pill("OK", "ok"); st.success(badge["detail"])

        # Botón de selección
        if st.button(f"✅ Seleccionar Opción {i+1}", key=f"pick_{i}"):
            st.session_state["selected"] = {"data": c, "safety": badge}
            st.success("Opción seleccionada. Abrí **4) Resultados**, **5) Comparar & Explicar** o **6) Pareto & Export**.")

        # Explicación en criollo (mini narrativa) — evitar anidar expanders
pop = st.popover("🧠 ¿Por qué esta receta pinta bien? (explicación en criollo)")
with pop:
    bullets = []
    bullets.append("• Sumamos puntos si **rigidez/estanqueidad** se acercan a lo que pediste.")
    bullets.append("• Restamos si se pasa en **agua/energía/tiempo** de la tripulación.")
    if problem_present:
        bullets.append("• Bonus porque esta opción **se come basura problemática** (¡la que más molesta en la base!).")
    if c.get('regolith_pct', 0) > 0:
        bullets.append("• Usa **MGS-1** (regolito) como carga mineral → eso es ISRU puro: menos dependencia de la Tierra.")
    st.markdown("\n".join(bullets))


# -------------------- Pie de guía / glosario --------------------
st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)
with st.expander("📚 Glosario ultra rápido", expanded=False):
    st.markdown("""
- **ISRU**: *In-Situ Resource Utilization*. Usar recursos del lugar (en Marte, el **regolito** MGS-1).
- **P02 – Press & Heat Lamination**: “plancha” y “fusiona” multicapa para dar forma.
- **P03 – Sinter with MGS-1**: mezcla con regolito y sinteriza → piezas rígidas, útiles para interiores.
- **P04 – CTB Reconfig**: reusar/transformar bolsas EVA/CTB con herrajes.
- **Score**: cuánto “cierra” la opción según tu objetivo y límites de recursos/tiempo.
""")
st.info("Sugerencia: generá varias opciones y pasá a **4) Resultados**, **5) Comparar** y **6) Pareto & Export** para cerrar tu plan.")
