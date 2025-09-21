# app/Home.py
# --- path guard ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------

import streamlit as st
from pathlib import Path

# ⚠️ PRIMER comando de Streamlit:
st.set_page_config(
    page_title="REX-AI Mars — Mission Hub",
    page_icon="🛰️",
    layout="wide"
)

# ---------- Estilos SpaceX-like ----------
st.markdown("""
<style>
:root{ --bd: rgba(130,140,160,.28);}
.hero{
  border:1px solid var(--bd);
  border-radius:22px;
  padding:28px;
  margin-bottom:20px;
  background: radial-gradient(1200px 400px at 20% -20%, rgba(80,120,255,.12), transparent);
}
.hero h1{margin:0 0 6px 0; font-size:2rem}
.hero .tagline{font-size:1.1rem; opacity:.9; margin-bottom:12px}
.grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap:18px; margin:20px 0;}
.card{border:1px solid var(--bd); border-radius:16px; padding:18px; background:rgba(255,255,255,.02);}
.card h3{margin:.1rem 0 .25rem 0;}
.kpi{border:1px solid var(--bd); border-radius:14px; padding:14px; text-align:center;}
.kpi h3{margin:0 0 6px 0; font-size:.95rem; opacity:.8}
.kpi .v{font-size:1.6rem; font-weight:800; letter-spacing:.2px}
.pill{display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.78rem;
      border:1px solid var(--bd); margin-right:6px}
.pill.ok{background:#e8f7ee; color:#136c3a; border-color:#b3e2c4}
.pill.info{background:#e7f1ff; color:#174ea6; border-color:#c6dcff}
.pill.warn{background:#fff3cd; color:#8a6d1d; border-color:#ffe69b}
.small{font-size:.92rem; opacity:.9}
.section{margin-top:30px; margin-bottom:18px;}
.section h2{margin-bottom:6px;}
</style>
""", unsafe_allow_html=True)

# ---------- Encabezado ----------
logo_svg = ROOT / "app" / "static" / "logo_rexai.svg"
cols = st.columns([0.15, 0.85])
with cols[0]:
    if logo_svg.exists():
        st.image(str(logo_svg), use_column_width=True)
with cols[1]:
    st.title("REX-AI Mars — Mission Hub")
    st.caption("Recycling & Experimentation eXpert — Jezero Base")

# ---------- Hero narrative ----------
st.markdown("""
<div class="hero">
  <h1>Conoce a REX-AI</h1>
  <div class="tagline">
    Nuestra inteligencia artificial que convierte basura espacial en recursos para la misión.
  </div>
  <div>
    <span class="pill info">Reciclaje inteligente</span>
    <span class="pill ok">Optimización multi-objetivo</span>
    <span class="pill warn">Uso seguro de MGS-1</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Qué hace (versión narrativa + técnica) ----------
st.markdown("### ¿Qué hace REX-AI?")
st.markdown("""
**En criollo**:  
La nave genera **basura inorgánica** — pouches multicapa, espumas, textiles, guantes de nitrilo, piezas de aluminio.  
Lo que hace **REX-AI** es agarrar ese lío y decirte: *“Con esto podés armar un contenedor, un utensilio, una herramienta…”*  
Y no lo hace a ojo: combina datos reales de procesos, costos de crew, y hasta regula la mezcla con **regolito MGS-1** del cráter Jezero.

**Para ingenieros**:  
- Analiza inventarios reales (NASA Non-Metabolic Waste Categories).  
- Selecciona procesos viables (Shredder, Heat Lamination, Sinter con MGS-1, Reuso CTB).  
- Calcula propiedades predichas (rigidez, estanqueidad, masa final) con modelos ligeros.  
- Puntúa cada candidato con un **score multi-objetivo**: compatibilidad con target + penalización por recursos + bonus por “consumir problemáticos”.  
- Devuelve recetas con trazabilidad: cada ID, categoría y flag queda registrado.
""")

# ---------- Estado del sistema ----------
st.markdown("### Estado actual")
c1, c2, c3, c4 = st.columns(4)
data_ok = (ROOT / "data" / "waste_inventory_sample.csv").exists()
proc_ok = (ROOT / "data" / "process_catalog.csv").exists()
tgt_ok  = (ROOT / "data" / "targets_presets.json").exists()
with c1: st.markdown(f'<div class="kpi"><h3>Inventario</h3><div class="v">{"✅" if data_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi"><h3>Procesos</h3><div class="v">{"✅" if proc_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi"><h3>Targets</h3><div class="v">{"✅" if tgt_ok else "❌"}</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="kpi"><h3>Modo</h3><div class="v">Demo</div></div>', unsafe_allow_html=True)

# ---------- Flujo ----------
st.markdown("### Flujo de misión")
st.markdown("""
<div class="grid">
  <div class="card"><h3>1) Inventario</h3><div class="small">Subí y editá los residuos. Detectamos los **problemáticos** y los marcamos.</div></div>
  <div class="card"><h3>2) Objetivo</h3><div class="small">Definí qué querés fabricar (ej: Container, Tool) y límites de recursos.</div></div>
  <div class="card"><h3>3) Generador</h3><div class="small">REX-AI arma recetas priorizando consumir problemáticos y elige procesos coherentes.</div></div>
  <div class="card"><h3>4) Resultados</h3><div class="small">Ves métricas, trade-offs, Sankey de flujo y checklist de fabricación.</div></div>
</div>
""", unsafe_allow_html=True)

# ---------- Explicación SpaceX-style ----------
st.markdown("### ¿Por qué importa?")
st.markdown("""
- **Nivel crew**: cada minuto de astronauta ahorrado vale millones. REX-AI aprende a minimizarlo.  
- **Nivel recursos**: el agua y energía en Marte son como oro; penalizamos cualquier exceso.  
- **Nivel materiales**: en lugar de mandar toneladas desde la Tierra, reusamos lo que ya está en el hábitat.  
- **Nivel misión**: más seguridad (sin incineración, sin PFAS), más resiliencia (aprovechar regolito local).  

**Ejemplo simple**:  
Si mezclamos espuma ZOTEK F30 con regolito MGS-1 → se logra un panel rígido que puede reforzar interiores.  
Si usamos pouches multicapa en laminar + presión → conseguimos láminas estancas para compartimentos.  
""")

# ---------- Ruta completa ----------
st.markdown("---")
st.caption(
    "Ruta: 1) Inventario → 2) Objetivo → 3) Generador → 4) Resultados → "
    "5) Comparar → 6) Pareto & Export → 7) Playbooks → 8) Feedback & Impact → 9) Capacity Simulator"
)

# === Extensión: Escalabilidad & Roadmap técnico (SpaceX-style) ===

st.markdown("## Escalabilidad y Roadmap técnico — cómo REX-AI crece sin techo 🚀")

st.markdown("""
**Resumen ejecutivo**  
REX-AI hoy es una **demo operativa** con módulos ligeros y trazabilidad real (IDs de residuo, categorías, flags y uso de MGS-1).  
Fue diseñada como **esqueleto escalable**: los mismos puntos de extensión que hoy alimentan el generador pueden acoplarse a
pipelines de datos, modelos avanzados y orquestación industrial sin reescribir la app.  
**Meta**: convertir la basura en *supply chain in-situ*, con IA que aprende en cada corrida y reduce agua/energía/tiempo/costo.

### 1) Estado actual (v1 demo)
- **Arquitectura modular**: `app/modules/*` separa UI, IO, generador, scoring, explicabilidad y export.
- **Trazabilidad**: cada candidato guarda `source_ids`, `source_categories`, `source_flags` y `regolith_pct`.
- **Compatibilidad de datos**: normalización robusta (nombres de columna variables), enfoque *schema-first* simple.
- **Explicabilidad**: score multi-objetivo transparente + desglose por componentes (función, agua, energía, crew, seguridad).
- **Seguridad**: reglas “hard-stop” (sin incineración, evitar PFAS/microplásticos), y checks de coherencia básica por proceso.

### 2) Arquitectura para escalar (cómo se vuelve cada vez más inteligente)
**Plano de datos (Data Plane)**
- **Ingesta**: CSV/JSON hoy → **Parquet** en **objeto/MinIO** o **S3**, versión de datos por corrida (data lineage).
- **Contratos de datos**: `pydantic`/`msgspec` para validar lotes, procesos y targets (rompe si no cumple -> evitar “data drift”).
- **Catálogo**: `PostgreSQL/pgvector` para búsquedas semánticas de materiales/flags y `DuckDB` para análisis *in-process*.
- **Streaming**: `Kafka/Redpanda` para telemetría de ensayos (tiempo real) y bitácoras de procesos (IoT/OPC-UA/ROS puenteados).

**Plano de modelos (Model Plane)**
- **Surrogates de propiedades**: de “heurísticas ligeras” → a **GNNs** (material graphs), **XGBoost/TabTransformer** (tabular),
  y **Physics-Informed ML** (PIML) para rigidez/porosidad/estanqueidad condicionadas al proceso.
- **Incertidumbre**: **ensembles**, **MC Dropout**, **conformal prediction** → graficar bandas de confianza y *risk-aware scoring*.
- **Optimización**: **Bayesian Optimization** (Ax/BoTorch) para receta/proceso bajo límites (agua/kWh/crew),
  más **solvers con restricciones** (MILP/CP-SAT) para factibilidad operativa (turnos, kg/lote, disponibilidad).
- **Ciclo activo**: **Active Learning** y **Bayesian Experimental Design** que selecciona el *próximo experimento* de mayor valor.

**Plano de control (Control Plane)**
- **MLOps**: entrenamiento/registro con **MLflow** o **Weights & Biases**, versionado de datasets y de artefactos.
- **Orquestación**: **Airflow/Prefect** para pipelines de ingesta, feature store, entrenamiento y despliegue continuo (CD).
- **Serving**: **FastAPI** + **ONNX Runtime/TensorRT** para inferencia acelerada (CPU/GPU/Jetson), cola con **Redis**.
- **Edge/Flight**: empaquetado **OCI** con perfiles reproducibles (SBOM), estrategia *graceful degradation* y *circuit breakers*.

### 3) Seguridad, fiabilidad y compliance (misión crítica)
- **Guardrails de proceso**: políticas *deny-by-default* para incineración y sustancias críticas; whitelists por hábitat.
- **Provenance**: hash de datasets y modelos, **audit log** por corrida, exportables a JSON/CSV (ya presente en demo).
- **Testing y validación**: *golden datasets*, *shadow mode* para nuevos modelos, *canary releases* y *rollbacks* atómicos.
- **Resiliencia**: timeouts, reintento exponencial para ingestas/sensórica, *data gap filling* y *late data handling*.

### 4) Interoperabilidad con infraestructura de misión
- **Protocolos**: OPC-UA/ROS para células robóticas, **ISA-95/88** para integración MES/SCADA.
- **Digital Twin**: simulación *in-silico* (DEM/FEM simplificada) para priorizar ensayos con mayor valor esperado.
- **Compatibilidad NASA**: mapeo a taxonomías *Non-Metabolic Waste*, ingeniería de materiales (MGS-1) y formatos estándar.

### 5) ¿Por qué puede revolucionar Marte y la Tierra?
- **Marte (ISRU real)**: convertir *lo que sobra* en *lo que falta* con costo marginal casi nulo en logística interplanetaria.
- **Tierra (economía circular)**: recetas trasladables a residuos complejos (multicapa/espumas), cierre de ciclos en bases remotas,
  minería urbana y descarbonización por **reducción de materia virgen + energía**.
- **Aprendizaje compuesto**: cada base/hábitat entrena un pedacito → federado, con privacidad y *model averaging*.

### 6) Roadmap por etapas (claro y accionable)
**T-0 (demo+)**  
- Persistir corridas en DuckDB/Parquet, MLflow local, bandas de confianza visuales, export de *experiment design*.

**T-1 (MVP productivo)**  
- FastAPI + ONNX serving, BO con restricciones, pgvector para materiales, Airflow para pipelines diarios, auditoría completa.

**T-2 (Flight-ready)**  
- Edge GPU (TensorRT), active learning en lazo cerrado, integración OPC-UA/ROS, digital twin ligero y canary en campo.

**T-3 (Programa)**  
- Federated learning entre hábitats, planificación multi-planta, optimización global de recursos y logística inversa.

### 7) Cómo REX-AI se “vuelve más inteligente” (de verdad)
1. **Captura** datos crudos de cada corrida (input → receta → proceso → salida).  
2. **Valida** con contratos de datos y los guarda versionados.  
3. **Etiqueta/Enriquece** (features, contexto, condiciones de proceso).  
4. **Re-entrena** surrogates con incertidumbre y compara vs. “golden”.  
5. **Optimiza** próximos experimentos (mínimo costo, máximo aprendizaje).  
6. **Despliega** modelos certificados; deja bitácora y *rollbacks*.  
7. **Repite** (cada ciclo reduce agua/kWh/crew y mejora score/seguridad).

> **Mensaje al jurado**: lo que hoy ves corriendo en Streamlit ya implementa trazabilidad, reglas y explicabilidad.
> Cambiar de “heurísticas” a **modelos avanzados** es un swap controlado en `modules/generator.py` y `modules/explain.py`,
> con el resto de la arquitectura (datos/MLOps/serving) ya preparada para crecer sin romper la UX ni la seguridad.
""")
