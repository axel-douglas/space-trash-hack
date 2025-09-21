# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import pandas as pd
from app.modules.io import load_waste_df, save_waste_df

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Inventario", page_icon="🧱", layout="wide")

st.title("1) Inventario de residuos")
st.caption("Este inventario refleja los ítems no-metabólicos que NASA identifica como problemáticos: empaques multicapa, espumas técnicas (ZOTEK F30), bolsas EVA/CTB, textiles/wipes técnicos y guantes de nitrilo.")

with st.expander("¿Por qué estos ítems? (resumen rápido)", expanded=True):
    st.markdown("""
- **Pouches PE–PET–Al** (multicapa térmico) → difíciles de reciclar por capas incompatibles.
- **Espumas ZOTEK F30 (PE reticulado)** → celdas cerradas, baja densidad, voluminosas.
- **EVA/CTB (Nomex/Nylon/Polyester)** → textiles técnicos con recubrimientos.
- **Guantes de nitrilo** → elastómeros con aditivos.
- **Estructuras de Al** → muy valiosas para **reuso** o como refuerzo en compuestos.
- **Regolito MGS-1** se usa como **carga mineral** en procesos de sinterizado/compuesto.
""")

# --- Carga y normalización ligera para edición amigable ---
df_raw = load_waste_df().copy()

# Aseguramos nombres amigables para edición sin romper downstream
df = df_raw.rename(columns={
    "id": "id",
    "category": "category",
    "material_family": "material_family",
    "mass_kg": "mass_kg",
    "volume_l": "volume_l",
    "flags": "flags",
})
# Derivados útiles para otras páginas (sin cambiar nombres originales)
df["_problematic"] = False

def _is_problematic(row: pd.Series) -> bool:
    cat = str(row.get("category", "")).lower()
    fam = str(row.get("material_family", "")).lower()
    flg = str(row.get("flags", "")).lower()
    # Heurística NASA: multicapa, foam técnico, EVA/CTB, nitrilo, wipes técnicos
    rules = [
        "pouches" in cat or "multilayer" in flg or "pe-pet-al" in fam,
        "foam" in cat or "zotek" in fam or "closed_cell" in flg,
        "eva" in cat or "ctb" in flg or "nomex" in fam or "nylon" in fam or "polyester" in fam,
        "glove" in cat or "nitrile" in fam,
        "wipe" in flg or "textile" in cat
    ]
    return any(rules)

df["_problematic"] = df.apply(_is_problematic, axis=1)

# Métricas rápidas para “sabor laboratorio”
c1,c2,c3,c4 = st.columns(4)
c1.metric("Ítems", len(df))
c2.metric("Masa total (kg)", f"{df['mass_kg'].sum():.2f}")
c3.metric("Volumen (L)", f"{df['volume_l'].sum():.1f}")
c4.metric("Problemáticos", int(df["_problematic"].sum()))

st.markdown("### Editar inventario")
st.caption("Tip: los **problemáticos** se resaltan en la vista previa. Podés ajustar `flags`/`category` para reflejar mejor la realidad de tu lote.")

edited = st.data_editor(
    df[["id","category","material_family","mass_kg","volume_l","flags"]],
    num_rows="dynamic",
    use_container_width=True,
    key="inv_editor",
)

colA, colB = st.columns([1,1])
with colA:
    if st.button("💾 Guardar inventario", type="primary"):
        # Persistimos SOLO columnas originales esperadas por el pipeline
        save_waste_df(edited)
        st.success("Inventario guardado.")

with colB:
    if st.button("↺ Recalcular 'problemáticos' para vista previa"):
        st.experimental_rerun()

st.markdown("---")
st.subheader("Vista previa con resaltado de 'problemáticos'")

# Re-generamos 'problematic' sobre la versión editada para la vista previa:
prev = edited.copy()
prev["_problematic"] = prev.apply(_is_problematic, axis=1)

def _row_style(s: pd.Series):
    return ['background-color: #FFF3CD' if s["_problematic"] else '' for _ in s]

styled = prev.style.apply(_row_style, axis=1).hide(axis="columns", subset=["_problematic"])
st.dataframe(styled, use_container_width=True)

st.info("**Siguiente paso** → Abrí **2) Objetivo** para definir el target. El generador priorizará consumir ítems 'problemáticos' y propondrá procesos que pueden mezclar **regolito MGS-1** (p.ej., *Sinter with MGS-1*).")
