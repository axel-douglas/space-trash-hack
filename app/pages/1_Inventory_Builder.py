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
st.caption("Este inventario refleja ítems no-metabólicos problemáticos para NASA: pouches multicapa, espumas (ZOTEK F30), bolsas EVA/CTB, textiles/wipes y guantes de nitrilo. El generador prioriza valorizarlos y, cuando corresponde, mezcla **regolito MGS-1**.")

# --------------------- helpers robustos de columnas ---------------------
def _pick_col(df: pd.DataFrame, names: list[str], default=None):
    for n in names:
        if n in df.columns:
            return n
    return default

def _is_problematic_row(row: pd.Series) -> bool:
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

# --------------------- cargar y normalizar a esquema estándar ---------------------
raw = load_waste_df().copy()

id_col     = _pick_col(raw, ["id"], default=None)
cat_col    = _pick_col(raw, ["category", "Category"], default=None)
mat_col    = _pick_col(raw, ["material_family", "material", "Material"], default=None)
mass_col   = _pick_col(raw, ["mass_kg", "kg", "Mass_kg"], default=None)
vol_col    = _pick_col(raw, ["volume_l", "Volume_L"], default=None)
flags_col  = _pick_col(raw, ["flags", "Flags"], default=None)

# Construimos un DF estándar para edición/guardado:
df = pd.DataFrame({
    "id": raw[id_col] if id_col else raw.index.astype(str),
    "category": raw[cat_col] if cat_col else "",
    "material_family": raw[mat_col] if mat_col else "",
    "mass_kg": pd.to_numeric(raw[mass_col], errors="coerce").fillna(0.0) if mass_col else 0.0,
    "volume_l": pd.to_numeric(raw[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0,
    "flags": raw[flags_col] if flags_col else "",
})

# Derivado “problemático” para métricas/vista
df["_problematic"] = df.apply(_is_problematic_row, axis=1)

# --------------------- métricas “sabor laboratorio” ---------------------
c1,c2,c3,c4 = st.columns(4)
c1.metric("Ítems", len(df))
c2.metric("Masa total (kg)", f"{df['mass_kg'].sum():.2f}")
c3.metric("Volumen (L)", f"{df['volume_l'].sum():.1f}")
c4.metric("Problemáticos", int(df["_problematic"].sum()))

with st.expander("¿Por qué estos ítems? (resumen rápido)", expanded=False):
    st.markdown("""
- **Pouches PE–PET–Al** (multicapa) → capas incompatibles, difícil reciclado.
- **Espumas ZOTEK F30 (PE reticulado)** → celdas cerradas, voluminosas.
- **EVA/CTB (Nomex/Nylon/Polyester)** → textiles técnicos.
- **Guantes de nitrilo** → elastómeros con aditivos.
- **Aluminio** → valioso para **reuso/refuerzo**.
- **MGS-1** → carga mineral para *Sinter with MGS-1*.
""")

st.markdown("### Editar inventario")
st.caption("Los **problemáticos** se resaltan más abajo. Podés ajustar `flags`/`category` para reflejar tus lotes.")

edited = st.data_editor(
    df[["id","category","material_family","mass_kg","volume_l","flags"]],
    num_rows="dynamic",
    use_container_width=True,
    key="inv_editor",
)

colA, colB = st.columns([1,1])
with colA:
    if st.button("💾 Guardar inventario", type="primary"):
        # Guardamos con el **esquema estándar** esperado por el pipeline:
        out = edited[["id","category","material_family","mass_kg","volume_l","flags"]].copy()
        save_waste_df(out)
        st.success("Inventario guardado.")

with colB:
    if st.button("↺ Recalcular 'problemáticos' para vista previa"):
        st.rerun()

st.markdown("---")
st.subheader("Vista previa con indicadores de 'problemáticos'")

preview = edited.copy()
preview["_problematic"] = preview.apply(_is_problematic_row, axis=1)

# Columna indicador no intrusiva (sin pintar el fondo)
preview.insert(
    0,
    "Indicador",
    preview["_problematic"].map(lambda v: "⚠️ Problemático" if v else "✓ OK")
)

# Estilo: solo color de texto en la columna Indicador (sin fondos que quiten contraste)
def _indicator_style(val: str):
    if isinstance(val, str) and "Problemático" in val:
        return "color: #d9534f; font-weight: 600;"  # rojo suave, buena lectura en temas claro/oscuro
    if isinstance(val, str) and "OK" in val:
        return "color: #1f9d55; font-weight: 600;"  # verde suave
    return ""

styled = (
    preview.style
    .map(_indicator_style, subset=["Indicador"])
    .hide(axis="columns", subset=["_problematic"])  # ocultamos helper
)

st.dataframe(styled, use_container_width=True)

st.info("**Siguiente paso** → Abrí **2) Objetivo**. El generador prioriza ítems problemáticos y usa **P03 (Sinter with MGS-1)** con regolito cuando corresponde.")

show_only_prob = st.toggle("Mostrar solo problemáticos", value=False)
preview_view = preview[preview["_problematic"]] if show_only_prob else preview
styled = (
    preview_view.style
    .map(_indicator_style, subset=["Indicador"])
    .hide(axis="columns", subset=["_problematic"])
)
st.dataframe(styled, use_container_width=True)

