# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import pandas as pd
from app.modules.io import load_waste_df, save_waste_df

# ⚠️ Primera llamada de Streamlit
st.set_page_config(page_title="Inventario", page_icon="🧱", layout="wide")

# --------------------------------------------------------------------
# Narrativa y explicación
# --------------------------------------------------------------------
st.title("1) Inventario de residuos inorgánicos")
st.caption("📦 **Panel de laboratorio del hábitat** — Aquí comienza todo: "
           "registrar con precisión qué basura espacial tenemos disponible.")

st.markdown("""
En cualquier misión, antes de fabricar **nuevas herramientas o piezas críticas**,  
los astronautas necesitan saber **qué materiales tienen a mano**.  
Este inventario es como **el almacén de repuestos del futuro**: plástico de empaques, 
fragmentos metálicos, textiles técnicos… todo puede transformarse en algo útil.  

👉 **Tu tarea ahora es completar este inventario como si fueras parte del equipo de la misión.**  
Cada fila representa un lote de residuo que podrá convertirse en materia prima.
""")

# --------------------------------------------------------------------
# Cargar y mostrar datos
# --------------------------------------------------------------------
df = load_waste_df()

st.subheader("📊 Inventario editable")
st.caption("Agregá, editá o duplicá filas para simular nuevos lotes. "
           "Guardá antes de continuar al siguiente paso.")

edited = st.data_editor(
    df, num_rows="dynamic", use_container_width=True,
    key="waste_editor",
    column_config={
        "material": st.column_config.TextColumn("Material"),
        "kg": st.column_config.NumberColumn("Masa (kg)", min_value=0.0, step=0.1),
        "notes": st.column_config.TextColumn("Notas")
    }
)

if st.button("💾 Guardar inventario", type="primary"):
    save_waste_df(edited)
    st.success("✅ Inventario guardado. Los datos están listos para los próximos cálculos.")

# --------------------------------------------------------------------
# Explicaciones y tips
# --------------------------------------------------------------------
with st.expander("ℹ️ ¿Por qué importa este paso? (explicación para no expertos)"):
    st.markdown("""
- Si el inventario está **incompleto o mal cargado**, el generador de procesos 
  va a proponer recetas poco realistas.  
- En la misión real, esto equivaldría a **perder tiempo y recursos** valiosos 
  tratando de fabricar algo con materiales que no existen.  
- Cargar bien este inventario es como darle a la IA una **foto fiel del almacén**: 
  cuanto más clara sea, mejores soluciones propondrá.  
    """)

st.info("💡 **Tip de misión**: duplicá una fila si querés simular que llegaron "
        "más lotes de un mismo residuo (ejemplo: varias bolsas de embalaje plástico).")

# --------------------------------------------------------------------
# Narrativa de cierre
# --------------------------------------------------------------------
st.markdown("---")
st.success("✅ **Inventario listo** — Abrí la página **2) Definir objetivo** para elegir qué fabricar.")

st.caption("🛰️ *Recordá: cada tornillo, cada retazo de plástico, puede ser la diferencia "
           "entre una misión fallida y un logro histórico.*")
