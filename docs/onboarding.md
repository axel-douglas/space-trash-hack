# Onboarding operativo Rex-AI

El home y la demo guiada se reconstruyeron para priorizar claridad operativa en
los laboratorios. Los componentes viven en `app.modules.ui_blocks` y los flows
usan microcopys breves pensados para tripulaciones mixtas (ingeniería + crew
ops). Esta guía resume qué preparar antes de un recorrido y cómo adaptar la
experiencia.

## 1. Brief operativo
- **Componente**: combinación de `surface`, `chipline` y `minimal_button`.
- **Narrativa**: enfocá el mensaje en la misión del turno (ej. "Validar receta
  EVA sin PFAS"). Evitá metáforas y usa verbos de acción.
- **Assets**: las imágenes o GIFs deben ir en `app/static/media`. Mantener peso
  < 3 MB para garantizar carga rápida en Streamlit Cloud.

## 2. Checklist escalonado
- **Layout**: `layout_stack` + `layout_block("layout-grid--dual")`.
- **Contenido**: tres pasos máximos. El primero siempre "Inventario", el
  segundo "Target" y el tercero "Generador" o "Feedback" según la demo.
- **Estado**: usá `pill` (`ok`, `warn`, `risk`) para señalar el estado de cada
  paso sin necesidad de texto largo.

## 3. Guía automatizada
- **Activación**: `minimal_button` con `key="guided_demo"` que setea
  `st.session_state["demo_active"]`.
- **Progresión**: usa `st_autorefresh` o callbacks livianos para avanzar cada
  6-8 segundos. La animación `enable_reveal_animation()` suaviza la transición.
- **Copy de pasos**: declaralo en un JSON (`data/onboarding_steps.json`) con
  `title`, `body` y `icon`. Así podemos traducir rápido o ajustar mensajes.

## 4. Laboratorio Target
- **Componente clave**: sliders configurados con
  `compute_target_limits(presets)`. Garantiza que el CSV de inventario tenga las
  columnas `_source_volume_l` y `kg` para calcular baselines NASA.
- **Tip**: guardá presets de prueba en `data/targets_presets.json` con valores
  que cuenten una historia (ej. "Contenedores herméticos" con poca agua y alta
  estanqueidad).

## 5. Recomendaciones de demostración
1. **Preparar datos**: deja cargados `waste_inventory_sample.csv` y
   `processes_sample.csv` para evitar esperas.
2. **Storytelling**: comienza con la pill de seguridad para alinear prioridades
   y luego mostrá el generador.
3. **Hand-off**: al terminar, exportá un CSV desde el generador para entregar un
   artefacto tangible.

## 6. Troubleshooting rápido
- Si un chip no se pinta con el tono esperado, revisá que `tone` tenga uno de
  los valores soportados (`positive`, `warning`, `danger`, `info`, `accent`).
- Si los sliders muestran límites raros, corré `pytest tests/pages` para validar
  que `compute_target_limits` siga leyendo el inventario.
- Para ajustar densidades o sombras, edita `app/static/styles/base.css` y
  sincronizá la documentación en `docs/ui-components.md`.
