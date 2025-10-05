# Onboarding operativo Rex-AI

Esta guía acompaña a facilitadores y tripulaciones mixtas (ingeniería + crew
ops) que recorren la demo de Rex-AI. El objetivo es explicar qué preparar antes
de la sesión, cómo guiar cada paso y qué hacer si necesitás ajustar la
experiencia sobre la marcha.

## 1. Antes de la sesión

1. **Prepará los datasets de ejemplo** en `data/` (`waste_inventory_sample.csv`
   y `processes_sample.csv`) para evitar esperas al cargar el generador.
2. **Chequeá el inventario** ejecutando `pytest tests/pages` o el módulo
   `app.modules.data_pipeline`. Verificá que existan las columnas `kg`,
   `_source_volume_l` y `moisture_pct` para habilitar métricas automáticas.
3. **Define el storytelling**: elegí un escenario (p. ej. Residence Renovations)
   y decidí qué mensaje enfatizarás (seguridad, consumo de recursos, logística).
4. **Actualizá assets** livianos (imágenes o GIFs < 3 MB) dentro de
   `app/static/media` si necesitás un brief visual personalizado.

## 2. Estructura del recorrido

| Paso | Objetivo | Componentes clave |
| --- | --- | --- |
| **Brief inicial** | Alinear a la tripulación con la misión del turno. | `layout_stack`, `chipline`, `minimal_button` |
| **Checklist escalonado** | Recordar los pasos críticos (Inventario → Target → Generador/Feedback). | `layout_block("layout-grid--dual")`, `pill` |
| **Guía automatizada** | Mostrar la demo sin intervención manual. | `minimal_button` (`key="guided_demo"`), callbacks livianos |
| **Laboratorio Target** | Ajustar rigidez, estanqueidad y límites operativos. | `compute_target_limits`, sliders |
| **Generador / Export** | Visualizar riesgo, recursos y entregar un artefacto tangible. | `layout_stack`, `action_button` |

### Tips narrativos

- Usá verbos directos (“Validar receta EVA sin PFAS”) y evitá metáforas.
- Mostrá primero la pill de seguridad para fijar prioridades y luego la vista de
  candidatos.
- Al cerrar, exportá un CSV/JSON para entregar evidencia tangible de la sesión.

## 3. Personalización rápida

- **Mensajes guiados**: definilos en `data/onboarding_steps.json` con `title`,
  `body` e `icon`. Esto permite traducir o ajustar copy sin tocar código.
- **Presets de Target**: guardá escenarios de práctica en
  `data/targets_presets.json` (por ejemplo, “Contenedores herméticos” con poca
  agua y alta estanqueidad).
- **Tema visual**: cualquier ajuste a sombras o densidades se hace en
  `app/static/styles/base.css`; reflejá el cambio en `docs/ui-components.md`.

## 4. Troubleshooting inmediato

| Síntoma | Revisión recomendada |
| --- | --- |
| Chips sin color esperado | Confirmá que `tone` sea `positive`, `warning`, `danger`, `info` o `accent`. |
| Sliders con límites extraños | Ejecutá `pytest tests/pages` para validar `compute_target_limits`. |
| Guía automatizada no avanza | Revisá que `st.session_state["demo_active"]` se actualice y que los callbacks usen tiempos de 6–8 s. |
| Mixers sin datos | Verificá que los CSV de inventario/curvas tengan columnas numéricas y ejecutá nuevamente la carga. |

Con estas prácticas el onboarding mantiene un tono claro para perfiles técnicos
y no técnicos, minimizando sorpresas durante demostraciones en laboratorio.
