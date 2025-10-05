# Biblioteca de componentes UI

`app/modules/ui_blocks.py` centraliza el layout y los microcomponentes de Rex-AI.
La meta es armar pantallas consistentes sin escribir HTML crudo y mantener un
tono directo para la tripulación.

## Inicializar tema y layout

```python
from app.modules.ui_blocks import configure_page, initialise_frontend, layout_stack

configure_page(page_title="Rex-AI", page_icon="🛰️")
initialise_frontend()

with layout_stack() as stack:
    stack.subheader("Panel de control Rex-AI")
    stack.write("Agrupá métricas clave dentro del contexto del laboratorio.")
```

- `configure_page` aplica título, favicon y opciones de Streamlit.
- `initialise_frontend` inyecta la hoja de estilos base y el tema compartido.
- `layout_stack`/`layout_block` construyen grillas responsivas sin escribir CSS.

## Patrones operativos

```python
from app.modules.ui_blocks import layout_stack, layout_block, micro_divider

with layout_stack() as stack:
    stack.markdown("### Checklist del turno")
    with layout_block("layout-grid layout-grid--dual", parent=stack):
        col_left, col_right = st.columns(2)
        col_left.success("Inventario NASA validado")
        col_right.info("Crew brief listo")
    micro_divider(parent=stack)
    stack.caption("Actualizá el estado antes de iniciar la corrida IA.")
```

## Pills y chips

```python
from app.modules.ui_blocks import pill, chipline

pill("🛡️ Seguridad OK", kind="ok")
chipline([
    {"label": "PFAS controlados", "icon": "🧪", "tone": "positive"},
    {"label": "Microplásticos mitigados", "icon": "🧴", "tone": "positive"},
    {"label": "Crew listo", "icon": "👩‍🚀"},
])
```

- `pill(kind="ok" | "warn" | "risk")` resume el estado operativo.
- `chipline` acepta strings o diccionarios con `label`, `icon`, `tone`.
- Pasá `render=False` para obtener el HTML y reutilizarlo en otros contenedores.

## Botones de acción

```python
from app.modules.ui_blocks import minimal_button

if minimal_button(
    "Iniciar mezcla IA",
    key="launch_blend",
    help_text="Ejecuta blending + validación NASA",
    status_hints={"loading": "Optimizando parámetros"},
):
    lanzar_pipeline()
```

`minimal_button` soporta estados `idle`, `loading`, `success`, `error` y evita
microinteracciones intrusivas. `action_button` añade soporte para descargas y
marcos más robustos.

## Sliders del Target Designer

`compute_target_limits` calcula límites y tooltips a partir de presets e
inventario NASA.

```python
from app.modules.target_limits import compute_target_limits
presets = load_targets()
limits = compute_target_limits(presets)
water_slider = st.slider(
    "Agua máxima (L)",
    limits["max_water_l"]["min"],
    limits["max_water_l"]["max"],
    limits["max_water_l"]["default"],
    limits["max_water_l"]["step"],
    help=limits["max_water_l"]["help"],
)
```

## Extender la biblioteca

1. Creá nuevos helpers en `ui_blocks.py` con parámetros explícitos y soporte para
   `render=False` (útil en tests).
2. Definí clases en `app/static/styles/base.css` siguiendo el prefijo `rex-` o
   `layout-` para mantener consistencia.
3. Documentá cambios relevantes en este archivo y en `docs/design-system.md`.

### Copy y tono

- Títulos cortos (máx. 4 palabras) y verbos de acción.
- Mensajes directos, sin metáforas ni jerga innecesaria.
- Siempre indicá qué debe hacer la tripulación a continuación.
