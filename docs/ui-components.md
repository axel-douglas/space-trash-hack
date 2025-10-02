# Biblioteca de componentes UI

La capa visual de Rex-AI se consolidó en `app.modules.ui_blocks`. El objetivo es
ofrecer bloques accesibles, fáciles de combinar y con un perfil visual
consistente para las demostraciones de laboratorio. El módulo inyecta un tema
ligero inspirado en paneles NASA y expone utilidades para construir layouts,
controles y estados informativos sin depender del antiguo paquete `luxe`.

> **Nota de copy**: mantené un tono directo y técnico. Los títulos deben ser
> cortos y los mensajes orientados a acciones concretas de la tripulación.

## Inicializar tema y HUD accesible

```python
from app.modules.ui_blocks import load_theme, surface, enable_reveal_animation

load_theme()  # inyecta CSS base, HUD y microinteracciones

with surface(tone="raised"):
    st.subheader("Panel de control Rex-AI")
    st.write("Agrupá métricas clave dentro del contexto del laboratorio.")

enable_reveal_animation()  # activa animaciones suaves cuando haya scroll
```

- `surface` y `glass_card` generan contenedores temáticos con paddings y
  sombras coherentes.
- El HUD accesible permite cambiar contraste, tipografía y modo daltónico sin
  tocar CSS.

## Layouts operativos

```python
from app.modules.ui_blocks import layout_stack, layout_block, micro_divider

with layout_stack() as stack:
    stack.markdown("### Checklist del turno")
    with layout_block("layout-grid layout-grid--dual", parent=stack):
        col1, col2 = st.columns(2)
        col1.success("Inventario NASA validado")
        col2.info("Crew brief listo")
    micro_divider(parent=stack)
    stack.caption("Actualizá el estado antes de iniciar la corrida IA.")
```

Los helpers utilizan clases definidas en `app/static/styles/base.css`. Evitá
escribir HTML crudo salvo que necesites un layout puntual; siempre que sea
posible extendé estas utilidades.

## Chips y pills para estados de laboratorio

```python
from app.modules.ui_blocks import chipline, pill

pill("🛡️ Seguridad · OK", kind="ok")

chipline(
    [
        {"label": "PFAS controlados", "icon": "🧪", "tone": "positive"},
        {"label": "Microplásticos mitigados", "icon": "🧴", "tone": "positive"},
        {"label": "Crew listo", "icon": "👩‍🚀"},
    ]
)
```

- `pill` admite `kind="ok" | "warn" | "risk"` para reflejar el semáforo
  operativo.
- `chipline` acepta strings o diccionarios con `label`, `icon` y `tone`. Usala
  para resumir medidas de mitigación, badges de inventario o flags EVA.

Si necesitás el HTML para incrustarlo en otro componente, pasá `render=False` y
recibirás el markup listo para reutilizar.

## Botones de acción compartidos

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

- `minimal_button` cubre el 90% de los CTAs operativos; ajustá `state` (`idle`,
  `loading`, `success`, `error`) según la respuesta del backend.
- `futuristic_button` mantiene la variante con partículas y sonido para demos
  públicas. Usa `mode="cinematic"` si querés el efecto completo.

## Marcadores de objetivo

Los sliders de la página **Target Designer** se alimentan con
`app.modules.target_limits.compute_target_limits`. El helper analiza los presets
de `data/targets_presets.json` y el inventario NASA para calcular límites,
pasos y mensajes de ayuda.

```python
from app.modules.target_limits import compute_target_limits

presets = load_targets()
slider_limits = compute_target_limits(presets)
st.slider(
    "Agua máxima (L)",
    slider_limits["max_water_l"]["min"],
    slider_limits["max_water_l"]["max"],
    slider_limits["max_water_l"]["min"],
    slider_limits["max_water_l"]["step"],
    help=slider_limits["max_water_l"]["help"],
)
```

> **Tip**: mantené el CSV de inventario actualizado; el helper toma el P90 de
> volumen y masa para alinear los límites con los baseline NASA.

## Extender el sistema

- Si necesitás un nuevo bloque, creadlo en `ui_blocks.py` siguiendo la misma
  filosofía: estilos inyectados, API declarativa y soporte para `render=False`
  en caso de tests o composición.
- Para variantes visuales definí clases en `app/static/styles/base.css` y
  referencialas mediante `use_token` o funciones auxiliares.
- Documentá los cambios en esta página para que el equipo de laboratorio tenga
  una referencia única del sistema operativo UI.
