# Rex-AI Data Visualization Theme

Guía para utilizar los temas registrados de Altair y Plotly en la consola
Rex-AI. Describe cómo activar la paleta, qué tokens están disponibles y buenas
prácticas para mantener consistencia visual.

## Activación

```python
from app.modules.ui_blocks import initialise_frontend
from app.modules.visual_theme import apply_global_visual_theme

initialise_frontend()
apply_global_visual_theme()
```

- El modo por defecto es oscuro; podés forzarlo con
  `REXAI_THEME_MODE=light|dark`.
- `get_palette()` expone los colores actuales para integraciones personalizadas.

## Tokens principales

| Token | Uso |
| --- | --- |
| `background` | Fondo global del dashboard. |
| `surface` | Paneles y tarjetas. |
| `panel` | Contenedores secundarios / overlays. |
| `text`, `muted` | Texto principal y secundario. |
| `accent`, `accent_soft` | Curvas activas, marcas destacadas. |
| `grid` | Líneas de referencia y ejes. |
| `categorical` | Lista de colores para series múltiples. |

Los valores cambian automáticamente entre modo claro y oscuro. Usá el helper
para obtenerlos:

```python
from app.modules.visual_theme import get_palette
palette = get_palette()
alt.themes.set_theme("rexai_dark")
```

## Gradiente eléctrico

El gradiente `palette.electric_gradient` (`#7CF4FF → #2AA8FF → #4C4CFF`) se usa
para highlights, heatmaps y selección de filas. En Plotly está disponible como
`colorscale="rexai_electric"`.

## Buenas prácticas

1. Evitá hardcodear colores; apoyate en el template `rexai_dark`/`rexai_light`.
2. Reutilizá `palette.categorical` para asegurar contraste adecuado.
3. Para exportar a PDF o slides con fondo claro activá `REXAI_THEME_MODE=light`
ante de ejecutar `streamlit run`.
4. Documentá cualquier escala adicional en este archivo para mantener el catálogo
   al día.

## Referencias

- Altair registra los temas como `rexai_dark` y `rexai_light`.
- Plotly establece `pio.templates.default = "rexai_dark"` (o light según modo).
- El CSS global se inyecta mediante `apply_global_visual_theme()`.
