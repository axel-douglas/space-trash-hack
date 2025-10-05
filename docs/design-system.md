# Sistema visual base

La app utiliza una única hoja de estilos (`app/static/styles/base.css`) para
aplicar la paleta inspirada en NASA, tipografía y grillas responsivas. El diseño
prioriza contraste alto y componentes reutilizables para que cada página se
perciba como parte de la misma consola.

## Variables de diseño

| Variable | Valor | Descripción |
| --- | --- | --- |
| `--mission-color-background` | `#09192f` | Fondo global. |
| `--mission-color-panel` | `#101724` | Paneles secundarios y tarjetas. |
| `--mission-color-surface` | `#112a45` | Superficies principales. |
| `--mission-color-text` | `#f8fafc` | Texto primario. |
| `--mission-color-accent` | `#5aa9ff` | CTA, estados activos, iconografía. |
| `--mission-space-sm/md/lg` | `0.75rem` / `1rem` / `1.5rem` | Gaps estándar entre componentes. |
| `--mission-radius-md` | `0.5rem` | Bordes redondeados para paneles. |
| `--mission-shadow-soft` | `0 6px 18px rgba(3, 13, 31, 0.35)` | Sombra base para profundidad. |

Extender el sistema sumando variables con el prefijo `--mission-` garantiza
consistencia entre páginas y facilita documentar cambios en `docs/ui-components.md`.

## Patrones reutilizables

- `.layout-grid` (`--dual`, `--flow`) → columnas fluidas y responsivas.
- `.layout-stack`, `.depth-stack` → pilas verticales con espaciado consistente.
- `.side-panel`, `.pane` → contenedores con borde, sombra y padding definidos.
- `.chipline`, `.badge-group` → etiquetas compactas para estado o filtros.
- `.rexai-minimal-button` → CTA sin microinteracciones complejas.

Todos los patrones incluyen media queries para pantallas < 768 px. Evitá
sobreescribirlos salvo que documentes el cambio.

## Ajustes recomendados

1. Actualizá colores desde la raíz (`:root`) y reflejá las modificaciones en
   `docs/ui-components.md`.
2. Para nuevos componentes, apoyate en `layout_stack` y `layout_block` desde
   `app/modules/ui_blocks.py` para heredar estilos automáticamente.
3. Validá contraste con herramientas WCAG al introducir combinaciones de color
   personalizadas.
