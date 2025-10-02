# Sistema visual base

La UI ahora depende de una única hoja de estilos ligera (`app/static/styles/base.css`) que aporta la paleta NASA-inspired y los grids responsivos compartidos por todas las páginas. El objetivo es mantener contraste alto sin HUDs ni toggles interactivos.

## Variables clave

| Variable | Valor | Uso |
| --- | --- | --- |
| `--color-background` | `#09192f` | Fondo general de la aplicación.
| `--color-surface` | `#112a45` | Paneles base y superficies neutras.
| `--color-surface-raised` | `#1c3554` | Tarjetas elevadas con sombra ligera.
| `--color-text` | `#f5f7fb` | Tipografía principal.
| `--color-accent` | `#63b3ff` | Links, estados activos y CTA.
| `--space-lg` | `1.5rem` | Espaciado vertical estándar entre bloques.
| `--shadow-soft` | `0 6px 18px rgba(3, 13, 31, 0.35)` | Profundidad suave para paneles.

> Si necesitás otro tono o espaciado, añadilo en `:root` siguiendo el prefijo `--color-` o `--space-` para mantener consistencia.

## Patrones disponibles

- `.layout-grid` + modificadores (`--dual`, `--flow`) resuelven columnas fluidas.
- `.layout-stack` y `.depth-stack` generan pilas verticales con gap consistente.
- `.rex-surface` y `.rex-glass` proveen superficies con bordes redondeados y sombras suaves.
- `.chipline`, `.badge-group` y `.hr-micro` mantienen micro-componentes alineados con la paleta.
- `.rexai-minimal-button` estiliza botones CTA sin depender de microinteracciones JS.

Todos los componentes se renderizan correctamente en móviles (<768px) gracias a los media queries incluidos; evitá sobreescribirlos salvo que sea imprescindible.
