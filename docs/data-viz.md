# Rex-AI Data Visualization Theme

La experiencia visual de Rex-AI está inspirada en tableros automotrices de gama
alta: superficies satinadas, tipografías técnicas y acentos eléctricos que
refuerzan precisión y dinamismo. Este documento resume las guías de uso,
paletas y configuraciones disponibles para Altair y Plotly.

## Activación

* La app registra los temas automáticamente desde `_bootstrap.py` mediante
  `apply_global_visual_theme`. Las páginas solo deben importar `_bootstrap`
  (ya ocurre en todos los módulos de `app/`).
* El modo por defecto es `dark`. Puede forzarse otra variante definiendo la
  variable de entorno `REXAI_THEME_MODE=light` o `REXAI_THEME_MODE=dark`.
* Ambos motores (Altair y Plotly) comparten tokens cromáticos a través del
  módulo `app.modules.visual_theme`. Para acceder a ellos:  
  ```python
  from app.modules.visual_theme import get_palette

  palette = get_palette()  # respeta el modo activo
  st.write(palette.categorical)
  ```

## Tipografía y layout

| Elemento               | Configuración                                            |
|------------------------|----------------------------------------------------------|
| Títulos / headings     | `Rajdhani`, semibold, espaciado 0.02em                    |
| Cuerpos / labels       | `Inter`, 12–14px según el motor                           |
| Fondos                 | Panel satinado (ver paleta) con grid en baja opacidad     |
| Bordes y gridlines     | Gris azulado translúcido (`grid` en la paleta)            |
| Hover / focus          | Fondos oscuros con borde en `accent`                      |
| Selecciones destacadas | Gradiente eléctrico (ver sección siguiente)               |

## Paletas

### Modo claro

| Token              | Valor                                            | Uso principal                                  |
|--------------------|--------------------------------------------------|------------------------------------------------|
| `background`       | `#F8FAFC`                                        | Fondo de aplicación / dashboards               |
| `surface`          | `#FFFFFF`                                        | Paneles, tarjetas y áreas de gráfica           |
| `panel`            | `#E2E8F0`                                        | Bordes suaves y barras de herramientas         |
| `text`             | `#0F172A`                                        | Texto principal                                |
| `muted`            | `#475569`                                        | Etiquetas secundarias                          |
| `accent`           | `#1D4ED8`                                        | Líneas, marcadores activos                     |
| `accent_soft`      | `#60A5FA`                                        | Rellenos suaves                                |
| `grid`             | `rgba(30,64,175,0.14)`                            | Gridlines y ejes                               |
| `categorical`      | `#0F76FF`, `#1DD3F8`, `#F59E0B`, `#14B8A6`, `#7C3AED`, `#F97316` | Series múltiples | 

### Modo oscuro

| Token              | Valor                                            | Uso principal                                  |
|--------------------|--------------------------------------------------|------------------------------------------------|
| `background`       | `#070B12`                                        | Fondo global                                   |
| `surface`          | `#0E141F`                                        | Paneles y tarjetas                             |
| `panel`            | `#192132`                                        | Delineados sutiles                             |
| `text`             | `#F8FAFC`                                        | Texto principal                                |
| `muted`            | `#94A3B8`                                        | Texto secundario                               |
| `accent`           | `#38BDF8`                                        | Curvas y puntos activos                        |
| `accent_soft`      | `#7DD3FC`                                        | Rellenos suaves                                |
| `grid`             | `rgba(148,163,184,0.22)`                          | Gridlines y ejes                               |
| `categorical`      | `#7DD3FC`, `#34D399`, `#FBBF24`, `#F472B6`, `#A855F7`, `#F97316` | Series múltiples | 

### Gradiente eléctrico (highlight)

El resalte para selecciones y transiciones se basa en un gradiente de energía
eléctrica aplicado en ambos motores como escala secuencial:

* `#7CF4FF`
* `#2AA8FF`
* `#4C4CFF`

Altair lo usa para `range["diverging"]`, `range["heatmap"]` y `range["ramp"]`,
mientras que Plotly lo asigna a `coloraxis` y a los `colorscale` por defecto en
`scatter`, `bar` y `heatmap`. Aplicarlo manualmente es tan simple como usar el
rango `"ramp"` o el template `"rexai_dark"/"rexai_light"` ya registrado.

## Buenas prácticas

1. **Evitar hardcodear colores**: el tema ya define escalas por defecto para
   Altair y Plotly. Solo sobrescribir cuando haya semántica crítica.
2. **Contraste**: sobre fondos personalizados, respetar las combinaciones
   `text` + `muted` para mantener accesibilidad WCAG AA.
3. **Highlight de selección**: para resaltar filas/barra activa, usar los
   colores del gradiente (`palette.electric_gradient`) en orden o aplicarlos a
   `marker.line.color` en Plotly.
4. **Impresión / exportación**: para fondos blancos usar `REXAI_THEME_MODE=light`
   en la línea de comandos antes de ejecutar `streamlit run`.

## Referencias rápidas

* Altair: tema registrado como `rexai_dark` y `rexai_light`.
* Plotly: template registrado como `rexai_dark` y `rexai_light`; el template
  activo queda en `pio.templates.default`.
* CSS global: se inyecta automáticamente mediante `apply_global_visual_theme`,
  reutilizando `app/static/styles/base.css`.
