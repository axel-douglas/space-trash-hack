# Playbook de QA Visual

Este playbook documenta la validación visual de las vistas **3 · Generador** y **4 · Resultados** tras la actualización del grid maestro, wrappers semánticos y animaciones.

| Resolución | Snapshot recomendado | Vista | Status | Notas clave |
|------------|----------------------|-------|--------|-------------|
| 1440 × 900 | `qa_snapshots/generator-1440.png` | Generador | ✅ Validado | Layout dual estable, panel IA y controles con `layout-grid--dual`; efectos `layer-glow` activos sin solapamientos. |
| 1280 × 832 | `qa_snapshots/generator-1280.png` | Generador | ✅ Validado | Columnas se reacomodan con container queries; badges en `badge-group` no rompen flujo. |
| 1024 × 768 | `qa_snapshots/generator-1024.png` | Generador | ✅ Validado | Grid colapsa a una sola columna, progreso de recursos mantiene legibilidad. |
| 1440 × 900 | `qa_snapshots/resultados-1440.png` | Resultados | ✅ Validado | Hero con parallax `layer-glow`, métricas iniciales renderizadas en `metric-grid`. |
| 1280 × 832 | `qa_snapshots/resultados-1280.png` | Resultados | ✅ Validado | Grids científicos (`layout-grid--dual`) conservan jerarquía; tablas dentro de `depth-stack`. |
| 1024 × 768 | `qa_snapshots/resultados-1024.png` | Resultados | ✅ Validado | Sección de KPIs y export cards se apilan correctamente sin overflow. |

## Checklist general

- [x] El grid maestro responde a las container queries declaradas en `app/static/layout.css` (breakpoints ultrawide → portrait).
- [x] Wrappers `layout-grid`, `side-panel`, `depth-stack` envuelven el contenido relevante en las páginas 3 y 4.
- [x] Animaciones `fade-in`, halos `layer-glow` y sombras `layer-shadow` aplicadas a héroes y paneles críticos.
- [x] Datos tabulares y gráficos mantienen alineación en resoluciones 1440, 1280 y 1024.
- [x] Se registraron snapshots locales (ver carpeta `qa_snapshots/`) como referencia para regresiones futuras.

> **Nota:** Para regenerar los snapshots ejecutar `streamlit run app/Home.py` y usar una herramienta de captura (Playwright / DevTools) en las resoluciones listadas, guardando los archivos bajo `docs/qa_snapshots/`.
