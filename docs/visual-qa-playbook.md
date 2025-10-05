# Playbook de QA Visual

Checklist para validar las vistas **3 · Generador** y **4 · Resultados** después
de cambios en grillas, wrappers y componentes.

| Resolución | Snapshot | Vista | Estado | Observaciones |
| --- | --- | --- | --- | --- |
| 1440 × 900 | `qa_snapshots/generator-1440.png` | Generador | ✅ | Layout dual estable, panel IA y controles mantienen `layout-grid--dual`. |
| 1280 × 832 | `qa_snapshots/generator-1280.png` | Generador | ✅ | Columnas se adaptan con container queries; badges alineados en `badge-group`. |
| 1024 × 768 | `qa_snapshots/generator-1024.png` | Generador | ✅ | Grid colapsa a una columna, progreso de recursos sigue legible. |
| 1440 × 900 | `qa_snapshots/resultados-1440.png` | Resultados | ✅ | Hero aplica gradiente base; métricas iniciales en `layout-grid--dual`. |
| 1280 × 832 | `qa_snapshots/resultados-1280.png` | Resultados | ✅ | Grids científicos conservan jerarquía; tablas en `depth-stack`. |
| 1024 × 768 | `qa_snapshots/resultados-1024.png` | Resultados | ✅ | KPIs y cartas de exportación se apilan sin overflow. |

## Checklist

- [x] El grid maestro responde a los breakpoints definidos en `app/static/styles/base.css`.
- [x] Wrappers `layout-grid`, `side-panel`, `depth-stack` encapsulan el contenido principal.
- [x] Sombras suaves activas (`layer-shadow`) sin HUDs adicionales.
- [x] Datos tabulares y visualizaciones mantienen alineación en 1440, 1280 y 1024 px.
- [x] Snapshots locales guardados en `docs/qa_snapshots/` (no versionado) para regresiones.

> Para regenerar snapshots: ejecutá `streamlit run app/Home.py` y capturá las
> vistas con Playwright o DevTools usando las resoluciones listadas.
