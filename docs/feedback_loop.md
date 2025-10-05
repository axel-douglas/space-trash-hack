# Sprint 3 — Feedback y aprendizaje activo

Descripción del flujo para capturar feedback operativo, incorporarlo al dataset
gold y reentrenar Rex-AI de forma incremental. Incluye cómo priorizar nuevas
mediciones con estrategias de active learning.

## 1. Registro de feedback operativo

1. **Capturá datos en la app**: cada formulario de **Feedback & Impact** genera
   `data/logs/feedback_*.parquet` y `data/logs/impact_*.parquet` con métricas y
   notas estructuradas.
2. **Complementá con ensayos físicos** cuando corresponda. Usá el esquema
   documentado en `datasets/feedback_schema.yaml` para exportar registros a
   `feedback/recipes.parquet` (no versionado).
3. **Campos mínimos recomendados**: `recipe_id`, `process_id`, mediciones reales,
   intervalos `conf_lo_*`/`conf_hi_*`, `label_source`, `label_weight`,
   `measurement_ts`, `operator_id` y `notes`.

## 2. Ingesta al dataset gold

```bash
python scripts/ingest_feedback.py \
  --feedback feedback/recipes.parquet \
  --gold-dir data/gold
```

El script valida columnas, fusiona nuevas mediciones con
`data/gold/features.parquet` y `data/gold/labels.parquet`, agrega `ingested_at`
para auditoría y crea directorios de destino si no existen. Usá `--dry-run` para
ver el diff antes de escribir.

## 3. Reentrenamiento incremental

```bash
python -m app.modules.retrain_from_feedback \
  --logs "data/logs/feedback_*.parquet" \
  --gold data/gold
```

El módulo reusa el pipeline principal (`train_and_save`) y genera un resumen
JSON con métricas, hashes de artefactos y pesos aplicados por `label_source`.

### Automatización sugerida

- Programá una corrida nocturna (`cron`, GitHub Actions o similar) que ejecute el
  comando anterior cuando se detecten nuevos Parquet.
- Enviá notificaciones a operaciones si el reentrenamiento incorpora feedback
  fresco o si `ModelRegistry.ready` cambia de estado.
- Archivá los archivos procesados en `data/logs/processed/` para reproducir
  entrenamientos anteriores.

## 4. Aprendizaje activo

`app/modules/active_learning.py` prioriza qué recetas conviene evaluar:

| Estrategia | Descripción |
| --- | --- |
| `uncertainty` | Ordena candidatos con intervalos de confianza más amplios para reducir incertidumbre. |
| `expected_improvement` | Combina score previsto, incertidumbre y mejor resultado histórico para maximizar ganancia esperada. |

Podés integrar estas sugerencias en la UI (pestaña de generador) o en un cuaderno
de planificación para decidir qué recetas ensayar a continuación. La propuesta
de experimentos extendidos permanece documentada en
[`docs/proposals/experiments_feedback_flow.md`](proposals/experiments_feedback_flow.md).
