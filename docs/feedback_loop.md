# Sprint 3 — Feedback y aprendizaje activo

Este documento describe el flujo operativo para cerrar el lazo entre la app,
los ensayos físicos/simulador y el reentrenamiento incremental del modelo.

## 1. Registro de feedback

1. Los operadores exportan resultados de cada receta validada al archivo
   `feedback/recipes.parquet` (no versionado, ver `feedback/README.md`) utilizando
   el esquema documentado en `datasets/feedback_schema.yaml`.
2. Cada fila debe incluir `recipe_id`, `process_id`, valores medidos,
   intervalos de confianza (`conf_lo_*`, `conf_hi_*`), `label_source` y
   `label_weight` acorde a la confiabilidad de la medición.
3. Se recomienda completar `notes`, `measurement_ts` y `operator_id` para
   asegurar trazabilidad.

## 2. Ingesta al dataset gold

```bash
python scripts/ingest_feedback.py --feedback feedback/recipes.parquet --gold-dir datasets/gold
```

El comando:

* Valida que el feedback contenga todas las columnas definidas para el modelo.
* Fusiona la información con `datasets/gold/features.parquet` y
  `datasets/gold/labels.parquet`, preservando la última medición por
  `(recipe_id, process_id)`.
* Añade una marca `ingested_at` para auditar cuándo se incorporó cada fila.

Para verificar el impacto sin escribir resultados usar `--dry-run`.

## 3. Reentrenamiento incremental

```bash
python -m app.modules.retrain_from_feedback --logs feedback/recipes.parquet --gold datasets/gold
```

El módulo reusa el pipeline principal (`train_and_save`) y acepta los mismos
parámetros de sampling. El output en JSON incluye métricas, hashes de modelos y
resumen de pesos por `label_source`.

### Automatización sugerida

* **Cron/manual**: programar una corrida nocturna en la estación de trabajo
  (ej. `0 2 * * * python -m app.modules.retrain_from_feedback --logs 'feedback/*.parquet'`).
* **Alertas**: si la corrida detecta nuevos `feedback_*.parquet`, enviar
  notificación al canal de operaciones.
* **Versionado**: archivar los Parquet procesados en `data/logs/` con marca de
  tiempo para poder reproducir entrenamientos anteriores.

## 4. Active learning

El módulo `app/modules/active_learning.py` ordena candidatos según dos
estrategias:

* `uncertainty`: prioriza recetas con intervalos de confianza amplios.
* `expected_improvement`: combina el score previsto con la incertidumbre y el
  mejor resultado observado.

Integrar las sugerencias en la UI o en un cuaderno de planificación permite
concentrar ensayos en recetas que maximizan aprendizaje por etiqueta añadida.
