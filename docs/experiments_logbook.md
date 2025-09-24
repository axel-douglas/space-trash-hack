# Registro de experimentos físicos / simulador

Para acompañar el esquema de feedback se define la convención de archivos
`datasets/raw/experiments_YYYYMMDD.parquet`. Cada lote documenta:

| Campo | Descripción |
| --- | --- |
| `experiment_id` | UUID o hash del ensayo (concordante con cuaderno físico). |
| `recipe_id`, `process_id` | Claves que vinculan el ensayo con la receta generada. |
| `environment` | `lab`, `iss_sim`, `analog_field`, etc. |
| `operator_id` | Astronauta/técnico responsable del ensayo. |
| `started_at`, `finished_at` | Timestamps ISO8601 para trazabilidad. |
| `raw_measurements` | JSON con lecturas crudas (temperatura, presión, torque…). |
| `verdict` | `pass`, `fail`, `inconclusive`. |
| `next_actions` | Texto breve con follow-up (p.ej. repetir con ajuste). |

## Flujo operativo

1. Registrar cada corrida en el Parquet del día usando la tabla anterior.
2. Al finalizar el turno, ejecutar `scripts/ingest_feedback.py` para promover las
   corridas con mediciones consolidadas al dataset gold.
3. Guardar los Parquet diarios en `data/logs/experiments/` para respaldos y
   auditorías.

Este registro asegura trazabilidad completa desde la planificación de la receta
hasta las mediciones que alimentan el modelo.
