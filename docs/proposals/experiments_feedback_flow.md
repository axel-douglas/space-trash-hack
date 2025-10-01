# Propuesta externa — Flujo de experimentos y registro

**Estado**: pendiente de confirmación del equipo de operaciones.

Este documento consolida el flujo sugerido para capturar resultados de ensayos
físicos o de simulador y preparar los datos para reentrenamientos del modelo.
Permanece fuera del plan activo hasta que el equipo confirme que contará con los
recursos para sostenerlo.

## Objetivo

Establecer una convención única para documentar experimentos y facilitar su
ingesta en el dataset "gold". El diseño prioriza:

* **Trazabilidad** entre receta, proceso y corrida ejecutada.
* **Compatibilidad** con los esquemas existentes en `datasets/feedback_schema.yaml`.
* **Reproducibilidad** de métricas y pesos aplicados durante el reentrenamiento.

## Registro propuesto

Mientras no exista un almacenamiento definitivo, se sugiere crear un directorio
ad hoc en el entorno de trabajo (por ejemplo
`datasets/raw/experiments_YYYYMMDD.parquet`) y guardar allí cada lote diario. Un
archivo Parquet debe incluir:

| Campo | Descripción |
| --- | --- |
| `experiment_id` | UUID o hash del ensayo (alineado con el cuaderno físico). |
| `recipe_id`, `process_id` | Claves que vinculan el ensayo con la receta generada. |
| `environment` | `lab`, `iss_sim`, `analog_field`, etc. |
| `operator_id` | Astronauta/técnico responsable del ensayo. |
| `started_at`, `finished_at` | Timestamps ISO8601 para trazabilidad. |
| `raw_measurements` | JSON con lecturas crudas (temperatura, presión, torque…). |
| `verdict` | `pass`, `fail`, `inconclusive`. |
| `next_actions` | Texto breve con follow-up (p.ej. repetir con ajuste). |

## Integración con pipelines

Una vez confirmada la implementación, los pasos previstos son:

1. Registrar cada corrida en el Parquet del día usando la tabla anterior.
2. Ejecutar `scripts/ingest_feedback.py` para promover las corridas con
   mediciones consolidadas al dataset gold (creando el directorio destino si
   hiciera falta).
3. Archivar los Parquet diarios en `data/logs/` o en el almacenamiento que se
   defina para auditorías y respaldos.

## Próximos pasos

* Validar con operaciones si habrá personal para cargar y mantener estos
  registros.
* Ajustar los scripts y automatizaciones existentes según el almacenamiento que
  seleccione el equipo.
* Integrar la documentación aprobada en los manuales operativos una vez que el
  flujo se ponga en marcha.
