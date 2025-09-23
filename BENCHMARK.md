# Benchmarks Rex-AI vs heurísticas

Este documento resume la comparación entre el modelo RandomForest multiobjetivo
y las reglas `heuristic_props` para tres escenarios fijos propuestos durante el
hackathon. El script [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py)
reproduce el experimento y guarda la evidencia en `data/benchmarks/`.

## Cómo ejecutar los benchmarks

1. Asegurá que los artefactos entrenados (`data/models/rexai_regressor.joblib`,
   clasificadores y metadata) están disponibles. Podés generarlos con
   `python -m app.modules.model_training` o descargar el bundle publicado en
   los releases.
2. Ejecutá el script:

   ```bash
   python scripts/run_benchmarks.py --format csv
   ```

   El comando produce dos archivos:

   - `data/benchmarks/scenario_predictions.csv`
   - `data/benchmarks/scenario_metrics.csv`

3. Opcionalmente, usá `--format both` para obtener también los Parquet.

## Escenarios evaluados

Los escenarios replican combinaciones frecuentes en la app:

| Escenario | Proceso | Regolito | Residuos (IDs) |
|-----------|---------|----------|----------------|
| Multicapa + Laminar | P02 – Press & Heat Lamination | 0% | W006, W007, W008 |
| Espuma + MGS-1 + Sinter | P03 – Sinter with MGS-1 | 20% | W001, W011, W015 |
| CTB Reconfig | P04 – CTB Kit Reconfig | 0% | W002, W009, W010 |

Las combinaciones se cargan desde `data/waste_inventory_sample.csv` y
`data/process_catalog.csv`, por lo que cualquier actualización en esos archivos
se reflejará automáticamente al rerunear el script.

## Resultados principales

Los errores se calculan tratando las heurísticas como baseline. El resumen por
escenario (promediado sobre las cinco métricas) es:

| Escenario | MAE medio | RMSE |
|-----------|-----------|------|
| Multicapa + Laminar | 169.25 | 360.89 |
| Espuma + MGS-1 + Sinter | 602.37 | 1 253.63 |
| CTB Reconfig | 535.19 | 1 163.39 |
| Global (los 3 escenarios) | 435.60 | 1 009.17 |

### Observaciones

* **Consumo de crew**: es la métrica con mayor desviación. Por ejemplo, el
  modelo estima 2 848.99 min para el escenario de espuma+sinter frente a
  50.57 min heurísticos, un error absoluto de 2 798.42 min.
* **Energía y agua**: también muestran gaps amplios en P03 (61.32 kWh y
  151.50 L por encima de la heurística). En P02 la energía queda mucho más
  alineada (3.43 kWh de diferencia).
* **Rigidez/estanqueidad**: las discrepancias son moderadas (≤0.35 en escala
  normalizada), útiles para validar calibración de targets mecánicos.

## Archivos generados

* `data/benchmarks/scenario_predictions.csv`: detalle por target con
  predicciones ML, heurística, error absoluto y bandas CI95.
* `data/benchmarks/scenario_metrics.csv`: métricas agregadas por escenario,
  por target y resumen global.

Ambos archivos pueden versionarse junto con el repositorio para documentar el
estado actual del modelo.
