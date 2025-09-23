# Benchmarks Rex-AI vs heurísticas

Este documento resume la comparación entre el modelo RandomForest multiobjetivo
y las reglas `heuristic_props` para tres escenarios fijos propuestos durante el
hackathon. El script [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py)
reproduce el experimento y guarda la evidencia en `data/benchmarks/`.

## Cómo ejecutar los benchmarks

1. Asegurá que los artefactos entrenados (`data/models/rexai_regressor.joblib`,
   clasificadores y metadata) están disponibles. Podés regenerarlos con:

   ```bash
   python -m app.modules.model_training --gold datasets/gold --append-logs "data/logs/feedback_*.parquet"
   ```

   Si preferís evitar el entrenamiento local, descargá el bundle
   `rexai_model_bundle_hybrid_v1.zip` desde la sección **Releases** (o el
   artifact equivalente de CI), descomprimilo y copiá su contenido en
   `data/models/` antes de continuar.
2. Ejecutá el script:

   ```bash
   python scripts/run_benchmarks.py --format csv
   ```

   El comando produce dos archivos:

   - `data/benchmarks/scenario_predictions.csv`
   - `data/benchmarks/scenario_metrics.csv`

3. Añadí `--with-ablation` para ejecutar barridos desactivando grupos de
   features (ver sección "Ablation de features"). Opcionalmente, usá
   `--format both` para obtener también los Parquet.

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
| Multicapa + Laminar | 197.76 | 381.53 |
| Espuma + MGS-1 + Sinter | 594.04 | 1 211.11 |
| CTB Reconfig | 632.74 | 1 192.88 |
| Global (los 3 escenarios) | 474.85 | 1 005.87 |

Los valores anteriores provienen de `data/benchmarks/scenario_metrics.csv` y se
actualizan automáticamente al rerunear el script.【F:data/benchmarks/scenario_metrics.csv†L2-L25】

### Observaciones

* **Consumo de crew**: continúa siendo la métrica con mayor desviación. En
  espuma+sinter el modelo estima 2 751.60 min frente a 50.57 min heurísticos,
  un error absoluto de 2 701.03 min.【F:data/benchmarks/scenario_predictions.csv†L11-L11】
* **Energía y agua**: en P03 la energía cae 168.31 kWh por debajo de la
  heurística, mientras que el agua sube 100.31 L; en P02 la energía se mantiene
  cercana con 80.81 kWh de gap.【F:data/benchmarks/scenario_predictions.csv†L4-L10】
* **Rigidez/estanqueidad**: los desvíos siguen siendo moderados (≤0.31), útiles
  para validar calibración de targets mecánicos.【F:data/benchmarks/scenario_predictions.csv†L2-L8】【F:data/benchmarks/scenario_predictions.csv†L12-L13】

## Archivos generados

* `data/benchmarks/scenario_predictions.csv`: detalle por target con
  predicciones ML, heurística, error absoluto y bandas CI95.
* `data/benchmarks/scenario_metrics.csv`: métricas agregadas por escenario,
  por target y resumen global.
* `data/benchmarks/ablation_predictions.csv`: detalle de cada corrida de
  ablation por escenario y target.
* `data/benchmarks/ablation_metrics.csv`: métricas agregadas por grupo de
  features desactivado.

Ambos archivos pueden versionarse junto con el repositorio para documentar el
estado actual del modelo.

## Ablation de features

Para estudiar la sensibilidad del modelo, el script repite los escenarios
desactivando tres grupos de features: composición del regolito MGS-1
(`oxide_*` y `regolith_pct`), banderas de materiales NASA (fracciones por
familia) e índices logísticos derivados de packaging y reutilización. En cada
variación se reutilizan las mismas heurísticas base y se recalculan
MAE/RMSE/CI95. Los resultados completos están en
`data/benchmarks/ablation_metrics.csv` y `data/benchmarks/ablation_predictions.csv`.

### Impacto global

| Grupo desactivado | MAE global | Δ vs base | RMSE global | Δ vs base |
|-------------------|-----------:|----------:|------------:|----------:|
| Índices logísticos | 468.27 | −6.58 | 995.75 | −10.12 |
| Composición MGS-1 | 479.65 | +4.80 | 1 009.44 | +3.57 |
| Banderas NASA | 477.73 | +2.88 | 1 008.34 | +2.46 |

Los índices logísticos reducen ligeramente el error medio al apagarlos, lo que
sugiere que ese subconjunto podría estar sobreajustado a los heurísticos.
En cambio, quitar la composición MGS-1 o las banderas NASA aumenta el MAE
global, confirmando que ambas familias aportan señal útil en el baseline
actual.【F:data/benchmarks/ablation_metrics.csv†L71-L73】【F:data/benchmarks/scenario_metrics.csv†L20-L25】

### Contribución por target

* **Crew_min**: quitar las banderas NASA incrementa el MAE a 2 073 min
  (+13.6 vs base), lo que indica que las fracciones de materiales blandos siguen
  modulando la estimación de mano de obra.【F:data/benchmarks/ablation_metrics.csv†L66-L69】【F:data/benchmarks/scenario_metrics.csv†L20-L23】
* **Energy_kwh**: la composición MGS-1 es clave; al apagarla el MAE sube a
  273.35 (+47.2), evidenciando que las cargas de óxidos aportan contexto sobre
  la energía de proceso.【F:data/benchmarks/ablation_metrics.csv†L61-L63】【F:data/benchmarks/scenario_metrics.csv†L21-L24】
* **Estanqueidad y rigidez**: al quitar MGS-1 la estanqueidad sube apenas a
  0.200 (+0.0008), mientras que las banderas NASA elevan la rigidez a 0.136
  (+0.024); ambos cambios son pequeños pero consistentes con lo observado en la
  app.【F:data/benchmarks/ablation_metrics.csv†L63-L69】【F:data/benchmarks/scenario_metrics.csv†L22-L23】
* **Water_l**: los índices logísticos parecen introducir ruido; al apagarlos el
  MAE cae a 51.35 L (−36.8 frente al baseline).【F:data/benchmarks/ablation_metrics.csv†L60-L65】【F:data/benchmarks/scenario_metrics.csv†L21-L24】

Estas observaciones facilitan priorizar futuras iteraciones del modelo (p.ej.
refinar los features logísticos o enriquecer las fracciones NASA para mejorar
rigidez).
