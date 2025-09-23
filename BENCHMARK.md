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
   `--format both` para obtener también los Parquet y `--output-dir` para
   versionar corridas separadas (por ejemplo `data/benchmarks/pre_feedback/`).

4. Repetí el comando después de incorporar feedback humano (reentrenamiento) y
   conservá ambos subdirectorios para analizar la evolución de métricas.

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
escenario (promediado sobre las cinco métricas) incorpora también el ancho
medio de los intervalos de confianza al 95% (`ci95_width_mean`):

| Escenario | MAE medio | RMSE | CI95 (ancho medio) |
|-----------|----------:|-----:|-------------------:|
| CTB Reconfig | 197 882 | 435 212 | 322 693 |
| Espuma + MGS-1 + Sinter | 203 085 | 447 205 | 308 786 |
| Multicapa + Laminar | 200 600 | 441 213 | 315 995 |
| Global (los 3 escenarios) | 200 522 | 441 237 | 315 825 |

Los valores anteriores provienen de `data/benchmarks/scenario_metrics.csv` y se
actualizan automáticamente al rerunear el script.【F:data/benchmarks/scenario_metrics.csv†L17-L25】

### Observaciones

* **Gap frente a las heurísticas**: el ensemble entrenado mantiene una distancia
  elevada respecto a las reglas determinísticas (MAE global 200 522 y RMSE
  441 237), por lo que no se observa una mejora frente al baseline en esta
  corrida.【F:data/benchmarks/scenario_metrics.csv†L17-L25】
* **Consumo de crew**: continúa siendo la métrica más conflictiva; el modelo
  entrega estimaciones del orden de 0.98–1.00 millones de minutos frente a
  heurísticas de 33–51 min, generando errores absolutos cercanos al millón en
  los tres escenarios.【F:data/benchmarks/scenario_predictions.csv†L6-L16】
* **Energía y agua**: los desvíos rondan 16 000 kWh y 166–182 L según el
  escenario, muy por encima de las reglas base.【F:data/benchmarks/scenario_predictions.csv†L4-L15】
* **Rigidez/estanqueidad**: las discrepancias siguen acotadas (≤0.29), lo que
  facilita auditar calibraciones mecánicas pese al desfasaje global.【F:data/benchmarks/scenario_predictions.csv†L2-L14】
* **CI95**: las bandas medias se ensanchan hasta ~316 000 unidades a nivel
  global, reflejando la gran dispersión asociada a las predicciones actuales de
  energía, agua y crew.【F:data/benchmarks/scenario_metrics.csv†L17-L25】

## Archivos generados

* `data/benchmarks/scenario_predictions.csv`: detalle por target con
  predicciones ML, heurística, error absoluto y bandas CI95 (valores mínimos,
  máximos y ancho).
* `data/benchmarks/scenario_metrics.csv`: métricas agregadas por escenario,
  por target y resumen global incluyendo los promedios de CI95 (`ci95_*`).
* `data/benchmarks/ablation_predictions.csv`: detalle de cada corrida de
  ablation por escenario y target.
* `data/benchmarks/ablation_metrics.csv`: métricas agregadas por grupo de
  features desactivado, con los anchos de CI95 posteriores a la intervención.

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

| Grupo desactivado | MAE global | Δ vs base | RMSE global | Δ vs base | CI95 (ancho) | Δ CI95 |
|-------------------|-----------:|----------:|------------:|----------:|-------------:|-------:|
| Índices logísticos | 199 918 | −605 | 439 883 | −1 354 | 317 484 | +1 659 |
| Composición MGS-1 | 200 522 | −1 | 441 237 | 0 | 315 824 | −1 |
| Banderas NASA | 199 313 | −1 209 | 438 603 | −2 634 | 318 659 | +2 834 |

Todos los grupos producen desvíos similares, pero ningún apagado consigue cerrar
la brecha respecto a las heurísticas: incluso las reducciones de MAE vienen
acompañadas por un ensanchamiento de las bandas de confianza o mejoras
marginales.【F:data/benchmarks/ablation_metrics.csv†L71-L73】【F:data/benchmarks/scenario_metrics.csv†L25-L25】

### Contribución por target

* **Índices logísticos**: al removerlos el MAE cae levemente en crew_min (983 461
  vs 986 445) y energía (16 009 vs 16 049), pero el CI95 se amplía en +8 210 min
  para crew y +84 kWh para energía.【F:data/benchmarks/ablation_metrics.csv†L56-L60】【F:data/benchmarks/scenario_metrics.csv†L20-L21】
* **Composición MGS-1**: apenas altera los promedios globales, aunque reduce el
  MAE de water_l a 113.79 L y estrecha la banda a 341.61 L (−2.22 vs la base).
  Rigidez y estanqueidad prácticamente no cambian.【F:data/benchmarks/ablation_metrics.csv†L63-L65】【F:data/benchmarks/scenario_metrics.csv†L22-L24】
* **Banderas NASA**: disminuyen el MAE de crew_min hasta 980 478 y de energía a
  15 969, a costa de CI95 más anchos (hasta 1 564 899 min y 28 049 kWh).【F:data/benchmarks/ablation_metrics.csv†L66-L70】【F:data/benchmarks/scenario_metrics.csv†L20-L21】

Estas observaciones ayudan a priorizar próximos ajustes: optimizar los índices
logísticos para no degradar la incertidumbre, revisar el escalado de energía/crew
cuando se combinan banderas NASA y evaluar si la composición MGS-1 sigue siendo
redundante bajo el modelo actual.

## Comparar mejoras post-feedback

Para medir el impacto real del feedback humano:

1. Guarda una corrida base antes de reentrenar (por ejemplo
   `python scripts/run_benchmarks.py --output-dir data/benchmarks/pre_feedback --format csv --with-ablation`).
2. Reentrena con los logs de feedback y vuelve a ejecutar el script apuntando a
   otro directorio (por ejemplo `data/benchmarks/post_feedback`).
3. Generá un gráfico interactivo de deltas con:

   ```bash
   python scripts/plot_benchmark_deltas.py \
     --before data/benchmarks/pre_feedback/scenario_metrics.csv \
     --after data/benchmarks/post_feedback/scenario_metrics.csv \
     --output data/benchmarks/feedback_deltas.html
   ```

El HTML resultante muestra, para cada escenario, cuánto se redujeron (positivos)
o aumentaron (negativos) el MAE, el RMSE y el ancho del CI95 respecto al modelo
anterior.【F:scripts/plot_benchmark_deltas.py†L1-L107】
