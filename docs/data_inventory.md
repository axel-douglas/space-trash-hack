# Inventario de datos Rex-AI

Mapa completo de datasets, artefactos intermedios y etiquetas utilizadas por el
pipeline de Rex-AI. Sirve como referencia para `scripts/build_gold_dataset.py`
y para coordinar nuevas ingestas.

## Directorios clave

| Ruta | Contenido | Uso principal |
| --- | --- | --- |
| `datasets/raw/` | CSV originales de NASA, UCF y catálogos internos. | Fuente de features físicos y logísticos. |
| `datasets/processed/` | Resultados intermedios (`rexai_training_dataset.parquet`, `ml/synthetic_runs.parquet`). | Control de consistencia y análisis sintético. |
| `data/gold/` | `features.parquet`, `labels.parquet` armonizados. | Dataset “gold” para entrenamiento y reentrenamiento. |
| `data/processed/` | Copias procesadas consumidas por la app (`rexai_training_dataset.parquet`). | Punto de validación para nuevas corridas. |
| `data/models/` | Artefactos entrenados (`rexai_regressor.joblib`, `metadata.json`, ensambles). | Base para `ModelRegistry`. |
| `data/` | Catálogo de procesos, presets, logs y configuraciones de UI. | Parametrización operativa y trazabilidad. |

## Fuentes crudas (`datasets/raw/`)

| Archivo | Columnas destacadas | Uso |
| --- | --- | --- |
| `nasa_waste_inventory.csv` | `id`, `category`, `mass_kg`, `volume_m3`, `moisture_pct`, `difficulty_factor`, `flags` | Base de residuos NASA; se expanden features de masa, volumen, humedad, dificultad y familias (`foam`, `eva`, `multilayer`). |
| `mgs1_properties.csv` | `simulant`, `property`, `value`, `units` | Propiedades físicas del regolito MGS-1 (densidad, granulometría). |
| `mgs1_composition.csv` | `mineral`, `wt_percent` | Fracciones minerales; se agregan a índices de compatibilidad. |
| `mgs1_oxides.csv` | `oxide`, `wt_percent` | Alimenta features `oxide_*` para análisis químico. |
| `nasa_trash_to_gas.csv` | `mission`, `kg_per_cm_day`, `gas_mix_*`, `water_makeup_kg` | Construye `gas_recovery_index` y puntajes Trash-to-Gas. |
| `logistics_to_living.csv` | `scenario`, `crew_days`, `goods_kg`, `outfitting_replaced_kg` | Deriva índices de reutilización logística. |

## Configuración operativa (`data/`)

| Archivo | Campos | Rol |
| --- | --- | --- |
| `process_catalog.csv` | Energía/agua/crew por proceso, notas y ubicación. | Parametriza targets esperados y features de contexto. |
| `targets_presets.json` | `name`, `rigidity`, `tightness`, `max_water_l`, `max_energy_kwh`, `max_crew_min`, `scenario` | Presets para el Target Designer y normalización de scores. |
| `waste_inventory_sample.csv` | Subconjunto de inventario NASA. | Dataset de demo y validación rápida. |
| `data/logs/feedback_*.parquet` | Correcciones de usuarios (rigidez, estanqueidad, penalizaciones). | Fuente de labels débiles para reentrenamiento. |
| `data/logs/impact_*.parquet` | Métricas reales de corridas. | Registro operativo y auditoría. |
| `benchmarks/*.csv` | MAE/RMSE históricos, predicciones y ablations. | Seguimiento de drift y validación. |

## Mapeo de features y targets

| Tipo | Destino | Fuente | Transformación |
| --- | --- | --- | --- |
| Feature numérico | `mass_input_kg`, `total_mass_kg` | `nasa_waste_inventory` + receta | Suma ponderada por masa. |
| Feature numérico | `moisture_frac` | `moisture_pct` | Normalización 0–1 ajustada por peso. |
| Feature numérico | `difficulty_index` | `difficulty_factor` | Promedio ponderado por masa. |
| Feature numérico | `foam_frac`, `eva_frac`, `multilayer_frac`, etc. | Flags de inventario | Conteo ponderado / one-hot de familias. |
| Feature numérico | `gas_recovery_index`, `trash_to_gas_score` | `nasa_trash_to_gas.csv` | Mapear misión → índice. |
| Feature numérico | `logistics_reuse_index` | `logistics_to_living.csv` | `outfitting_replaced_kg / packaging_kg`. |
| Feature numérico | `oxide_*` | `mgs1_oxides.csv` | Uso directo de fracciones. |
| Feature numérico | `regolith_pct` | UI/generador | Porcentaje de MGS-1 agregado a la receta. |
| Feature categórico | `process_id` | `process_catalog.csv` | One-hot / embeddings tabulares. |
| Targets continuos | `rigidez`, `estanqueidad`, `energy_kwh`, `water_l`, `crew_min` | Heurísticas calibradas + feedback operativo | Escala 0–1 para rigidez/estanqueidad; valores físicos para recursos. |
| Flags auxiliares | `tightness_pass`, `process_risk` | Clasificadores y feedback | Se derivan en el pipeline para explicar decisiones. |

## Próximas acciones

1. Normalizar IDs de residuo (`id`, `material`) para generar `recipe_id` único y
aumentar la trazabilidad.
2. Publicar `datasets/schema.yaml` con tipos esperados para `features.parquet`
y `labels.parquet`.
3. Inventariar nuevas fuentes (simulador térmico, ensayos físicos) antes de
alimentar `feedback/`.
4. Documentar mapeos `flag → feature` en un diccionario central para evitar
interpretaciones divergentes.
