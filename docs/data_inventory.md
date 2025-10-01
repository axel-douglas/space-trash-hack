# Inventario de datos Rex-AI (Sprint 0)

Este documento mapea los datos existentes en el repositorio a los *features*, *targets* y señales auxiliares del pipeline predictivo. Sirve como punto de partida para el script `build_gold_dataset.py` y para coordinar nuevas ingestas.

## Directorios clave

| Ruta | Contenido | Uso en pipeline |
| --- | --- | --- |
| `datasets/raw/` | Archivos CSV provenientes de NASA, UCF y catálogos internos. | Fuente primaria para enriquecer features físicos (composición, oxidos) y perfiles de residuo. |
| `datasets/processed/` | Espacio reservado para datasets intermedios (`rexai_training_dataset.parquet`). | Entrada legacy del modelo actual; servirá como referencia para validaciones de consistencia. |
| `data/gold/` | Directorio destino para `features.parquet` y `labels.parquet`. | Dataset "gold" armonizado que alimentará el entrenamiento/re-entrenamiento. |
| `datasets/processed/` | Espacio reservado para datasets intermedios (`rexai_training_dataset.parquet`, `ml/synthetic_runs.parquet`). | Entrada legacy del modelo actual; servirá como referencia para validaciones de consistencia y análisis de corridas sintéticas. |
| `data/processed/` | Artefactos intermedios generados por el pipeline (`rexai_training_dataset.parquet`, `ml/synthetic_runs.parquet`). | Fuente principal para validar que el reentrenamiento produzca datasets consistentes. |
| `datasets/gold/` | Directorio destino para `features.parquet` y `labels.parquet`. | Dataset "gold" armonizado que alimentará el entrenamiento/re-entrenamiento. |
| `data/` | Configuración operativa (catálogo de procesos, logs, presets de objetivos) y modelos empaquetados. | Cruce de parámetros de proceso, restricciones operacionales y mediciones previas. |
| `data/models/` | Modelos actuales (`rexai_regressor.joblib`, `metadata.json`, clasificadores). | Sirven para validar compatibilidad de features/targets y para bootstrap mientras se migra al nuevo pipeline. |

## Fuentes crudas (`datasets/raw/`)

| Archivo | Columnas principales | Tipo | Cómo se usa |
| --- | --- | --- | --- |
| `mgs1_properties.csv` | `simulant`, `property`, `value`, `units`, `notes` | Long table | Referencia de propiedades físicas del regolito (densidad, distribución de grano). Se pivoteará para features como `regolith_density` o controles de mezclado. |
| `mgs1_composition.csv` | `simulant`, `mineral`, `wt_percent` | Long table | Define fracciones minerales base. Se agregará por mineral para features agregados (`plagioclase_pct`, etc.) o se integrará a un índice de compatibilidad. |
| `mgs1_oxides.csv` | `simulant`, `oxide`, `wt_percent` | Long table | Fuente directa para features `oxide_*` (p.ej. `oxide_sio2`, `oxide_feot`, `oxide_mgo`, `oxide_cao`, `oxide_so3`, `oxide_h2o`). |
| `nasa_waste_inventory.csv` | `id`, `category`, `item`, `key_materials`, `mass_kg`, `volume_m3`, `moisture_pct`, `pct_mass`, `pct_volume`, `commercial_equivalent`, `difficulty_factor`, `flags`, `notes` | Catálogo tabular | Base para features de mezcla de residuos: `mass_input_kg`, `difficulty_index`, fracciones por familia (`foam_frac`, `eva_frac`, `textile_frac`, etc.) y *flags* (`closed_cell`, `ctb`, `eva`, `multilayer`). |
| `nasa_trash_to_gas.csv` | `mission`, `category`, `kg_per_cm_day`, `gas_mix_*`, `o2_ch4_yield_kg`, `water_makeup_kg`, `isp_*`, `delta_v_ms` | Indicadores por misión | Se utilizará para construir índices `gas_recovery_index` y `trash_to_gas_score` vinculados a cada mezcla candidata. |
| `logistics_to_living.csv` | `scenario`, `crew_days`, `crew_count`, `goods_kg`, `packaging_kg`, `ctb_count`, `reuse_plan`, `outfitting_replaced_kg`, `residual_waste_kg` | Escenarios logísticos | Alimenta el índice `logistics_reuse_index` y metas de reducción de residuo. |

## Configuración operativa (`data/`)

| Archivo | Columnas / Campos | Cómo contribuye |
| --- | --- | --- |
| `process_catalog.csv` | `process_id`, `name`, `location`, `energy_kwh_per_kg`, `water_l_per_kg`, `crew_min_per_batch`, `notes` | Define parámetros base por proceso (P01–P04). Se usa para poblar targets esperados (`energy_kwh`, `water_l`, `crew_min`) y como *features* de contexto (`process_id`, `location`). |
| `waste_inventory_sample.csv` | Similar a `nasa_waste_inventory` pero resumido | Dataset de prueba para UI/benchmark. Útil para validar mapeos de flags a features. |
| `impact_log.csv` | `ts_iso`, `scenario`, `target_name`, `materials`, `weights`, `process_id`, ... | Historial de ejecuciones de la app. Campos `materials`/`weights` permiten reconstruir recetas generadas manualmente (servirán como `weak labels`). |
| `feedback_log.csv` | `ts_iso`, `astronaut`, `scenario`, `target_name`, `option_idx`, `rigidity_ok`, `ease_ok`, `issues`, `notes`, `extra` | Feedback cualitativo de usuarios. Se convertirá en etiquetas *weak* (`label_source=weak`) con mapeos booleanos a `rigidez`/`estanqueidad`. |
| `targets_presets.json` | Lista de presets: `name`, `rigidity`, `tightness`, `max_water_l`, `max_energy_kwh`, `max_crew_min` | Sirve para inicializar pesos multiobjetivo y normalizar *score* de usuario. |
| `benchmarks/*.csv` | Métricas y predicciones históricas | Para comparar baseline heurístico vs IA; no entran directo al dataset gold pero ayudan en validación. |
| `models/metadata.json` | Descripción del bundle actual (features, targets, métricas) | Referencia para garantizar compatibilidad de columnas en el nuevo esquema. |

## Mapeo preliminar de features y targets

| Tipo | Columna objetivo | Fuente | Transformación |
| --- | --- | --- | --- |
| Feature numérico | `regolith_pct` | Control de receta (UI / generador) | Porcentaje de MGS-1 mezclado (derivado de inputs de usuario). |
| Feature categórico | `process_id` | `process_catalog.csv` / receta | Codificar como categoría (one-hot). |
| Feature numérico | `mass_input_kg`, `total_mass_kg` | `nasa_waste_inventory.csv` + cantidades de receta | Suma ponderada de masas de cada residuo. |
| Feature numérico | `density_kg_m3` | `mgs1_properties.csv` + mezcla | Usar `density_bulk` ajustado por humedad. |
| Feature numérico | `moisture_frac` | `nasa_waste_inventory.csv` (`moisture_pct`) | Normalizar a 0–1 según composición de la receta. |
| Feature numérico | `difficulty_index` | `nasa_waste_inventory.csv` (`difficulty_factor`) | Promedio ponderado por masa. |
| Feature numérico | `problematic_mass_frac`, `problematic_item_frac` | Flags de `nasa_waste_inventory.csv` | Clasificar flags críticos (`hazard`, `sharp`, etc.) y agregarlos como fracciones. |
| Feature numérico | `foam_frac`, `eva_frac`, `textile_frac`, `multilayer_frac`, `glove_frac`, `polyethylene_frac`, `carbon_fiber_frac`, `hydrogen_rich_frac`, `packaging_frac` | Flags + `key_materials` | Etiquetado binario/ponderado según presencia de cada familia en la receta. |
| Feature numérico | `gas_recovery_index`, `trash_to_gas_score` | `nasa_trash_to_gas.csv` | Mapear misión/escenario al índice correspondiente. |
| Feature numérico | `logistics_reuse_index` | `logistics_to_living.csv` | Normalizar `outfitting_replaced_kg` / `packaging_kg`. |
| Feature numérico | `oxide_sio2`, `oxide_feot`, `oxide_mgo`, `oxide_cao`, `oxide_so3`, `oxide_h2o` | `mgs1_oxides.csv` | Usar directamente las fracciones de óxidos del regolito. |
| Target continuo | `rigidez`, `estanqueidad` | Etiquetas reales/simuladas (`feedback` o ensayos) | Escala 0–1. Inicialmente se poblará con heurísticas calibradas. |
| Target continuo | `energy_kwh`, `water_l`, `crew_min` | `process_catalog.csv` + mediciones reales | Ajustar según masa del batch y telemetría del proceso. |
| Flag | `tightness_pass`, `process_risk` | Clasificadores auxiliares / feedback | Derivados de `estanqueidad` y análisis de proceso. |
| Meta | `score_objective` | Configuración de usuario | Función multiobjetivo (no se persiste como label, pero se registra para ranking). |

## Próximas acciones sobre los datos

1. Normalizar IDs de residuo (`id` vs `material`) para que cada receta tenga `recipe_id` único y trazable.
2. Definir diccionario de flags → features (ej.: `ctb` → `eva_frac`, `multilayer`; `closed_cell` → `foam_frac`).
3. Documentar en `datasets/schema.yaml` el orden y tipo de columnas esperadas para `features.parquet` y `labels.parquet`.
4. Inventariar cualquier fuente externa adicional (simulador térmico, experimentos físicos) antes de poblar `feedback/`.
