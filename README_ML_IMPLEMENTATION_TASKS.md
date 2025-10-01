# Rex-AI — Backlog de implementación para el pipeline predictivo

Este backlog transforma el plan técnico en sprints accionables. Cada tarea se puede convertir en issue o ticket. Las dependencias indican el orden sugerido y qué entregables alimentan a otros equipos (app, ciencia de materiales, operaciones).

## Leyenda

| Campo | Descripción |
| --- | --- |
| Prioridad | 🔴 crítico, 🟠 alto, 🟡 medio, 🟢 deseable |
| Tipo | `data`, `ml`, `app`, `ops`, `infra` |
| Entregable | Artefacto, script o documentación que debe quedar en el repo o en almacenamiento externo |

---

## Sprint 0 — Preparativos (día 0)

| ID | Tarea | Prioridad | Tipo | Dependencias | Entregable |
| --- | --- | --- | --- | --- | --- |
| S0.1 | Inventariar datasets existentes (`datasets/`, `data/`, `feedback/`) y mapear columnas a features/targets | 🔴 | data | — | `docs/data_inventory.md` |
| S0.2 | Definir esquema Parquet unificado (`features.parquet`, `labels.parquet`) con `recipe_id`, `process_id`, `label_source`, `label_weight` | 🔴 | data | S0.1 | `datasets/schema.yaml` |
| S0.3 | Configurar entorno (dependencias scikit-learn, xgboost, torch opcional) y verificar ejecución de `python -m app.modules.model_training --help` | 🔴 | infra | — | Log de instalación + `requirements-lock.txt` opcional |
| S0.4 | Crear tablero de seguimiento (Notion/Jira) enlazando estas tareas e identificando owners | 🟠 | ops | — | URL del tablero (ver `docs/CHANGELOG.md`) |

## Sprint 1 — Pipeline de datos + modelo base (días 1–2)

| ID | Tarea | Prioridad | Tipo | Dependencias | Entregable |
| --- | --- | --- | --- | --- | --- |
| S1.1 | Implementar script ETL `scripts/build_gold_dataset.py` que combine fuentes y genere `data/gold/{features,labels}.parquet` | 🔴 | data | S0.2 | Script + Parquet de muestra |
| S1.2 | Extender `compute_feature_vector()` para incluir fracciones de óxidos, flags EVA/CTB/multilayer, índices Trash-to-Gas y parámetros de proceso | 🔴 | ml | S1.1 | PR con cambios + pruebas unitarias |
| S1.3 | Añadir targets `rigidez`, `estanqueidad`, `energy_kwh`, `water_l`, `crew_min` en el pipeline de entrenamiento (RandomForest multisalida) | 🔴 | ml | S1.1 | `data/models/rexai_regressor.joblib`, `metadata.json` |
| S1.4 | Calcular intervalos CI95 usando varianza entre árboles + residuales; persistir en `metadata.json` | 🔴 | ml | S1.3 | Campos `confidence_interval` por target |
| S1.5 | Registrar `feature_importances_` y logging por receta (`ml_prediction`, `uncertainty`, `label_source`) | 🟠 | ml | S1.3 | Actualización en `app/modules/model_registry.py` |
| S1.6 | Crear notebook `notebooks/validate_model.ipynb` con métricas MAE/RMSE y cobertura | 🟠 | ml | S1.3 | Notebook con gráficos guardados |

## Sprint 1 — Integración en la app (día 2)

| ID | Tarea | Prioridad | Tipo | Dependencias | Entregable |
| --- | --- | --- | --- | --- | --- |
| S1.7 | Actualizar `ModelRegistry` para cargar `metadata.json` y mostrar CI + label_source en la UI | 🔴 | app | S1.3, S1.4, S1.5 | PR en `app/` + capturas UI |
| S1.8 | Implementar selector de modo (`ml_mode` vs `heuristic_mode`) con fallback controlado | 🟠 | app | S1.7 | PR + prueba manual documentada |
| S1.9 | Mostrar explicabilidad (top features) junto a cada receta candidata | 🟡 | app | S1.7 | PR + captura |

## Sprint 2 — Generación y ranking (días 3–4)

| ID | Tarea | Prioridad | Tipo | Dependencias | Entregable |
| --- | --- | --- | --- | --- | --- |
| S2.1 | Implementar generador combinatorio de recetas (`scripts/generate_candidates.py`) con límites de search space | 🔴 | ml | S1.2 | Script + JSON de candidatos |
| S2.2 | Integrar scoring multiobjetivo configurable por usuario (weights + penalizaciones) | 🔴 | ml | S2.1 | Función `score_recipe()` con tests |
| S2.3 | Clasificadores auxiliares para flags de seguridad/proceso (`passes_seal`, `process_risk`) | 🟠 | ml | S1.1 | Modelos guardados + métricas |
| S2.4 | Pipeline de ranking que filtre top 10–20 recetas con CI y explicabilidad | 🟠 | ml | S2.2, S2.3 | `scripts/rank_candidates.py` |
| S2.5 | UI: mostrar tabla de candidatos con bandas de confianza y tags de riesgo | 🟠 | app | S2.4 | PR + captura |

## Sprint 3 — Feedback y aprendizaje activo (días 5–6)

| ID | Tarea | Prioridad | Tipo | Dependencias | Entregable |
| --- | --- | --- | --- | --- | --- |
| S3.1 | Diseñar formato `feedback/recipes.parquet` con campos `label_source`, `label_weight`, `conf_lo_*`, `conf_hi_*` | 🔴 | data | S0.2 | Esquema + ejemplo |
| S3.2 | Script `scripts/ingest_feedback.py` para incorporar nuevos labels al gold dataset | 🔴 | data | S3.1, S1.1 | Script + test |
| S3.3 | Automatizar re-entrenamiento (`python -m app.modules.retrain_from_feedback`) con cron/manual | 🟠 | ml | S3.2 | Job o documentación |
| S3.4 | Implementar estrategia de active learning (uncertainty sampling / expected improvement) para sugerir próximos ensayos | 🟡 | ml | S2.4 | Módulo `app/modules/active_learning.py` |
| S3.5 | Registrar experimentos físicos/simulador en `datasets/raw/experiments_*.parquet` con trazabilidad | 🟠 | ops | S3.1 | Guía operativa |

## Sprint 4 — Extensiones opcionales (días 7–10)

| ID | Tarea | Prioridad | Tipo | Dependencias | Entregable |
| --- | --- | --- | --- | --- | --- |
| S4.1 | Entrenar modelos XGBoost por target y comparar performance | 🟡 | ml | S1.3 | Resultados en notebook + modelos |
| S4.2 | Autoencoder tabular para detección de duplicados en espacio latente | 🟡 | ml | S1.2 | `autoencoder.pt`, documentación |
| S4.3 | Optimización en espacio latente (Bayesian Optimization / grid adaptativo) | 🟢 | ml | S4.2 | Script `optimize_latent.py` |
| S4.4 | CI/CD: workflow que empaquete modelos y publique release ZIP (`package_model_bundle`) | 🟡 | infra | S1.3 | Archivo YAML de pipeline |

## Métricas de éxito y checkpoints

1. **Cobertura de CI95 ≥ 85 %** en validación holdout para `energy_kwh` y `water_l`.
2. **MAE relativo ≤ 15 %** vs baseline heurístico en `crew_min`.
3. **Filtro automático ≥ 99 %** de recetas inviables en pruebas masivas.
4. **Tiempo de entrenamiento reproducible** (< 10 min en hardware objetivo) registrado en `metadata.json`.
5. **Dashboard para el jurado** con comparativa heurística vs IA antes/después de 10 labels nuevos.

## Próximos pasos inmediatos

1. Asignar owners a tareas Sprint 0 y Sprint 1.
2. Crear issues en GitHub/Linear usando este backlog como plantilla.
3. Programar daily de 15 min para seguimiento durante el hackathon.
