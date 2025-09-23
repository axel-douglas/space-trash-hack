# REX-AI — Plan para apagar heurísticas y operar "solo IA"

Este documento sintetiza el plan operativo para sustituir las heurísticas como fuente de etiquetas, versionar los artefactos de IA y dejar a Rex-AI funcionando en modo IA con trazabilidad y fallback controlado. Está pensado como guía de hackathon (48–72 h) pero suficientemente robusto para continuar después del evento.

## 0) Objetivo final

* Las etiquetas `rigidez`, `estanqueidad`, `energy_kwh`, `water_l` y `crew_min` ya no provienen de `heuristic_props`; se originan en mediciones, simuladores validados o etiquetado débil revisado por humanos.
* Los artefactos entrenados (`rexai_regressor.joblib`, opcionales como `xgboost.joblib`, `autoencoder.pt`) se versionan en `data/models/` y `ModelRegistry.ready == True`.
* La app muestra procedencia ML, bandas de confianza y deja el fallback heurístico únicamente como modo degradado explícito.

## 1) Datos de entrada disponibles y uso recomendado

* **Inventario NASA + flags (Non-Metabolic Waste)**: base para features de materiales y penalizaciones.
* **Simulantes marcianos (MGS-1, Jezero, etc.)**: incorporar composición mineralógica, densidad, granulometría y humedad como features de carga mineral.
* **Procesos P02/P03/P04**: usar duraciones, temperatura/presión, rendimientos estimados y parámetros operativos como features de proceso.
* **Procesos Trash-to-Gas / Logistics-to-Living**: emplear índices energéticos, rendimientos de gas y ahorro logístico cuando estén disponibles.

➡️ Acción inmediata: extender `compute_feature_vector()` para incluir todas las variables anteriores (entrenamiento e inferencia comparten esta función). No es necesario tocar la UI.

## 2) Estrategia de etiquetado (sustitución de heurísticas)

Priorizar fuentes en orden A → C y combinar todas las posibles:

* **A. Etiquetas medidas**: registrar ensayos reales (propiedades mecánicas, estanqueidad, minutos de crew, consumos energéticos/hídricos) en `datasets/raw/experiments_*.parquet`. Mapear `recipe_id`/`process_id` + composición a los targets; constituyen la clase "gold".
* **B. Etiquetas simuladas validadas**: construir gemelos digitales ligeros (FEM/DEM para rigidez/porosidad, balances térmicos para energía/agua, diagramas de operaciones para tiempos). Documentar ecuaciones y supuestos, y almacenar `confidence` del simulador por target.
* **C. Etiquetado débil con revisión humana**: cuando falten A/B, usar reglas como etiquetas iniciales, revisarlas en un panel técnico (HIL) y persistir `label_source = weak_human_ok`.

Formato de datasets:

* `datasets/bronze/*.parquet`: ingesta cruda.
* `datasets/silver/*.parquet`: datos curados y sin fugas entre splits.
* `datasets/gold/*.parquet`: splits de entrenamiento/validación con `label_source` y `label_weight`.

## 3) Entrenamiento reproducible y medible

* **Modelo base**: `RandomForestRegressor` multisalida (scikit-learn). Calcular incertidumbre mediante varianza entre árboles + residuales.
* **Alternativas**: modelos XGBoost por target y autoencoder tabular para embeddings (opcional si PyTorch está disponible; la app ya trata a Torch como dependencia opcional).
* **Ponderación**: `sample_weight` según `label_source` (p.ej., measured = 1.0, sim = 0.7, weak = 0.4).
* **Métricas**: MAE/RMSE y cobertura de intervalos al 95 % por target y por `label_source`.
* **Compresión + optimización** (opcional): guardar PCA/UMAP o embeddings del autoencoder y habilitar un optimizador de proporciones en ese espacio comprimido (patrón "feature compression + optimizer" para encontrar fórmulas objetivo).

Comando sugerido:

```bash
python -m app.modules.model_training --use_labels gold --weighting
```

Salida esperada: `data/models/rexai_regressor.joblib`, `metadata.json`, y opcionales (`xgboost.joblib`, `autoencoder.pt`).

## 4) Registro y empaquetado de artefactos

* Versionar únicamente `metadata.json` en Git; publicar los binarios (`.joblib/.pt`) mediante releases o almacenamiento externo.
* Asegurarse de que `metadata.json` incluya `feature_names`, estadísticas (`means`, `std`), `residual_std` por target e `importances`.
* Automatizar un flujo (CI/CD) que entrene con `datasets/gold`, publique artefactos y ejecute una prueba de inferencia (`predict() != {}`).

## 5) Integración en la app

* En el generador, mantener el cálculo heurístico pero sustituirlo por `ModelRegistry.predict(features)` cuando `ModelRegistry.ready` y la predicción sea válida.
* Exponer en la UI la procedencia (`label_source`), fecha de entrenamiento y cobertura del CI.
* Controlar el modo con `st.session_state["ml_mode"]` para que el fallback heurístico solo se active manualmente o ante ausencia de modelos.

## 6) Feedback y Human-in-the-Loop

* Mantener `impact.jsonl` y `feedback.jsonl` como receptores de feedback ampliado.
* Cada aprobación/rechazo o ajuste de receta debe volver a `datasets/raw/` con targets actualizados o pseudo-targets.
* Definir un formato estructurado de receta/proceso (lista de pasos atómicos) para entrenar, a futuro, un generador de procesos que se refine con feedback humano.

## 7) Validación verificable por el jurado

Crear `notebooks/validate_model.ipynb` con:

* Distribución por `label_source` y curvas de calibración de intervalos.
* Tabla de métricas por target y familia de material/proceso.
* Comparativa heurística vs IA en un holdout.
* Tres recetas recomendadas con experimentos o simulaciones comparativas.

## 8) Roadmap técnico (dos sprints)

**Sprint 1 (ahora)**

1. Unificar datasets (`bronze/silver/gold`) y entrenar RF multisalida con ponderación.
2. Publicar `rexai_regressor.joblib` en un release y actualizar el pipeline de inferencia.
3. Encender el "modo IA" en la app (UI ya lista para mostrar CI/importancias).

**Sprint 2 (próximo)**

1. Añadir modelos XGBoost por target y comparar en la UI.
2. Incorporar autoencoder para embeddings (si Torch disponible).
3. Sustituir reglas de energía/agua/tiempo por simuladores dedicados.
4. Activar el optimizador sobre el espacio comprimido para sugerir formulaciones con límites de recursos.

## Apéndice A — Inspiración metodológica

* **Espacio de features común + compresión + optimización**: construir un "screening" en D dimensiones con descriptores físico-químicos y de proceso, comprimirlo (PCA/autoencoder) y usar un optimizador para hallar proporciones que cumplan objetivos. Equivale a "mezcla de residuos + MGS-1" → "propiedades target" con un generador de fórmulas.
* **Generador de proceso + feedback humano**: entrenar un generador de pasos de proceso con recetas existentes y refinarlo con el panel de feedback (`Feedback & Impact`). Esto cierra el lazo fórmula → proceso → validación para aprendizaje continuo.

## Apéndice B — Checklist inmediato

1. Crear `datasets/{bronze,silver,gold}/` y scripts ETL → features + targets con `label_source`.
2. Ejecutar el pipeline con ponderación; guardar `rexai_regressor.joblib` + `metadata.json`.
3. Publicar los artefactos y actualizar `MODEL_DIR` o la URL remota.
4. Configurar la app para que `ModelRegistry.ready == True` y mostrar procedencia + CI; fallback solo si `predict()` devuelve `{}`.
5. Elaborar notebook de validación con métricas y recetas demo.
6. Consolidar `impact.jsonl`/`feedback.jsonl` en `datasets/raw/` y re-entrenar periódicamente (`python -m app.modules.retrain_from_feedback`).

---

**Pregunta clave**: ¿este plan apaga las heurísticas?

**Sí.** Al reemplazar las etiquetas generadas por reglas con datos medidos/simulados/revisados, entrenar y empaquetar modelos reales, y dejar la app en modo IA por defecto (con fallback controlado), Rex-AI deja de depender de `heuristic_props` tanto en entrenamiento como en inferencia.
