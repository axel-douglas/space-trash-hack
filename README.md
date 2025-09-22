# Space Trash Hack — Streamlit Demo (2025)

Demo ligera que muestra la lógica del "cerebro de reciclaje" para Marte:
1) Inventario de residuos (NASA non-metabolic waste, simplificado)
2) Diseño de objetivo (TargetSpec)
3) Generación de recetas (combinaciones + proceso)
4) Resultados y trade-offs (Pareto, Sankey, métricas)

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt`

## Entrenar y generar artefactos de IA

El módulo `app/modules/model_training.py` permite entrenar un modelo de
regresión multi-salida (Random Forest) a partir de corridas sintéticas. Esto
produce:

- Dataset versionado en `data/processed/ml/synthetic_runs.parquet`.
- Pipeline empaquetado en `data/models/rexai_regressor.joblib` con metadatos
  (`data/models/metadata.json`).

Para regenerar el modelo:

```bash
python -m app.modules.model_training
```

Los artefactos resultantes (`.joblib`, `.parquet`, etc.) se generan de forma
local y están listados en `.gitignore` para evitar commitear binarios. La
aplicación cargará automáticamente el modelo y reemplazará las predicciones
heurísticas por las del pipeline cuando los artefactos estén disponibles.

## Ejecutar
```bash
streamlit run app/Home.py
