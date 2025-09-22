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

El pipeline de entrenamiento consume datasets físicos/químicos alineados con
los documentos de NASA y UCF:

- `datasets/raw/nasa_waste_inventory.csv`: taxonomía de residuos no-metabólicos
  (pouches, espumas, EVA/CTB, textiles, nitrilo, etc.).
- `datasets/raw/mgs1_composition.csv` / `mgs1_oxides.csv` / `mgs1_properties.csv`:
  composición mineralógica y propiedades de MGS-1 (regolito de referencia).
- `datasets/raw/nasa_trash_to_gas.csv`: rendimiento de procesos Trash-to-Gas y
  Trash-to-Supply-Gas para Gateway/Mars.
- `datasets/raw/logistics_to_living.csv`: eficiencia de reuso de CTB (Logistics-2-Living).

`app/modules/model_training.py` construye combinaciones realistas mezclando esos
datasets con el catálogo de procesos y genera:

- Dataset procesado en `datasets/processed/rexai_training_dataset.parquet`.
- Pipeline RandomForest multi-salida (`data/models/rexai_regressor.joblib`) con
  estimación de incertidumbre (desvío entre árboles + residuales de validación).
- Ensemble de modelos de "wow effect":
  - `data/models/rexai_xgboost.joblib` (boosting por target).
  - `data/models/rexai_tabtransformer.pt` (transformer tabular ligero).
- Autoencoder para embeddings latentes (`data/models/rexai_autoencoder.pt`).
- Metadatos en `data/models/metadata.json` (features, targets, fecha, métricas,
  importancias de features, residuales, paths de artefactos).

Para regenerar todos los artefactos:

```bash
python -m app.modules.model_training
```

> Nota: la optimización bayesiana con Ax/BoTorch es opcional. El entorno Streamlit
> detecta automáticamente si `ax-platform` y `botorch` están instalados; en caso
> contrario utiliza el optimizador heurístico integrado. Para habilitarla basta con
> `pip install ax-platform botorch` antes de ejecutar la app.

Los binarios (`.joblib`, `.pt`, `.parquet`) permanecen ignorados por Git para
mantener el repo liviano. Cuando existen localmente, la app reemplaza las
predicciones heurísticas por las del modelo Rex-AI (RandomForest + XGBoost +
TabTransformer), expone bandas de confianza 95%, importancias promedio y el
vector latente entrenado sobre mezclas MGS-1 + residuos NASA.

## Ejecutar
```bash
streamlit run app/Home.py
