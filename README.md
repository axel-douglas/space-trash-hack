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
regresión multi-salida a partir de datasets físicos/químicos alineados con
documentos de NASA y UCF. Mezcla el inventario de residuos no-metabólicos
(pouches, espumas, EVA/CTB, textiles, nitrilo, etc.), la composición mineralógica
y propiedades de MGS-1, y rendimientos de procesos Trash-to-Gas y
Logistics-to-Living. 

Esto produce:
- Dataset procesado en `datasets/processed/rexai_training_dataset.parquet`.
- Pipeline empaquetado (`data/models/rexai_regressor.joblib`) con metadatos
  en `data/models/metadata.json`.
- Autoencoder para embeddings latentes en `data/models/rexai_autoencoder.pt`.

Para regenerar todos los artefactos:

```bash
python -m app.modules.model_training
