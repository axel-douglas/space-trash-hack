# Feedback de experimentos

Coloca en esta carpeta los archivos Parquet con feedback humano o de simuladores
validados. La aplicación espera un archivo principal llamado
`recipes.parquet`, pero se recomienda mantenerlo fuera del control de
versiones (ver `.gitignore`).

## Estructura requerida

El esquema completo está documentado en
[`datasets/feedback_schema.yaml`](../datasets/feedback_schema.yaml). Cada fila
contiene la receta (`recipe_id`, `process_id`), los targets medidos,
clasificadores auxiliares (`tightness_pass`, `rigidity_level`), bandas de
confianza (`conf_lo_*`, `conf_hi_*`) y metadatos de trazabilidad
(`label_source`, `label_weight`, `measurement_ts`, `operator_id`, `notes`).

### Ejemplo mínimo

El siguiente ejemplo muestra dos filas con valores ilustrativos. Se puede copiar
en un cuaderno o script de Pandas para generar un Parquet localmente:

```python
import pandas as pd
from pathlib import Path

rows = [
    {
        "recipe_id": "RX-001",
        "process_id": "P02",
        "label_source": "mission",
        "label_weight": 5.0,
        "rigidez": 0.78,
        "estanqueidad": 0.71,
        "energy_kwh": 3.1,
        "water_l": 1.6,
        "crew_min": 18.5,
        "tightness_pass": True,
        "rigidity_level": 3,
        "conf_lo_rigidez": 0.72,
        "conf_hi_rigidez": 0.84,
        "conf_lo_energy_kwh": 2.9,
        "conf_hi_energy_kwh": 3.5,
        "measurement_ts": "2025-09-24T12:00:00Z",
        "operator_id": "astro-a",
        "notes": "Validación térmica",
    },
    {
        "recipe_id": "RX-002",
        "process_id": "P03",
        "label_source": "simulated",
        "label_weight": 0.8,
        "rigidez": 0.65,
        "estanqueidad": 0.62,
        "energy_kwh": 2.8,
        "water_l": 1.4,
        "crew_min": 17.2,
        "tightness_pass": False,
        "rigidity_level": 2,
        "conf_lo_rigidez": 0.6,
        "conf_hi_rigidez": 0.7,
        "conf_lo_energy_kwh": 2.5,
        "conf_hi_energy_kwh": 3.0,
        "measurement_ts": "2025-09-25T09:30:00Z",
        "operator_id": "astro-b",
        "notes": "Simulador estructural",
    },
]

# Incorporar columnas de features requeridas por el pipeline
from app.modules import model_training
for row in rows:
    for column in model_training.FEATURE_COLUMNS:
        row.setdefault(column, 0.0)

frame = pd.DataFrame(rows)
Path("feedback/recipes.parquet").parent.mkdir(exist_ok=True)
frame.to_parquet("feedback/recipes.parquet", index=False)
```

> **Nota**: El script anterior genera valores por defecto (`0.0`) en las
> columnas de features para que el pipeline pueda validar la estructura antes
de reemplazarlos con mediciones reales.

## Flujo operativo

1. Exporta los resultados en `feedback/recipes.parquet` (no subirlo al repo).
2. Ejecuta `scripts/ingest_feedback.py --feedback feedback/recipes.parquet` para
   fusionar las mediciones con el dataset gold.
3. Lanza `python -m app.modules.retrain_from_feedback --logs feedback/*.parquet`
   para reentrenar el modelo usando las mediciones recientes.
