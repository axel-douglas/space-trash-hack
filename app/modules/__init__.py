# app/modules/__init__.py
"""
Exportes livianos para la app Streamlit.

Notas:
- NO importamos `model_training` al cargar el paquete (lazy import).
- Re-exportamos solo runtime estable (IO, generator, ML registry).
- Si se quiere entrenar desde la UI/CLI: usar `get_train_and_save()`.
"""

from __future__ import annotations

# --- Runtime estable (ligero) ---
from .ml_models import MODEL_REGISTRY, ModelRegistry, PredictionResult
from .io import (
    load_waste_df,
    save_waste_df,
    load_targets,
    load_process_catalog,
)
from .generator import generate_candidates, PredProps
from .ranking import rank_candidates, score_recipe, derive_auxiliary_signals

__all__ = [
    # IO
    "load_waste_df",
    "save_waste_df",
    "load_targets",
    "load_process_catalog",
    # Generación
    "generate_candidates",
    "PredProps",
    "rank_candidates",
    "score_recipe",
    "derive_auxiliary_signals",
    # ML
    "MODEL_REGISTRY",
    "ModelRegistry",
    "PredictionResult",
    # Entrenamiento (lazy)
    "get_train_and_save",
]

def get_train_and_save():
    """
    Import diferido del pipeline de entrenamiento.
    Evita errores/sobrecargas cuando solo se usa la app.
    """
    try:
        from .model_training import train_and_save  # import tardío
        return train_and_save
    except Exception as exc:
        def _stub(*_a, **_kw):
            raise RuntimeError(f"Training pipeline no disponible: {exc}")
        return _stub
