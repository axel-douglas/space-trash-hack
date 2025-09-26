# app/modules/process_planner.py
from __future__ import annotations
import pandas as pd

# Reglas simples de idoneidad por residuo (pueden expandirse)
SUITABILITY = {
    "pouches": ["P02"],                 # Press & Heat Lamination
    "foam": ["P02", "P03"],             # laminar o sinter con MGS-1
    "EVA_bag": ["P04", "P02"],          # reutilizar CTB / laminar
    "glove": ["P01", "P02"],            # triturar + laminar
    "aluminum": ["P04"],                # reconfiguración herrajes/struts
    "textiles": ["P02"],                # laminar en capas
}

FLAG_BOOST = {
    "multilayer": ["P02"],
    "thermal": ["P02"],
    "ctb": ["P04"],
    "closed_cell": ["P03", "P02"],
    "nitrile": ["P01", "P02"],
    "struts": ["P04"],
}

def choose_process(target_name: str, proc_df: pd.DataFrame,
                   scenario: str|None = None,
                   crew_time_low: bool = False) -> pd.DataFrame:
    df = proc_df.copy()

    # Penalización/bonificación según tiempo de tripulación
    if crew_time_low:
        # preferir procesos con menor crew_min_per_batch
        df["crew_bias"] = - df["crew_min_per_batch"]
    else:
        df["crew_bias"] = 0.0

    # En esta función solo devolvemos el catálogo; la adecuación por residuo
    # sucede durante la generación, cuando conocemos las filas del inventario.
    return df
