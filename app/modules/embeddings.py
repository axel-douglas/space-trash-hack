import numpy as np
import pandas as pd

MATERIAL_BASE = {
    "PE-PET-Al": [0.6, 0.5, 0.7, 0.2],   # [rigidity, tightness, density proxy, recycle_score]
    "polyester": [0.4, 0.3, 0.4, 0.6],
    "ZOTEK_F30": [0.3, 0.2, 0.2, 0.8],
    "nylon-polyester": [0.5, 0.4, 0.5, 0.5],
    "nitrile": [0.5, 0.6, 0.6, 0.4],
    "Al": [0.9, 0.8, 0.9, 0.7]
}

def waste_to_embedding_row(row: pd.Series) -> np.ndarray:
    base = MATERIAL_BASE.get(row["material_family"], [0.4,0.4,0.4,0.4])
    # escala por masa/volumen para reflejar disponibilidad relativa
    mass_factor = min(1.0, float(row["mass_kg"])/4.0)
    vol_factor  = min(1.0, float(row["volume_l"])/25.0)
    return np.array(base + [mass_factor, vol_factor], dtype=float)

def make_waste_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([waste_to_embedding_row(r) for _, r in df.iterrows()])
