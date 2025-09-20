import numpy as np
import pandas as pd

def pareto_front(df: pd.DataFrame, minimize_cols=("Energía (kWh)","Agua (L)","Crew (min)"), maximize_cols=("Score",)):
    """
    Devuelve índices de filas no dominadas (Pareto front).
    Minimiza energía/agua/crew y maximiza score.
    """
    data = df.copy()
    for c in minimize_cols:
        data[f"min_{c}"] = data[c]
    for c in maximize_cols:
        data[f"min_{c}"] = -data[c]  # maximizar = minimizar el negativo

    X = data[[f"min_{c}" for c in minimize_cols] + [f"min_{c}" for c in maximize_cols]].to_numpy()
    is_dominated = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        if is_dominated[i]: 
            continue
        is_dominated |= np.all(X <= X[i], axis=1) & np.any(X < X[i], axis=1)
    front_idx = np.where(~is_dominated)[0]
    return data.iloc[front_idx].index.tolist()
