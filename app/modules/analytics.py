import numpy as np
import pandas as pd


def pareto_front(
    df: pd.DataFrame,
    minimize_cols=("Energía (kWh)", "Agua (L)", "Crew (min)"),
    maximize_cols=("Score",),
):
    """Devuelve los índices de las filas no dominadas (frente de Pareto).

    La función mantiene la interfaz existente (lista de índices del ``DataFrame``
    original) pero ahora opera directamente sobre ``numpy`` sin crear columnas
    auxiliares.  Se construye una matriz con las columnas a minimizar y el
    negativo de las columnas a maximizar, y se utilizan operaciones vectorizadas
    para evaluar la dominancia entre todas las combinaciones de filas.  Esta
    versión evita bucles en Python y reduce el coste de copiar datos,
    proporcionando un mejor rendimiento en ``DataFrames`` grandes.
    """

    metrics = []
    if minimize_cols:
        metrics.append(df.loc[:, list(minimize_cols)].to_numpy())
    if maximize_cols:
        metrics.append(-df.loc[:, list(maximize_cols)].to_numpy())

    if not metrics:
        return df.index.tolist()

    X = np.hstack(metrics) if len(metrics) > 1 else metrics[0]

    less_equal = X[:, None, :] <= X[None, :, :]
    strictly_less = X[:, None, :] < X[None, :, :]
    dominates = np.all(less_equal, axis=2) & np.any(strictly_less, axis=2)
    is_dominated = dominates.any(axis=0)
    front_idx = np.nonzero(~is_dominated)[0]
    return df.index[front_idx].tolist()
