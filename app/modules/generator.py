import numpy as np
import pandas as pd
from .predictors import predict_properties

def score_candidate(props, target, crew_time_low: bool = False):
    """
    Score multiobjetivo. Si crew_time_low=True, aumenta el peso de tiempo de tripulación.
    """
    w_func = 0.4
    w_agua = 0.2
    w_ener = 0.2
    w_time = 0.1
    w_safe = 0.1

    if crew_time_low:
        w_time = 0.2
        w_func = 0.35  # pequeña redistribución

    f = w_func*((props.rigidity+props.tightness)/2.0)
    a = w_agua*(1 - min(props.water_l / max(target["max_water_l"],1e-6), 1))
    e = w_ener*(1 - min(props.energy_kwh / max(target["max_energy_kwh"],1e-6), 1))
    t = w_time*(1 - min(props.crew_min / max(target["max_crew_min"],1e-6), 1))
    s = w_safe  # seguridad base (las banderas duras se validan fuera)

    return max(0.0, f+a+e+t+s)

def generate_candidates(waste_df: pd.DataFrame, process_df: pd.DataFrame, target: dict, n=6, seed=7, crew_time_low: bool=False):
    rng = np.random.default_rng(seed)
    mats = waste_df["id"].tolist()
    results = []
    if len(mats) == 0 or process_df.empty:
        return results

    for _ in range(n):
        k = int(rng.integers(3, min(4, len(mats))+1))
        chosen = rng.choice(mats, size=k, replace=False)
        weights = rng.random(k); weights = weights/weights.sum()
        proc_row = process_df.sample(1, random_state=int(rng.integers(0,1e6))).iloc[0]
        props = predict_properties(weights, [proc_row["energy_kwh_per_kg"], proc_row["water_l_per_kg"], proc_row["crew_min_per_batch"]])
        sc = score_candidate(props, target, crew_time_low=crew_time_low)
        results.append({
            "materials": list(chosen),
            "weights": list(np.round(weights,3)),
            "process_id": proc_row["process_id"],
            "process_name": proc_row["name"],
            "props": props,
            "score": round(sc,3)
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
