import numpy as np
import pandas as pd
from .predictors import predict_properties

def score_candidate(props, target):
    # score multiobjetivo
    f = 0.4*((props.rigidity+props.tightness)/2.0)
    a = 0.2*(1 - min(props.water_l / max(target["max_water_l"],1e-6), 1))
    e = 0.2*(1 - min(props.energy_kwh / max(target["max_energy_kwh"],1e-6), 1))
    t = 0.1*(1 - min(props.crew_min / max(target["max_crew_min"],1e-6), 1))
    s = 0.1  # seguridad fija en demo (sin PFAS/microplásticos)
    return max(0.0, f+a+e+t+s)

def generate_candidates(waste_df: pd.DataFrame, process_df: pd.DataFrame, target: dict, n=6, seed=7):
    rng = np.random.default_rng(seed)
    mats = waste_df["id"].tolist()
    results = []
    # elige 3-4 materiales por candidato
    for _ in range(n):
        k = rng.integers(3, min(4, len(mats))+1)
        chosen = rng.choice(mats, size=k, replace=False)
        weights = rng.random(k); weights = weights/weights.sum()
        proc_row = process_df.sample(1, random_state=int(rng.integers(0,1e6))).iloc[0]
        props = predict_properties(weights, [proc_row["energy_kwh_per_kg"], proc_row["water_l_per_kg"], proc_row["crew_min_per_batch"]])
        sc = score_candidate(props, target)
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
