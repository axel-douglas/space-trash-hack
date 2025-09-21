# app/modules/generator.py
from __future__ import annotations
import random
import pandas as pd
from dataclasses import dataclass

@dataclass
class PredProps:
    rigidity: float
    tightness: float
    mass_final_kg: float
    energy_kwh: float
    water_l: float
    crew_min: float

def _problem_mass_weight(row):
    return row.get("_problematic", False) * float(row["kg"])

def _pick_materials(waste_df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    # Muestreamos con probabilidad proporcional al kg y boost si es problemático
    w = waste_df["kg"].clip(lower=0.01) + waste_df["_problematic"].astype(int)*2.0
    return waste_df.sample(n=min(n, len(waste_df)), weights=w, replace=False, random_state=None)

def generate_candidates(waste_df: pd.DataFrame, proc_df: pd.DataFrame,
                        target: dict, n: int = 6, crew_time_low: bool = False):
    out = []
    for _ in range(n):
        picks = _pick_materials(waste_df, n=random.choice([2,3]))
        weights = (picks["kg"] / picks["kg"].sum()).round(2).tolist()
        used_ids = picks["_source_id"].tolist()
        used_cats= picks["_source_category"].tolist()
        used_flags= picks["_source_flags"].tolist()

        # Elegimos proceso por heurística simple: si hay flags/categorías conocidas, favorecer P02/P03/P04
        proc = proc_df.sample(1).iloc[0]
        if any("pouches" in str(c).lower() for c in used_cats):
            proc = proc_df[proc_df["process_id"]=="P02"].iloc[0]
        if any("foam" in str(c).lower() for c in used_cats):
            proc = proc_df[proc_df["process_id"].isin(["P03","P02"])].sample(1).iloc[0]
        if any("EVA" in str(c) for c in used_cats) or any("CTB" in str(f) for f in used_flags):
            proc = proc_df[proc_df["process_id"]=="P04"].iloc[0]

        # “Predicciones” ligeras (demo) basadas en catálogo del proceso
        mass_final = picks["kg"].sum() * 0.9
        energy = float(proc["energy_kwh_per_kg"]) * mass_final
        water  = float(proc["water_l_per_kg"]) * mass_final
        crew   = float(proc["crew_min_per_batch"])

        # Propiedades (dummy razonable)
        rigidity  = min(1.0, 0.5 + 0.2*("aluminum" in " ".join(used_cats)))
        tightness = min(1.0, 0.5 + 0.2*("pouches" in " ".join(used_cats)))

        props = PredProps(rigidity, tightness, mass_final, energy, water, crew)

        # Scoring: objetivo + bono por “consumir problema”
        score = 0.0
        # similitud al target
        score += 1.0 - abs(props.rigidity  - float(target["rigidity"]))
        score += 1.0 - abs(props.tightness - float(target["tightness"]))
        # penalizaciones por recursos
        score -= max(0.0, (energy  - float(target["max_energy_kwh"])) / max(0.1, target["max_energy_kwh"]))
        score -= max(0.0, (water   - float(target["max_water_l"])) / max(0.1, target["max_water_l"]))
        score -= max(0.0, (crew    - float(target["max_crew_min"])) / max(1.0, target["max_crew_min"]))

        # BONO clave: masa problemática consumida
        prob_mass = picks.apply(_problem_mass_weight, axis=1).sum()
        score += 0.5 * (prob_mass / max(0.1, picks["kg"].sum()))

        out.append({
            "materials": picks["material"].tolist(),
            "weights": weights,
            "process_id": proc["process_id"],
            "process_name": proc["name"],
            "props": props,
            "score": round(float(score), 3),

            # Rastro NASA:
            "source_ids": used_ids,
            "source_categories": used_cats,
            "source_flags": used_flags,
        })
    # Ordenar por score
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
