import pandas as pd

SCENARIO_TO_PROCESS = {
    "Residence Renovations": ["P04", "P02", "P01", "P03"],  # Reuse CTB / Lamination / Shredder / Sinter
    "Cosmic Celebrations":   ["P02", "P01"],                 # Lamination / Shredder (encapsulado)
    "Daring Discoveries":    ["P03", "P02", "P01"],          # Sinter + Regolito / Lamination / Shredder
}

def choose_process(target_name: str, process_df: pd.DataFrame, scenario: str | None = None, crew_time_low: bool = False):
    """
    Devuelve subset de procesos recomendados considerando escenario y modo 'Crew-time Low'.
    """
    if scenario and scenario in SCENARIO_TO_PROCESS:
        ids = SCENARIO_TO_PROCESS[scenario]
        subset = process_df.loc[process_df["process_id"].isin(ids)].copy()
    else:
        # fallback por tipo de target
        if target_name in ("Container", "Utensil"):
            subset = process_df.loc[process_df["process_id"].isin(["P01","P02"])].copy()
        elif target_name == "Interior":
            subset = process_df.loc[process_df["process_id"].isin(["P04","P03"])].copy()
        else:
            subset = process_df.loc[process_df["process_id"].isin(["P01","P03"])].copy()

    # Modo Crew-time Low: prioriza procesos con menor 'crew_min_per_batch'
    if crew_time_low and not subset.empty:
        subset = subset.sort_values("crew_min_per_batch", ascending=True).head(max(2, len(subset)//2)).copy()
    return subset

def process_vector(row):
    return [float(row["energy_kwh_per_kg"]), float(row["water_l_per_kg"]), float(row["crew_min_per_batch"])]
