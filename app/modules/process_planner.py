import pandas as pd

def choose_process(target_name: str, process_df: pd.DataFrame):
    # reglas súper simples para demo
    if target_name in ("Container","Utensil"):
        return process_df.loc[process_df["process_id"].isin(["P01","P02"])].copy()
    if target_name == "Interior":
        return process_df.loc[process_df["process_id"].isin(["P04","P03"])].copy()
    return process_df.loc[process_df["process_id"].isin(["P01","P03"])].copy()

def process_vector(row):
    return [float(row["energy_kwh_per_kg"]), float(row["water_l_per_kg"]), float(row["crew_min_per_batch"])]
