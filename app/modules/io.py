from pathlib import Path
import pandas as pd
import json

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_waste_df():
    return pd.read_csv(DATA_DIR / "waste_inventory_sample.csv")

def load_process_df():
    return pd.read_csv(DATA_DIR / "process_catalog.csv")

def load_targets():
    with open(DATA_DIR / "targets_presets.json","r") as f:
        return json.load(f)

def save_waste_df(df):
    df.to_csv(DATA_DIR / "waste_inventory_sample.csv", index=False)
