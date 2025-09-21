# app/modules/generator.py
from __future__ import annotations
import random
from dataclasses import dataclass
import pandas as pd

@dataclass
class PredProps:
    rigidity: float
    tightness: float
    mass_final_kg: float
    energy_kwh: float
    water_l: float
    crew_min: float

# --- utilidades de compatibilidad con DF (nombres de columnas) ---
def _col(df: pd.DataFrame, candidates: list[str], default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def _ensure_compat(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve copia con columnas estandarizadas:
       id, category, material, kg, volume_l, flags, _problematic
    """
    out = df.copy()
    # ID
    if "id" not in out.columns:
        out["id"] = out.index.astype(str)
    # Category
    cat_col = _col(out, ["category","Category"])
    if cat_col != "category":
        out["category"] = out[cat_col] if cat_col else ""
    # Material
    mat_col = _col(out, ["material","material_family","Material"])
    if mat_col != "material":
        out["material"] = out[mat_col] if mat_col else ""
    # Masa (kg)
    kg_col = _col(out, ["kg","mass_kg","Mass_kg"])
    if kg_col != "kg":
        out["kg"] = pd.to_numeric(out[kg_col], errors="coerce").fillna(0.0) if kg_col else 0.0
    # Volumen (L)
    vol_col = _col(out, ["volume_l","Volume_L"])
    if vol_col != "volume_l":
        out["volume_l"] = pd.to_numeric(out[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0
    # Flags
    flg_col = _col(out, ["flags","Flags"])
    if flg_col != "flags":
        out["flags"] = out[flg_col] if flg_col else ""
    # Problemáticos
    if "_problematic" not in out.columns:
        out["_problematic"] = out.apply(_is_problematic, axis=1)
    # Origen/huella NASA
    out["_source_id"] = out["id"].astype(str)
    out["_source_category"] = out["category"].astype(str)
    out["_source_flags"] = out["flags"].astype(str)
    return out

def _is_problematic(row: pd.Series) -> bool:
    cat = str(row.get("category", "")).lower()
    fam = str(row.get("material", "")).lower() + " " + str(row.get("material_family","")).lower()
    flg = str(row.get("flags", "")).lower()
    rules = [
        "pouches" in cat or "multilayer" in flg or "pe-pet-al" in fam,
        "foam" in cat or "zotek" in fam or "closed_cell" in flg,
        "eva" in cat or "ctb" in flg or "nomex" in fam or "nylon" in fam or "polyester" in fam,
        "glove" in cat or "nitrile" in fam,
        "wipe" in flg or "textile" in cat
    ]
    return any(rules)

def _pick_materials(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    # Preferimos masa y problemáticos (boost)
    w = df["kg"].clip(lower=0.01) + df["_problematic"].astype(int) * 2.0
    return df.sample(n=min(n, len(df)), weights=w, replace=False, random_state=None)

def generate_candidates(waste_df: pd.DataFrame, proc_df: pd.DataFrame,
                        target: dict, n: int = 6, crew_time_low: bool = False):
    """
    Genera candidatos priorizando:
    - Consumir masa de ítems 'problemáticos' definidos por NASA.
    - Elegir procesos coherentes: P02 (laminado multicapa), P03 (sinter + MGS-1), P04 (reuso CTB).
    - Insertar explícitamente **regolito MGS-1** cuando el proceso sea P03.
    """
    if waste_df is None or proc_df is None or len(waste_df) == 0 or len(proc_df) == 0:
        return []

    df = _ensure_compat(waste_df)
    out = []

    for _ in range(n):
        picks = _pick_materials(df, n=random.choice([2,3]))
        total_kg = max(0.001, float(picks["kg"].sum()))
        weights = (picks["kg"] / total_kg).round(2).tolist()

        used_ids   = picks["_source_id"].tolist()
        used_cats  = picks["_source_category"].tolist()
        used_flags = picks["_source_flags"].tolist()
        used_mats  = picks["material"].tolist()

        # Heurística de proceso:
        proc = proc_df.sample(1).iloc[0]
        cats_join = " ".join([str(c).lower() for c in used_cats])
        flags_join = " ".join([str(f).lower() for f in used_flags])

        # Multicapa/pouches → favorecer laminación
        if ("pouches" in cats_join) or ("multilayer" in flags_join) or ("pe-pet-al" in " ".join(used_mats).lower()):
            cand = proc_df[proc_df["process_id"]=="P02"]
            if not cand.empty:
                proc = cand.iloc[0]

        # Espumas → laminar o sinterizar con MGS-1
        if ("foam" in cats_join) or ("zotek" in " ".join(used_mats).lower()):
            cand = proc_df[proc_df["process_id"].isin(["P03","P02"])]
            if not cand.empty:
                proc = cand.sample(1).iloc[0]

        # EVA/CTB → reconfigurar kit (reuso)
        if ("eva" in cats_join) or ("ctb" in flags_join):
            cand = proc_df[proc_df["process_id"]=="P04"]
            if not cand.empty:
                proc = cand.iloc[0]

        # Cálculo recursos base
        mass_final = total_kg * 0.90  # pérdida por recorte/mermas
        energy = float(proc["energy_kwh_per_kg"]) * mass_final
        water  = float(proc["water_l_per_kg"]) * mass_final
        crew   = float(proc["crew_min_per_batch"])

        # Propiedades (demo razonable)
        # - Si hay Al → sube rigidez
        # - Si hay pouches multilayer → sube estanqueidad
        mats_join = " ".join(used_mats).lower()
        rigidity  = min(1.0, 0.5 + (0.2 if "al" in mats_join or "aluminum" in mats_join else 0.0))
        tightness = min(1.0, 0.5 + (0.2 if "pouches" in cats_join or "pe-pet-al" in mats_join else 0.0))

        regolith_pct = 0.0
        materials_for_plan = used_mats.copy()
        weights_for_plan = weights.copy()

        # Si el proceso es P03, **inyectar regolito MGS-1** como carga mineral (10–30%)
        if str(proc["process_id"]).upper() == "P03":
            regolith_pct = 0.2  # 20% por defecto (podés hacer slider más adelante)
            materials_for_plan.append("MGS-1_regolith")
            weights_for_plan = [round(w*(1.0 - regolith_pct), 2) for w in weights_for_plan]
            weights_for_plan.append(round(regolith_pct, 2))
            # Re-normalizar por cualquier redondeo
            s = sum(weights_for_plan)
            if s > 0:
                weights_for_plan = [round(w/s, 2) for w in weights_for_plan]

            # Sutilezas: al sumar carga mineral, suele subir rigidez y bajar estanqueidad
            rigidity = min(1.0, rigidity + 0.1)
            tightness = max(0.0, tightness - 0.05)

        props = PredProps(
            rigidity=rigidity,
            tightness=tightness,
            mass_final_kg=mass_final,
            energy_kwh=energy,
            water_l=water,
            crew_min=crew
        )

        # Scoring multi-objetivo (simple y transparente)
        score = 0.0
        score += 1.0 - abs(props.rigidity  - float(target["rigidity"]))
        score += 1.0 - abs(props.tightness - float(target["tightness"]))
        # Penalizaciones por recursos vs límites
        def _pen(v, lim, eps):
            return max(0.0, (v - float(lim)) / max(eps, float(lim)))
        score -= _pen(props.energy_kwh, target["max_energy_kwh"], 0.1)
        score -= _pen(props.water_l,     target["max_water_l"],   0.1)
        score -= _pen(props.crew_min,    target["max_crew_min"],  1.0)

        # Bono por **masa problemática** consumida
        prob_mass = float((picks["_problematic"].astype(int) * picks["kg"]).sum())
        score += 0.5 * (prob_mass / max(0.1, total_kg))

        out.append({
            "materials": materials_for_plan,
            "weights": weights_for_plan,
            "process_id": str(proc["process_id"]),
            "process_name": str(proc["name"]),
            "props": props,
            "score": round(float(score), 3),

            # Trazabilidad NASA
            "source_ids": used_ids,
            "source_categories": used_cats,
            "source_flags": used_flags,

            # Señal explícita de uso de regolito
            "regolith_pct": regolith_pct
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out
