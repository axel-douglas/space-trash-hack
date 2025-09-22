# app/modules/exporters.py
import json
import io
import pandas as pd

def _trace_fields(selected: dict):
    """Obtiene campos de trazabilidad si existen, sino listas vacías."""
    ids   = selected.get("source_ids", []) or []
    cats  = selected.get("source_categories", []) or []
    flags = selected.get("source_flags", []) or []
    return ids, cats, flags

def candidate_to_json(selected: dict, target: dict, safety: dict) -> bytes:
    p = selected["props"]
    ids, cats, flags = _trace_fields(selected)
    feature_importance = selected.get("feature_importance", {})
    payload = {
        "target": target,
        "candidate": {
            "materials": selected["materials"],
            "weights": selected["weights"],
            "process": {"id": selected["process_id"], "name": selected["process_name"]},
            "predictions": {
                "rigidity": p.rigidity,
                "tightness": p.tightness,
                "mass_final_kg": p.mass_final_kg,
                "energy_kwh": p.energy_kwh,
                "water_l": p.water_l,
                "crew_min": p.crew_min
            },
            "score": selected["score"],
            "safety": safety,
            # --- Trazabilidad NASA ---
            "traceability": {
                "source_ids": ids,
                "source_categories": cats,
                "source_flags": flags,
            },
            "explainability": {
                "feature_importance": feature_importance,
                "notes": "Valores positivos elevan el score; negativos indican penalizaciones."
            }
        }
    }
    return json.dumps(payload, indent=2).encode("utf-8")

def candidate_to_csv(selected: dict) -> bytes:
    p = selected["props"]
    ids, cats, flags = _trace_fields(selected)
    row = {
        "materials": "|".join(selected["materials"]),
        "weights": "|".join(map(str, selected["weights"])),
        "process_id": selected["process_id"],
        "process_name": selected["process_name"],
        "rigidity": p.rigidity,
        "tightness": p.tightness,
        "mass_final_kg": p.mass_final_kg,
        "energy_kwh": p.energy_kwh,
        "water_l": p.water_l,
        "crew_min": p.crew_min,
        "score": selected["score"],
        # --- Trazabilidad NASA (columnas planas para Excel/Sheets) ---
        "source_ids": "|".join(map(str, ids)),
        "source_categories": "|".join(map(str, cats)),
        "source_flags": "|".join(map(str, flags)),
    }
    for key, value in (selected.get("feature_importance", {}) or {}).items():
        row[f"shap_{key}"] = value
    df = pd.DataFrame([row])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
