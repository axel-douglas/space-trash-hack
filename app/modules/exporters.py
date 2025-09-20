import json
import io
import pandas as pd

def candidate_to_json(selected: dict, target: dict, safety: dict) -> bytes:
    p = selected["props"]
    payload = {
        "target": target,
        "candidate": {
            "materials": selected["materials"],
            "weights": selected["weights"],
            "process": {"id": selected["process_id"], "name": selected["process_name"]},
            "predictions": {
                "rigidity": p.rigidity, "tightness": p.tightness,
                "mass_final_kg": p.mass_final_kg,
                "energy_kwh": p.energy_kwh, "water_l": p.water_l, "crew_min": p.crew_min
            },
            "score": selected["score"],
            "safety": safety
        }
    }
    return json.dumps(payload, indent=2).encode("utf-8")

def candidate_to_csv(selected: dict) -> bytes:
    p = selected["props"]
    df = pd.DataFrame([{
        "materials": "|".join(selected["materials"]),
        "weights": "|".join(map(str, selected["weights"])),
        "process_id": selected["process_id"],
        "process_name": selected["process_name"],
        "rigidity": p.rigidity, "tightness": p.tightness,
        "mass_final_kg": p.mass_final_kg,
        "energy_kwh": p.energy_kwh, "water_l": p.water_l, "crew_min": p.crew_min,
        "score": selected["score"]
    }])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
