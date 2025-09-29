import numpy as np
import pandas as pd

def score_breakdown(props, target, crew_time_low: bool=False):
    # pesos como en generator.score_candidate
    w_func = 0.4; w_agua = 0.2; w_ener = 0.2; w_time = 0.1; w_safe = 0.1
    if crew_time_low:
        w_func = 0.35; w_time = 0.2
    f = w_func*((props.rigidity+props.tightness)/2.0)
    a = w_agua*(1 - min(props.water_l / max(target["max_water_l"],1e-6), 1))
    e = w_ener*(1 - min(props.energy_kwh / max(target["max_energy_kwh"],1e-6), 1))
    t = w_time*(1 - min(props.crew_min / max(target["max_crew_min"],1e-6), 1))
    s = w_safe
    parts = pd.DataFrame({
        "component":["Función (rigidez+estanqueidad)","Agua","Energía","Tiempo tripulación","Seguridad base"],
        "weight":[w_func,w_agua,w_ener,w_time,w_safe],
        "contribution":[f,a,e,t,s]
    })
    parts["pct"] = (parts["contribution"]/max(parts["contribution"].sum(),1e-9)).round(3)
    return parts

def compare_table(cands, target, crew_time_low: bool=False):
    rows=[]
    for i,c in enumerate(cands, start=1):
        p=c["props"]
        rows.append({
            "Opción": i,
            "Score": c["score"],
            "Rigidez": round(p.rigidity,3),
            "Estanqueidad": round(p.tightness,3),
            "Energía (kWh)": round(p.energy_kwh,3),
            "Agua (L)": round(p.water_l,3),
            "Crew (min)": round(p.crew_min,1),
            "Masa (kg)": round(getattr(p, "mass_final_kg", np.nan),3),
            "Proceso": f'{c["process_id"]} {c["process_name"]}',
            "Materiales": ", ".join(c["materials"])
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False)
