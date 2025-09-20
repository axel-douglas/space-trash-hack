import numpy as np
import pandas as pd

def score_breakdown(props, target):
    # mismo score que generator.score_candidate, pero desglosado
    f = 0.4*((props.rigidity+props.tightness)/2.0)
    a = 0.2*(1 - min(props.water_l / max(target["max_water_l"],1e-6), 1))
    e = 0.2*(1 - min(props.energy_kwh / max(target["max_energy_kwh"],1e-6), 1))
    t = 0.1*(1 - min(props.crew_min / max(target["max_crew_min"],1e-6), 1))
    s = 0.1
    parts = pd.DataFrame({
        "component":["Función (rigidez+estanqueidad)","Agua","Energía","Tiempo tripulación","Seguridad base"],
        "weight":[0.4,0.2,0.2,0.1,0.1],
        "contribution":[f,a,e,t,s]
    })
    parts["pct"] = (parts["contribution"]/parts["contribution"].sum()).round(3)
    return parts

def compare_table(cands, target):
    rows=[]
    for i,c in enumerate(cands, start=1):
        p=c["props"]
        rows.append({
            "Opción": i,
            "Score": c["score"],
            "Rigidez": p.rigidity,
            "Estanqueidad": p.tightness,
            "Energía (kWh)": p.energy_kwh,
            "Agua (L)": p.water_l,
            "Crew (min)": p.crew_min,
            "Proceso": f'{c["process_id"]} {c["process_name"]}',
            "Materiales": ", ".join(c["materials"])
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False)
