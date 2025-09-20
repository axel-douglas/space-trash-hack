import numpy as np
from dataclasses import dataclass

@dataclass
class PredictedProps:
    rigidity: float
    tightness: float
    mass_final_kg: float
    energy_kwh: float
    water_l: float
    crew_min: float

def predict_properties(mix_vector, process_vec):
    # mix_vector: np.array of per-material weights (sum=1)
    # process_vec: [energy_per_kg, water_per_kg, crew_min]
    base_rigidity = np.dot(mix_vector, np.linspace(0.4,0.9,len(mix_vector)))
    base_tight    = np.dot(mix_vector, np.linspace(0.5,0.8,len(mix_vector))[::-1])
    mass_final    = 0.95  # pérdida ~5%
    energy = process_vec[0] * mass_final
    water  = process_vec[1] * mass_final
    crew   = process_vec[2]
    # normalizamos a [0,1]
    return PredictedProps(
        rigidity=float(np.clip(base_rigidity,0,1)),
        tightness=float(np.clip(base_tight,0,1)),
        mass_final_kg=mass_final,
        energy_kwh=float(energy),
        water_l=float(water),
        crew_min=float(crew)
    )
