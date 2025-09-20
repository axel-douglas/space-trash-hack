from dataclasses import dataclass
import math

@dataclass
class LineConfig:
    # parámetros simples de línea/equipo en hábitat
    batches_per_shift: int       # lotes por turno
    kg_per_batch: float          # kg por lote
    energy_kwh_per_batch: float  # kWh por lote
    water_l_per_batch: float     # L por lote
    crew_min_per_batch: float    # min de tripulación por lote

def simulate(line: LineConfig, shifts_per_sol: int, num_sols: int):
    # resultados agregados en ventanas de producción
    total_batches = line.batches_per_shift * shifts_per_sol * num_sols
    total_kg = total_batches * line.kg_per_batch
    total_kwh = total_batches * line.energy_kwh_per_batch
    total_water = total_batches * line.water_l_per_batch
    total_crew_min = total_batches * line.crew_min_per_batch
    return {
        "batches": int(total_batches),
        "kg": float(total_kg),
        "kwh": float(total_kwh),
        "water_l": float(total_water),
        "crew_min": float(total_crew_min)
    }
