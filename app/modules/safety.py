from dataclasses import dataclass
from typing import List, Dict
import re

PFAS_HINTS = ["ptfe","fluoro","pfas","pfos","pfoa","fep","pvdf"]
MICRO_RISK_FAMILIES = ["polyester","nylon","pe","pet","pe-pet-al","zotek_f30","foam","textiles"]
INCINERATION_HINTS = ["inciner","burn","combust"]

@dataclass
class SafetyFlags:
    pfas: bool
    microplastics: bool
    incineration: bool
    notes: List[str]

def _norm(x:str) -> str:
    return re.sub(r"[^a-z0-9]+","", x.lower())

def check_safety(material_families: List[str],
                 process_name: str,
                 process_id: str) -> SafetyFlags:
    fams = [_norm(f) for f in material_families]
    pname = _norm(process_name + " " + process_id)

    notes = []
    pfas = any(any(h in f for h in PFAS_HINTS) for f in fams)
    if pfas: notes.append("Se detectaron familias con indicios fluorados (posible PFAS).")

    # Microplásticos: riesgo si hay familias poliméricas + proceso de shredder sin encapsulado/laminado/sinterizado
    micro = any(any(k in f for k in MICRO_RISK_FAMILIES) for f in fams)
    shred = "shredder" in pname
    laminated = ("lamination" in pname or "press" in pname or "sinter" in pname or "ctb" in pname)
    micro_risk = bool(micro and shred and not laminated)
    if micro_risk: notes.append("Triturado sin encapsulado puede liberar microplásticos.")

    incin = any(h in pname for h in INCINERATION_HINTS)
    if incin: notes.append("Incineración está prohibida por la consigna.")

    return SafetyFlags(pfas=pfas, microplastics=micro_risk, incineration=incin, notes=notes)

def safety_badge(flags: SafetyFlags) -> Dict[str,str]:
    level = "OK"
    if flags.incineration or flags.pfas or flags.microplastics:
        level = "Riesgo"
    detail = "; ".join(flags.notes) if flags.notes else "Sin hallazgos."
    return {"level": level, "detail": detail}
