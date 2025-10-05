from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping

from app.modules.utils import format_number, safe_float

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


def _extract_numeric(candidate: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(candidate, Mapping):
        return None

    props = candidate.get("props")
    sources: list[Any] = []
    if props is not None:
        sources.append(props)
    sources.append(candidate)

    for source in sources:
        if source is None:
            continue
        if hasattr(source, key):
            value = getattr(source, key)
        elif isinstance(source, Mapping):
            value = source.get(key)
        else:
            continue
        number = safe_float(value)
        if number is not None:
            return number
    return None


def build_safety_compliance(
    candidate: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
    flags: SafetyFlags,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deriva checklists de cumplimiento ambiental y de recursos."""

    compliance_rows: List[Dict[str, Any]] = []

    incineration_ok = not flags.incineration
    compliance_rows.append(
        {
            "label": "No incineración",
            "ok": incineration_ok,
            "icon": "✅" if incineration_ok else "⚠️",
            "message": (
                "Proceso evita pasos de combustión directa."
                if incineration_ok
                else "Replanteá rutas mecánicas/químicas antes de incinerar."
            ),
        }
    )

    micro_ok = not flags.microplastics
    compliance_rows.append(
        {
            "label": "Filtrado microfibra activo",
            "ok": micro_ok,
            "icon": "✅" if micro_ok else "⚠️",
            "message": (
                "Contención y filtrado de partículas verificado."
                if micro_ok
                else "Instalá filtros HEPA/microfibra en shredder y ventilación."
            ),
        }
    )

    pfas_ok = not flags.pfas
    compliance_rows.append(
        {
            "label": "Alternativa sin PFAS",
            "ok": pfas_ok,
            "icon": "✅" if pfas_ok else "⚠️",
            "message": (
                "Receta libre de compuestos fluorados identificados."
                if pfas_ok
                else "Sustituí químicos fluorados o aislá el flujo con monitoreo."
            ),
        }
    )

    target_mapping: Mapping[str, Any] = target or {}
    candidate_mapping: Mapping[str, Any] | None = candidate if isinstance(candidate, Mapping) else None

    resource_rows: List[Dict[str, Any]] = []
    resource_specs = [
        ("energy_kwh", "Energía", "max_energy_kwh", "kWh", 2),
        ("water_l", "Agua", "max_water_l", "L", 1),
        ("crew_min", "Crew", "max_crew_min", "min", 0),
    ]

    for attr, label, limit_key, unit, precision in resource_specs:
        value = _extract_numeric(candidate_mapping, attr)
        limit = safe_float(target_mapping.get(limit_key))
        value_text = format_number(value, precision=precision)
        limit_text = format_number(limit, precision=precision)

        if limit is None:
            icon = "⚠️"
            ok: bool | None = None
            message = f"Definí un límite de {label.lower()} para evaluar compliance."
        elif value is None:
            icon = "⚠️"
            ok = None
            message = f"Sin estimación de {label.lower()}; completá datos antes de certificar."
        else:
            ok = value <= limit
            if ok:
                icon = "✅"
                message = (
                    f"Consumo proyectado {value_text} {unit} ≤ objetivo {limit_text} {unit}."
                )
            else:
                icon = "⚠️"
                message = (
                    f"{value_text} {unit} supera el límite {limit_text} {unit}; ajustá receta o topes."
                )

        resource_rows.append(
            {
                "label": f"{label} ({unit})",
                "ok": ok,
                "icon": icon,
                "message": message,
                "value": value,
                "limit": limit,
                "value_text": value_text,
                "limit_text": limit_text,
            }
        )

    return {"compliance": compliance_rows, "resource_compliance": resource_rows}
