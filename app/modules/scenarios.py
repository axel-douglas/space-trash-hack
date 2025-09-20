from dataclasses import dataclass
from typing import List

@dataclass
class Step:
    title: str
    detail: str

@dataclass
class Playbook:
    name: str
    summary: str
    steps: List[Step]

PLAYBOOKS = {
"Residence Renovations": Playbook(
    name="Residence Renovations",
    summary="Reuso de CTB/estructuras + laminación de espumas y films para outfitting robusto con mínimo tiempo de tripulación.",
    steps=[
        Step("Auditar marcos y puntales de aluminio",
             "Reconfigurar longitudes y unir con herrajes CTB para formar estantes/particiones."),
        Step("Aprovechar espumas (ZOTEK/bubble wrap)",
             "Laminarlas por presión/calor para crear paneles livianos y skins protectoras."),
        Step("Refuerzo con regolito (opcional)",
             "Sinterizar mezcla polímero+MGS-1 para bases o bordes con mayor resistencia a impacto."),
        Step("Checklist de seguridad",
             "Evitar incineración; preferir encapsulado para minimizar microplásticos.")
]) ,
"Cosmic Celebrations": Playbook(
    name="Cosmic Celebrations",
    summary="Laminados textiles + films multicapa para utilería y decoración segura sin agua.",
    steps=[
        Step("Seleccionar textiles y wipes limpios",
             "Priorizar poliéster/nylon para mayor estabilidad dimensional."),
        Step("Encapsular films/plásticos delgados",
             "Prensado térmico para evitar desprendimientos de microplásticos."),
        Step("Cortes modulares",
             "Plantillas con esquinas redondeadas; fijación con herrajes CTB o clips."),
        Step("Checklist de seguridad",
             "Nada de llamas abiertas; verificar olores volátiles tras enfriado.")
]),
"Daring Discoveries": Playbook(
    name="Daring Discoveries",
    summary="Uso del carbono sobrante como carga/refuerzo y redes/filtros como mallas.",
    steps=[
        Step("Clasificar carbono y meshes",
             "Separar granulometrías; reservar mallas metálicas/poliméricas para refuerzo superficial."),
        Step("Mezcla con polímeros disponibles",
             "Añadir 5–20% carbono en laminado/compactación; o sinter con MGS-1 para piezas rígidas."),
        Step("Protección frente a polvo",
             "Preferir procesos indoor; sellos perimetrales posproceso."),
        Step("Checklist de seguridad",
             "Evitar generadores de humo; registrar cualquier residuo fino en filtros HEPA.")
])
}
