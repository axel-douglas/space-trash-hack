from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Step:
    title: str
    detail: str


@dataclass
class RecipeComponent:
    material: str
    quantity: str
    role: Optional[str] = None


@dataclass
class ExampleRecipe:
    process_id: str
    mix: List[RecipeComponent]
    batch_notes: str
    generator_filters: Optional[Dict[str, bool]] = None


@dataclass
class Playbook:
    name: str
    summary: str
    product_label: str
    product_end_use: str
    example_recipe: ExampleRecipe
    metadata: Dict[str, str] = field(default_factory=dict)
    steps: List[Step]
    generator_filters: Optional[Dict[str, bool]] = None


PLAYBOOKS = {
    "Residence Renovations": Playbook(
        name="Residence Renovations",
        summary=(
            "Reuso de CTB/estructuras + laminación de espumas y films para outfitting "
            "robusto con mínimo tiempo de tripulación."
        ),
        product_label="Kit de estantería modular",
        product_end_use=(
            "Contenedor: módulos de almacenamiento rígidos para reorganizar camarotes."
        ),
        example_recipe=ExampleRecipe(
            process_id="RENOV-FOAM-01",
            batch_notes=(
                "Laminar espuma ZOTEK con film burbuja y fijar sobre bastidores CTB "
                "reforzados para un módulo de 1.5 m."
            ),
            mix=[
                RecipeComponent(
                    "Segmentos CTB de aluminio reutilizados",
                    "4 uds (≈2.8 kg)",
                    "Estructura principal",
                ),
                RecipeComponent(
                    "Paneles ZOTEK M (espuma)",
                    "3 láminas (≈1.5 kg)",
                    "Superficies y aislamiento",
                ),
                RecipeComponent(
                    "Film bubble wrap + film protector",
                    "5 m (≈0.6 kg)",
                    "Laminado protector",
                ),
                RecipeComponent(
                    "Herrajes y cinchos CTB",
                    "Set completo (≈0.4 kg)",
                    "Anclaje y unión",
                ),
            ],
            generator_filters={
                "showroom_only_safe": True,
                "showroom_limit_energy": True,
                "showroom_limit_water": True,
                "showroom_limit_crew": True,
            },
        ),
        metadata={
            "ID de proceso": "RENOV-FOAM-01",
            "Tiempo estimado": "95 min",
            "Tripulación": "2 astronautas",
        },
        steps=[
            Step(
                "Auditar marcos y puntales de aluminio",
                "Reconfigurar longitudes y unir con herrajes CTB para formar estantes/"
                "particiones.",
            ),
            Step(
                "Aprovechar espumas (ZOTEK/bubble wrap)",
                "Laminarlas por presión/calor para crear paneles livianos y skins "
                "protectoras.",
            ),
            Step(
                "Refuerzo con regolito (opcional)",
                "Sinterizar mezcla polímero+MGS-1 para bases o bordes con mayor "
                "resistencia a impacto.",
            ),
            Step(
                "Checklist de seguridad",
                "Evitar incineración; preferir encapsulado para minimizar microplásticos.",
            ),
        ],
        generator_filters={
            "showroom_only_safe": True,
            "showroom_limit_energy": True,
            "showroom_limit_water": True,
            "showroom_limit_crew": True,
        },
    ),
    "Cosmic Celebrations": Playbook(
        name="Cosmic Celebrations",
        summary=(
            "Laminados textiles + films multicapa para utilería y decoración segura sin "
            "agua."
        ),
        product_label="Set de decoración reconfigurable",
        product_end_use="Utensilio: backdrops y props livianos para celebraciones.",
        example_recipe=ExampleRecipe(
            process_id="CELE-FAB-07",
            batch_notes=(
                "Consolidar textiles limpios con films metalizados para paneles flexibles "
                "de decoración sin desprendimiento."
            ),
            mix=[
                RecipeComponent(
                    "Textiles de poliéster/nylon",
                    "2 mantas (≈1.2 kg)",
                    "Capa base y estética",
                ),
                RecipeComponent(
                    "Films multicapa metalizados",
                    "4 m² (≈0.5 kg)",
                    "Realce visual y reflejo",
                ),
                RecipeComponent(
                    "Wipes limpios",
                    "20 uds (≈0.2 kg)",
                    "Refuerzos y bordes",
                ),
                RecipeComponent(
                    "Clips CTB reutilizables",
                    "12 uds",
                    "Fijación modular",
                ),
            ],
            generator_filters={
                "showroom_only_safe": True,
                "showroom_limit_energy": True,
                "showroom_limit_water": True,
                "showroom_limit_crew": False,
            },
        ),
        metadata={
            "ID de proceso": "CELE-FAB-07",
            "Tiempo estimado": "70 min",
            "Tripulación": "1 astronauta",
        },
        steps=[
            Step(
                "Seleccionar textiles y wipes limpios",
                "Priorizar poliéster/nylon para mayor estabilidad dimensional.",
            ),
            Step(
                "Encapsular films/plásticos delgados",
                "Prensado térmico para evitar desprendimientos de microplásticos.",
            ),
            Step(
                "Cortes modulares",
                "Plantillas con esquinas redondeadas; fijación con herrajes CTB o clips.",
            ),
            Step(
                "Checklist de seguridad",
                "Nada de llamas abiertas; verificar olores volátiles tras enfriado.",
            ),
        ],
        generator_filters={
            "showroom_only_safe": True,
            "showroom_limit_energy": True,
            "showroom_limit_water": True,
            "showroom_limit_crew": False,
        },
    ),
    "Daring Discoveries": Playbook(
        name="Daring Discoveries",
        summary=(
            "Uso del carbono sobrante como carga/refuerzo y redes/filtros como mallas."
        ),
        product_label="Liner conductivo para cápsula de muestras",
        product_end_use="Liner: recubrimiento antiestático para contenedor científico.",
        example_recipe=ExampleRecipe(
            process_id="DISC-LIN-03",
            batch_notes=(
                "Compactar mezcla polimérica con 15% de carbono y reforzar con mesh "
                "metálico para liner semicircular."
            ),
            mix=[
                RecipeComponent(
                    "Polímero base (PLA/PEEK recuperado)",
                    "2.5 kg pellet",
                    "Matriz estructural",
                ),
                RecipeComponent(
                    "Carbono amorfo reciclado",
                    "0.45 kg (15%)",
                    "Conductividad y refuerzo",
                ),
                RecipeComponent(
                    "Mesh metálico fino",
                    "1 m²",
                    "Refuerzo superficial",
                ),
                RecipeComponent(
                    "Regolito MGS-1 tamizado",
                    "0.3 kg",
                    "Carga mineral opcional",
                ),
            ],
            generator_filters={
                "showroom_only_safe": False,
                "showroom_limit_energy": True,
                "showroom_limit_water": False,
                "showroom_limit_crew": True,
            },
        ),
        metadata={
            "ID de proceso": "DISC-LIN-03",
            "Tiempo estimado": "120 min",
            "Tripulación": "2 astronautas",
        },
        steps=[
            Step(
                "Clasificar carbono y meshes",
                "Separar granulometrías; reservar mallas metálicas/poliméricas para "
                "refuerzo superficial.",
            ),
            Step(
                "Mezcla con polímeros disponibles",
                "Añadir 5–20% carbono en laminado/compactación; o sinter con MGS-1 para "
                "piezas rígidas.",
            ),
            Step(
                "Protección frente a polvo",
                "Preferir procesos indoor; sellos perimetrales posproceso.",
            ),
            Step(
                "Checklist de seguridad",
                "Evitar generadores de humo; registrar cualquier residuo fino en filtros "
                "HEPA.",
            ),
        ],
        generator_filters={
            "showroom_only_safe": False,
            "showroom_limit_energy": True,
            "showroom_limit_water": False,
            "showroom_limit_crew": True,
        },
    ),
}
