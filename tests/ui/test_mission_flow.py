import sys
import types

import pytest

if "joblib" not in sys.modules:
    sys.modules["joblib"] = types.ModuleType("joblib")

pytest.importorskip("numpy")
pytest.importorskip("plotly")

from app.modules.luxe_components import ActionCard, HeroFlowStage, MissionFlowShowcase


@pytest.fixture
def sample_stages() -> list[HeroFlowStage]:
    return [
        HeroFlowStage(
            key="inventory",
            order=2,
            name="Inventario",
            hero_headline="Inventario",
            hero_copy="Normalizá residuos en minutos.",
            card_body="Normalizá residuos y marcá flags EVA.",
            compact_card_body="Normalizá residuos EVA.",
            icon="🧱",
            timeline_label="Inventario",
            timeline_description="Carga CSV y marca flags críticos.",
            footer="Dataset NASA",
        ),
        HeroFlowStage(
            key="generator",
            order=1,
            name="Generador",
            hero_headline="Generá y valida",
            hero_copy="Rex-AI mezcla ítems.",
            card_body="Rex-AI compara heurística vs IA en vivo.",
            compact_card_body="Compará heurística vs IA.",
            icon="🤖",
            timeline_label="Generador IA",
            timeline_description="Explorá mezclas óptimas.",
            footer="Cooperativo",
        ),
        HeroFlowStage(
            key="report",
            order=3,
            name="Resultados",
            hero_headline="Reportar",
            hero_copy="Exportá todo",
            card_body="Exportá Sankey y feedback listos para ingeniería.",
            compact_card_body=None,
            icon="📦",
            timeline_label="Export",
            timeline_description="Entregá reportes completos.",
            footer=None,
        ),
    ]


def test_mission_flow_stage_titles_are_unique_and_sorted(sample_stages: list[HeroFlowStage]) -> None:
    showcase = MissionFlowShowcase(
        stages=sample_stages,
        primary_actions=[
            ActionCard(title="Acción", body="CTA principal", icon="🚀"),
        ],
        insights=["Insight"],
    )

    html = showcase.markup()
    titles = showcase.stage_titles()
    last_index = -1
    for title in titles:
        occurrences = html.count(title)
        assert occurrences == 1, f"{title} should appear once, found {occurrences}"
        current_index = html.index(title)
        assert current_index > last_index
        last_index = current_index


def test_mission_flow_copy_sequences(sample_stages: list[HeroFlowStage]) -> None:
    showcase = MissionFlowShowcase(stages=sample_stages)

    desktop_copy = showcase.copy_sequence("desktop")
    mobile_copy = showcase.copy_sequence("mobile")

    assert desktop_copy == [
        "Rex-AI compara heurística vs IA en vivo.",
        "Normalizá residuos y marcá flags EVA.",
        "Exportá Sankey y feedback listos para ingeniería.",
    ]
    assert mobile_copy == [
        "Compará heurística vs IA.",
        "Normalizá residuos EVA.",
        "Exportá Sankey y feedback listos para ingeniería.",
    ]
