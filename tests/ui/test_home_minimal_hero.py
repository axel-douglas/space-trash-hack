from pathlib import Path

from app.modules.luxe_components import HeroFlowStage, MinimalHero


def test_home_minimal_hero_snapshot() -> None:
    flow = [
        HeroFlowStage(
            key="inventory",
            order=1,
            name="Inventario",
            hero_headline="Calibrá el inventario",
            hero_copy="Normalizá residuos, detectá flags EVA y estructuras multi-layer.",
            card_body="Normalizá residuos y marcá flags problemáticos (multilayer, EVA, nitrilo).",
            icon="🧱",
            timeline_label="Inventario en vivo",
            timeline_description="Ingerí CSV NASA, normalizá unidades y marca riesgos EVA desde la cabina.",
            footer="Dataset NASA + crew flags",
        ),
        HeroFlowStage(
            key="target",
            order=2,
            name="Target",
            hero_headline="Seleccioná objetivo",
            hero_copy="Define límites de agua, energía y logística con presets marcianos.",
            card_body="Elegí producto final y límites de agua, energía y crew para la misión.",
            icon="🎯",
            timeline_label="Target marciano",
            timeline_description="Seleccioná producto final, límites de agua y energía, o usa presets homologados.",
            footer="Presets o límites manuales",
        ),
    ]

    metrics = [
        {
            "key": "status",
            "label": "Estado",
            "value": "✅ Modelo listo",
            "caption": "Nombre: rexai-rf-ensemble",
            "icon": "🛰️",
            "stage_key": "inventory",
            "tone": "accent",
        },
        {
            "key": "training",
            "label": "Entrenamiento",
            "value": "12 May 2045 10:00 UTC",
            "caption": "Procedencia: NASA · Muestras: 128",
            "icon": "🧪",
            "stage_key": "target",
            "tone": "info",
        },
        {
            "key": "uncertainty",
            "label": "Incertidumbre",
            "value": "±4.2%",
            "caption": "Bandas 95% expuestas",
            "icon": "📈",
            "stage_key": "results",
        },
    ]

    hero = MinimalHero(
        title="Rex-AI orquesta el reciclaje orbital y marciano",
        subtitle=(
            "Un loop autónomo que mezcla regolito MGS-1, polímeros EVA y residuos de carga "
            "para fabricar piezas listas para misión."
        ),
        icon="🛰️",
        chips=[
            {"label": "RandomForest multisalida", "tone": "accent"},
            {"label": "Comparadores heurísticos", "tone": "info"},
            {"label": "Crew telemetry ready", "tone": "accent"},
        ],
        gradient="linear-gradient(135deg, rgba(59,130,246,0.28), rgba(14,165,233,0.08))",
        glow="rgba(96,165,250,0.45)",
        density="roomy",
        metrics=metrics,
        flow=flow,
    )

    markup = hero.markup.strip()
    snapshot_path = Path(__file__).with_name("__snapshots__") / "test_home_minimal_hero.py.snap"
    expected = snapshot_path.read_text(encoding="utf-8").strip()

    assert markup == expected
    assert markup.count("class='luxe-hero luxe-hero--minimal'") == 1
    assert markup.count("class='luxe-hero__kpi'") == 2
    assert "luxe-hero__layer" not in markup
    assert "video" not in markup
