from pathlib import Path

from app.modules.luxe_components import BriefingCard, HeroFlowStage, TeslaHero


def test_home_hero_snapshot() -> None:
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
        HeroFlowStage(
            key="generator",
            order=3,
            name="Generador",
            hero_headline="Generá y valida",
            hero_copy="Rex-AI mezcla, explica contribuciones y exporta procesos listos para la tripulación.",
            card_body="Rex-AI mezcla ítems, sugiere proceso y explica cada predicción en vivo.",
            icon="🤖",
            timeline_label="Generador IA",
            timeline_description="Explorá mezclas óptimas, revisá contribuciones y bandas de confianza en segundos.",
            footer="ML + heurística cooperativa",
        ),
        HeroFlowStage(
            key="results",
            order=4,
            name="Resultados",
            hero_headline="Reportá y exportá",
            hero_copy="Trade-offs, confianza 95% y comparativa heurística listos para ingeniería.",
            card_body="Trade-offs, confianza 95%, comparación heurística vs IA y export final.",
            icon="📊",
            timeline_label="Resultados y export",
            timeline_description="Compará heurísticas vs IA, exportá recetas y registra feedback para retraining.",
            footer="Listo para experimentos",
        ),
    ]

    cards = [
        BriefingCard(
            title="Crew Ops + IA",
            body="La cabina recibe datos del inventario NASA, restricciones de crew-time y energía en tiempo real.",
            accent="#38bdf8",
        ),
        BriefingCard(
            title="Trazabilidad total",
            body="Cada decisión enlaza features, flags de riesgo y la receta final exportable a ingeniería.",
            accent="#a855f7",
        ),
        BriefingCard(
            title="Seguridad primero",
            body="Bandas de confianza, monitoreo de toxicidad EVA y comparadores heurísticos siempre visibles.",
            accent="#f97316",
        ),
    ]

    chips = [
        {"label": "RandomForest multisalida", "tone": "accent"},
        {"label": "Comparadores: XGBoost / Tabular", "tone": "info"},
        {"label": "Bandas de confianza 95%", "tone": "accent"},
        {"label": "Telemetría NASA · Crew safe", "tone": "info"},
    ]

    scene = TeslaHero.with_briefing(
        title="Rex-AI orquesta el reciclaje orbital y marciano",
        subtitle=(
            "Un loop autónomo que mezcla regolito MGS-1, polímeros EVA y residuos de carga "
            "para fabricar piezas listas para misión. El copiloto gestiona riesgos, "
            "energía y trazabilidad sin perder contexto."
        ),
        tagline="Sincronizá sensores, crew y modelo para reciclar basura orbital en hardware vital.",
        video_url="https://cdn.coverr.co/videos/coverr-into-the-blue-nebula-9071/1080p.mp4",
        chips=chips,
        icon="🛰️",
        gradient="linear-gradient(135deg, rgba(59,130,246,0.28), rgba(14,165,233,0.08))",
        glow="rgba(96,165,250,0.45)",
        density="roomy",
        parallax_icons=[
            {"icon": "🛰️", "top": "8%", "left": "74%", "size": "4.8rem", "speed": "22s"},
            {"icon": "🪐", "top": "62%", "left": "80%", "size": "5.2rem", "speed": "28s"},
            {"icon": "✨", "top": "20%", "left": "12%", "size": "3.2rem", "speed": "18s"},
        ],
        flow=flow,
        briefing_cards=cards,
        metrics=[],
        render=False,
    )

    markup = scene.markup.strip()
    snapshot_path = Path(__file__).with_name("__snapshots__") / "test_home_hero.py.snap"
    expected = snapshot_path.read_text(encoding="utf-8").strip()

    assert markup == expected
    assert markup.count("class='luxe-hero'") == 1
    assert "Rex-AI orquesta el reciclaje orbital y marciano" in markup
    assert "Sincronizá sensores, crew y modelo" in markup
    for stage in flow:
        assert stage.hero_headline in markup
