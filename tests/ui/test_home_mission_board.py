from app.modules.luxe_components import MissionBoard


def test_mission_board_markup_highlights_active_step() -> None:
    payload = [
        {
            "key": "inventory",
            "title": "Inventario",
            "description": "Normalizá residuos NASA y marcá flags EVA o multilayer.",
            "href": "./?page=1_Inventory_Builder",
            "icon": "🧱",
        },
        {
            "key": "target",
            "title": "Target",
            "description": "Define objetivo y límites de agua/energía.",
            "href": "./?page=2_Target_Designer",
            "icon": "🎯",
        },
        {
            "key": "generator",
            "title": "Generador",
            "description": "Compará recetas IA vs heurística y valida contribuciones.",
            "href": "./?page=3_Generator",
            "icon": "🤖",
        },
    ]

    board = MissionBoard.from_payload(payload, title="Próxima acción")
    markup = board.markup(highlight_key="generator")

    assert markup.count("<li") == len(payload)
    assert "<ol" in markup and "</ol>" in markup
    assert "data-key='generator'" in markup
    active_segment = markup.split("data-key='generator'", 1)[0].split("<li")[-1]
    assert "is-active" in active_segment
    assert "./?page=3_Generator" in markup
    assert "class='mission-board__badge'>1<" in markup
