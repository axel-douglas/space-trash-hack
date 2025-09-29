import importlib


def test_ui_blocks_smoke() -> None:
    ui_blocks = importlib.import_module("app.modules.ui_blocks")

    # Should not raise when injecting theme without HUD elements
    ui_blocks.load_theme(show_hud=False)

    # Surface context manager should be usable without errors
    with ui_blocks.surface():
        pass
