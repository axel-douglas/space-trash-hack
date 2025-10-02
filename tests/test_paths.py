from __future__ import annotations

from app.modules import paths, ui_blocks


def test_path_constants_align_with_repository_structure() -> None:
    """Smoke test ensuring filesystem constants remain in sync."""

    assert paths.DATA_ROOT.is_dir()
    assert paths.MODELS_DIR.is_dir()
    assert paths.LOGS_DIR.parent == paths.DATA_ROOT
    assert paths.LOGS_DIR.name == "logs"


def test_initialise_frontend_invokes_theme_helpers(monkeypatch) -> None:
    """``initialise_frontend`` should cascade to theme loaders in order."""

    calls: list[str] = []

    def fake_load_theme() -> None:
        calls.append("load")

    def fake_apply() -> None:
        calls.append("apply")

    monkeypatch.setattr(ui_blocks, "load_theme", fake_load_theme)
    monkeypatch.setattr(ui_blocks, "apply_global_visual_theme", fake_apply)

    ui_blocks.initialise_frontend()

    assert calls == ["load", "apply"]
