from __future__ import annotations

from pathlib import Path

import pytest

from app.modules.paths import DATA_ROOT

from tests.ui.test_mission_overview_page import _run_home_app, _load_inventory_fixture


def test_home_page_shows_last_modified_caption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = DATA_ROOT / "waste_inventory_sample.csv"
    assert csv_path.exists(), "El dataset de inventario debe existir bajo DATA_ROOT para la prueba"

    monkeypatch.chdir(tmp_path)

    inventory_df = _load_inventory_fixture()
    app = _run_home_app(monkeypatch, inventory_loader=lambda: inventory_df.copy(deep=True))

    caption_texts = [caption.body for caption in app.caption]
    assert any("Actualizado:" in text for text in caption_texts)
