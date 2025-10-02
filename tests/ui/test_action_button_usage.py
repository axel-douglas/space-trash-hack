from __future__ import annotations

import sys
import types
from importlib import util
from pathlib import Path
from typing import Any

import pytest


for _missing in ("joblib", "plotly"):
    sys.modules.setdefault(_missing, types.ModuleType(_missing))
sys.modules.setdefault("plotly.graph_objects", types.ModuleType("plotly.graph_objects"))

_polars = types.ModuleType("polars")


class _DummyExpr:
    def cast(self, *_args, **_kwargs):  # noqa: ANN002
        return self

    def alias(self, *_args, **_kwargs):  # noqa: ANN002
        return self

    def cum_sum(self, *_args, **_kwargs):  # noqa: ANN002
        return self

    def __truediv__(self, *_args, **_kwargs):
        return self

    def __rtruediv__(self, *_args, **_kwargs):
        return self

    def __rsub__(self, *_args, **_kwargs):
        return self

    def __gt__(self, *_args, **_kwargs):
        return self


class _DummyFrame:
    def __init__(self) -> None:
        self.height = 0

    def rename(self, *_args, **_kwargs):  # noqa: ANN002
        return self

    def select(self, *_args, **_kwargs):  # noqa: ANN002
        return self

    def sort(self, *_args, **_kwargs):  # noqa: ANN002
        return self

    def with_columns(self, *_args, **_kwargs):  # noqa: ANN002
        return self

    def filter(self, *_args, **_kwargs):  # noqa: ANN002
        return self


_polars.read_csv = lambda *_args, **_kwargs: _DummyFrame()  # type: ignore[attr-defined]
_polars.DataFrame = lambda *_args, **_kwargs: _DummyFrame()  # type: ignore[attr-defined]
_polars.Series = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
_polars.Float64 = float  # type: ignore[attr-defined]
_polars.col = lambda *_args, **_kwargs: _DummyExpr()  # type: ignore[attr-defined]
sys.modules.setdefault("polars", _polars)


@pytest.fixture()
def ui_blocks(monkeypatch):
    spec = util.spec_from_file_location(
        "ui_blocks_test_module",
        Path(__file__).resolve().parents[2] / "app/modules/ui_blocks.py",
    )
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_theme", lambda show_hud=False: None)
    return module


def test_action_button_uses_streamlit_primitives(ui_blocks, monkeypatch):
    button_calls: list[dict[str, Any]] = []
    status_events: list[dict[str, Any]] = []
    captions: list[str] = []

    def fake_button(label, **kwargs):  # noqa: ANN001
        call = {"label": label, **kwargs}
        button_calls.append(call)
        return False

    class DummyStatus:
        def __init__(self, label: str, state: str = "info") -> None:
            status_events.append({"event": "create", "label": label, "state": state})

        def update(self, label: str | None = None, state: str | None = None, expanded: bool | None = None) -> None:
            status_events.append(
                {
                    "event": "update",
                    "label": label,
                    "state": state,
                    "expanded": expanded,
                }
            )

    monkeypatch.setattr(ui_blocks.st, "button", fake_button)
    monkeypatch.setattr(ui_blocks.st, "status", lambda label, state="info": DummyStatus(label, state))
    monkeypatch.setattr(ui_blocks.st, "caption", lambda text: captions.append(text))

    ui_blocks.action_button(
        "Lanzar\nsecuencia orbital",
        key="demo_fx_cta",
        icon="🚀",
        state="loading",
        loading_label="Sincronizando cápsula…",
        success_label="Órbita establecida",
        status_hints={
            "idle": "Listo para despegar",
            "loading": "Ajustando vector",
            "success": "Secuencia completada",
            "error": "Interrupción detectada",
        },
        help_text="Demo de secuencia orbital",
    )

    assert button_calls, "se esperaba que action_button llamara a st.button"
    call = button_calls[0]
    assert call["label"] == "🚀 Sincronizando cápsula…"
    assert call["key"] == "demo_fx_cta"
    assert call["disabled"] is True
    assert call["use_container_width"] is True
    assert captions == ["Demo de secuencia orbital"]

    assert status_events[0] == {"event": "create", "label": "Ajustando vector", "state": "running"}
    assert status_events[1] == {"event": "update", "label": "Ajustando vector", "state": "running", "expanded": False}


def test_action_button_supports_downloads(ui_blocks, monkeypatch):
    button_calls: list[dict[str, Any]] = []
    status_events: list[dict[str, Any]] = []

    def fake_download(label, **kwargs):  # noqa: ANN001
        call = {"label": label, **kwargs}
        button_calls.append(call)
        return True

    class DummyStatus:
        def __init__(self, label: str, state: str = "info") -> None:
            status_events.append({"event": "create", "label": label, "state": state})

        def update(self, label: str | None = None, state: str | None = None, expanded: bool | None = None) -> None:
            status_events.append(
                {
                    "event": "update",
                    "label": label,
                    "state": state,
                    "expanded": expanded,
                }
            )

    monkeypatch.setattr(ui_blocks.st, "download_button", fake_download)
    monkeypatch.setattr(ui_blocks.st, "status", lambda label, state="info": DummyStatus(label, state))

    result = ui_blocks.action_button(
        "Exportar inventario",
        key="export_inventory",
        icon="💾",
        state="success",
        success_label="Inventario guardado",
        status_hints={"success": "Inventario actualizado"},
        download_data=b"id,name\n1,Demo",
        download_file_name="inventory.csv",
        download_mime="text/csv",
    )

    assert result is True
    assert button_calls and button_calls[0]["label"] == "💾 Inventario guardado"
    assert button_calls[0]["file_name"] == "inventory.csv"
    assert status_events[0] == {"event": "create", "label": "Inventario actualizado", "state": "complete"}
    assert status_events[1] == {"event": "update", "label": "Inventario actualizado", "state": "complete", "expanded": False}
