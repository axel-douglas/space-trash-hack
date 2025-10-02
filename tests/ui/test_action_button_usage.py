from __future__ import annotations

import sys
import types
from importlib import util
from pathlib import Path
from typing import Any

import pytest


sys.modules.setdefault("joblib", types.ModuleType("joblib"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

modules_pkg = types.ModuleType("app.modules")
modules_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("app.modules", modules_pkg)

visual_theme_stub = types.ModuleType("app.modules.visual_theme")
visual_theme_stub.apply_global_visual_theme = lambda: None  # type: ignore[attr-defined]
sys.modules.setdefault("app.modules.visual_theme", visual_theme_stub)
modules_pkg.visual_theme = visual_theme_stub  # type: ignore[attr-defined]

try:  # prefer the real dependency when available
    import plotly  # type: ignore[import]

    # Ensure the submodules used in the UI helpers are importable during the test
    import plotly.graph_objects  # noqa: F401  # type: ignore[import]
    import plotly.io  # noqa: F401  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    stub = types.ModuleType("plotly")
    sys.modules.setdefault("plotly", stub)
    sys.modules.setdefault("plotly.graph_objects", types.ModuleType("plotly.graph_objects"))
    sys.modules.setdefault("plotly.io", types.ModuleType("plotly.io"))

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


class _DummyStatusWidget:
    def update(self, *_args, **_kwargs):  # noqa: ANN002
        return None


class _DummyDeltaGenerator:
    def container(self):  # noqa: D401
        return self

    def markdown(self, *_args, **_kwargs):  # noqa: ANN002
        return None

    def caption(self, *_args, **_kwargs):  # noqa: ANN002
        return None

    def columns(self, *_args, **_kwargs):  # noqa: ANN002
        return []

    def expander(self, *_args, **_kwargs):  # noqa: ANN002
        return _DummyContext()


class _DummyContext(_DummyDeltaGenerator):
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, *_args):  # noqa: ANN002
        return False


_streamlit = types.ModuleType("streamlit")
_streamlit.button = lambda *args, **kwargs: False  # type: ignore[assignment]
_streamlit.download_button = lambda *args, **kwargs: False  # type: ignore[assignment]
_streamlit.status = lambda *args, **kwargs: _DummyStatusWidget()  # type: ignore[assignment]
_streamlit.caption = lambda *args, **kwargs: None  # type: ignore[assignment]
_streamlit.markdown = lambda *args, **kwargs: None  # type: ignore[assignment]
_streamlit.container = lambda *args, **kwargs: _DummyDeltaGenerator()  # type: ignore[assignment]
_streamlit.columns = lambda *args, **kwargs: []  # type: ignore[assignment]
_streamlit.empty = lambda *args, **kwargs: _DummyDeltaGenerator()  # type: ignore[assignment]
_streamlit.spinner = lambda *args, **kwargs: _DummyContext()  # type: ignore[assignment]
_streamlit.session_state = {}
sys.modules.setdefault("streamlit", _streamlit)

_delta_mod = types.ModuleType("streamlit.delta_generator")
_delta_mod.DeltaGenerator = _DummyDeltaGenerator  # type: ignore[attr-defined]
sys.modules.setdefault("streamlit.delta_generator", _delta_mod)


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


def test_action_button_loading_state_uses_status(ui_blocks, monkeypatch):
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


def test_action_button_error_state_shows_feedback(ui_blocks, monkeypatch):
    button_calls: list[dict[str, Any]] = []
    status_events: list[dict[str, Any]] = []

    def fake_button(label, **kwargs):  # noqa: ANN001
        button_calls.append({"label": label, **kwargs})
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

    ui_blocks.action_button(
        "Intentar de nuevo",
        key="retry_button",
        state="error",
        error_label="Fallo",
        status_hints={"error": "Se produjo un error"},
    )

    assert button_calls
    assert status_events[0] == {"event": "create", "label": "Se produjo un error", "state": "error"}
    assert status_events[1] == {"event": "update", "label": "Se produjo un error", "state": "error", "expanded": False}
