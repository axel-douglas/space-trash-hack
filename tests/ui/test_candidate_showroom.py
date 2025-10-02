"""Behavioural tests for the candidate showroom UI component."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

from pytest_streamlit import StreamlitRunner


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

_io_stub = types.ModuleType("app.modules.io")


def _noop(*_args, **_kwargs):  # noqa: ANN002
    return None


for _name in (
    "load_waste_df",
    "save_waste_df",
    "load_targets",
    "load_process_catalog",
    "invalidate_waste_cache",
    "invalidate_process_cache",
    "invalidate_targets_cache",
    "invalidate_all_io_caches",
):
    setattr(_io_stub, _name, _noop)

sys.modules.setdefault("app.modules.io", _io_stub)

def _showroom_app() -> None:
    from types import SimpleNamespace as _NS

    import streamlit as st

    from app.modules import candidate_showroom as showroom

    candidate = {
        "process_id": "RX-01",
        "process_name": "Moldeado orbital",
        "score": 0.82,
        "props": _NS(
            rigidity=0.76,
            tightness=0.64,
            energy_kwh=1.5,
            water_l=0.38,
            crew_min=12,
        ),
        "heuristic_props": _NS(rigidity=0.72, tightness=0.6),
        "confidence_interval": {"rigidez": (0.7, 0.8), "estanqueidad": (0.6, 0.7)},
        "uncertainty": {"Rigidez": 0.04},
        "auxiliary": {"passes_seal": True},
        "materials": ["polímero"],
    }

    target = {
        "rigidity": 0.9,
        "tightness": 0.85,
        "max_energy_kwh": 2.5,
        "max_water_l": 1.0,
        "max_crew_min": 18,
    }

    st.session_state["__fixture_candidate__"] = candidate
    st.session_state["__fixture_target__"] = target

    showroom.render_candidate_showroom([candidate], target)


def test_candidate_showroom_confirm_updates_success(monkeypatch) -> None:
    from app.modules import candidate_showroom as showroom

    def _fake_check_safety(materials, process_name, process_id):  # noqa: ANN001
        return []

    def _fake_safety_badge(flags):  # noqa: ANN001
        return {"level": "Seguro", "detail": "Nominal"}

    def _fake_action_button(label, key, state="idle", **kwargs):  # noqa: ANN001, D401
        import streamlit as st

        st.session_state.setdefault("__action_states__", {})[key] = state
        return st.button(label, key=key)

    @contextmanager
    def _fake_modal(*_args, **_kwargs):  # noqa: ANN001
        yield showroom.st.container()

    monkeypatch.setattr(showroom, "check_safety", _fake_check_safety)
    monkeypatch.setattr(showroom, "safety_badge", _fake_safety_badge)
    monkeypatch.setattr(showroom, "action_button", _fake_action_button)
    monkeypatch.setattr(showroom.st, "modal", _fake_modal, raising=False)

    def _session_get(app_test, key, default=None):  # noqa: ANN001
        try:
            return app_test.session_state[key]
        except KeyError:
            return default

    runner = StreamlitRunner(_showroom_app)
    app = runner.run()

    fx_states = _session_get(app, "__action_states__", {})
    assert fx_states and fx_states.get("showroom_select_0") == "idle"

    app = app.button(key="showroom_select_0").click().run()
    fx_states = _session_get(app, "__action_states__", {})
    assert _session_get(app, "showroom_modal") == "0"

    app = app.button(key="confirm_0").click().run()
    app = app.run()
    fx_states = _session_get(app, "__action_states__", {})
    success_payload = _session_get(app, showroom._SUCCESS_KEY)  # noqa: SLF001

    assert isinstance(success_payload, dict)
    assert success_payload.get("candidate_key") == "0"
    assert "Moldeado orbital" in success_payload.get("message", "")
    assert fx_states.get("showroom_select_0") == "success"
    assert _session_get(app, "showroom_modal") is None
    selected = _session_get(app, "selected", {})
    expected_candidate = _session_get(app, "__fixture_candidate__", {})
    assert selected.get("data", {}).get("process_id") == expected_candidate.get("process_id")
