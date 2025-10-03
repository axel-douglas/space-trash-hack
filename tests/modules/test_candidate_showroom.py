import pytest

import sys
import types

sys.modules.setdefault("joblib", types.ModuleType("joblib"))
class _FakeNumpy(types.ModuleType):
    def isscalar(self, obj):  # pragma: no cover - simple heuristic
        return not isinstance(obj, (list, tuple, dict, set))

    bool_ = bool


sys.modules.setdefault("numpy", _FakeNumpy("numpy"))
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("polars", types.ModuleType("polars"))


class _FakePandas(types.ModuleType):
    def DataFrame(self, data):  # pragma: no cover - simple stub
        self.last_dataframe_input = data
        return {"rows": data}


sys.modules.setdefault("pandas", _FakePandas("pandas"))


class _StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.column_config = types.SimpleNamespace(
            NumberColumn=lambda *args, **kwargs: {"label": args[0] if args else ""},
            TextColumn=lambda *args, **kwargs: {"label": args[0] if args else ""},
            ListColumn=lambda *args, **kwargs: {"label": args[0] if args else ""},
        )
        self.__path__ = []

    def cache_resource(self, *args, **kwargs):  # pragma: no cover - simple identity decorator
        def _decorator(func):
            return func

        return _decorator

    def __getattr__(self, name):  # pragma: no cover - simple stub
        def _noop(*args, **kwargs):
            return None

        return _noop


sys.modules.setdefault("streamlit", _StreamlitStub())
streamlit_delta = types.ModuleType("streamlit.delta_generator")


class _DeltaGenerator:  # pragma: no cover - simple stub
    pass


streamlit_delta.DeltaGenerator = _DeltaGenerator
sys.modules.setdefault("streamlit.delta_generator", streamlit_delta)

ml_models_stub = types.ModuleType("app.modules.ml_models")
ml_models_stub.MODEL_REGISTRY = object()


class _DummyModelRegistry:  # pragma: no cover - simple stub
    ready = False


ml_models_stub.ModelRegistry = _DummyModelRegistry
ml_models_stub.PredictionResult = type("PredictionResult", (), {})


class _DummyRegistryCache:  # pragma: no cover - simple stub
    def __call__(self):
        return _DummyModelRegistry()

    def clear(self):
        return None


ml_models_stub.get_model_registry = _DummyRegistryCache()
sys.modules["app.modules.ml_models"] = ml_models_stub

io_stub = types.ModuleType("app.modules.io")


def _noop_io(*_args, **_kwargs):  # pragma: no cover - simple stub
    return None


for _name in [
    "load_waste_df",
    "save_waste_df",
    "load_targets",
    "load_process_catalog",
    "invalidate_waste_cache",
    "invalidate_process_cache",
    "invalidate_targets_cache",
    "invalidate_all_io_caches",
]:
    setattr(io_stub, _name, _noop_io)


sys.modules["app.modules.io"] = io_stub
sys.modules.setdefault("app.modules.mission_overview", types.ModuleType("app.modules.mission_overview"))
visual_theme_stub = types.ModuleType("app.modules.visual_theme")
visual_theme_stub.apply_global_visual_theme = lambda: None
sys.modules["app.modules.visual_theme"] = visual_theme_stub

from app.modules import candidate_showroom
from app.modules.candidate_showroom import (
    _collect_badges,
    _normalize_success,
    _prepare_rows,
    render_candidate_showroom,
)


def _base_candidate(**overrides):
    candidate = {
        "score": 0.8,
        "props": {
            "rigidity": 0.5,
            "water_l": 1.0,
            "energy_kwh": 2.0,
            "crew_min": 4,
        },
        "materials": [],
        "auxiliary": {},
        "timeline_badges": [],
        "process_id": "A1",
        "process_name": "Proceso Seguro",
    }
    candidate.update(overrides)
    return candidate


def test_prepare_rows_filters_by_score_and_safety():
    candidates = [
        _base_candidate(),
        _base_candidate(
            score=0.6,
            process_id="B1",
            process_name="Proceso Riesgo",
            materials=["PTFE"],
        ),
    ]

    rows = _prepare_rows(
        candidates,
        score_threshold=0.7,
        only_safe=True,
        threshold_active=True,
        resource_limits={"energy": 3.0, "water": 2.0, "crew": 6.0},
    )

    assert len(rows) == 1
    assert rows[0]["candidate"]["process_id"] == "A1"
    assert rows[0]["is_safe"] is True
    assert "🎯 Score ≥ 0.70" in rows[0]["badges"]


def test_prepare_rows_applies_resource_limits():
    candidates = [
        _base_candidate(score=0.9, props={"rigidity": 1.0, "water_l": 1.0, "energy_kwh": 4.0, "crew_min": 3}),
        _base_candidate(score=0.85, props={"rigidity": 0.8, "water_l": 1.2, "energy_kwh": 2.5, "crew_min": 3}),
    ]

    rows = _prepare_rows(
        candidates,
        score_threshold=0.5,
        only_safe=False,
        threshold_active=False,
        resource_limits={"energy": 3.0},
    )

    assert len(rows) == 1
    assert rows[0]["candidate"]["score"] == pytest.approx(0.85)


def test_normalize_success_variants():
    assert _normalize_success({"message": "ok", "candidate_idx": 2}) == {
        "message": "ok",
        "candidate_key": "2",
    }
    assert _normalize_success(" listo ") == {"message": " listo ", "candidate_key": None}
    assert _normalize_success(5) == {"message": "", "candidate_key": None}


def test_collect_badges_sources_and_auxiliary():
    cand = {
        "regolith_pct": 10,
        "source_categories": ["multilayer"],
    }
    aux = {"passes_seal": True}
    badges = _collect_badges(cand, aux)

    assert "⛰️ ISRU MGS-1" in badges
    assert "♻️ Valorización problemáticos" in badges
    assert "🛡️ Seal ready" in badges


def test_render_candidate_actions_uses_pill_and_chipline(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(candidate_showroom, "st", fake_st)
    monkeypatch.setattr(candidate_showroom, "action_button", lambda *_, **__: False)

    pill_calls: list[tuple[str, str]] = []

    def _record_pill(label, *, kind="ok", render=True):
        del render
        pill_calls.append((label, kind))
        return label

    chip_calls: list[list[str]] = []

    def _record_chipline(labels):
        chip_calls.append(list(labels))
        return ""

    monkeypatch.setattr(candidate_showroom, "pill", _record_pill)
    monkeypatch.setattr(candidate_showroom, "chipline", _record_chipline)

    row = {
        "process_name": "Proceso Seguro",
        "process_id": "A1",
        "score": 0.9,
        "rigidity": 0.5,
        "energy": 1.2,
        "water": 0.8,
        "crew": 3.0,
        "is_safe": True,
        "safety": {"detail": "Sin hallazgos", "level": "OK"},
        "badges": ["🎯 Score ≥ 0.80"],
        "key": "0",
        "candidate": _base_candidate(),
    }

    candidate_showroom._render_candidate_actions(1, row, {"candidate_key": None}, scenario="Test")

    assert pill_calls and pill_calls[0][0].startswith("Seguridad"), "La pastilla debe mostrar la seguridad"
    assert chip_calls and chip_calls[0] == ["🎯 Score ≥ 0.80"], "Las etiquetas deben mostrarse como chips"

class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ColumnConfig:
    class NumberColumn:  # pragma: no cover - simple data holder
        def __init__(self, label: str, **kwargs):
            self.label = label
            self.options = kwargs

    class TextColumn:  # pragma: no cover - simple data holder
        def __init__(self, label: str, **kwargs):
            self.label = label
            self.options = kwargs

    class ListColumn:  # pragma: no cover - simple data holder
        def __init__(self, label: str, **kwargs):
            self.label = label
            self.options = kwargs


class _FakeColumn:
    def __init__(self, store):
        self._store = store

    def metric(self, label, value):
        self._store.append({"label": label, "value": value})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self):
        self.session_state: dict[str, object] = {}
        self.slider_values: list[object] = []
        self.checkbox_values: list[object] = []
        self.slider_calls: list[dict[str, object]] = []
        self.checkbox_calls: list[dict[str, object]] = []
        self.caption_texts: list[str] = []
        self.subheaders: list[str] = []
        self.dataframe_calls: list[tuple[object, dict]] = []
        self.metric_calls: list[tuple[str, object]] = []
        self.column_calls: list[tuple[int, str]] = []
        self.warning_messages: list[str] = []
        self.info_messages: list[str] = []
        self.success_messages: list[str] = []
        self.column_config = _ColumnConfig

    def info(self, message):
        self.info_messages.append(message)

    def slider(self, label, *, min_value=None, max_value=None, value=None, step=None, key=None):
        call = {
            "label": label,
            "min": min_value,
            "max": max_value,
            "value": value,
            "step": step,
            "key": key,
        }
        self.slider_calls.append(call)
        if self.slider_values:
            return self.slider_values.pop(0)
        return value

    def checkbox(self, label, *, value=False, key=None, help=None):
        call = {"label": label, "value": value, "key": key, "help": help}
        self.checkbox_calls.append(call)
        if self.checkbox_values:
            return self.checkbox_values.pop(0)
        return value

    def columns(self, n, gap="small"):
        self.column_calls.append((n, gap))
        return [_FakeColumn(self.metric_calls) for _ in range(n)]

    def caption(self, text):
        self.caption_texts.append(str(text))

    def subheader(self, text):
        self.subheaders.append(str(text))

    def dataframe(self, data, **kwargs):
        self.dataframe_calls.append((data, kwargs))

    def warning(self, message):
        self.warning_messages.append(str(message))

    def success(self, message):
        self.success_messages.append(str(message))

    def metric(self, *args, **kwargs):
        label = args[0] if args else kwargs.get("label", "")
        value = args[1] if len(args) > 1 else kwargs.get("value")
        self.metric_calls.append({"label": label, "value": value})

    # Compatibility helpers for patched actions
    def expander(self, *_, **__):
        return _NullContext()

    def modal(self, *_, **__):
        return _NullContext()

    def markdown(self, *_, **__):
        return None


def test_render_candidate_showroom_native_mode(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.slider_values = [0.82, 2.3, 1.1, 5]
    fake_st.checkbox_values = [True, True, True, True]

    monkeypatch.setattr(candidate_showroom, "st", fake_st)
    monkeypatch.setattr(candidate_showroom, "_render_candidate_actions", lambda *_, **__: None)

    chip_calls: list[list[str]] = []

    def _record_chipline(labels):
        chip_calls.append(list(labels))
        return ""

    monkeypatch.setattr(candidate_showroom, "chipline", _record_chipline)

    candidates = [_base_candidate(score=0.88)]
    target = {
        "max_energy_kwh": 2.5,
        "max_water_l": 1.5,
        "max_crew_min": 5,
        "scenario": "Daring Discoveries",
    }

    visible = render_candidate_showroom(candidates, target)

    assert visible == candidates
    assert fake_st.subheaders and "Ranking" in fake_st.subheaders[0]
    assert fake_st.dataframe_calls, "Should render native dataframe"
    df_kwargs = fake_st.dataframe_calls[0][1]
    assert df_kwargs.get("hide_index") is True
    assert "column_config" in df_kwargs
    assert any("Score mínimo" in item["label"] for item in fake_st.metric_calls)
    assert "Filtros activos" in fake_st.caption_texts
    assert chip_calls, "Chipline should be used for filtros activos"
