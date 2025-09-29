from __future__ import annotations
import sys
import types
from types import SimpleNamespace

import pytest

for _missing in ("joblib", "polars", "plotly", "numpy", "pandas", "requests"):
    sys.modules.setdefault(_missing, types.ModuleType(_missing))
sys.modules.setdefault("plotly.graph_objects", types.ModuleType("plotly.graph_objects"))

numpy_stub = sys.modules.get("numpy")
if numpy_stub is not None and not hasattr(numpy_stub, "isscalar"):
    numpy_stub.isscalar = lambda obj: isinstance(obj, (int, float))
if numpy_stub is not None and not hasattr(numpy_stub, "bool_"):
    numpy_stub.bool_ = bool

if "streamlit" not in sys.modules:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.session_state = {}
    streamlit_stub.markdown = lambda *a, **k: None
    streamlit_stub.info = lambda *a, **k: None
    streamlit_stub.success = lambda *a, **k: None
    streamlit_stub.warning = lambda *a, **k: None
    streamlit_stub.slider = lambda *a, **k: 0.0
    streamlit_stub.columns = lambda *a, **k: []
    streamlit_stub.caption = lambda *a, **k: None
    streamlit_stub.button = lambda *a, **k: False
    streamlit_stub.tabs = lambda labels: [SimpleNamespace() for _ in labels]
    streamlit_stub.empty = lambda: SimpleNamespace(markdown=lambda *a, **k: None)
    streamlit_stub.subheader = lambda *a, **k: None
    streamlit_stub.selectbox = lambda *a, **k: ""
    streamlit_stub.toggle = lambda *a, **k: False
    streamlit_stub.columns = lambda *a, **k: []
    streamlit_stub.modal = lambda *a, **k: SimpleNamespace(__enter__=lambda self: SimpleNamespace(), __exit__=lambda self, exc_type, exc, tb: None)
    def _cache_resource_stub(*_a, **_k):  # noqa: ANN001
        def _decorator(func):  # noqa: ANN001
            setattr(func, "clear", lambda: None)
            return func

        return _decorator

    streamlit_stub.cache_resource = _cache_resource_stub
    components_stub = types.ModuleType("streamlit.components")
    v1_stub = types.ModuleType("streamlit.components.v1")
    v1_stub.html = lambda *a, **k: None
    components_stub.v1 = v1_stub
    streamlit_stub.components = components_stub
    sys.modules["streamlit.components"] = components_stub
    sys.modules["streamlit.components.v1"] = v1_stub
    delta_stub = types.ModuleType("streamlit.delta_generator")
    delta_stub.DeltaGenerator = SimpleNamespace
    streamlit_stub.delta_generator = delta_stub
    sys.modules["streamlit.delta_generator"] = delta_stub
    sys.modules["streamlit"] = streamlit_stub

from app.modules import candidate_showroom as showroom
from app.modules import luxe_components


@pytest.fixture()
def fake_streamlit(monkeypatch):
    outputs: list[str] = []
    state: dict[str, object] = {}

    def _capture_markdown(value, *_, **__):  # noqa: ANN001
        outputs.append(value)

    def _noop_info(*args, **kwargs):  # noqa: ANN001
        if args:
            outputs.append(str(args[0]))

    stub = SimpleNamespace(
        markdown=_capture_markdown,
        info=_noop_info,
        session_state=state,
    )
    monkeypatch.setattr(luxe_components, "st", stub)
    return outputs, state


def test_timeline_hologram_builder(fake_streamlit):
    outputs, state = fake_streamlit

    candidate = {
        "process_id": "RX-01",
        "process_name": "Moldeado orbital",
        "score": 0.82,
        "props": SimpleNamespace(rigidity=0.76, water_l=0.38),
        "timeline_badges": ["🛡️ Seal ready"],
        "auxiliary": {"icon": "🛠️"},
    }

    target = {"rigidity": 0.9, "max_water_l": 1.0}

    hologram = showroom._build_timeline_hologram(  # noqa: SLF001
        [candidate],
        target,
        priority=0.6,
    )

    assert isinstance(hologram, luxe_components.TimelineHologram)
    assert hologram.priority_value == pytest.approx(0.6)
    assert hologram.items[0].icon == "🛠️"

    hologram.render()

    markup = "".join(fragment for fragment in outputs if "timeline-hologram" in fragment)
    assert "timeline-hologram__item" in markup
    assert "data-enhanced='false'" in markup
    assert "Rigidez" in markup and "Agua" in markup
    assert luxe_components._TIMELINE_HOLOGRAM_KEY in state

    outputs.clear()
    hologram.render()
    assert any("timeline-hologram" in fragment for fragment in outputs)
    # Script solo se inyecta una vez: el estado evita duplicados
    assert state[luxe_components._TIMELINE_HOLOGRAM_KEY] is True
