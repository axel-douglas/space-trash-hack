import pandas as pd
import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner

from app.modules.visualizations import ConvergenceScene


def test_convergence_scene_chart_spec() -> None:
    history = pd.DataFrame(
        {
            "iteration": [2, 0, 1],
            "hypervolume": [0.55, 0.2, 0.4],
            "dominance_ratio": [0.75, 0.2, 0.5],
            "pareto_size": [6, 3, 5],
            "score": [0.92, float("nan"), 0.78],
            "penalty": [0.03, float("nan"), 0.08],
        }
    )

    scene = ConvergenceScene(history)
    prepared = scene._prepared

    assert list(prepared["iteration"]) == [0, 1, 2]
    assert pytest.approx(prepared.iloc[-1]["dominance_pct"], rel=1e-6) == 75.0
    assert prepared.iloc[-1]["pareto_size"] == 6

    spec = scene.build_chart().to_dict()
    layers = spec.get("layer", [])
    assert len(layers) == 2
    assert spec.get("resolve", {}).get("scale", {}).get("y") == "independent"

    hv_tooltips: set[str] | None = None
    dominance_tooltips: set[str] | None = None
    for layer in layers:
        encoding = layer.get("encoding", {})
        y_encoding = encoding.get("y", {})
        field = y_encoding.get("field")
        if field == "hypervolume" and hv_tooltips is None:
            hv_tooltips = {item["field"] for item in encoding.get("tooltip", [])}
        if field == "dominance_pct" and dominance_tooltips is None:
            dominance_tooltips = {item["field"] for item in encoding.get("tooltip", [])}

    assert hv_tooltips == {"iteration", "hypervolume"}
    assert dominance_tooltips is not None
    assert dominance_tooltips == {"iteration", "dominance_ratio"}


def _convergence_app(history: pd.DataFrame) -> None:
    import streamlit as st  # noqa: F401

    scene = ConvergenceScene(history)
    scene.render()


def test_convergence_scene_render_uses_metrics() -> None:
    history = pd.DataFrame(
        {
            "iteration": [0, 1],
            "hypervolume": [0.3, 0.6],
            "dominance_ratio": [0.2, 0.7],
        }
    )

    runner = StreamlitRunner(lambda: _convergence_app(history))
    app = runner.run()

    markup = "".join(
        getattr(block, "body", "")
        for block in getattr(app, "markdown", [])
        if isinstance(getattr(block, "body", ""), str)
    )
    assert "<div class='convergence-badge'>" not in markup
