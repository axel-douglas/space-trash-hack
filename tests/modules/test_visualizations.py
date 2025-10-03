import pandas as pd

from app.modules.visualizations import ConvergenceScene


class DummyTarget:
    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, message: str) -> None:  # pragma: no cover - simple collector
        self.infos.append(message)


def test_convergence_scene_render_handles_missing_metrics():
    history = pd.DataFrame(
        {
            "iteration": [0, 1, 2],
            "score": [0.5, 0.6, 0.7],
        }
    )

    scene = ConvergenceScene(history)
    target = DummyTarget()

    scene.render(container=target)

    assert target.infos == [
        "Sin datos de convergencia todavía. Ejecutá el optimizador para graficar su progreso."
    ]
