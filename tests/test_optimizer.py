import random
import types

from app.modules import execution, optimizer


def _make_candidate(score: float) -> optimizer.Candidate:
    return {
        "score": score,
        "props": types.SimpleNamespace(
            energy_kwh=0.1,
            water_l=0.1,
            crew_min=10.0,
        ),
    }


def test_optimize_candidates_parallel_heuristic(monkeypatch):
    monkeypatch.setattr(optimizer, "_PARALLEL_THRESHOLD", 1)

    class DummyBackend(execution.ExecutionBackend):
        def __init__(self):
            super().__init__(max_workers=4)
            self.map_calls = 0
            self.submit_calls = 0
            self.shutdown_called = False

        def map(self, func, iterable):
            self.map_calls += 1
            return [func(item) for item in iterable]

        def submit(self, func, *args, **kwargs):
            self.submit_calls += 1
            result = func(*args, **kwargs)
            return types.SimpleNamespace(result=lambda: result)

        def shutdown(self):
            self.shutdown_called = True

    backend = DummyBackend()

    random.seed(42)

    def sampler(_override):
        return _make_candidate(random.random())

    initial = [_make_candidate(0.5)]
    pareto, history = optimizer.optimize_candidates(
        initial,
        sampler,
        target={},
        n_evals=3,
        backend=backend,
    )

    assert len(history) == 4
    assert history.iteration.tolist() == [0, 1, 2, 3]
    assert backend.map_calls >= 2
    assert backend.submit_calls == 0
    assert backend.shutdown_called is False
    assert pareto and all(isinstance(item, dict) for item in pareto)


def test_optimize_candidates_parallel_ax(monkeypatch):
    monkeypatch.setattr(optimizer, "_PARALLEL_THRESHOLD", 1)

    class DummyBackend(execution.ExecutionBackend):
        def __init__(self):
            super().__init__(max_workers=4)
            self.map_calls = 0
            self.submit_calls = 0
            self.shutdown_called = False

        def map(self, func, iterable):
            self.map_calls += 1
            return [func(item) for item in iterable]

        def submit(self, func, *args, **kwargs):
            self.submit_calls += 1
            result = func(*args, **kwargs)
            return types.SimpleNamespace(result=lambda: result)

        def shutdown(self):
            self.shutdown_called = True

    backend = DummyBackend()

    class DummyAxClient:
        def __init__(self, enforce_sequential_optimization: bool = True):
            self.enforce_sequential_optimization = enforce_sequential_optimization
            self.calls = 0
            self.completed: list[tuple[int, float]] = []

        def create_experiment(self, **kwargs):
            return None

        def get_next_trial(self):
            params = {
                "problematic_bias": 1.0 + 0.1 * self.calls,
                "regolith_pct": 0.05 * self.calls,
                "process_choice": "P02",
            }
            idx = self.calls
            self.calls += 1
            return params, idx

        def complete_trial(self, trial_index: int, raw_data: float):
            self.completed.append((trial_index, raw_data))

    monkeypatch.setattr(optimizer, "AX_AVAILABLE", True)
    monkeypatch.setattr(optimizer, "AxClient", DummyAxClient)

    def sampler(override):
        score = float(override.get("problematic_bias", 1.0))
        return _make_candidate(score)

    pareto, history = optimizer.optimize_candidates(
        initial_candidates=[],
        sampler=sampler,
        target={},
        n_evals=2,
        process_ids=["P01"],
        backend=backend,
    )

    assert len(history) == 3
    assert history.iteration.tolist() == [0, 1, 2]
    assert backend.submit_calls >= 2
    assert backend.map_calls >= 1
    assert backend.shutdown_called is False
    assert pareto and all(isinstance(item, dict) for item in pareto)
