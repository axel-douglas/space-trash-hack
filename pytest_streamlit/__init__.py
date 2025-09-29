"""Lightweight helpers to drive Streamlit apps in tests.

This local shim emulates the minimal API surface from the external
``pytest-streamlit`` package that our tests expect.  It is intentionally
small: we only expose a ``StreamlitRunner`` wrapper built on top of
``streamlit.testing.v1.AppTest`` so tests can trigger reruns and inspect
session state without spinning a browser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from streamlit.testing.v1 import AppTest

__all__ = ["StreamlitRunner"]


@dataclass
class StreamlitRunner:
    """Minimal runner to exercise Streamlit apps in pytest.

    Parameters
    ----------
    app : Callable
        A callable that builds the Streamlit UI.
    args : Iterable[Any] | None
        Positional arguments forwarded to ``app``.
    kwargs : Mapping[str, Any] | None
        Keyword arguments forwarded to ``app``.
    """

    app: Callable[..., Any]
    args: Iterable[Any] | None = None
    kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:  # pragma: no cover - defensive wiring
        self._test = AppTest.from_function(
            self.app,
            args=tuple(self.args or ()),
            kwargs=dict(self.kwargs or {}),
        )

    def run(self) -> AppTest:
        """Run or rerun the wrapped Streamlit app and return the ``AppTest``."""

        return self._test.run()

    @property
    def app_test(self) -> AppTest:
        """Expose the underlying ``AppTest`` instance for advanced usage."""

        return self._test

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._test, item)
