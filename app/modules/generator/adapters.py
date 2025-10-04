"""Adapter utilities for optional machine learning dependencies."""
from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from typing import Any, Callable, NamedTuple

from app.modules import logging_utils


class _JaxNamespace(NamedTuple):
    """Namespace container for the optional :mod:`jax` dependency."""

    jnp: Any
    jit: Callable[[Callable[..., Any]], Callable[..., Any]]


class _PyArrowNamespace(NamedTuple):
    """Namespace container for the optional :mod:`pyarrow` dependency."""

    pa: Any
    pq: Any


@lru_cache(maxsize=1)
def _load_jax_namespace() -> _JaxNamespace | None:
    """Return the imported :mod:`jax` namespace when available."""

    if importlib.util.find_spec("jax") is None:
        return None

    try:
        jax_module = importlib.import_module("jax")
        jnp_module = importlib.import_module("jax.numpy")
    except Exception:
        return None

    jit_impl = getattr(jax_module, "jit", None)
    if not callable(jit_impl):

        def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        jit_impl = _identity

    return _JaxNamespace(jnp=jnp_module, jit=jit_impl)


@lru_cache(maxsize=1)
def _load_torch_module() -> Any | None:
    """Return the :mod:`torch` module if it can be imported."""

    if importlib.util.find_spec("torch") is None:
        return None

    try:
        return importlib.import_module("torch")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_pyarrow_namespace() -> _PyArrowNamespace | None:
    """Return the :mod:`pyarrow` namespace when present."""

    pa_mod = getattr(logging_utils, "pa", None)
    pq_mod = getattr(logging_utils, "pq", None)
    if pa_mod is not None and pq_mod is not None:
        return _PyArrowNamespace(pa=pa_mod, pq=pq_mod)

    if importlib.util.find_spec("pyarrow") is None:
        return None

    try:
        pa_mod = importlib.import_module("pyarrow")
        pq_mod = importlib.import_module("pyarrow.parquet")
    except Exception:
        return None

    return _PyArrowNamespace(pa=pa_mod, pq=pq_mod)


def optional_jnp() -> Any | None:
    """Return the ``jax.numpy`` module when :mod:`jax` is installed."""

    namespace = _load_jax_namespace()
    return None if namespace is None else namespace.jnp


def optional_jit() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return :func:`jax.jit` when available, otherwise a passthrough."""

    namespace = _load_jax_namespace()
    if namespace is None:
        return lambda fn: fn
    return namespace.jit


__all__ = [
    "_load_jax_namespace",
    "_load_pyarrow_namespace",
    "_load_torch_module",
    "optional_jit",
    "optional_jnp",
]
