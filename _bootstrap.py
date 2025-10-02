"""Compatibility shim ensuring legacy `import _bootstrap` resolves from any page."""

from app._bootstrap import PROJECT_ROOT  # noqa: F401

__all__ = ["PROJECT_ROOT"]
