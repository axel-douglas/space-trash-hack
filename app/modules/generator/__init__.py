"""Generator package exposing candidate production services."""
from __future__ import annotations

from .adapters import optional_jit, optional_jnp
from .assembly import CandidateAssembler
from .normalization import build_match_key, normalize_category, normalize_item, token_set
from . import service as _service
from .service import GeneratorService, generate_candidates

__all__ = [
    "CandidateAssembler",
    "GeneratorService",
    "build_match_key",
    "generate_candidates",
    "normalize_category",
    "normalize_item",
    "optional_jit",
    "optional_jnp",
    "token_set",
]


def __getattr__(name: str) -> object:
    return getattr(_service, name)


__all__ = sorted(set(__all__ + list(getattr(_service, "__all__", []))))
