"""Ensure project utilities (paths + shared assets) are ready for the Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MICROINTERACTIONS_PATH = PROJECT_ROOT / "app" / "static" / "microinteractions.js"


def load_microinteractions_script() -> str:
    """Return the bundled microinteractions JavaScript (cached in ``app/static``)."""

    try:
        return _MICROINTERACTIONS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


__all__ = ["PROJECT_ROOT", "load_microinteractions_script"]
try:
    from app.modules.visual_theme import apply_global_visual_theme
except Exception:  # pragma: no cover - theme setup should not break imports
    pass
else:
    apply_global_visual_theme()
