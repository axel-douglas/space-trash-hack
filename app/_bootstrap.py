"""Ensure the project root is available on ``sys.path`` when running from ``app/``."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app.modules.visual_theme import apply_global_visual_theme
except Exception:  # pragma: no cover - theme setup should not break imports
    pass
else:
    apply_global_visual_theme()
