"""Mission overview entrypoint consolidating mission status panels."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.Home import render_page

render_page()
