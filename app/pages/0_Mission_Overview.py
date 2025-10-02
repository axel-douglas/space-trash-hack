"""Mission overview entrypoint consolidating mission status panels."""

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.Home import render_page

render_page()
