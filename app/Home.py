from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

_PROJECT_ROOT = ensure_streamlit_entrypoint(__file__)

__doc__ = """Streamlit entrypoint that mirrors the mission overview page."""

from app.modules import mission_overview


def render_page() -> None:
    """Render the combined home + mission overview experience."""

    mission_overview.render_overview_dashboard()


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    render_page()
