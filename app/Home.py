from _entrypoint_utils import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

__doc__ = """Streamlit entrypoint that mirrors the mission overview page."""

from app.modules import mission_overview


def render_page() -> None:
    """Render the combined home + mission overview experience."""

    mission_overview.render_overview_dashboard()


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    render_page()
