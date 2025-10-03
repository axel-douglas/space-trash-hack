from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

__doc__ = """Mission overview entrypoint consolidating mission status panels."""

from app.Home import render_page

render_page()
