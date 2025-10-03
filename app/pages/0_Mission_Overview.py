"""Mission overview entrypoint consolidating mission status panels."""

from app.bootstrap import ensure_streamlit_imports

ensure_streamlit_imports(__file__)

from app.bootstrap import ensure_streamlit_path

ensure_streamlit_path(__file__)

from app.Home import render_page

render_page()
