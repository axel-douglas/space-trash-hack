from pathlib import Path


def test_generator_layout_copy() -> None:
    """Smoke test that ensures the generator page uses the simplified layout."""
    content = Path("app/pages/3_Generator.py").read_text(encoding="utf-8")

    assert "st.header(\"Generador asistido por IA\")" in content
    assert "Generar lote" in content
    assert "control.expander(\"Opciones avanzadas\")" in content
    assert "TeslaHero" not in content
