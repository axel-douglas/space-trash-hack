from app.modules import navigation


def test_format_stepper_uses_spanish_labels() -> None:
    step = navigation.get_step("results")
    summary = navigation.format_stepper(step)

    assert summary.startswith("Paso 4 de")
    assert summary.endswith("Resultados (legacy)")


def test_set_active_step_invalid_key_raises(monkeypatch) -> None:
    # Bypass Streamlit session state to focus on validation.
    monkeypatch.setattr(navigation, "st", type("S", (), {"session_state": {}})())

    try:
        navigation.set_active_step("unknown")
    except KeyError as exc:
        assert "not defined" in str(exc)
    else:  # pragma: no cover - sanity guard
        raise AssertionError("set_active_step should reject unknown keys")
