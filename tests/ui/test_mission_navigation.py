from app.modules import navigation


def test_breadcrumb_labels_stop_at_active_step() -> None:
    step = navigation.get_step("generator")
    labels = navigation.breadcrumb_labels(step)

    assert labels[-1] == "Generador asistido"
    assert labels[:2] == ["Home", "Definir objetivo"]
    assert "Scenario Playbooks" not in " ".join(labels)


def test_iter_steps_respects_order() -> None:
    keys = [step.key for step in navigation.iter_steps()]
    assert keys == [
        "home",
        "target",
        "generator",
        "results",
        "compare",
        "export",
        "playbooks",
        "feedback",
        "capacity",
    ]
