from app.modules import paths


def test_path_constants_align_with_repository_structure() -> None:
    """Smoke test ensuring filesystem constants remain in sync."""

    assert paths.DATA_ROOT.is_dir()
    assert paths.MODELS_DIR.is_dir()
    assert paths.LOGS_DIR.is_dir()
    assert paths.LOGS_DIR.parent == paths.DATA_ROOT
