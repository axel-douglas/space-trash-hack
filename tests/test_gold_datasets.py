"""Hermetic tests for :mod:`scripts.build_gold_dataset`."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pytest

from app.modules import label_mapper


gold_module = pytest.importorskip("scripts.build_gold_dataset")


def _invoke_builder(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Execute ``build_gold_dataset`` ensuring artefacts live under ``tmp_path``."""

    build_gold_dataset = getattr(gold_module, "build_gold_dataset")

    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((tmp_path,), {"return_frames": True}),
        ((), {"output_dir": tmp_path, "return_frames": True}),
        ((tmp_path,), {}),
        ((), {"output_dir": tmp_path}),
    ]

    result: Any = None
    call_args: tuple[Any, ...] | None = None
    call_kwargs: dict[str, Any] | None = None
    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            result = build_gold_dataset(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        else:
            call_args = args
            call_kwargs = kwargs
            break

    if call_args is None:
        raise RuntimeError("build_gold_dataset signature not supported") from last_error

    output_dir = _detect_output_dir(tmp_path, call_args, call_kwargs)
    features_df, labels_df = _extract_frames(result, output_dir)
    return features_df, labels_df, output_dir


def _detect_output_dir(
    tmp_path: Path, args: tuple[Any, ...], kwargs: dict[str, Any] | None
) -> Path:
    """Infer output directory based on invocation arguments."""

    if kwargs:
        for key, value in kwargs.items():
            if "dir" in key or "path" in key:
                if isinstance(value, (str, Path)):
                    return Path(value)

    if args:
        # ``build_gold_dataset`` historically accepted the target directory as the
        # first positional argument.
        candidate = args[0]
        if isinstance(candidate, (str, Path)):
            return Path(candidate)

    return tmp_path


def _extract_frames(
    result: Any, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise the return value into feature/label dataframes."""

    features_df: pd.DataFrame | None = None
    labels_df: pd.DataFrame | None = None

    if isinstance(result, tuple) and len(result) >= 2:
        features_df, labels_df = result[:2]
    elif isinstance(result, dict):
        features_df = result.get("features") or result.get("features_df")
        labels_df = result.get("labels") or result.get("labels_df")
    else:
        features_df = getattr(result, "features", None)
        labels_df = getattr(result, "labels", None)

    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.read_parquet(output_dir / "features.parquet")
    if not isinstance(labels_df, pd.DataFrame):
        labels_df = pd.read_parquet(output_dir / "labels.parquet")

    return features_df, labels_df


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        elif pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def _validate_model(model_cls: type[Any], record: dict[str, Any]) -> Any:
    payload = _clean_record(record)
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)  # type: ignore[attr-defined]
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(payload)  # type: ignore[attr-defined]
    return model_cls(**payload)


def test_build_gold_dataset_hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the gold dataset builder works entirely inside a temporary folder."""

    build_gold_dataset = getattr(gold_module, "build_gold_dataset")
    GoldFeatureRow = getattr(gold_module, "GoldFeatureRow")
    GoldLabelRow = getattr(gold_module, "GoldLabelRow")

    gold_dir = Path("datasets") / "gold"
    preexisting = set()
    if gold_dir.exists():
        preexisting = {path.resolve() for path in gold_dir.glob("*.parquet")}

    features_df, labels_df, output_dir = _invoke_builder(tmp_path)

    assert isinstance(features_df, pd.DataFrame)
    assert isinstance(labels_df, pd.DataFrame)
    assert not features_df.empty
    assert not labels_df.empty

    for record in features_df.to_dict(orient="records"):
        instance = _validate_model(GoldFeatureRow, record)
        assert getattr(instance, "recipe_id", None), "recipe_id must be populated"

    for record in labels_df.to_dict(orient="records"):
        instance = _validate_model(GoldLabelRow, record)
        source = getattr(instance, "label_source", None)
        if source is None and isinstance(instance, Mapping):
            source = instance.get("label_source")
        assert isinstance(source, str) and source.strip()

    labels_path = output_dir / "labels.parquet"
    monkeypatch.setattr(label_mapper, "GOLD_LABELS_PATH", labels_path, raising=False)
    monkeypatch.setattr(label_mapper, "_LABELS_CACHE", None, raising=False)
    monkeypatch.setattr(label_mapper, "_LABELS_CACHE_PATH", None, raising=False)

    sample_rows = labels_df.head(min(5, len(labels_df))).to_dict(orient="records")
    for record in sample_rows:
        recipe_id = str(record.get("recipe_id"))
        process_id = str(record.get("process_id"))
        targets, metadata = label_mapper.lookup_labels(
            materials=None,
            process_id=process_id,
            params={"recipe_id": recipe_id, "process_id": process_id},
        )
        assert metadata.get("label_source") == record.get("label_source")
        assert targets, "lookup_labels must return curated targets"

    if gold_dir.exists():
        post_run = {path.resolve() for path in gold_dir.glob("*.parquet")}
        assert post_run == preexisting


def test_builder_cleans_temporary_directory(tmp_path: Path) -> None:
    """The builder should not leave residual artefacts outside the temp folder."""

    features_df, labels_df, output_dir = _invoke_builder(tmp_path)

    assert output_dir == tmp_path
    assert (output_dir / "features.parquet").exists()
    assert (output_dir / "labels.parquet").exists()

    for extra in output_dir.iterdir():
        if extra.suffix.lower() == ".parquet":
            continue
        if extra.is_dir():
            # Allow builders that create intermediate folders but ensure they are empty.
            assert not any(extra.iterdir())

    assert features_df.equals(pd.read_parquet(output_dir / "features.parquet"))
    assert labels_df.equals(pd.read_parquet(output_dir / "labels.parquet"))
