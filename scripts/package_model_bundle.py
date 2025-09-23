"""Package the trained Rex-AI artefacts into a distributable zip bundle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"
METADATA_PATH = MODEL_DIR / "metadata.json"
LEGACY_METADATA_PATH = MODEL_DIR / "metadata_gold.json"
DIST_DIR = ROOT / "dist"


def _load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if LEGACY_METADATA_PATH.exists():
        return json.loads(LEGACY_METADATA_PATH.read_text(encoding="utf-8"))
    raise SystemExit("No metadata.json found. Run the training pipeline first.")


def _normalise_name(trained_at: str | None) -> str:
    if not trained_at:
        return "rexai-models"
    safe = re.sub(r"[^0-9T]", "", trained_at)
    if not safe:
        return "rexai-models"
    return f"rexai-models-{safe}"


def _gather_paths(metadata: dict) -> List[Path]:
    paths: List[Path] = []
    artefacts = metadata.get("artifacts", {})
    pipeline = artefacts.get("pipeline")
    if pipeline:
        paths.append(ROOT / pipeline)

    xgb = artefacts.get("xgboost", {}).get("path")
    if xgb:
        paths.append(ROOT / xgb)

    for optional_key in ("autoencoder", "tabtransformer"):
        opt = artefacts.get(optional_key, {})
        opt_path = opt.get("path")
        if opt_path:
            paths.append(ROOT / opt_path)

    classifiers = metadata.get("classifiers", {})
    for payload in classifiers.values():
        clf_path = payload.get("path")
        if clf_path:
            paths.append(ROOT / clf_path)

    # Always include metadata files
    paths.append(METADATA_PATH)
    if LEGACY_METADATA_PATH.exists():
        paths.append(LEGACY_METADATA_PATH)

    return paths


def _validate(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise SystemExit(f"Cannot build bundle; missing artefacts:\n{joined}")


def build_bundle(output: Path | None = None) -> Path:
    metadata = _load_metadata()
    bundle_name = _normalise_name(metadata.get("trained_at")) + ".zip"
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output or (DIST_DIR / bundle_name)

    paths = _gather_paths(metadata)
    _validate(paths)

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            arcname = path.relative_to(ROOT)
            archive.write(path, arcname)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the generated bundle")
    args = parser.parse_args()

    bundle = build_bundle(args.output)
    print(json.dumps({"bundle": str(bundle), "size_bytes": bundle.stat().st_size}, indent=2))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
