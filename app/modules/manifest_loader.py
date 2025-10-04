"""Reusable helpers for manifest ingestion workflows.

The original *Aduana Inteligente* page bundled the CSV template creation and
file parsing logic directly in the Streamlit script.  Moving that
responsibility into a module makes it easier to reuse the manifest helpers
from multiple pages while keeping the UI layer lean.

The utilities defined here intentionally avoid any Streamlit dependencies so
they can be imported from unit tests or background jobs without pulling the UI
stack.  Callers are expected to handle user feedback (spinners, error
messages) on their side.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import IO, Any, Mapping, Sequence

import pandas as pd

from app.modules.generator import GeneratorService

MANIFEST_TEMPLATE_COLUMNS: tuple[str, ...] = (
    "item",
    "category",
    "mass_kg",
    "tg_loss_pct",
    "ega_loss_pct",
    "water_l_per_kg",
    "energy_kwh_per_kg",
)


def build_manifest_template() -> pd.DataFrame:
    """Return a small example manifest used as upload template.

    The dataset mirrors the layout expected by :meth:`GeneratorService.analyze_manifest`
    so operators can download it, tweak the rows and re-upload the file without
    memorising the column naming conventions.
    """

    return pd.DataFrame(
        [
            {
                "item": "HDPE packaging film",
                "category": "Packaging",
                "mass_kg": 12.5,
                "tg_loss_pct": 4.0,
                "ega_loss_pct": 0.5,
                "water_l_per_kg": 0.1,
                "energy_kwh_per_kg": 0.9,
            },
            {
                "item": "Nomex insulation",
                "category": "Structural elements",
                "mass_kg": 8.2,
                "tg_loss_pct": 2.5,
                "ega_loss_pct": 0.2,
                "water_l_per_kg": 0.0,
                "energy_kwh_per_kg": 0.4,
            },
        ],
        columns=list(MANIFEST_TEMPLATE_COLUMNS),
    )


def manifest_template_csv_bytes() -> bytes:
    """Serialize the manifest template to CSV encoded as UTF-8 bytes."""

    buffer = io.StringIO()
    build_manifest_template().to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def load_manifest_from_upload(uploaded_file: IO[bytes] | IO[str]) -> pd.DataFrame:
    """Parse the uploaded manifest file into a dataframe.

    The file pointer is rewound before returning so Streamlit keeps a pristine
    copy in case the caller wants to reuse it for another operation (for
    instance, persisting the raw payload).  ``ValueError`` is raised when the
    handle is ``None`` so callers can surface a user-friendly warning.
    """

    if uploaded_file is None:  # pragma: no cover - guarded by UI logic
        raise ValueError("Se esperaba un archivo de manifiesto para analizar")

    try:
        dataframe = pd.read_csv(uploaded_file)
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:  # pragma: no cover - defensive guard for BytesIO variants
            pass
    return dataframe


def run_policy_analysis(
    service: GeneratorService,
    manifest: pd.DataFrame | Mapping[str, Sequence[object]] | Sequence[Mapping[str, object]] | str | Path,
    *,
    include_pdf: bool = False,
) -> dict[str, Any]:
    """Delegate the heavy lifting to :class:`GeneratorService`.

    The wrapper primarily exists to keep the call-site expressive and offers a
    single import that bundles manifest parsing and policy analysis.  The
    service is injected so tests can provide a stub implementation.
    """

    return service.analyze_manifest(manifest, include_pdf=include_pdf)


__all__ = [
    "MANIFEST_TEMPLATE_COLUMNS",
    "build_manifest_template",
    "manifest_template_csv_bytes",
    "load_manifest_from_upload",
    "run_policy_analysis",
]
