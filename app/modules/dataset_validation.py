"""Dataset validation helpers used across IO and data build modules."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


REQUIRED_WASTE_COLUMNS = {
    "id",
    "category",
    "mass_kg",
    "flags",
}


class InvalidWasteDatasetError(ValueError):
    """Raised when the waste inventory dataset does not match the expected schema."""

    def __init__(self, message: str, *, issues: Iterable[str] | None = None) -> None:
        self.issues = tuple(issues or [])
        super().__init__(message)


class _WasteInventoryRow(BaseModel):
    """Pydantic model enforcing types and ranges for the NASA waste dataset."""

    id: str
    category: str
    material_family: str | None = None
    mass_kg: float = Field(ge=0)
    volume_l: float | None = None
    flags: str | None = None

    @field_validator("volume_l")
    @classmethod
    def _non_negative_volume(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("debe ser mayor o igual a 0")
        return value

    @field_validator("mass_kg")
    @classmethod
    def _non_negative_mass(cls, value: float) -> float:
        if value < 0:
            raise ValueError("debe ser mayor o igual a 0")
        return value

    model_config = {
        "extra": "ignore",
    }


def _format_validation_errors(error: ValidationError) -> tuple[str, ...]:
    messages: list[str] = []
    for issue in error.errors():
        location = issue.get("loc", ("?",))
        column = location[0] if location else "?"
        msg = issue.get("msg", "valor inválido")
        messages.append(f"{column}: {msg}")
    return tuple(messages)


def validate_waste_inventory(
    frame: pd.DataFrame,
    *,
    dataset_label: str = "el inventario de residuos",
) -> pd.DataFrame:
    """Validate *frame* using :mod:`pydantic` before it is prepared downstream."""

    missing = REQUIRED_WASTE_COLUMNS.difference(frame.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        message = (
            f"Faltan columnas obligatorias en {dataset_label}: {columns}. "
            "Verificá que el archivo tenga el esquema NASA original."
        )
        raise InvalidWasteDatasetError(message, issues=sorted(missing))

    validated_columns = ["id", "category", "material_family", "mass_kg", "volume_l", "flags"]
    subset = frame.reindex(columns=validated_columns, fill_value=None)
    subset = subset.astype(object)
    subset = subset.where(pd.notna(subset), None)
    records = subset.to_dict(orient="records")

    try:
        for record in records:
            _WasteInventoryRow.model_validate(record)
    except ValidationError as error:
        issues = _format_validation_errors(error)
        joined = "; ".join(issues)
        message = (
            f"{dataset_label.capitalize()} contiene valores inválidos: {joined}. "
            "Corregí los datos antes de volver a cargar el archivo."
        )
        raise InvalidWasteDatasetError(message, issues=issues) from error

    return frame


__all__ = [
    "InvalidWasteDatasetError",
    "validate_waste_inventory",
]
