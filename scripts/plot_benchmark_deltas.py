"""Plot benchmark deltas before vs after incorporar feedback humano."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.express as px


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de métricas: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(
        f"Formato no soportado para {path}. Usá CSV o Parquet generados con run_benchmarks."
    )


def _prepare_delta_frame(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["scenario", "target", "level"]
    metrics = ["mae", "rmse", "ci95_width_mean"]

    merged = before.merge(
        after,
        on=key_cols,
        suffixes=("_before", "_after"),
        how="inner",
    )

    records = []
    for _, row in merged.iterrows():
        for metric in metrics:
            before_value = float(row[f"{metric}_before"])
            after_value = float(row[f"{metric}_after"])
            delta = before_value - after_value
            records.append(
                {
                    "scenario": row["scenario"],
                    "target": row["target"],
                    "level": row["level"],
                    "metric": metric,
                    "before": before_value,
                    "after": after_value,
                    "delta": delta,
                }
            )

    delta_df = pd.DataFrame.from_records(records)
    if delta_df.empty:
        raise ValueError(
            "No se encontraron filas coincidentes entre las métricas base y las posteriores."
            " Verificá que ambos archivos provienen del mismo script run_benchmarks."
        )

    return delta_df


def _scenario_overall(delta_df: pd.DataFrame) -> pd.DataFrame:
    scenario_mask = (delta_df["level"] == "scenario") & (delta_df["target"] == "overall")
    scoped = delta_df[scenario_mask].copy()
    if scoped.empty:
        raise ValueError(
            "Los archivos no contienen el agregado 'target=overall, level=scenario'."
            " Ejecutá run_benchmarks >= v2 para generar columnas completas."
        )
    return scoped


def _plot(delta_df: pd.DataFrame, output: Path) -> None:
    fig = px.bar(
        delta_df,
        x="scenario",
        y="delta",
        color="metric",
        text="delta",
        barmode="group",
        title="Mejoras tras incorporar feedback (positivas = error reduce)",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis_title="Δ (Before - After)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un gráfico interactivo con las mejoras de métricas",
    )
    parser.add_argument("--before", type=Path, required=True, help="CSV/Parquet antes del feedback")
    parser.add_argument("--after", type=Path, required=True, help="CSV/Parquet después del feedback")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/feedback_deltas.html"),
        help="Archivo HTML donde guardar el gráfico (por defecto en data/benchmarks)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    before_df = _load_metrics(args.before)
    after_df = _load_metrics(args.after)
    delta_df = _prepare_delta_frame(before_df, after_df)
    scenario_df = _scenario_overall(delta_df)
    _plot(scenario_df, args.output)
    print(
        "Gráfico generado en",
        args.output,
        "(Δ positivos indican reducción de error vs heurísticas)",
    )
    return args.output


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

