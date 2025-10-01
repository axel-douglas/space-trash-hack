"""Reusable visualization helpers for Rex-AI dashboards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import altair as alt
import pandas as pd
import streamlit as st


def _format_value(value: float | int | None, fmt: str, *, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{format(float(value), fmt)}{suffix}"


@dataclass
class ConvergenceScene:
    """Render a convergence chart with KPI metrics and microcopy."""

    history_df: pd.DataFrame
    title: str = "Convergencia del optimizador"
    subtitle: str = "Seguimiento en vivo de hipervolumen y dominancia del frente Pareto."
    microcopy: Sequence[str] | None = None
    height: int = 320
    _prepared: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._prepared = self._prepare_history(self.history_df)

    @staticmethod
    def _prepare_history(history_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(history_df, pd.DataFrame):
            return pd.DataFrame()

        df = history_df.copy()
        if df.empty:
            return df

        for column in ("iteration", "hypervolume", "dominance_ratio"):
            if column not in df.columns:
                df[column] = pd.NA

        df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
        if df["iteration"].isna().all():
            df["iteration"] = range(len(df))
        df["iteration"] = df["iteration"].ffill().bfill()

        df["hypervolume"] = pd.to_numeric(df["hypervolume"], errors="coerce")
        df["dominance_ratio"] = pd.to_numeric(df["dominance_ratio"], errors="coerce")

        if "pareto_size" not in df.columns:
            df["pareto_size"] = pd.NA
        if "score" not in df.columns:
            df["score"] = pd.NA
        if "penalty" not in df.columns:
            df["penalty"] = pd.NA

        df = df.sort_values("iteration").reset_index(drop=True)
        df["dominance_pct"] = df["dominance_ratio"] * 100
        return df.dropna(subset=["hypervolume", "dominance_ratio"], how="all")

    def build_chart(self) -> alt.Chart:
        if self._prepared.empty:
            return alt.Chart(pd.DataFrame({"iteration": [], "value": []}))

        data = self._prepared
        base = alt.Chart(data).encode(
            x=alt.X(
                "iteration:Q",
                title="Iteración",
                axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8"),
            )
        )

        hypervolume_line = base.mark_line(color="#38bdf8", strokeWidth=2.4).encode(
            y=alt.Y(
                "hypervolume:Q",
                title="Hipervolumen (↑ mejor)",
                axis=alt.Axis(titleColor="#38bdf8", labelColor="#cbd5f5"),
            ),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteración", format="d"),
                alt.Tooltip("hypervolume:Q", title="Hipervolumen", format=".3f"),
            ],
        )
        dominance_line = base.mark_line(color="#c084fc", strokeWidth=2.2).encode(
            y=alt.Y(
                "dominance_pct:Q",
                title="Dominancia Pareto (%)",
                axis=alt.Axis(titleColor="#c084fc", orient="right", format=".0f"),
            ),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteración", format="d"),
                alt.Tooltip("dominance_ratio:Q", title="Dominancia", format=".1%"),
            ],
        )
        layered = alt.layer(hypervolume_line, dominance_line)
        layered = layered.resolve_scale(y="independent").properties(height=self.height)
        layered = layered.configure_axis(
            grid=False,
            labelColor="#cbd5f5",
            titleFontWeight="bold",
            titleFontSize=12,
            ticks=False,
        ).configure_view(strokeOpacity=0)

        return layered.interactive()

    def render(self, container: st.delta_generator | None = None) -> None:
        target = container or st
        if self._prepared.empty:
            target.info("Sin datos de convergencia todavía. Ejecutá el optimizador para graficar su progreso.")
            return

        target.subheader(self.title)
        if self.subtitle:
            target.caption(self.subtitle)

        last = self._prepared.dropna(subset=["hypervolume", "dominance_ratio"], how="all").iloc[-1]
        hv = last.get("hypervolume")
        dom_ratio = last.get("dominance_ratio")
        pareto_size = last.get("pareto_size")

        col_hv, col_dom = target.columns(2)
        col_hv.metric("Hipervolumen", _format_value(hv, ".3f"))
        col_dom.metric("Dominancia Pareto", _format_value(dom_ratio, ".1%"))

        copy_lines: Sequence[str]
        if self.microcopy is not None:
            copy_lines = list(self.microcopy)[:2]
        else:
            default_lines: list[str] = []
            iteration = int(last.get("iteration", 0))
            if pd.notna(pareto_size):
                default_lines.append(
                    f"Pareto activo con **{int(float(pareto_size))}** soluciones en la iteración **{iteration}**."
                )
            else:
                default_lines.append(f"Iteración **{iteration}** con recalculo dinámico del frente Pareto.")

            score = last.get("score")
            penalty = last.get("penalty")
            if pd.notna(score):
                penalty_text = _format_value(penalty, '.3f') if pd.notna(penalty) else "—"
                default_lines.append(
                    f"Último score evaluado: **{float(score):.3f}** (penalización {penalty_text})."
                )
            else:
                default_lines.append(
                    "El hipervolumen resume la expansión del frente multiobjetivo y la dominancia el porcentaje superado."
                )
            copy_lines = default_lines[:2]

        for line in copy_lines:
            target.markdown(f"- {line}")

        chart = self.build_chart()
        target.altair_chart(chart, use_container_width=True)
