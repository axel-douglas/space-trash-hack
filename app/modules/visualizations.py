"""Reusable visualization helpers for Rex-AI dashboards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import altair as alt
import pandas as pd
import streamlit as st


_BADGE_CSS = """
.convergence-badges{display:flex;flex-wrap:wrap;gap:12px;margin:0.4rem 0 0.8rem;}
.convergence-badge{position:relative;padding:0.65rem 0.95rem;border-radius:14px;backdrop-filter:blur(18px);color:#e2e8f0;min-width:180px;box-shadow:0 18px 36px rgba(15,23,42,0.25);background:linear-gradient(135deg,rgba(59,130,246,0.52),rgba(30,64,175,0.32));overflow:hidden;}
.convergence-badge[data-tone="iris"]{background:linear-gradient(135deg,rgba(168,85,247,0.55),rgba(67,56,202,0.32));}
.convergence-badge strong{display:block;font-size:1.4rem;font-weight:700;letter-spacing:0.01em;}
.convergence-badge span{display:block;font-size:0.78rem;letter-spacing:0.08em;text-transform:uppercase;color:rgba(226,232,240,0.82);margin-bottom:0.2rem;}
.convergence-badge::after{content:"";position:absolute;inset:-60%;background:radial-gradient(circle at center,rgba(255,255,255,0.18) 0,rgba(255,255,255,0) 65%);opacity:0;animation:badgePulse 6s ease-in-out infinite;}
.convergence-badge[data-tone="iris"]::after{animation-delay:1.2s;}
@keyframes badgePulse{0%,100%{opacity:0;transform:scale(0.65);}45%{opacity:0.8;transform:scale(1.05);}60%{opacity:0;transform:scale(1.2);}}
.convergence-copy{font-size:0.85rem;color:rgba(226,232,240,0.76);margin-bottom:0.4rem;}
"""


def _ensure_badge_css() -> None:
    key = "__convergence_badge_css__"
    if st.session_state.get(key):
        return
    st.markdown(f"<style>{_BADGE_CSS}</style>", unsafe_allow_html=True)
    st.session_state[key] = True


def _format_value(value: float | int | None, fmt: str, *, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{format(float(value), fmt)}{suffix}"


@dataclass
class ConvergenceScene:
    """Render a convergence chart with animated badges and microcopy."""

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
        base = alt.Chart(data).encode(x=alt.X("iteration:Q", title="Iteración"))

        hypervolume_line = base.mark_line(color="#38bdf8", strokeWidth=3).encode(
            y=alt.Y(
                "hypervolume:Q",
                title="Hipervolumen (↑ mejor)",
                axis=alt.Axis(titleColor="#38bdf8"),
            ),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteración", format="d"),
                alt.Tooltip("hypervolume:Q", title="Hipervolumen", format=".3f"),
            ],
        )
        hypervolume_points = base.mark_circle(color="#93c5fd", size=85, opacity=0.95).encode(
            y="hypervolume:Q",
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteración", format="d"),
                alt.Tooltip("hypervolume:Q", title="Hipervolumen", format=".3f"),
            ],
        )

        dominance_line = base.mark_line(color="#c084fc", strokeDash=[6, 3], strokeWidth=2.6).encode(
            y=alt.Y(
                "dominance_pct:Q",
                title="Dominancia Pareto (%)",
                axis=alt.Axis(titleColor="#c084fc", orient="right", format=".0f"),
            ),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteración", format="d"),
                alt.Tooltip("dominance_ratio:Q", title="Dominancia", format=".2%"),
                alt.Tooltip("dominance_pct:Q", title="Dominancia (%)", format=".1f"),
            ],
        )
        dominance_points = base.mark_square(color="#f0abfc", size=80, opacity=0.9).encode(
            y="dominance_pct:Q",
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteración", format="d"),
                alt.Tooltip("dominance_ratio:Q", title="Dominancia", format=".2%"),
                alt.Tooltip("dominance_pct:Q", title="Dominancia (%)", format=".1f"),
            ],
        )

        layered = alt.layer(hypervolume_line, hypervolume_points, dominance_line, dominance_points)
        layered = layered.resolve_scale(y="independent").properties(height=self.height)
        layered = layered.configure_axis(
            grid=True,
            gridOpacity=0.15,
            labelColor="#cbd5f5",
            titleFontWeight="bold",
            titleFontSize=12,
        ).configure_view(strokeOpacity=0)

        return layered.interactive()

    def render(self, container: st.delta_generator | None = None) -> None:
        target = container or st
        if self._prepared.empty:
            target.info("Sin datos de convergencia todavía. Ejecutá el optimizador para graficar su progreso.")
            return

        _ensure_badge_css()

        target.subheader(self.title)
        if self.subtitle:
            target.caption(self.subtitle)

        last = self._prepared.dropna(subset=["hypervolume", "dominance_ratio"], how="all").iloc[-1]
        hv = last.get("hypervolume")
        dom_pct = last.get("dominance_pct")
        pareto_size = last.get("pareto_size")

        badges = """
        <div class=\"convergence-badges\">
          <div class=\"convergence-badge\" data-tone=\"aqua\">
            <span>Hipervolumen</span>
            <strong>{_format_value(hv, '.3f')}</strong>
          </div>
          <div class=\"convergence-badge\" data-tone=\"iris\">
            <span>Dominancia Pareto</span>
            <strong>{_format_value(dom_pct, '.1f', suffix='%')}</strong>
          </div>
        </div>
        """
        target.markdown(badges, unsafe_allow_html=True)

        copy_lines: Iterable[str]
        if self.microcopy is not None:
            copy_lines = self.microcopy
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
            default_lines.append(
                "El hipervolumen refleja cuánto se expande el frente multiobjetivo; la dominancia indica qué porcentaje del "
                "pool queda superado."
            )
            copy_lines = default_lines

        for line in copy_lines:
            target.markdown(f"<div class='convergence-copy'>• {line}</div>", unsafe_allow_html=True)

        chart = self.build_chart()
        target.altair_chart(chart, use_container_width=True)
