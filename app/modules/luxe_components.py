"""High fidelity UI components for the Rex-AI Streamlit app.

This module centralises visually rich components (hero, metrics, glass cards)
so that pages can consume declarative helpers instead of raw HTML blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import streamlit as st


# ---------------------------------------------------------------------------
# CSS bootstrap
# ---------------------------------------------------------------------------
_LUXE_CSS = """
<style>
:root {
  --luxe-surface: rgba(12, 17, 27, 0.78);
  --luxe-border: rgba(148, 163, 184, 0.32);
  --luxe-border-strong: rgba(148, 163, 184, 0.48);
  --luxe-ink: #e9f0ff;
  --luxe-muted: rgba(226, 232, 240, 0.76);
  --luxe-accent: #60a5fa;
  --luxe-positive: #34d399;
  --luxe-warning: #f59e0b;
  --luxe-danger: #f87171;
}

@keyframes heroGlow {
  0% { box-shadow: 0 25px 70px rgba(96, 165, 250, 0.24); }
  50% { box-shadow: 0 35px 110px rgba(56, 189, 248, 0.45); }
  100% { box-shadow: 0 25px 70px rgba(96, 165, 250, 0.24); }
}

@keyframes parallaxDrift {
  0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.45; }
  50% { transform: translate3d(12px, -10px, 0) scale(1.05); opacity: 0.75; }
  100% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.45; }
}

@keyframes sparkleShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.luxe-hero {
  position: relative;
  overflow: hidden;
  border-radius: 32px;
  border: 1px solid var(--luxe-border);
  background: var(--hero-gradient, linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(14, 165, 233, 0.06)));
  padding: var(--hero-padding, 2.6rem 3.1rem);
  color: var(--luxe-ink);
  isolation: isolate;
  backdrop-filter: blur(18px);
  box-shadow: 0 24px 60px rgba(8, 15, 35, 0.45);
  animation: heroGlow 14s ease-in-out infinite;
}

.luxe-hero::after {
  content: "";
  position: absolute;
  inset: -30%;
  background: radial-gradient(circle at 20% 20%, var(--hero-glow, rgba(96, 165, 250, 0.4)), transparent 62%);
  z-index: -2;
  filter: blur(2px);
  animation: sparkleShift 28s ease infinite;
}

.luxe-hero__layer {
  position: absolute;
  pointer-events: none;
  opacity: 0.55;
  font-size: var(--layer-size, 4rem);
  animation: parallaxDrift var(--layer-speed, 18s) ease-in-out infinite;
  mix-blend-mode: screen;
}

.luxe-hero__content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.luxe-hero__icon {
  font-size: 2.2rem;
  align-self: flex-start;
  filter: drop-shadow(0 0 10px rgba(148, 163, 184, 0.45));
}

.luxe-hero h1 {
  font-size: clamp(2.15rem, 4vw, 2.9rem);
  margin: 0;
  letter-spacing: 0.015em;
}

.luxe-hero p {
  margin: 0;
  color: var(--luxe-muted);
  font-size: 1.05rem;
  max-width: 46rem;
}

.luxe-chip-row {
  display: flex;
  gap: var(--chip-gap, 0.6rem);
  flex-wrap: wrap;
  margin-top: var(--chip-margin-top, 0.6rem);
}

.luxe-chip {
  border-radius: 999px;
  padding: var(--chip-padding, 0.38rem 0.9rem);
  border: 1px solid var(--chip-border, rgba(148, 163, 184, 0.35));
  background: var(--chip-bg, rgba(15, 23, 42, 0.55));
  color: var(--chip-ink, var(--luxe-ink));
  font-size: var(--chip-size, 0.82rem);
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  backdrop-filter: blur(6px);
}

.luxe-chip[data-tone="accent"] { background: rgba(96, 165, 250, 0.16); border-color: rgba(125, 211, 252, 0.45); }
.luxe-chip[data-tone="info"] { background: rgba(14, 165, 233, 0.16); border-color: rgba(56, 189, 248, 0.5); }
.luxe-chip[data-tone="positive"] { background: rgba(52, 211, 153, 0.14); border-color: rgba(52, 211, 153, 0.45); }
.luxe-chip[data-tone="warning"] { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.45); }

.luxe-metric-galaxy {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--metric-min, 13rem), 1fr));
  gap: var(--metric-gap, 1rem);
}

.luxe-metric {
  border-radius: 22px;
  border: 1px solid var(--luxe-border);
  background: rgba(13, 17, 23, 0.76);
  padding: var(--metric-padding, 1.2rem 1.4rem);
  box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.1), 0 18px 40px rgba(8, 15, 35, 0.38);
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  position: relative;
  overflow: hidden;
}

.luxe-metric[data-glow="true"]::after {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 15% 20%, rgba(96, 165, 250, 0.24), transparent 65%);
  opacity: 0.85;
  z-index: 0;
  pointer-events: none;
}

.luxe-metric__icon {
  font-size: 1.15rem;
  opacity: 0.7;
}

.luxe-metric__label {
  font-size: 0.78rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  opacity: 0.72;
}

.luxe-metric__value {
  font-size: 1.48rem;
  font-weight: 700;
}

.luxe-metric__delta {
  font-size: 0.82rem;
  opacity: 0.75;
}

.luxe-metric__caption {
  font-size: 0.82rem;
  color: var(--luxe-muted);
}

.luxe-stack {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--stack-min, 16rem), 1fr));
  gap: var(--stack-gap, 1.1rem);
  margin: var(--stack-margin, 1.6rem 0 0);
}

.luxe-card {
  position: relative;
  border-radius: 22px;
  border: 1px solid var(--luxe-border);
  background: rgba(12, 17, 27, 0.72);
  padding: var(--card-padding, 1.3rem 1.4rem);
  box-shadow: 0 18px 40px rgba(8, 15, 35, 0.35);
  overflow: hidden;
  backdrop-filter: blur(18px);
  color: var(--luxe-ink);
}

.luxe-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.14), transparent 55%);
  opacity: 0.75;
  pointer-events: none;
}

.luxe-card__icon {
  font-size: 1.4rem;
  margin-bottom: 0.4rem;
}

.luxe-card__title {
  font-size: 1.05rem;
  margin: 0 0 0.45rem 0;
}

.luxe-card__body {
  font-size: 0.92rem;
  color: var(--luxe-muted);
}

.luxe-card__footer {
  margin-top: 0.7rem;
  font-size: 0.8rem;
  color: rgba(226, 232, 240, 0.64);
}
</style>
"""


def _inject_css() -> None:
  """Inject the shared CSS rules once per session."""
  if st.session_state.get("_luxe_css_injected"):
      return
  st.markdown(_LUXE_CSS, unsafe_allow_html=True)
  st.session_state["_luxe_css_injected"] = True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _merge_styles(base: Mapping[str, str], extra: Mapping[str, str] | None = None) -> str:
  merged: dict[str, str] = dict(base)
  if extra:
      merged.update({k: v for k, v in extra.items() if v is not None})
  return "; ".join(f"{k}: {v}" for k, v in merged.items())


def ChipRow(
    chips: Sequence[str | Mapping[str, str]],
    *,
    tone: str | None = None,
    size: str = "md",
    render: bool = True,
    gap: str | None = None,
) -> str:
  """Render a reusable pill row.

  Parameters
  ----------
  chips:
      Sequence with either plain strings or dicts containing ``label`` and
      optional ``icon``/``tone`` overrides.
  tone:
      Base tone applied to chips (``accent``, ``info``, ``positive``...).
  size:
      ``"sm"``, ``"md"`` or ``"lg"``; controls padding and font size.
  render:
      When ``True`` the HTML is pushed to Streamlit, otherwise only returned.
  gap:
      Optional override for the spacing between chips.
  """
  _inject_css()

  size_map: Mapping[str, tuple[str, str]] = {
      "sm": ("0.32rem 0.8rem", "0.74rem"),
      "md": ("0.38rem 0.9rem", "0.82rem"),
      "lg": ("0.45rem 1.05rem", "0.92rem"),
  }
  padding, font_size = size_map.get(size, size_map["md"])
  row_style = {
      "--chip-padding": padding,
      "--chip-size": font_size,
  }
  if gap:
      row_style["--chip-gap"] = gap

  html = [f"<div class='luxe-chip-row' style='{_merge_styles(row_style, {})}'>"]
  for item in chips:
      if isinstance(item, Mapping):
          label = item.get("label", "")
          icon = item.get("icon")
          chip_tone = item.get("tone", tone)
      else:
          label = str(item)
          icon = None
          chip_tone = tone
      icon_fragment = f"<span>{icon}</span>" if icon else ""
      html.append(
          f"<span class='luxe-chip' data-tone='{chip_tone or ''}'>"
          f"{icon_fragment}<span>{label}</span>"
          "</span>"
      )
  html.append("</div>")
  html_markup = "".join(html)
  if render:
      st.markdown(html_markup, unsafe_allow_html=True)
  return html_markup


# ---------------------------------------------------------------------------
# TeslaHero
# ---------------------------------------------------------------------------
@dataclass
class TeslaHero:
  title: str
  subtitle: str
  chips: Sequence[str | Mapping[str, str]] = field(default_factory=list)
  icon: str | None = None
  gradient: str | None = None
  glow: str | None = None
  density: str = "cozy"
  parallax_icons: Sequence[Mapping[str, str]] = field(default_factory=list)

  def render(self) -> None:
      _inject_css()
      padding_map = {
          "compact": "1.9rem 2.2rem",
          "cozy": "2.5rem 2.9rem",
          "roomy": "3.1rem 3.4rem",
      }
      padding = padding_map.get(self.density, padding_map["cozy"])
      hero_style = {
          "--hero-padding": padding,
      }
      if self.gradient:
          hero_style["--hero-gradient"] = self.gradient
      if self.glow:
          hero_style["--hero-glow"] = self.glow

      layers = []
      for idx, layer in enumerate(self.parallax_icons):
          icon = layer.get("icon", "✦")
          top = layer.get("top", f"{10 + idx * 12}%")
          left = layer.get("left", f"{55 + idx * 8}%")
          size = layer.get("size", "4rem")
          speed = layer.get("speed", f"{16 + idx * 4}s")
          layers.append(
              f"<span class='luxe-hero__layer' style='top:{top};left:{left};--layer-size:{size};--layer-speed:{speed};'>"
              f"{icon}</span>"
          )

      chips_html = ChipRow(self.chips, render=False) if self.chips else ""
      icon_html = f"<div class='luxe-hero__icon'>{self.icon}</div>" if self.icon else ""

      html = f"""
      <div class='luxe-hero' style='{_merge_styles(hero_style, {})}'>
        {''.join(layers)}
        <div class='luxe-hero__content'>
          {icon_html}
          <h1>{self.title}</h1>
          <p>{self.subtitle}</p>
          {chips_html}
        </div>
      </div>
      """
      st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MetricGalaxy
# ---------------------------------------------------------------------------
@dataclass
class MetricItem:
  label: str
  value: str
  caption: str | None = None
  delta: str | None = None
  icon: str | None = None
  tone: str | None = None


@dataclass
class MetricGalaxy:
  metrics: Sequence[MetricItem]
  glow: bool = True
  density: str = "cozy"
  min_width: str = "13rem"

  def render(self) -> None:
      _inject_css()
      padding_map = {
          "compact": "1rem 1.15rem",
          "cozy": "1.2rem 1.4rem",
          "roomy": "1.45rem 1.65rem",
      }
      gap_map = {
          "compact": "0.8rem",
          "cozy": "1rem",
          "roomy": "1.3rem",
      }
      style = {
          "--metric-padding": padding_map.get(self.density, padding_map["cozy"]),
          "--metric-gap": gap_map.get(self.density, gap_map["cozy"]),
          "--metric-min": self.min_width,
      }
      html = [f"<div class='luxe-metric-galaxy' style='{_merge_styles(style, {})}'>"]
      for metric in self.metrics:
          tone = metric.tone or ("positive" if (metric.delta and metric.delta.startswith("+")) else "")
          icon_html = f"<div class='luxe-metric__icon'>{metric.icon}</div>" if metric.icon else ""
          delta_html = f"<div class='luxe-metric__delta'>{metric.delta}</div>" if metric.delta else ""
          caption_html = f"<div class='luxe-metric__caption'>{metric.caption}</div>" if metric.caption else ""
          html.append(
              f"<div class='luxe-metric' data-glow='{str(self.glow).lower()}' data-tone='{tone}'>"
              f"{icon_html}"
              f"<div class='luxe-metric__label'>{metric.label}</div>"
              f"<div class='luxe-metric__value'>{metric.value}</div>"
              f"{delta_html}"
              f"{caption_html}"
              "</div>"
          )
      html.append("</div>")
      st.markdown("".join(html), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# GlassStack
# ---------------------------------------------------------------------------
@dataclass
class GlassCard:
  title: str
  body: str
  icon: str | None = None
  footer: str | None = None


@dataclass
class GlassStack:
  cards: Sequence[GlassCard]
  columns_min: str = "16rem"
  density: str = "cozy"

  def render(self) -> None:
      _inject_css()
      padding_map = {
          "compact": "1.05rem 1.15rem",
          "cozy": "1.3rem 1.4rem",
          "roomy": "1.6rem 1.75rem",
      }
      gap_map = {
          "compact": "0.9rem",
          "cozy": "1.1rem",
          "roomy": "1.45rem",
      }
      style = {
          "--card-padding": padding_map.get(self.density, padding_map["cozy"]),
          "--stack-gap": gap_map.get(self.density, gap_map["cozy"]),
          "--stack-min": self.columns_min,
      }
      html = [f"<div class='luxe-stack' style='{_merge_styles(style, {})}'>"]
      for card in self.cards:
          icon_html = f"<div class='luxe-card__icon'>{card.icon}</div>" if card.icon else ""
          footer_html = f"<div class='luxe-card__footer'>{card.footer}</div>" if card.footer else ""
          html.append(
              f"<div class='luxe-card'>"
              f"{icon_html}"
              f"<h3 class='luxe-card__title'>{card.title}</h3>"
              f"<div class='luxe-card__body'>{card.body}</div>"
              f"{footer_html}"
              "</div>"
          )
      html.append("</div>")
      st.markdown("".join(html), unsafe_allow_html=True)


__all__ = [
  "TeslaHero",
  "MetricGalaxy",
  "MetricItem",
  "GlassStack",
  "GlassCard",
  "ChipRow",
]
