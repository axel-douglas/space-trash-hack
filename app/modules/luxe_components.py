"""High-fidelity UI primitives for the Rex-AI Streamlit experience."""

from __future__ import annotations

import base64
import json
import math
import unicodedata
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


_CSS_KEY = "__rexai_luxe_css__"
_TIMELINE_HOLOGRAM_KEY = "__timeline_hologram_assets__"
_FRAMER_MOTION_SRC = "https://cdn.jsdelivr.net/npm/framer-motion@11.0.5/dist/framer-motion.umd.js"


# ---------------------------------------------------------------------------
# CSS snippets
# ---------------------------------------------------------------------------
_BRIEFING_AND_TARGET_CSS = """
<style>
.luxe-hero-scene {
    display: grid;
    gap: 32px;
    margin-bottom: 16px;
}
.luxe-hero-scene__lead {
    display: grid;
    gap: 18px;
}
.luxe-hero-scene .briefing-grid {
    margin-top: 0;
}
.luxe-hero__tagline {
    margin: 0;
    font-size: 1.02rem;
    color: rgba(226, 232, 240, 0.78);
}
.briefing-grid {
    display: grid;
    grid-template-columns: minmax(280px, 1fr) minmax(320px, 1fr);
    gap: 32px;
    align-items: stretch;
    margin-top: 18px;
}
.briefing-video {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 32px 80px -40px rgba(15, 23, 42, 0.8);
    min-height: 280px;
    background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.4), rgba(15, 23, 42, 0.85));
}
.briefing-video video {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: saturate(1.05) contrast(1.05);
}
.briefing-fallback {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(226, 232, 240, 0.88);
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: linear-gradient(120deg, rgba(56, 189, 248, 0.25), rgba(59, 130, 246, 0.05));
    animation: aurora 16s linear infinite;
}
.briefing-cards {
    display: grid;
    gap: 18px;
}
.briefing-card {
    position: relative;
    padding: 22px 24px;
    border-radius: 24px;
    border: 1px solid rgba(96, 165, 250, 0.18);
    background: rgba(15, 23, 42, 0.78);
    color: var(--ink);
    box-shadow: 0 24px 48px -32px rgba(30, 64, 175, 0.55);
    overflow: hidden;
}
.briefing-card::after {
    content: "";
    position: absolute;
    inset: -60% 40% 20% -40%;
    background: var(--card-accent);
    opacity: 0.16;
    filter: blur(60px);
    transform: rotate(8deg);
    transition: transform 600ms ease, opacity 600ms ease;
}
.briefing-card h3 {
    margin: 0 0 6px;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
}
.briefing-card p {
    margin: 0;
    color: rgba(226, 232, 240, 0.86);
    font-size: 0.95rem;
}
.briefing-card:hover::after {
    opacity: 0.32;
    transform: rotate(-6deg) scale(1.08);
}
.briefing-stepper {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 24px;
}
.briefing-step {
    display: grid;
    grid-template-columns: 44px 1fr;
    gap: 12px;
    align-items: start;
    padding: 12px 14px;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.16);
    animation: rise-in 460ms ease backwards;
}
.briefing-step span {
    width: 44px;
    height: 44px;
    border-radius: 999px;
    display: grid;
    place-items: center;
    font-weight: 700;
    background: rgba(56, 189, 248, 0.18);
    color: var(--ink);
}
.briefing-step strong {
    display: block;
    font-size: 0.98rem;
    margin-bottom: 4px;
}
.briefing-step small {
    color: rgba(226, 232, 240, 0.78);
    font-size: 0.85rem;
}
.orbital-timeline {
    perspective: 1200px;
    margin: 32px 0 12px;
}
.orbital-track {
    position: relative;
    transform-style: preserve-3d;
    transform: rotateX(18deg);
    display: flex;
    gap: 32px;
    padding: 32px 18px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(15, 23, 42, 0.85));
    border: 1px solid rgba(96, 165, 250, 0.22);
    overflow-x: auto;
}
.orbital-track::-webkit-scrollbar {height: 6px;}
.orbital-track::-webkit-scrollbar-thumb {
    background: rgba(59, 130, 246, 0.3);
    border-radius: 999px;
}
.orbital-node {
    min-width: 220px;
    padding: 18px 20px;
    border-radius: 20px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 18px 32px -20px rgba(30, 64, 175, 0.7);
    position: relative;
    transform: translateZ(var(--depth, 0px));
    transition: transform 400ms ease, box-shadow 400ms ease;
}
.orbital-node::after {
    content: "";
    position: absolute;
    top: 50%;
    right: -16px;
    width: 32px;
    height: 2px;
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.0), rgba(56, 189, 248, 0.65));
}
.orbital-node:last-child::after {display: none;}
.orbital-node:hover {
    transform: translateZ(calc(var(--depth, 0px) + 28px));
    box-shadow: 0 30px 60px -28px rgba(37, 99, 235, 0.85);
}
.orbital-node span {
    font-size: 1.5rem;
}
.orbital-node h4 {
    margin: 12px 0 6px;
    font-size: 1.05rem;
}
.orbital-node p {
    margin: 0;
    color: rgba(226, 232, 240, 0.8);
    font-size: 0.9rem;
}
.guided-overlay {
    position: fixed;
    inset: 0;
    pointer-events: none;
    display: grid;
    place-items: center;
    z-index: 900;
}
.guided-overlay.hidden {display: none;}
.guided-panel {
    pointer-events: auto;
    max-width: 420px;
    padding: 26px 28px;
    border-radius: 24px;
    background: rgba(15, 23, 42, 0.94);
    border: 1px solid rgba(96, 165, 250, 0.24);
    box-shadow: 0 40px 120px -60px rgba(15, 23, 42, 0.9);
    backdrop-filter: blur(12px);
    text-align: center;
    animation: pulse 820ms ease-in-out infinite alternate;
}
.guided-panel h3 {
    margin: 0 0 10px;
}
.guided-panel p {
    margin: 0;
    color: rgba(226, 232, 240, 0.85);
}
.guided-panel footer {
    margin-top: 18px;
    font-size: 0.85rem;
    color: rgba(148, 163, 184, 0.85);
}
@keyframes aurora {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
@keyframes rise-in {
    from {opacity: 0; transform: translateY(18px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes pulse {
    from {transform: scale(1);}
    to {transform: scale(1.02);}
}
.luxe-card-grid {display:flex; gap:1rem; flex-wrap:wrap;}
.luxe-card {
    border-radius: 18px;
    padding: 1rem;
    width: 220px;
    background: linear-gradient(145deg, rgba(15,25,36,0.9), rgba(48,74,102,0.85));
    color: #f5f9ff;
    position: relative;
    box-shadow: 0 25px 45px rgba(2,12,27,0.45);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 1px solid rgba(255,255,255,0.08);
}
.luxe-card.is-active {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 35px 60px rgba(0, 150, 255, 0.45);
    border-color: rgba(94,174,255,0.8);
}
.luxe-card h4 {margin: 0; font-size: 1.1rem;}
.luxe-card .tagline {opacity: 0.85; font-size: 0.85rem; margin-top: 0.4rem;}
.luxe-card .image {
    width: 100%;
    height: 120px;
    border-radius: 14px;
    margin-bottom: 0.8rem;
    background: radial-gradient(circle at top left, rgba(120,195,255,0.8), rgba(8,15,27,0.6));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    text-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.target-3d-card {
    perspective: 1200px;
}
.target-3d-card .inner {
    background: linear-gradient(160deg, rgba(8,21,35,0.9), rgba(45,90,120,0.88));
    border-radius: 24px;
    padding: 1.5rem;
    min-height: 260px;
    color: #eaf4ff;
    box-shadow: 0 35px 55px rgba(3, 12, 32, 0.55);
    border: 1px solid rgba(255,255,255,0.08);
    transform: rotateY(-12deg) rotateX(6deg);
    transform-style: preserve-3d;
    position: relative;
}
.target-3d-card .inner::after {
    content: "";
    position: absolute;
    inset: 10px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    pointer-events: none;
}
.circular-indicator {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    background: conic-gradient(var(--accent-color, #59b1ff) calc(var(--value, 0) * 1%), rgba(255,255,255,0.08) 0);
    color: #0e2137;
    margin-right: 0.6rem;
}
.circular-indicator span {
    font-size: 0.85rem;
}
.slider-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}
.slider-row .stSlider {flex: 1;}
.feedback-pill {
    display: inline-flex;
    gap: 0.3rem;
    align-items: center;
    padding: 0.35rem 0.6rem;
    border-radius: 999px;
    background: rgba(88, 184, 255, 0.12);
    color: #cfe9ff;
    font-size: 0.78rem;
}
</style>
"""

_LUXE_COMPONENT_CSS = """
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

.luxe-hero__video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  z-index: 0;
  filter: saturate(1.25) brightness(0.65);
}

.luxe-hero__veil {
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.12) 10%, rgba(2, 6, 23, 0.88) 100%);
  z-index: 1;
  pointer-events: none;
}

.luxe-hero::after {
  content: "";
  position: absolute;
  inset: -30%;
  background: radial-gradient(circle at 20% 20%, var(--hero-glow, rgba(96, 165, 250, 0.4)), transparent 62%);
  filter: blur(40px);
  opacity: 0.9;
  pointer-events: none;
  animation: sparkleShift 18s ease-in-out infinite;
}

.luxe-hero__content {
  position: relative;
  z-index: 2;
  max-width: 520px;
}

.luxe-hero__content h1 {
  font-size: clamp(2.5rem, 4vw, 3.2rem);
  margin-bottom: 0.7rem;
}

.luxe-hero__content p {
  font-size: 1.05rem;
  color: rgba(226, 232, 240, 0.82);
  margin: 0 0 1rem 0;
}

.luxe-hero__icon {
  font-size: 2rem;
  margin-bottom: 1rem;
  opacity: 0.85;
}

.luxe-hero__layer {
  position: absolute;
  top: 20%;
  left: 60%;
  font-size: var(--layer-size, 3.5rem);
  opacity: 0.5;
  filter: drop-shadow(0 18px 25px rgba(15, 23, 42, 0.45));
  animation: parallaxDrift var(--layer-speed, 18s) ease-in-out infinite;
  z-index: 1;
  pointer-events: none;
}

.luxe-chip-row {
  display: inline-flex;
  flex-wrap: wrap;
  gap: var(--chip-gap, 0.6rem);
  margin-top: 1.2rem;
}

.luxe-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: var(--chip-padding, 0.4rem 0.9rem);
  font-size: var(--chip-size, 0.82rem);
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: rgba(15, 23, 42, 0.6);
  color: var(--luxe-ink);
  letter-spacing: 0.04em;
}

.luxe-chip[data-tone="accent"] { background: rgba(96, 165, 250, 0.16); border-color: rgba(125, 211, 252, 0.45); }
.luxe-chip[data-tone="info"] { background: rgba(14, 165, 233, 0.16); border-color: rgba(56, 189, 248, 0.5); }
.luxe-chip[data-tone="positive"] { background: rgba(52, 211, 153, 0.14); border-color: rgba(52, 211, 153, 0.45); }
.luxe-chip[data-tone="warning"] { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.45); }
.luxe-chip[data-tone="danger"] { background: rgba(248, 113, 113, 0.16); border-color: rgba(248, 113, 113, 0.45); }

.luxe-metric-galaxy {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--metric-min, 13rem), 1fr));
  gap: var(--metric-gap, 1rem);
  margin-top: 1.8rem;
}

.luxe-metric {
  position: relative;
  border-radius: 20px;
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

.timeline-hologram {
  position: relative;
  display: grid;
  gap: 1.2rem;
  padding: clamp(1.2rem, 2.5vw, 1.6rem);
  border-radius: 24px;
  border: 1px solid color-mix(in srgb, var(--luxe-border) 82%, transparent);
  background: linear-gradient(
    145deg,
    color-mix(in srgb, var(--luxe-surface) 92%, transparent),
    color-mix(in srgb, var(--luxe-accent) 12%, transparent)
  );
  color: var(--luxe-ink);
  box-shadow: 0 28px 60px rgba(8, 15, 35, 0.45);
  overflow: hidden;
}

.timeline-hologram::after {
  content: "";
  position: absolute;
  inset: -30% -40% 40% 20%;
  background: radial-gradient(circle at top, rgba(96, 165, 250, 0.28), transparent 65%);
  opacity: 0.45;
  pointer-events: none;
  filter: blur(48px);
}

.timeline-hologram__header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  position: relative;
  z-index: 1;
}

.timeline-hologram__priority {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--luxe-accent) 40%, transparent);
  background: color-mix(in srgb, var(--luxe-accent) 12%, transparent);
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: color-mix(in srgb, var(--luxe-ink) 92%, transparent);
}

.timeline-hologram__priority strong {
  font-size: 0.84rem;
}

.timeline-hologram__priority em {
  font-style: normal;
  opacity: 0.82;
}

.timeline-hologram__caption {
  flex: 1 1 240px;
  margin: 0;
  color: color-mix(in srgb, var(--luxe-muted) 92%, transparent);
  font-size: 0.92rem;
}

.timeline-hologram__list {
  position: relative;
  z-index: 1;
  display: grid;
  gap: clamp(0.8rem, 2vw, 1rem);
  margin: 0;
  padding: 0;
  list-style: none;
}

.timeline-hologram__item {
  position: relative;
  border-radius: 20px;
  border: 1px solid color-mix(in srgb, var(--luxe-border) 78%, transparent);
  background: color-mix(in srgb, var(--luxe-surface) 92%, transparent);
  padding: clamp(0.95rem, 2.4vw, 1.2rem);
  display: grid;
  gap: 0.55rem;
  transition: border 220ms ease, box-shadow 220ms ease, transform 220ms ease;
  outline: none;
}

.timeline-hologram__item[data-active="true"] {
  border-color: color-mix(in srgb, var(--luxe-accent) 65%, transparent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--luxe-accent) 22%, transparent);
}

.timeline-hologram__item:focus-visible {
  outline: 2px solid color-mix(in srgb, var(--luxe-accent) 70%, transparent);
  outline-offset: 2px;
}

.timeline-hologram__head {
  display: flex;
  gap: 0.6rem;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
}

.timeline-hologram__icon {
  font-size: 1.4rem;
  width: 2.2rem;
  height: 2.2rem;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--luxe-accent) 22%, transparent);
  color: var(--luxe-ink);
  box-shadow: 0 12px 22px rgba(96, 165, 250, 0.3);
}

.timeline-hologram__title {
  display: grid;
  gap: 0.25rem;
  min-width: 0;
  flex: 1;
}

.timeline-hologram__title strong {
  font-size: 1.05rem;
}

.timeline-hologram__title span {
  font-size: 0.88rem;
  color: color-mix(in srgb, var(--luxe-muted) 90%, transparent);
}

.timeline-hologram__score {
  font-family: "Space Grotesk", "Inter", sans-serif;
  font-size: 1.18rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}

.timeline-hologram__badges {
  display: inline-flex;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.timeline-hologram__badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--luxe-border) 75%, transparent);
  background: color-mix(in srgb, var(--luxe-surface) 88%, transparent);
  font-size: 0.74rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.timeline-hologram__metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.timeline-hologram__metric {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.32rem 0.6rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--luxe-border) 72%, transparent);
  background: color-mix(in srgb, var(--luxe-surface) 90%, transparent);
  font-size: 0.82rem;
  color: color-mix(in srgb, var(--luxe-muted) 95%, transparent);
}

.timeline-hologram__metric[data-tone="warning"] {
  border-color: color-mix(in srgb, var(--luxe-warning) 55%, transparent);
  background: color-mix(in srgb, var(--luxe-warning) 12%, transparent);
  color: color-mix(in srgb, var(--luxe-warning) 90%, transparent);
}

.timeline-hologram__metric[data-tone="info"] {
  border-color: color-mix(in srgb, var(--luxe-accent) 55%, transparent);
  background: color-mix(in srgb, var(--luxe-accent) 15%, transparent);
  color: color-mix(in srgb, var(--luxe-accent) 88%, transparent);
}

.timeline-hologram__metric[data-tone="positive"] {
  border-color: color-mix(in srgb, var(--luxe-positive) 50%, transparent);
  background: color-mix(in srgb, var(--luxe-positive) 14%, transparent);
  color: color-mix(in srgb, var(--luxe-positive) 92%, transparent);
}

.timeline-hologram__metric strong {
  font-size: 0.88rem;
  color: color-mix(in srgb, var(--luxe-ink) 95%, transparent);
}

.timeline-hologram[data-enhanced="true"] .timeline-hologram__item {
  will-change: transform, opacity;
}

@media (prefers-reduced-motion: reduce) {
  .timeline-hologram__item {
    transition: none;
  }
}

.luxe-mission-panel {
  position: sticky;
  top: 84px;
  border-radius: 22px;
  border: 1px solid color-mix(in srgb, var(--border-soft) 72%, transparent);
  padding: var(--mission-panel-padding, 24px 26px);
  background: color-mix(in srgb, var(--surface-panel) 92%, transparent);
  backdrop-filter: blur(18px);
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.luxe-mission-panel__title {
  margin: 0;
}

.luxe-mission-panel__metrics {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.luxe-mission-metric {
  border-radius: 16px;
  border: 1px solid color-mix(in srgb, var(--border-soft) 60%, transparent);
  background: color-mix(in srgb, var(--surface-card) 85%, transparent);
  padding: var(--mission-metric-padding, 14px 16px);
  transition: border 280ms ease, box-shadow 280ms ease;
  display: grid;
  gap: 0.35rem;
}

.luxe-mission-metric.is-active {
  border-color: color-mix(in srgb, var(--accent) 60%, transparent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 18%, transparent);
}

.luxe-mission-metric__icon {
  width: 32px;
  height: 32px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--accent) 18%, transparent);
  color: var(--accent);
  font-size: 1.1rem;
  margin-bottom: 2px;
}

.luxe-mission-metric h5 {
  margin: 0;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: color-mix(in srgb, var(--muted) 90%, transparent);
}

.luxe-mission-metric strong {
  font-size: 1.32rem;
}

.luxe-mission-metric p {
  margin: 0;
  color: var(--muted);
  font-size: 0.85rem;
}

.luxe-mission-metric__caption {
  margin: 0;
  color: color-mix(in srgb, var(--muted) 85%, transparent);
  font-size: 0.8rem;
}

.luxe-mission-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--mission-grid-min, 14rem), 1fr));
  gap: var(--mission-grid-gap, 1rem);
  margin-top: 1.5rem;
}

.luxe-mission-grid .luxe-mission-metric {
  border-radius: 18px;
  padding: var(--mission-grid-card-padding, 18px 20px);
  background: var(--metric-bg, color-mix(in srgb, var(--surface-panel) 88%, transparent));
  border: 1px solid var(--metric-border, color-mix(in srgb, var(--border-soft) 65%, transparent));
}

.luxe-carousel {
  display: flex;
  gap: var(--carousel-gap, 1rem);
  overflow-x: auto;
  padding-bottom: 0.5rem;
  scroll-snap-type: x mandatory;
}

.luxe-carousel::-webkit-scrollbar {
  height: 6px;
}

.luxe-carousel::-webkit-scrollbar-thumb {
  background: color-mix(in srgb, var(--accent) 45%, transparent);
  border-radius: 999px;
}

.luxe-carousel-card {
  min-width: var(--carousel-min, 15rem);
  scroll-snap-align: start;
  border-radius: 18px;
  border: 1px solid color-mix(in srgb, var(--border-soft) 64%, transparent);
  padding: var(--carousel-card-padding, 18px);
  background: color-mix(in srgb, var(--surface-card) 82%, transparent);
  display: grid;
  gap: 0.4rem;
}

.luxe-carousel-card h4 {
  margin: 0;
  font-size: 1.02rem;
}

.luxe-carousel-card__value {
  font-size: 1.3rem;
  font-weight: 700;
}

.luxe-carousel-card__description {
  margin: 0;
  font-size: 0.85rem;
  color: var(--muted);
}

.luxe-carousel-card__icon {
  font-size: 1.4rem;
  opacity: 0.75;
}

.luxe-action-deck {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--action-min, 16rem), 1fr));
  gap: var(--action-gap, 1.1rem);
}

.luxe-action-card {
  position: relative;
  border-radius: 22px;
  border: 1px solid color-mix(in srgb, var(--accent) 28%, transparent);
  padding: var(--action-card-padding, 22px 24px);
  background: linear-gradient(160deg, color-mix(in srgb, var(--accent) 16%, transparent), color-mix(in srgb, var(--surface-panel) 78%, transparent));
  color: var(--ink);
  box-shadow: 0 24px 48px -32px rgba(15, 23, 42, 0.55);
  overflow: hidden;
}

.luxe-action-card__icon {
  position: absolute;
  top: 16px;
  right: 18px;
  font-size: 1.4rem;
  opacity: 0.85;
}

.luxe-action-card__title {
  margin: 0 0 0.5rem 0;
  font-size: 1.16rem;
}

.luxe-action-card__body {
  margin: 0;
  color: var(--muted);
  font-size: 0.94rem;
}

.luxe-action-card__tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.32rem 0.75rem;
  border-radius: 999px;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: color-mix(in srgb, var(--accent) 18%, transparent);
  color: color-mix(in srgb, var(--accent) 90%, transparent);
  margin-bottom: 0.8rem;
}

.luxe-stack {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--stack-min, 16rem), 1fr));
  gap: var(--stack-gap, 1.1rem);
  margin: var(--stack-margin, 1.6rem 0 0);
}

.mission-flow-showcase {
  position: relative;
  display: grid;
  gap: clamp(1.4rem, 2.8vw, 1.8rem);
  padding: clamp(1.4rem, 2.6vw, 1.9rem);
  border-radius: 26px;
  border: 1px solid color-mix(in srgb, var(--luxe-border) 75%, transparent);
  background: linear-gradient(150deg, rgba(15, 23, 42, 0.78), rgba(30, 41, 59, 0.82));
  box-shadow: 0 26px 60px rgba(8, 15, 35, 0.42);
  color: var(--luxe-ink);
}

.mission-flow-showcase::after {
  content: "";
  position: absolute;
  inset: -40% -30% 40% 10%;
  background: radial-gradient(circle at top right, rgba(96, 165, 250, 0.32), transparent 70%);
  opacity: 0.55;
  filter: blur(60px);
  pointer-events: none;
}

.mission-flow-showcase__header {
  position: relative;
  z-index: 1;
  display: grid;
  gap: 0.45rem;
}

.mission-flow-showcase__header h3 {
  margin: 0;
  font-size: clamp(1.35rem, 2.4vw, 1.6rem);
}

.mission-flow-showcase__header p {
  margin: 0;
  color: rgba(226, 232, 240, 0.78);
  font-size: 0.95rem;
}

.mission-flow-showcase__content {
  position: relative;
  z-index: 1;
  display: grid;
  gap: clamp(1.2rem, 2vw, 1.6rem);
}

@media (min-width: 1100px) {
  .mission-flow-showcase__content {
    grid-template-columns: 1.6fr 1fr;
    align-items: start;
  }
}

.mission-flow-showcase__steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--mission-step-min, 15rem), 1fr));
  gap: var(--mission-step-gap, 1.1rem);
}

.mission-flow-showcase__step {
  position: relative;
  display: grid;
  gap: 0.6rem;
  padding: var(--mission-step-padding, 1.25rem 1.35rem);
  border-radius: 22px;
  border: 1px solid color-mix(in srgb, var(--luxe-border) 75%, transparent);
  background: linear-gradient(165deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.78));
  box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.08), 0 20px 40px rgba(8, 15, 35, 0.38);
}

.mission-flow-showcase__step-head {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.9rem;
  align-items: start;
}

.mission-flow-showcase__step-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.6rem;
  height: 2.6rem;
  border-radius: 999px;
  background: rgba(96, 165, 250, 0.16);
  border: 1px solid rgba(148, 163, 184, 0.28);
  font-size: 1.3rem;
}

.mission-flow-showcase__step h4 {
  margin: 0;
  font-size: 1.08rem;
}

.mission-flow-showcase__copy {
  margin: 0.35rem 0 0;
  font-size: 0.96rem;
  color: rgba(226, 232, 240, 0.78);
}

.mission-flow-showcase__copy [data-viewport="mobile"] {
  display: none;
}

.mission-flow-showcase__step-footer {
  margin: 0;
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(148, 163, 184, 0.78);
}

@media (max-width: 900px) {
  .mission-flow-showcase {
    padding: clamp(1.2rem, 3vw, 1.5rem);
  }

  .mission-flow-showcase__steps {
    gap: var(--mission-step-gap-mobile, 0.95rem);
  }

  .mission-flow-showcase__step {
    padding: var(--mission-step-padding-mobile, 1.05rem 1.1rem);
  }

  .mission-flow-showcase__copy [data-viewport="desktop"] {
    display: none;
  }

  .mission-flow-showcase__copy [data-viewport="mobile"] {
    display: inline;
  }
}

.mission-flow-showcase__timeline {
  display: grid;
  gap: 0.9rem;
}

.mission-flow-showcase__timeline h4 {
  margin: 0;
  font-size: 1.05rem;
}

.mission-flow-showcase__timeline .orbital-timeline {
  margin: 0;
}

.mission-flow-showcase__insights {
  margin: 0;
  padding-left: 1.1rem;
  display: grid;
  gap: 0.35rem;
  color: rgba(226, 232, 240, 0.76);
  font-size: 0.9rem;
}

.mission-flow-showcase__insights li {
  margin: 0;
}

.mission-flow-showcase__actions {
  margin-top: 1rem;
}

.mission-flow-showcase__actions + .mission-flow-showcase__actions {
  margin-top: 1.2rem;
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


_TIMELINE_HOLOGRAM_SCRIPT = f"""
<script>
(function() {{
  if (window.__timelineHologramEnhancer) {{
    return;
  }}

  const prefersReduced = () => (
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );

  const ensureMotion = () => {{
    if (window.Motion && typeof window.Motion.animate === 'function') {{
      return Promise.resolve(window.Motion);
    }}
    return new Promise((resolve, reject) => {{
      const script = document.createElement('script');
      script.src = '{_FRAMER_MOTION_SRC}';
      script.async = true;
      script.onload = () => resolve(window.Motion || null);
      script.onerror = reject;
      document.head.appendChild(script);
    }});
  }};

  const markEnhanced = (section) => section.setAttribute('data-enhanced', 'true');

  const enhance = () => {{
    const sections = Array.from(
      document.querySelectorAll('.timeline-hologram[data-enhanced="false"]')
    );
    if (!sections.length) {{
      return;
    }}

    if (prefersReduced()) {{
      sections.forEach(markEnhanced);
      return;
    }}

    ensureMotion()
      .then((Motion) => {{
        if (!Motion || typeof Motion.animate !== 'function') {{
          sections.forEach(markEnhanced);
          return;
        }}
        const {{ animate, stagger }} = Motion;
        sections.forEach((section) => {{
          const items = Array.from(section.querySelectorAll('.timeline-hologram__item'));
          if (!items.length) {{
            markEnhanced(section);
            return;
          }}
          markEnhanced(section);
          animate(
            items,
            {{ opacity: [0, 1], transform: ['translateY(24px)', 'translateY(0px)'] }},
            {{ duration: 0.65, delay: stagger(0.08), ease: 'ease-out' }}
          );
        }});
      }})
      .catch(() => sections.forEach(markEnhanced));
  }};

  const observer = new MutationObserver(() => enhance());
  if (document.body) {{
    observer.observe(document.body, {{ childList: true, subtree: true }});
  }}

  document.addEventListener('DOMContentLoaded', enhance);
  window.addEventListener('load', enhance);
  window.__timelineHologramEnhancer = {{ enhance }};
}})();
</script>
"""

_RANKING_COCKPIT_CSS = """
<style>
.ranking-cockpit {
  display: grid;
  gap: 1.2rem;
  margin-top: 0.4rem;
}

.ranking-cockpit__controls {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  align-items: end;
}

.ranking-cockpit__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
}

.ranking-card {
  position: relative;
  padding: 1.25rem 1.4rem;
  border-radius: 24px;
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: linear-gradient(145deg, rgba(15, 23, 42, 0.78), rgba(30, 41, 59, 0.78));
  box-shadow: 18px 18px 38px rgba(8, 15, 35, 0.55), -18px -18px 36px rgba(59, 130, 246, 0.05);
  color: var(--luxe-ink);
  transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
  overflow: hidden;
}

.ranking-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 20% -20%, rgba(96, 165, 250, 0.18), transparent 55%);
  opacity: 0.75;
  pointer-events: none;
}

.ranking-card.selected {
  border-color: rgba(96, 165, 250, 0.65);
  transform: translateY(-6px);
  box-shadow: 0 28px 60px rgba(37, 99, 235, 0.48);
}

.ranking-card.tone-high {
  border-color: rgba(239, 68, 68, 0.55);
}

.ranking-card.tone-med {
  border-color: rgba(245, 158, 11, 0.55);
}

.ranking-card__header {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 0.8rem;
  align-items: center;
  margin-bottom: 0.85rem;
}

.ranking-card__rank {
  font-size: 1.35rem;
  font-weight: 700;
  color: rgba(148, 197, 255, 0.95);
}

.ranking-card__title {
  font-weight: 600;
  line-height: 1.2;
}

.ranking-card__origin {
  display: block;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(226, 232, 240, 0.62);
}

.ranking-card__score {
  text-align: right;
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(226, 232, 240, 0.7);
}

.ranking-card__score strong {
  display: block;
  font-size: 1.25rem;
  letter-spacing: normal;
  color: var(--luxe-ink);
}

.ranking-card__chips {
  display: flex;
  gap: 0.45rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}

.ranking-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.32rem 0.75rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border: 1px solid rgba(148, 163, 184, 0.35);
  background: rgba(148, 163, 184, 0.12);
  color: rgba(226, 232, 240, 0.78);
}

.ranking-chip.seal-ok {
  border-color: rgba(34, 197, 94, 0.45);
  background: rgba(34, 197, 94, 0.18);
  color: rgba(15, 23, 42, 0.9);
}

.ranking-chip.seal-warn {
  border-color: rgba(245, 158, 11, 0.5);
  background: rgba(245, 158, 11, 0.18);
  color: rgba(15, 23, 42, 0.92);
}

.ranking-chip.risk-low {
  border-color: rgba(96, 165, 250, 0.45);
  background: rgba(96, 165, 250, 0.16);
  color: rgba(226, 232, 240, 0.92);
}

.ranking-chip.risk-med {
  border-color: rgba(245, 158, 11, 0.55);
  background: rgba(245, 158, 11, 0.22);
  color: rgba(15, 23, 42, 0.9);
}

.ranking-chip.risk-high {
  border-color: rgba(239, 68, 68, 0.6);
  background: rgba(239, 68, 68, 0.22);
  color: rgba(15, 23, 42, 0.95);
}

.ranking-card__metrics {
  display: grid;
  gap: 0.75rem;
}

.ranking-metric {
  display: grid;
  gap: 0.3rem;
}

.ranking-metric__label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(226, 232, 240, 0.66);
}

.ranking-metric__value {
  font-size: 0.98rem;
  font-weight: 600;
  color: var(--luxe-ink);
}

.ranking-bar {
  position: relative;
  height: 0.65rem;
  border-radius: 999px;
  background: linear-gradient(145deg, rgba(12, 18, 34, 0.92), rgba(20, 32, 52, 0.92));
  box-shadow: inset 2px 3px 6px rgba(0, 0, 0, 0.45), inset -2px -3px 6px rgba(59, 130, 246, 0.25);
  overflow: hidden;
}

.ranking-bar__fill {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: var(--fill, 0%);
  border-radius: inherit;
  background: linear-gradient(90deg, rgba(96, 165, 250, 0.9), rgba(56, 189, 248, 0.55));
  box-shadow: 0 6px 16px rgba(37, 99, 235, 0.55);
  transition: width 0.45s ease;
}

.ranking-bar__fill[data-tone='low'] {
  background: linear-gradient(90deg, rgba(34, 197, 94, 0.9), rgba(59, 130, 246, 0.55));
}

.ranking-bar__fill[data-tone='med'] {
  background: linear-gradient(90deg, rgba(245, 158, 11, 0.85), rgba(249, 115, 22, 0.55));
}

.ranking-bar__fill[data-tone='high'] {
  background: linear-gradient(90deg, rgba(239, 68, 68, 0.88), rgba(249, 115, 22, 0.55));
  box-shadow: 0 6px 16px rgba(239, 68, 68, 0.55);
}

.ranking-empty {
  padding: 1rem 1.2rem;
  border-radius: 18px;
  background: rgba(15, 23, 42, 0.72);
  border: 1px dashed rgba(148, 163, 184, 0.3);
  color: rgba(226, 232, 240, 0.82);
}
</style>
"""

_UTILITY_CSS = """
<style>
.card {
  border-radius: 18px;
  padding: 1rem 1.25rem;
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.22);
  color: rgba(226, 232, 240, 0.92);
  box-shadow: 0 22px 45px -24px rgba(15, 23, 42, 0.75);
}
.card-flat {
  box-shadow: none;
  border-color: rgba(148, 163, 184, 0.35);
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border: 1px solid rgba(94, 234, 212, 0.45);
  background: rgba(45, 212, 191, 0.16);
  color: rgba(15, 23, 42, 0.88);
}
.pill.ok {
  border-color: rgba(34, 197, 94, 0.45);
  background: rgba(34, 197, 94, 0.18);
  color: rgba(15, 23, 42, 0.9);
}
.pill.warn,
.pill.med {
  border-color: rgba(245, 158, 11, 0.55);
  background: rgba(245, 158, 11, 0.18);
}
.pill.risk,
.pill.bad {
  border-color: rgba(239, 68, 68, 0.6);
  background: rgba(239, 68, 68, 0.2);
  color: rgba(15, 23, 42, 0.92);
}
.pill-solid {
  color: rgba(226, 232, 240, 0.95);
  background: rgba(15, 23, 42, 0.75);
  border-color: rgba(226, 232, 240, 0.45);
}
.small {
  font-size: 0.85rem;
  opacity: 0.82;
}
</style>
"""


def _load_css() -> None:
    """Inject shared CSS snippets once per session."""

    if st.session_state.get(_CSS_KEY):
        return

    for css in (
        _BRIEFING_AND_TARGET_CSS,
        _LUXE_COMPONENT_CSS,
        _RANKING_COCKPIT_CSS,
        _UTILITY_CSS,
    ):
        st.markdown(css, unsafe_allow_html=True)

    st.session_state[_CSS_KEY] = True


def _inject_timeline_hologram_assets() -> None:
    """Load the motion enhancer for the TimelineHologram component once."""

    if st.session_state.get(_TIMELINE_HOLOGRAM_KEY):
        return

    st.markdown(_TIMELINE_HOLOGRAM_SCRIPT, unsafe_allow_html=True)
    st.session_state[_TIMELINE_HOLOGRAM_KEY] = True


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _current_theme() -> str:
    return st.session_state.get("hud_theme", "dark")


def _is_high_contrast() -> bool:
    return "high-contrast" in _current_theme()


def _is_colorblind_mode() -> bool:
    return st.session_state.get("hud_colorblind", "normal") != "normal"


def _class_names(*tokens: Iterable[str]) -> str:
    classes: list[str] = []
    for token in tokens:
        if isinstance(token, str):
            classes.append(token)
        else:
            classes.extend(t for t in token if t)
    return " ".join(cls for cls in classes if cls)


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if _is_nan(number) else number
    if isinstance(value, np.generic):
        number = float(value)
        return None if _is_nan(number) else number

    text = str(value).strip()
    if not text:
        return None
    sanitized = text.replace("%", "")
    try:
        number = float(sanitized)
    except ValueError:
        return None
    return None if _is_nan(number) else number


def _normalize_str(value: Any, placeholder: str = "—") -> str:
    if value is None:
        return placeholder
    text = str(value).strip()
    return text if text else placeholder


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def render_card(title: str, body: str = "") -> str:
    """Return a luxe card block, injecting CSS if necessary."""

    _load_css()

    classes = ["card"]
    if _is_high_contrast():
        classes.append("card-flat")

    body_html = f'<div class="small">{body}</div>' if body else ""
    return f'<div class="{_class_names(classes)}"><h4>{title}</h4>{body_html}</div>'


def render_pill(label: str, kind: Literal["ok", "warn", "risk"] = "ok") -> str:
    """Return a status pill supporting tone variants."""

    _load_css()

    tone_map = {
        "ok": ("ok",),
        "warn": ("warn", "med"),
        "risk": ("risk", "bad"),
    }
    classes = ["pill", *tone_map.get(kind, ("ok",))]
    if _is_colorblind_mode():
        classes.append("pill-solid")

    return f'<span class="{_class_names(classes)}">{label}</span>'


# ---------------------------------------------------------------------------
# Data descriptors
# ---------------------------------------------------------------------------


@dataclass
class BriefingCard:
    """Descriptor for each animated briefing card."""

    title: str
    body: str
    accent: str = "#38bdf8"


@dataclass
class TimelineMilestone:
    """Descriptor for an orbital timeline milestone."""

    label: str
    description: str
    icon: str = "🛰️"


@dataclass(frozen=True)
class TimelineHologramMetric:
    """Small badge-like metric shown inside the TimelineHologram component."""

    label: str
    value: str
    tone: Literal["neutral", "info", "warning", "positive"] = "neutral"
    sr_label: str | None = None


@dataclass
class TimelineHologramItem:
    """Element displayed in the hologram timeline with optional accent badges."""

    title: str
    subtitle: str
    score: float | None = None
    icon: str = "🛰️"
    rank: int | None = None
    badges: Sequence[str] = field(default_factory=tuple)
    metrics: Sequence[TimelineHologramMetric] = field(default_factory=tuple)
    emphasis: bool = False
    aria_label: str | None = None


@dataclass(frozen=True)
class HeroFlowStage:
    """Unified description for the mission flow used across hero layouts."""

    key: str
    order: int
    name: str
    hero_headline: str
    hero_copy: str
    card_body: str
    compact_card_body: str | None = None
    icon: str
    timeline_label: str
    timeline_description: str
    footer: str | None = None

    def as_step(self) -> tuple[str, str]:
        return (self.hero_headline, self.hero_copy)

    @property
    def card_title(self) -> str:
        return f"{self.order} · {self.name}"

    def copy_for_viewport(self, viewport: Literal["desktop", "mobile"] = "desktop") -> str:
        if viewport == "mobile" and self.compact_card_body:
            return self.compact_card_body
        return self.card_body

@dataclass
class TargetPresetMeta:
    """Metadata used to render the Tesla-style preset cards."""

    icon: str
    tagline: str


_PRESET_META: Dict[str, TargetPresetMeta] = {
    "Container": TargetPresetMeta(
        icon="📦",
        tagline="Para almacenaje hermético y modular sin sacrificar estilo.",
    ),
    "Utensil": TargetPresetMeta(
        icon="🍴",
        tagline="Diseñado para tareas delicadas con acabado pulido lunar.",
    ),
    "Interior": TargetPresetMeta(
        icon="🛋️",
        tagline="Habitáculos confortables que optimizan espacio y calor.",
    ),
    "Tool": TargetPresetMeta(
        icon="🛠️",
        tagline="Robustez industrial lista para cualquier misión orbital.",
    ),
}


# ---------------------------------------------------------------------------
# Mission briefing and guided flows
# ---------------------------------------------------------------------------

def _video_as_base64(video_path: Path) -> Optional[str]:
    try:
        data = video_path.read_bytes()
    except FileNotFoundError:
        return None
    if not data:
        return None
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


def mission_briefing(
    *,
    title: str,
    tagline: str,
    video_path: Optional[Path | str] = None,
    cards: Sequence[BriefingCard] = (),
    steps: Sequence[tuple[str, str]] = (),
) -> None:
    """Render the mission briefing hero with media loop and animated cards."""

    _load_css()

    st.markdown(f"## {title}")
    st.caption(tagline)

    media_src: Optional[str] = None
    if video_path:
        media_src = _video_as_base64(Path(video_path))

    st.markdown("<div class='briefing-grid'>", unsafe_allow_html=True)

    media_html = (
        f"<video autoplay loop muted playsinline src='{media_src}'></video>"
        if media_src
        else "<div class='briefing-fallback'>Simulación orbital</div>"
    )

    st.markdown(
        f"<div class='briefing-video'>{media_html}</div>",
        unsafe_allow_html=True,
    )

    cards_html = "".join(
        f"""
        <div class='briefing-card' style="--card-accent: {card.accent};">
            <h3>{card.title}</h3>
            <p>{card.body}</p>
        </div>
        """
        for card in cards
    )

    steps_html = "".join(
        f"""
        <div class='briefing-step' style="animation-delay: {idx * 120}ms;">
            <span>{idx + 1}</span>
            <div>
                <strong>{step_title}</strong>
                <small>{copy}</small>
            </div>
        </div>
        """
        for idx, (step_title, copy) in enumerate(steps)
    )

    st.markdown(
        f"""
        <div class='briefing-cards'>
            {cards_html}
            <div class='briefing-stepper'>
                {steps_html}
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def orbital_timeline(milestones: Iterable[TimelineMilestone]) -> None:
    """Render the animated orbital timeline."""

    _load_css()

    st.markdown("<div class='orbital-timeline'>", unsafe_allow_html=True)
    nodes = []
    for depth, milestone in enumerate(milestones):
        nodes.append(
            f"""
            <div class='orbital-node' style="--depth: {depth * 18}px;">
                <span>{milestone.icon}</span>
                <h4>{milestone.label}</h4>
                <p>{milestone.description}</p>
            </div>
            """
        )
    st.markdown(
        f"<div class='orbital-track'>{''.join(nodes)}</div></div>",
        unsafe_allow_html=True,
    )


@dataclass
class TimelineHologram:
    """Composable holographic timeline with motion and accessibility baked in."""

    items: Sequence[TimelineHologramItem]
    priority_label: str | None = None
    priority_value: float | None = None
    priority_detail: str | None = None
    caption: str | None = None

    def render(self) -> None:
        if not self.items:
            st.info("No hay elementos para mostrar en la timeline holográfica.")
            return

        _load_css()
        _inject_timeline_hologram_assets()
        st.markdown(self._build_html(), unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_html(self) -> str:
        container_attrs = [
            "class='timeline-hologram'",
            "data-enhanced='false'",
            "aria-live='polite'",
        ]
        html: list[str] = [f"<section {' '.join(container_attrs)}>"]

        header_html = self._build_header()
        if header_html:
            html.append(header_html)

        html.append("<ol class='timeline-hologram__list' role='list'>")
        for idx, item in enumerate(self.items):
            html.append(self._render_item(idx, item))
        html.append("</ol></section>")
        return "".join(html)

    def _build_header(self) -> str:
        chunks: list[str] = []

        pill_html = ""
        if self.priority_label and self.priority_value is not None:
            clamped = max(0.0, min(1.0, float(self.priority_value)))
            rigidez_pct = clamped * 100
            agua_pct = (1.0 - clamped) * 100
            detail_text = f" {self.priority_detail}" if self.priority_detail else ""
            aria_text = (
                f"{self.priority_label}: {rigidez_pct:.0f}% rigidez y {agua_pct:.0f}% agua.{detail_text}"
            )
            pill_html = (
                "<span class='timeline-hologram__priority' "
                f"aria-label='{escape(aria_text)}'>"
                f"<strong>{escape(self.priority_label)}</strong>"
                f"<em>{rigidez_pct:.0f}% R · {agua_pct:.0f}% A</em>"
                "</span>"
            )

        caption_html = ""
        if self.caption:
            caption_html = f"<p class='timeline-hologram__caption'>{escape(self.caption)}</p>"

        if pill_html or caption_html:
            chunks.append("<header class='timeline-hologram__header'>")
            if pill_html:
                chunks.append(pill_html)
            if caption_html:
                chunks.append(caption_html)
            chunks.append("</header>")

        return "".join(chunks)

    def _render_item(self, index: int, item: TimelineHologramItem) -> str:
        rank = item.rank if item.rank is not None else index + 1
        title = escape(item.title)
        subtitle = escape(item.subtitle)
        icon = escape(item.icon)
        aria_label = item.aria_label or f"Opción {rank:02d}: {item.title}. {item.subtitle}."
        aria_label = escape(aria_label)

        active = item.emphasis or index == 0
        aria_current = " aria-current='true'" if active else ""
        badges_html = "".join(
            f"<span class='timeline-hologram__badge'>{escape(badge)}</span>" for badge in item.badges
        )

        metrics_html = "".join(
            self._render_metric(metric)
            for metric in item.metrics
        )

        score_html = ""
        if item.score is not None:
            score_html = f"<div class='timeline-hologram__score'>{float(item.score):.3f}</div>"

        item_html = [
            "<li class='timeline-hologram__item' role='listitem' tabindex='0'",
            f" data-active={'true' if active else 'false'}",
            f" aria-label='{aria_label}'{aria_current}>",
            "<div class='timeline-hologram__head'>",
            f"<span class='timeline-hologram__icon'>{icon}</span>",
            "<div class='timeline-hologram__title'>",
            f"<strong>#{rank:02d} {title}</strong>",
            f"<span>{subtitle}</span>",
            "</div>",
            score_html,
            "</div>",
        ]

        if badges_html:
            item_html.append(f"<div class='timeline-hologram__badges'>{badges_html}</div>")

        if metrics_html:
            item_html.append(f"<div class='timeline-hologram__metrics'>{metrics_html}</div>")

        item_html.append("</li>")
        return "".join(item_html)

    def _render_metric(self, metric: TimelineHologramMetric) -> str:
        tone = metric.tone if metric.tone in {"neutral", "info", "warning", "positive"} else "neutral"
        aria_attr = ""
        if metric.sr_label:
            aria_attr = f" aria-label='{escape(metric.sr_label)}'"
        return (
            f"<span class='timeline-hologram__metric' data-tone='{tone}'{aria_attr}>"
            f"<strong>{escape(metric.label)}</strong>"
            f"<span>{escape(metric.value)}</span>"
            "</span>"
        )


def guided_demo(
    *,
    steps: Sequence[TimelineMilestone],
    session_key: str = "guided_demo_step",
    step_duration: float = 8.0,
) -> Optional[TimelineMilestone]:
    """Render an overlay with rotating guidance steps."""

    _load_css()

    if not steps:
        return None

    state = st.session_state.setdefault(session_key, {"index": 0, "last_tick": 0.0})
    current_index = state["index"]

    with st.sidebar:
        st.write("### Demo asistida")
        option_labels = [step.label for step in steps]
        chosen_label = st.radio(
            "Seleccioná el foco actual",
            option_labels,
            index=current_index,
            key=f"{session_key}_selector",
        )
        current_index = option_labels.index(chosen_label)
        state["index"] = current_index

    placeholder = st.empty()
    active_step = steps[current_index]
    payload = json.dumps(
        {
            "label": active_step.label,
            "description": active_step.description,
            "icon": active_step.icon,
            "duration": step_duration,
        }
    )
    placeholder.markdown(
        f"""
        <div class='guided-overlay'>
            <div class='guided-panel' data-payload='{payload}'>
                <h3>{active_step.label}</h3>
                <p>{active_step.description}</p>
                <footer>Actualiza cada {step_duration:.1f} segundos</footer>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return active_step


# ---------------------------------------------------------------------------
# Target configurator
# ---------------------------------------------------------------------------

def _render_preset_cards(presets: Iterable[Dict]) -> str:
    st.markdown("<div class='luxe-card-grid'>", unsafe_allow_html=True)
    columns = st.columns(len(presets))
    selected_name = st.session_state.get("_target_selected", "")

    for column, preset in zip(columns, presets):
        meta = _PRESET_META.get(
            preset.get("category", ""),
            TargetPresetMeta(icon="🛰️", tagline="Optimizado para desafíos orbitales."),
        )
        is_selected = selected_name == preset["name"]
        with column:
            if st.button(
                f"{preset['name']}",
                key=f"preset_{preset['name']}",
                help=meta.tagline,
            ):
                selected_name = preset["name"]
            st.markdown(
                f"""
                <div class='luxe-card {'is-active' if is_selected else ''}'>
                    <div class='image'>{meta.icon}</div>
                    <h4>{preset['name']}</h4>
                    <div class='tagline'>{meta.tagline}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.session_state["_target_selected"] = selected_name or presets[0]["name"]
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state["_target_selected"]


def _render_indicator(value: float, max_value: float, accent: str, label: str) -> None:
    percentage = 100 * value / max_value if max_value else 0
    st.markdown(
        f"<div class='circular-indicator' style='--value:{percentage}; --accent-color:{accent};'>"
        f"<span>{percentage:.0f}%</span></div><div class='feedback-pill'>{label}: {value:.2f}</div>",
        unsafe_allow_html=True,
    )


def _gauge(title: str, value: float, max_value: float, unit: str, color: str = "#59b1ff") -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": unit},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "bgcolor": "rgba(6,20,33,0.65)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
            },
            title={"text": title},
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eaf4ff"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _build_audio_clip() -> bytes:
    """Create a short sine beep encoded as WAV bytes."""

    sample_rate = 44100
    duration = 0.25
    frequency = 880
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = np.int16(tone * 32767)

    from io import BytesIO
    import wave

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()


def target_configurator(
    presets: List[Dict],
    scenario_options: Tuple[str, ...] | List[str] | None = None,
) -> Dict:
    """Render the luxe target configurator and return the resulting target spec."""

    if not presets:
        st.warning("No hay presets disponibles para configurar el objetivo.")
        return {}

    _load_css()
    scenario_options = tuple(scenario_options or ())

    st.subheader("Target Configurator ✨")
    st.caption("Seleccioná un preset y refiná el objetivo con feedback inmediato.")

    selected_name = _render_preset_cards(presets)
    selected_preset = next(p for p in presets if p["name"] == selected_name)

    current_target = st.session_state.get("target")
    default_scenario = scenario_options[0] if scenario_options else ""
    if current_target is None:
        st.session_state["target"] = {
            **selected_preset,
            "scenario": default_scenario,
            "crew_time_low": False,
        }
    elif current_target.get("name") != selected_name:
        st.session_state["target"] = {
            **selected_preset,
            "scenario": current_target.get("scenario", default_scenario),
            "crew_time_low": current_target.get("crew_time_low", False),
        }

    current_target = st.session_state["target"]

    previous_values = st.session_state.get("_target_prev_values", {})

    base = {
        key: float(selected_preset[key]) if "max_" not in key else selected_preset[key]
        for key in ("rigidity", "tightness", "max_water_l", "max_energy_kwh", "max_crew_min")
    }

    main_col, summary_col = st.columns([3, 1])

    with main_col:
        preview_col, controls_col = st.columns([1.2, 1.8])
        with preview_col:
            st.markdown(
                f"""
                <div class="target-3d-card">
                    <div class="inner">
                        <h3>{selected_name}</h3>
                        <p>Render conceptual en vivo del objeto seleccionado.</p>
                        <div style="margin-top:2rem; font-size:0.85rem; opacity:0.85;">
                            Rigidez base: {base['rigidity']:.2f}<br/>
                            Estanqueidad base: {base['tightness']:.2f}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            scenario = ""
            if scenario_options:
                default_scenario = current_target.get("scenario", scenario_options[0])
                if default_scenario not in scenario_options:
                    default_idx = 0
                else:
                    default_idx = scenario_options.index(default_scenario)
                scenario = st.selectbox(
                    "Escenario del reto",
                    scenario_options,
                    index=default_idx,
                )
            crew_low = st.toggle(
                "Crew-time Low",
                value=current_target.get("crew_time_low", False),
                help="Prioriza procesos con poco tiempo de tripulación.",
            )
        with controls_col:
            st.markdown("#### Ajustes dinámicos")
            rigidity = st.slider(
                "Rigidez deseada",
                0.0,
                1.0,
                float(current_target.get("rigidity", selected_preset["rigidity"])),
                0.05,
            )
            _render_indicator(rigidity, 1.0, "#4ecdc4", "Rigidez")

            tightness = st.slider(
                "Estanqueidad deseada",
                0.0,
                1.0,
                float(current_target.get("tightness", selected_preset["tightness"])),
                0.05,
            )
            _render_indicator(tightness, 1.0, "#ff8c69", "Estanqueidad")

            max_water = st.slider(
                "Agua máxima (L)",
                0.0,
                3.0,
                float(current_target.get("max_water_l", selected_preset["max_water_l"])),
                0.1,
            )
            _render_indicator(max_water, 3.0, "#59b1ff", "Agua")

            max_energy = st.slider(
                "Energía máxima (kWh)",
                0.0,
                3.0,
                float(current_target.get("max_energy_kwh", selected_preset["max_energy_kwh"])),
                0.1,
            )
            _render_indicator(max_energy, 3.0, "#f8d66d", "Energía")

            max_crew = st.slider(
                "Tiempo máximo de tripulación (min)",
                5,
                60,
                int(current_target.get("max_crew_min", selected_preset["max_crew_min"])),
                1,
            )
            _render_indicator(max_crew, 60.0, "#d277ff", "Crew")

            audio_enabled = st.checkbox(
                "Audio feedback", value=st.session_state.get("_target_audio", False)
            )
            haptic_enabled = st.checkbox(
                "Vibración háptica", value=st.session_state.get("_target_haptic", False)
            )
            st.session_state["_target_audio"] = audio_enabled
            st.session_state["_target_haptic"] = haptic_enabled

            if audio_enabled:
                st.audio(_build_audio_clip(), format="audio/wav", sample_rate=44100)

    with main_col:
        st.markdown("#### Simulación visual")
        gauges = st.columns(3)
        with gauges[0]:
            _gauge("Agua", max_water, 3.0, " L")
        with gauges[1]:
            _gauge("Energía", max_energy, 3.0, " kWh", color="#f8d66d")
        with gauges[2]:
            _gauge("Crew", max_crew, 60.0, " min", color="#d277ff")

        feedback_area = st.empty()

    current_values = {
        "rigidity": rigidity,
        "tightness": tightness,
        "max_water_l": max_water,
        "max_energy_kwh": max_energy,
        "max_crew_min": max_crew,
    }

    if previous_values and current_values != previous_values:
        messages = []
        if st.session_state.get("_target_audio"):
            messages.append("🔊 Audio feedback (simulado)")
        if st.session_state.get("_target_haptic"):
            messages.append("🤲 Haptic pulse (simulado)")
        if messages:
            feedback_area.success(" ".join(messages))
    else:
        feedback_area.empty()

    st.session_state["_target_prev_values"] = current_values

    with summary_col:
        st.markdown("### Resumen")
        st.caption("Comparación contra el preset seleccionado.")

        def metric(label: str, value: float, base_value: float, unit: str = "") -> None:
            delta = value - base_value
            st.metric(label, f"{value:.2f}{unit}", f"{delta:+.2f}{unit}")

        metric("Rigidez", rigidity, base["rigidity"], "")
        metric("Estanqueidad", tightness, base["tightness"], "")
        metric("Agua máx.", max_water, base["max_water_l"], " L")
        metric("Energía máx.", max_energy, base["max_energy_kwh"], " kWh")
        metric("Crew máx.", max_crew, base["max_crew_min"], " min")

    target = {
        "name": selected_name,
        "rigidity": rigidity,
        "tightness": tightness,
        "max_water_l": max_water,
        "max_energy_kwh": max_energy,
        "max_crew_min": max_crew,
        "scenario": scenario if scenario_options else current_target.get("scenario", ""),
        "crew_time_low": crew_low,
    }

    st.session_state["target"] = target
    return target


# ---------------------------------------------------------------------------
# Luxe hero components
# ---------------------------------------------------------------------------

def _merge_styles(base: Mapping[str, str], extra: Mapping[str, str] | None = None) -> str:
    merged: dict[str, str] = dict(base)
    if extra:
        merged.update({k: v for k, v in extra.items() if v is not None})
    return "; ".join(f"{k}: {v}" for k, v in merged.items())


def _density_value(density: str, mapping: Mapping[str, str]) -> str:
    if density in mapping:
        return mapping[density]
    if "cozy" in mapping:
        return mapping["cozy"]
    return next(iter(mapping.values()))


def ChipRow(
    chips: Sequence[str | Mapping[str, str]],
    *,
    tone: str | None = None,
    size: str = "md",
    render: bool = True,
    gap: str | None = None,
) -> str:
    """Render a compact list of chips with luxe styling."""

    if render:
        _load_css()

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


@dataclass
class TeslaHero:
    title: str
    subtitle: str
    video_url: str | None = None
    chips: Sequence[str | Mapping[str, str]] = field(default_factory=list)
    icon: str | None = None
    gradient: str | None = None
    glow: str | None = None
    density: str = "cozy"
    parallax_icons: Sequence[Mapping[str, str]] = field(default_factory=list)

    def render(self) -> None:
        _load_css()
        st.markdown(self._render_markup(), unsafe_allow_html=True)

    def _render_markup(self) -> str:
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
        video_markup = ""
        if self.video_url:
            video_markup = (
                f"<video class='luxe-hero__video' autoplay muted loop playsinline>"
                f"<source src='{self.video_url}' type='video/mp4' />"
                "</video>"
                "<div class='luxe-hero__veil'></div>"
            )
        layers_html = "".join(layers)

        return f"""
        <div class='luxe-hero' style='{_merge_styles(hero_style, {})}'>
          {video_markup}
          {layers_html}
          <div class='luxe-hero__content'>
            {icon_html}
            <h1>{self.title}</h1>
            <p>{self.subtitle}</p>
            {chips_html}
          </div>
        </div>
        """

    @classmethod
    def with_briefing(
        cls,
        *,
        title: str,
        subtitle: str,
        tagline: str,
        video_url: str | None = None,
        chips: Sequence[str | Mapping[str, str]] = (),
        icon: str | None = None,
        gradient: str | None = None,
        glow: str | None = None,
        density: str = "cozy",
        parallax_icons: Sequence[Mapping[str, str]] = (),
        flow: Sequence[HeroFlowStage] = (),
        briefing_video_path: Path | str | None = None,
        briefing_cards: Sequence[BriefingCard] = (),
        steps: Sequence[tuple[str, str]] | None = None,
        metrics: Sequence[Mapping[str, Any]] = (),
        render: bool = True,
    ) -> "TeslaHeroBriefingScene":
        hero = cls(
            title=title,
            subtitle=subtitle,
            video_url=video_url,
            chips=chips,
            icon=icon,
            gradient=gradient,
            glow=glow,
            density=density,
            parallax_icons=parallax_icons,
        )

        normalized_steps: Sequence[tuple[str, str]]
        if steps is not None:
            normalized_steps = steps
        elif flow:
            normalized_steps = [stage.as_step() for stage in flow]
        else:
            normalized_steps = ()

        media_src: str | None = None
        if briefing_video_path:
            path = Path(briefing_video_path)
            media_src = _video_as_base64(path)

        scene = TeslaHeroBriefingScene(
            hero=hero,
            tagline=tagline,
            steps=normalized_steps,
            cards=list(briefing_cards),
            media_src=media_src,
            flow=list(flow),
            metrics=[dict(metric) for metric in metrics],
        )

        if render:
            scene.render()
        return scene


@dataclass
class TeslaHeroBriefingScene:
    hero: TeslaHero
    tagline: str
    steps: Sequence[tuple[str, str]]
    cards: Sequence[BriefingCard]
    media_src: str | None
    flow: Sequence[HeroFlowStage] = field(default_factory=tuple)
    metrics: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    _markup: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self._markup = self._compose_markup()

    @property
    def markup(self) -> str:
        return self._markup

    def _compose_markup(self) -> str:
        hero_markup = self.hero._render_markup()
        tagline_html = (
            f"<p class='luxe-hero__tagline'>{self.tagline}</p>" if self.tagline else ""
        )
        media_html = (
            f"<video autoplay loop muted playsinline src='{self.media_src}'></video>"
            if self.media_src
            else "<div class='briefing-fallback'>Simulación orbital</div>"
        )
        cards_html = "".join(
            f"""
            <div class='briefing-card' style="--card-accent: {card.accent};">
                <h3>{card.title}</h3>
                <p>{card.body}</p>
            </div>
            """
            for card in self.cards
        )
        steps_html = "".join(
            f"""
            <div class='briefing-step' style="animation-delay: {idx * 120}ms;">
                <span>{idx + 1}</span>
                <div>
                    <strong>{title}</strong>
                    <small>{copy}</small>
                </div>
            </div>
            """
            for idx, (title, copy) in enumerate(self.steps)
        )

        return f"""
        <section class='luxe-hero-scene'>
          <div class='luxe-hero-scene__lead'>
            {hero_markup}
            {tagline_html}
          </div>
          <div class='briefing-grid'>
            <div class='briefing-video'>{media_html}</div>
            <div class='briefing-cards'>
              {cards_html}
              <div class='briefing-stepper'>
                {steps_html}
              </div>
            </div>
          </div>
        </section>
        """

    def render(self) -> None:
        _load_css()
        st.markdown(self._markup, unsafe_allow_html=True)

    def glass_cards(self) -> list['GlassCard']:
        cards: list['GlassCard'] = []
        for stage in self.flow:
            cards.append(
                GlassCard(
                    title=stage.card_title,
                    body=stage.card_body,
                    icon=stage.icon,
                    footer=stage.footer,
                )
            )
        return cards

    def timeline_milestones(self) -> list[TimelineMilestone]:
        return [
            TimelineMilestone(
                label=stage.timeline_label,
                description=stage.timeline_description,
                icon=stage.icon,
            )
            for stage in self.flow
        ]

    def metric_items(self) -> list['MetricItem']:
        items: list['MetricItem'] = []
        for metric in self.metrics:
            items.append(
                MetricItem(
                    label=str(metric.get("label", "")),
                    value=str(metric.get("value", "")),
                    caption=metric.get("caption"),
                    delta=metric.get("delta"),
                    icon=metric.get("icon"),
                    tone=metric.get("tone"),
                )
            )
        return items

    def metrics_payload(self) -> list[Mapping[str, Any]]:
        return [dict(metric) for metric in self.metrics]

    def stage_key_for_label(self, label: str) -> str | None:
        for stage in self.flow:
            if stage.timeline_label == label:
                return stage.key
        return None


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
        _load_css()
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
# Ranking cockpit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    fmt: Callable[[Any], str] | str | None = "{:.2f}"
    unit: str | None = None
    higher_is_better: bool = True

    def format_value(self, value: Any) -> str:
        if value is None or _is_nan(value):
            return "—"
        formatter = self.fmt
        if callable(formatter):
            try:
                rendered = formatter(value)
            except Exception:  # noqa: BLE001
                rendered = str(value)
        elif isinstance(formatter, str):
            try:
                rendered = formatter.format(value)
            except Exception:  # noqa: BLE001
                rendered = str(value)
        else:
            rendered = str(value)
        if self.unit and self.unit not in rendered:
            return f"{rendered} {self.unit}".strip()
        return rendered

    def numeric_value(self, entry: Mapping[str, Any]) -> float | None:
        return _coerce_numeric(entry.get(self.key))


@dataclass
class RankingCockpit:
    entries: Sequence[Mapping[str, Any]]
    metric_specs: Sequence[MetricSpec] = field(default_factory=list)
    key: str = "ranking_cockpit"
    rank_key: str = "Rank"
    label_key: str = "Proceso"
    score_key: str = "Score"
    score_label: str | None = None
    score_fmt: Callable[[Any], str] | str | None = "{:.3f}"
    seal_key: str | None = "Seal"
    risk_key: str | None = "Riesgo"
    selection_label: str = "📌 Foco del cockpit"
    empty_message: str = "No hay candidatos para mostrar."

    def render(self, container: st.delta_generator.DeltaGenerator | None = None) -> Mapping[str, Any] | None:  # type: ignore[name-defined]
        _load_css()
        target = container if container is not None else st
        panel = target.container()

        entries = [dict(entry) for entry in self.entries]
        if not entries:
            panel.markdown(
                f"<div class='ranking-empty'>{escape(self.empty_message)}</div>",
                unsafe_allow_html=True,
            )
            return None

        sort_labels = self._sort_label_map()
        sort_options = list(sort_labels.keys())
        widget_prefix = f"{self.key}__"

        col_sort, col_dir, col_risk, col_seal = panel.columns([2.2, 1.1, 1.4, 1.4])
        sort_key = col_sort.selectbox(
            "Ordenar por",
            sort_options,
            key=f"{widget_prefix}sort",
            format_func=lambda opt: sort_labels.get(opt, opt),
        )
        direction = col_dir.radio(
            "Dirección",
            ("desc", "asc"),
            horizontal=True,
            key=f"{widget_prefix}direction",
            format_func=lambda opt: "Desc ↓" if opt == "desc" else "Asc ↑",
        )

        risk_filters: list[str] = []
        if self.risk_key:
            risk_options = self._collect_options(entries, self.risk_key)
            if risk_options:
                risk_filters = col_risk.multiselect(
                    "Riesgo",
                    risk_options,
                    default=risk_options,
                    key=f"{widget_prefix}risk",
                    format_func=self._format_risk_option,
                )
            else:
                col_risk.empty()
        else:
            col_risk.empty()

        seal_filters: list[str] = []
        if self.seal_key:
            seal_options = self._collect_options(entries, self.seal_key)
            if seal_options:
                seal_filters = col_seal.multiselect(
                    "Sellado",
                    seal_options,
                    default=seal_options,
                    key=f"{widget_prefix}seal",
                    format_func=self._format_seal_option,
                )
            else:
                col_seal.empty()
        else:
            col_seal.empty()

        filtered = [
            entry
            for entry in entries
            if self._passes_filters(entry, risk_filters, seal_filters)
        ]
        if not filtered:
            panel.markdown(
                "<div class='ranking-empty'>No hay candidatos que coincidan con los filtros.</div>",
                unsafe_allow_html=True,
            )
            return None

        descending = direction == "desc"
        self._sort_entries(filtered, sort_key, descending)

        selection_key = f"{widget_prefix}selection"
        options = list(range(len(filtered)))
        if selection_key not in st.session_state:
            st.session_state[selection_key] = 0
        elif st.session_state[selection_key] >= len(filtered):
            st.session_state[selection_key] = 0

        selected_idx = panel.selectbox(
            self.selection_label,
            options,
            key=selection_key,
            format_func=lambda idx: self._selection_label(filtered[idx], idx),
        )
        if not isinstance(selected_idx, int) or selected_idx >= len(filtered):
            selected_idx = 0

        scales = self._metric_scales(filtered)
        prepared = self._prepare_entries(filtered, scales)
        markup = self._build_cards(prepared, selected_idx)
        panel.markdown(markup, unsafe_allow_html=True)

        return prepared[selected_idx]["entry"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sort_label_map(self) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        if self.score_key:
            labels[self.score_key] = self.score_label or self.score_key
        for spec in self.metric_specs:
            labels[spec.key] = spec.label
        return labels

    def _collect_options(self, entries: Sequence[Mapping[str, Any]], key: str | None) -> list[str]:
        values: set[str] = set()
        if key is None:
            return []
        for entry in entries:
            values.add(_normalize_str(entry.get(key)))
        return sorted(values)

    def _passes_filters(
        self,
        entry: Mapping[str, Any],
        risk_filters: Sequence[str],
        seal_filters: Sequence[str],
    ) -> bool:
        if self.risk_key and risk_filters:
            label = _normalize_str(entry.get(self.risk_key))
            if label not in risk_filters:
                return False
        if self.seal_key and seal_filters:
            label = _normalize_str(entry.get(self.seal_key))
            if label not in seal_filters:
                return False
        return True

    def _sort_entries(self, entries: list[Mapping[str, Any]], key: str, descending: bool) -> None:
        def sort_value(entry: Mapping[str, Any]) -> Any:
            numeric = _coerce_numeric(entry.get(key))
            if numeric is not None:
                return numeric
            raw = entry.get(key)
            if raw is None:
                return float("-inf") if descending else float("inf")
            return str(raw)

        entries.sort(key=sort_value, reverse=descending)

    def _metric_scales(self, entries: Sequence[Mapping[str, Any]]) -> Dict[str, tuple[float, float] | None]:
        scales: Dict[str, tuple[float, float] | None] = {}
        for spec in self.metric_specs:
            values = [spec.numeric_value(entry) for entry in entries]
            numeric_values = [val for val in values if val is not None]
            if not numeric_values:
                scales[spec.key] = None
                continue
            scales[spec.key] = (min(numeric_values), max(numeric_values))
        return scales

    def _prepare_entries(
        self,
        entries: Sequence[Mapping[str, Any]],
        scales: Mapping[str, tuple[float, float] | None],
    ) -> list[Dict[str, Any]]:
        prepared: list[Dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            metrics_payload: list[Dict[str, Any]] = []
            for spec in self.metric_specs:
                bounds = scales.get(spec.key)
                numeric_value = spec.numeric_value(entry)
                fill_pct = self._metric_fill(numeric_value, bounds, spec.higher_is_better)
                tone = self._bar_tone(fill_pct)
                metrics_payload.append(
                    {
                        "label": spec.label,
                        "display": spec.format_value(entry.get(spec.key)),
                        "fill": fill_pct,
                        "tone": tone,
                        "key": spec.key,
                    }
                )

            risk_label = _normalize_str(entry.get(self.risk_key)) if self.risk_key else ""
            risk_tone = self._risk_tone(risk_label)
            risk_display = self._format_risk_display(risk_label, risk_tone)

            seal_label = _normalize_str(entry.get(self.seal_key)) if self.seal_key else ""
            seal_status, seal_display = self._seal_display(seal_label)

            prepared.append(
                {
                    "entry": entry,
                    "position": idx + 1,
                    "label": _normalize_str(entry.get(self.label_key)),
                    "score": entry.get(self.score_key),
                    "score_display": self._format_score(entry.get(self.score_key)),
                    "metrics": metrics_payload,
                    "risk_label": risk_label,
                    "risk_tone": risk_tone,
                    "risk_display": risk_display,
                    "seal_label": seal_label,
                    "seal_status": seal_status,
                    "seal_display": seal_display,
                    "original_rank": entry.get(self.rank_key),
                }
            )
        return prepared

    def _metric_fill(
        self,
        value: float | None,
        bounds: tuple[float, float] | None,
        higher_is_better: bool,
    ) -> float:
        if value is None or bounds is None:
            return 0.0
        lower, upper = bounds
        if math.isclose(lower, upper):
            width = 100.0
        else:
            width = (value - lower) / (upper - lower) * 100.0
        width = max(0.0, min(100.0, width))
        if not higher_is_better:
            width = 100.0 - width
        return width

    def _bar_tone(self, width: float) -> str:
        if width >= 66:
            return "high"
        if width >= 33:
            return "med"
        return "low"

    def _risk_tone(self, label: str) -> str:
        if not label or label == "—":
            return "low"
        lowered = label.lower()
        cleaned = _strip_accents(lowered)
        if any(token in cleaned for token in ("crit", "critico", "alto", "high", "red", "rojo")) or "⚠" in label:
            return "high"
        if any(token in cleaned for token in ("med", "medio", "moderado", "amarillo", "amber")):
            return "med"
        return "low"

    def _format_risk_display(self, label: str, tone: str) -> str:
        if not label or label == "—":
            label = "Sin dato"
        icon_map = {"high": "⚠️", "med": "🟡", "low": "🛡️"}
        icon = icon_map.get(tone, "🛡️")
        return label if icon in label else f"{icon} {label}"

    def _seal_display(self, label: str) -> tuple[str, str]:
        normalized = label
        lowered = normalized.lower()
        cleaned = _strip_accents(lowered)
        if not normalized or normalized == "—":
            return ("neutral", "🔍 Sin dato")
        if "⚠" in normalized or "❌" in normalized or any(
            token in cleaned for token in ("riesgo", "warn", "fail", "falla", "no")):
            return ("warn", "⚠️ Revisar sellado" if normalized in {"⚠️", "❌"} else normalized)
        if "✅" in normalized or "✔" in normalized or any(
            token in cleaned for token in ("ok", "pass", "sellado ok", "sellado", "si", "yes", "true")):
            return ("ok", "✅ Sellado OK" if normalized in {"✅", "✔️", "✔"} else normalized)
        if "revis" in cleaned:
            return ("warn", normalized)
        return ("neutral", normalized)

    def _format_seal_option(self, label: str) -> str:
        status, display = self._seal_display(label)
        return display

    def _format_risk_option(self, label: str) -> str:
        tone = self._risk_tone(label)
        return self._format_risk_display(label, tone)

    def _format_score(self, value: Any) -> str:
        if value is None or _is_nan(value):
            return "—"
        formatter = self.score_fmt
        if callable(formatter):
            try:
                return formatter(value)
            except Exception:  # noqa: BLE001
                return str(value)
        if isinstance(formatter, str):
            try:
                return formatter.format(value)
            except Exception:  # noqa: BLE001
                return str(value)
        return str(value)

    def _selection_label(self, entry: Mapping[str, Any], idx: int) -> str:
        label = _normalize_str(entry.get(self.label_key))
        score_display = self._format_score(entry.get(self.score_key))
        return f"#{idx + 1} · {label} ({score_display})"

    def _build_cards(self, prepared: Sequence[Mapping[str, Any]], selected_idx: int) -> str:
        cards: list[str] = []
        score_label = escape(self.score_label or self.score_key)
        for idx, item in enumerate(prepared):
            classes = ["ranking-card"]
            risk_tone = item.get("risk_tone", "low")
            if risk_tone in {"high", "med"}:
                classes.append(f"tone-{risk_tone}")
            if idx == selected_idx:
                classes.append("selected")

            chips: list[str] = []
            seal_status = item.get("seal_status", "neutral")
            seal_display = item.get("seal_display", "")
            if seal_display:
                seal_class = f" seal-{seal_status}" if seal_status in {"ok", "warn"} else ""
                chips.append(
                    f"<span class='ranking-chip{seal_class}'>{escape(seal_display)}</span>"
                )
            risk_display = item.get("risk_display", "")
            if risk_display:
                chips.append(
                    f"<span class='ranking-chip risk-{risk_tone}'>{escape(risk_display)}</span>"
                )
            chips_block = (
                f"<div class='ranking-card__chips'>{''.join(chips)}</div>" if chips else ""
            )

            metrics_html = "".join(
                f"""
                <div class='ranking-metric'>
                  <div class='ranking-metric__label'>
                    <span>{escape(metric['label'])}</span>
                    <span class='ranking-metric__value'>{escape(metric['display'])}</span>
                  </div>
                  <div class='ranking-bar'>
                    <div class='ranking-bar__fill' data-metric='{escape(metric['key'])}' data-rank='{item['position']}' style='--fill:{metric['fill']:.1f}%;' data-tone='{metric['tone']}'></div>
                  </div>
                </div>
                """
                for metric in item.get("metrics", [])
            )

            origin_html = ""
            original_rank = item.get("original_rank")
            if original_rank not in (None, "", item.get("position")):
                origin_html = (
                    f"<span class='ranking-card__origin'>Rank base #{escape(str(original_rank))}</span>"
                )

            card_html = f"""
            <article class='{_class_names(classes)}' data-order='{idx}' data-rank='{item['position']}'>
              <div class='ranking-card__header'>
                <span class='ranking-card__rank'>#{item['position']}</span>
                <div class='ranking-card__title'>
                  {escape(item.get('label', '—'))}
                  {origin_html}
                </div>
                <div class='ranking-card__score'>
                  {score_label}
                  <strong>{escape(item.get('score_display', '—'))}</strong>
                </div>
              </div>
              {chips_block}
              <div class='ranking-card__metrics'>
                {metrics_html}
              </div>
            </article>
            """
            cards.append(card_html)

        return "<div class='ranking-cockpit__grid'>" + "".join(cards) + "</div>"


@dataclass
class MissionMetric:
    key: str
    label: str
    value: str
    details: Sequence[str] = field(default_factory=tuple)
    caption: str | None = None
    icon: str | None = None
    stage_key: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MissionMetric":
        details: Sequence[str] | str | None = payload.get("details")
        if details is None:
            normalized: Sequence[str] = ()
        elif isinstance(details, str):
            normalized = (details,)
        else:
            normalized = tuple(str(item) for item in details)
        return cls(
            key=str(payload.get("key", payload.get("stage_key", ""))),
            label=str(payload.get("label", "")),
            value=str(payload.get("value", "")),
            details=normalized,
            caption=payload.get("caption"),
            icon=payload.get("icon"),
            stage_key=payload.get("stage_key"),
        )


@dataclass
class MissionMetrics:
    metrics: Sequence[MissionMetric]
    title: str = "Panel de misión"
    panel_density: str = "cozy"
    grid_density: str = "cozy"
    columns_min: str = "14rem"
    animate: bool = True

    @classmethod
    def from_payload(
        cls,
        payload: Sequence[Mapping[str, Any]],
        *,
        title: str = "Panel de misión",
        panel_density: str = "cozy",
        grid_density: str = "cozy",
        columns_min: str = "14rem",
        animate: bool = True,
    ) -> "MissionMetrics":
        metrics = [MissionMetric.from_mapping(item) for item in payload]
        return cls(
            metrics=metrics,
            title=title,
            panel_density=panel_density,
            grid_density=grid_density,
            columns_min=columns_min,
            animate=animate,
        )

    def markup(
        self,
        *,
        highlight_key: str | None = None,
        layout: Literal["panel", "grid"] = "panel",
        detail_limit: int | None = None,
        show_title: bool | None = None,
    ) -> str:
        variant = layout
        highlight = highlight_key
        limit = max(detail_limit, 0) if detail_limit is not None else None

        def metric_markup(metric: MissionMetric) -> str:
            classes = ["luxe-mission-metric"]
            if highlight and metric.stage_key == highlight:
                classes.append("is-active")
            icon_html = (
                f"<span class='luxe-mission-metric__icon'>{metric.icon}</span>"
                if metric.icon
                else ""
            )
            details = metric.details
            if limit is not None:
                details = tuple(details)[:limit]
            detail_html = "".join(f"<p>{text}</p>" for text in details)
            caption_html = (
                f"<p class='luxe-mission-metric__caption'>{metric.caption}</p>"
                if metric.caption and limit is None
                else ""
            )
            return (
                f"<div class='{_class_names(classes)}' data-key='{metric.key}'>"
                f"{icon_html}"
                f"<h5>{metric.label}</h5>"
                f"<strong>{metric.value}</strong>"
                f"{detail_html}"
                f"{caption_html}"
                "</div>"
            )

        if variant == "grid":
            gap_map = {"compact": "0.85rem", "cozy": "1rem", "roomy": "1.3rem"}
            padding_map = {"compact": "16px 18px", "cozy": "18px 20px", "roomy": "20px 22px"}
            style = {
                "--mission-grid-min": self.columns_min,
                "--mission-grid-gap": gap_map.get(self.grid_density, gap_map["cozy"]),
                "--mission-grid-card-padding": padding_map.get(
                    self.grid_density, padding_map["cozy"]
                ),
            }
            classes = ["luxe-mission-grid"]
            if self.animate:
                classes.append("reveal")
            metrics_html = "".join(metric_markup(metric) for metric in self.metrics)
            return (
                f"<div class='{_class_names(classes)}' style='{_merge_styles(style, {})}'>"
                f"{metrics_html}"
                "</div>"
            )

        padding_map = {"compact": "20px 22px", "cozy": "24px 26px", "roomy": "28px 30px"}
        metric_padding_map = {"compact": "12px 14px", "cozy": "14px 16px", "roomy": "16px 18px"}
        style = {
            "--mission-panel-padding": padding_map.get(self.panel_density, padding_map["cozy"]),
            "--mission-metric-padding": metric_padding_map.get(
                self.panel_density, metric_padding_map["cozy"]
            ),
        }
        classes = ["luxe-mission-panel"]
        if self.animate:
            classes.append("reveal")
        show_heading = self.title if show_title is None else show_title
        heading_html = (
            f"<h3 class='luxe-mission-panel__title'>{self.title}</h3>"
            if show_heading
            else ""
        )
        metrics_html = "".join(metric_markup(metric) for metric in self.metrics)
        return (
            f"<aside class='{_class_names(classes)}' id='sticky-metrics' style='{_merge_styles(style, {})}'>"
            f"{heading_html}"
            "<div class='luxe-mission-panel__metrics'>"
            f"{metrics_html}"
            "</div>"
            "</aside>"
        )

    def render(
        self,
        *,
        highlight_key: str | None = None,
        layout: Literal["panel", "grid"] = "panel",
        detail_limit: int | None = None,
        show_title: bool | None = None,
    ) -> None:
        _load_css()
        st.markdown(
            self.markup(
                highlight_key=highlight_key,
                layout=layout,
                detail_limit=detail_limit,
                show_title=show_title,
            ),
            unsafe_allow_html=True,
        )


@dataclass
class CarouselItem:
    title: str
    value: str | None = None
    description: str | None = None
    icon: str | None = None

    def markup(self) -> str:
        icon_html = (
            f"<div class='luxe-carousel-card__icon'>{self.icon}</div>"
            if self.icon
            else ""
        )
        value_html = (
            f"<div class='luxe-carousel-card__value'>{self.value}</div>"
            if self.value
            else ""
        )
        description_html = (
            f"<p class='luxe-carousel-card__description'>{self.description}</p>"
            if self.description
            else ""
        )
        return (
            "<div class='luxe-carousel-card'>"
            f"{icon_html}"
            f"<h4>{self.title}</h4>"
            f"{value_html}"
            f"{description_html}"
            "</div>"
        )


@dataclass
class CarouselRail:
    items: Sequence[CarouselItem]
    data_track: str | None = None
    reveal: bool = True
    min_width: str = "15rem"
    density: str = "cozy"

    def markup(self) -> str:
        gap_map = {"compact": "0.75rem", "cozy": "1rem", "roomy": "1.25rem"}
        padding_map = {"compact": "16px", "cozy": "18px", "roomy": "20px"}
        style = {
            "--carousel-min": self.min_width,
            "--carousel-gap": gap_map.get(self.density, gap_map["cozy"]),
            "--carousel-card-padding": padding_map.get(self.density, padding_map["cozy"]),
        }
        classes = ["luxe-carousel"]
        if self.reveal:
            classes.append("reveal")
        metrics_html = "".join(item.markup() for item in self.items)
        data_attr = f" data-carousel='{self.data_track}'" if self.data_track else ""
        return (
            f"<div class='{_class_names(classes)}' style='{_merge_styles(style, {})}'{data_attr}>"
            f"{metrics_html}"
            "</div>"
        )

    def render(self) -> None:
        _load_css()
        st.markdown(self.markup(), unsafe_allow_html=True)


@dataclass
class ActionCard:
    title: str
    body: str
    icon: str | None = None
    tag: str | None = None

    def markup(self) -> str:
        tag_html = (
            f"<div class='luxe-action-card__tag'>{self.tag}</div>"
            if self.tag
            else ""
        )
        icon_html = (
            f"<span class='luxe-action-card__icon'>{self.icon}</span>"
            if self.icon
            else ""
        )
        return (
            "<div class='luxe-action-card'>"
            f"{icon_html}"
            f"{tag_html}"
            f"<h3 class='luxe-action-card__title'>{self.title}</h3>"
            f"<p class='luxe-action-card__body'>{self.body}</p>"
            "</div>"
        )


@dataclass
class ActionDeck:
    cards: Sequence[ActionCard]
    columns_min: str = "16rem"
    density: str = "cozy"
    gap: str | None = None
    reveal: bool = True

    def markup(self) -> str:
        gap_map = {"compact": "0.9rem", "cozy": "1.1rem", "roomy": "1.4rem"}
        padding_map = {"compact": "18px 20px", "cozy": "22px 24px", "roomy": "26px 28px"}
        style = {
            "--action-min": self.columns_min,
            "--action-gap": self.gap or gap_map.get(self.density, gap_map["cozy"]),
            "--action-card-padding": padding_map.get(self.density, padding_map["cozy"]),
        }
        classes = ["luxe-action-deck"]
        if self.reveal:
            classes.append("reveal")
        cards_html = "".join(card.markup() for card in self.cards)
        return (
            f"<div class='{_class_names(classes)}' style='{_merge_styles(style, {})}'>"
            f"{cards_html}"
            "</div>"
        )

    def render(self) -> None:
        _load_css()
        st.markdown(self.markup(), unsafe_allow_html=True)


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

    def markup(self) -> str:
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
        return "".join(html)

    def render(self) -> None:
        _load_css()
        st.markdown(self.markup(), unsafe_allow_html=True)


@dataclass
class MissionFlowShowcase:
    stages: Sequence[HeroFlowStage]
    primary_actions: Sequence[ActionCard] = field(default_factory=list)
    secondary_actions: Sequence[ActionCard] = field(default_factory=list)
    title: str | None = None
    subtitle: str | None = None
    stage_density: str = "cozy"
    mobile_stage_density: str = "compact"
    action_density: str = "cozy"
    action_columns_min: str = "15rem"
    secondary_action_columns_min: str | None = None
    timeline_title: str | None = None
    insights: Sequence[str] = field(default_factory=list)
    stage_min_width: str = "15rem"

    _ordered_stages: tuple[HeroFlowStage, ...] = field(init=False)

    def __post_init__(self) -> None:
        self._ordered_stages = tuple(sorted(self.stages, key=lambda stage: stage.order))

    def copy_sequence(self, viewport: Literal["desktop", "mobile"] = "desktop") -> list[str]:
        return [stage.copy_for_viewport(viewport) for stage in self._ordered_stages]

    def stage_titles(self) -> list[str]:
        return [stage.card_title for stage in self._ordered_stages]

    def markup(self) -> str:
        style = {
            "--mission-step-gap": _density_value(
                self.stage_density,
                {"compact": "0.85rem", "cozy": "1.1rem", "roomy": "1.35rem"},
            ),
            "--mission-step-padding": _density_value(
                self.stage_density,
                {
                    "compact": "1rem 1.05rem",
                    "cozy": "1.25rem 1.35rem",
                    "roomy": "1.5rem 1.65rem",
                },
            ),
            "--mission-step-gap-mobile": _density_value(
                self.mobile_stage_density,
                {"compact": "0.8rem", "cozy": "0.95rem", "roomy": "1.15rem"},
            ),
            "--mission-step-padding-mobile": _density_value(
                self.mobile_stage_density,
                {
                    "compact": "0.9rem 0.95rem",
                    "cozy": "1.05rem 1.1rem",
                    "roomy": "1.25rem 1.35rem",
                },
            ),
            "--mission-step-min": self.stage_min_width,
        }
        header_html = self._render_header()
        content_html = self._render_content()
        actions_html = self._render_secondary_actions()
        return (
            f"<section class='mission-flow-showcase reveal' style='{_merge_styles(style, {})}'>"
            f"{header_html}{content_html}{actions_html}"
            "</section>"
        )

    def render(self) -> None:
        _load_css()
        st.markdown(self.markup(), unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _render_header(self) -> str:
        if not self.title and not self.subtitle:
            return ""
        title_html = f"<h3>{self.title}</h3>" if self.title else ""
        subtitle_html = f"<p>{self.subtitle}</p>" if self.subtitle else ""
        return f"<header class='mission-flow-showcase__header'>{title_html}{subtitle_html}</header>"

    def _render_content(self) -> str:
        left_column = self._render_stage_column()
        right_column = self._render_right_column()
        return (
            "<div class='mission-flow-showcase__content'>"
            f"{left_column}{right_column}"
            "</div>"
        )

    def _render_stage_column(self) -> str:
        steps = []
        for stage in self._ordered_stages:
            desktop_copy = stage.copy_for_viewport("desktop")
            mobile_copy = stage.copy_for_viewport("mobile")
            footer_html = (
                f"<p class='mission-flow-showcase__step-footer'>{stage.footer}</p>"
                if stage.footer
                else ""
            )
            steps.append(
                "<article class='mission-flow-showcase__step' "
                f"data-stage='{escape(stage.key)}'>"
                "<div class='mission-flow-showcase__step-head'>"
                f"<span class='mission-flow-showcase__step-icon'>{escape(stage.icon)}</span>"
                "<div>"
                f"<h4>{escape(stage.card_title)}</h4>"
                "<p class='mission-flow-showcase__copy'>"
                f"<span data-viewport='desktop'>{desktop_copy}</span>"
                f"<span data-viewport='mobile'>{mobile_copy}</span>"
                "</p>"
                "</div>"
                "</div>"
                f"{footer_html}"
                "</article>"
            )
        steps_html = "".join(steps)
        return (
            "<div class='mission-flow-showcase__column mission-flow-showcase__column--left'>"
            "<div class='mission-flow-showcase__steps'>"
            f"{steps_html}"
            "</div>"
            "</div>"
        )

    def _render_right_column(self) -> str:
        blocks: list[str] = []
        timeline = self._render_timeline()
        if timeline:
            blocks.append(timeline)
        if self.primary_actions:
            blocks.append(
                self._render_action_deck(
                    self.primary_actions,
                    columns_min=self.action_columns_min,
                )
            )
        if not blocks:
            return ""
        return (
            "<div class='mission-flow-showcase__column mission-flow-showcase__column--right'>"
            f"{''.join(blocks)}"
            "</div>"
        )

    def _render_timeline(self) -> str:
        if not self._ordered_stages:
            return ""
        nodes = []
        for depth, stage in enumerate(self._ordered_stages):
            nodes.append(
                "<div class='orbital-node' "
                f"style='--depth: {depth * 18}px;'>"
                f"<span>{escape(stage.icon)}</span>"
                f"<h4>{stage.timeline_label}</h4>"
                f"<p>{stage.timeline_description}</p>"
                "</div>"
            )
        timeline_header = (
            f"<h4>{self.timeline_title}</h4>" if self.timeline_title else ""
        )
        insights_html = ""
        if self.insights:
            bullet_items = "".join(f"<li>{item}</li>" for item in self.insights)
            insights_html = (
                f"<ul class='mission-flow-showcase__insights'>{bullet_items}</ul>"
            )
        return (
            "<aside class='mission-flow-showcase__timeline'>"
            f"{timeline_header}"
            "<div class='orbital-timeline'>"
            f"<div class='orbital-track'>{''.join(nodes)}</div>"
            "</div>"
            f"{insights_html}"
            "</aside>"
        )

    def _render_action_deck(
        self,
        cards: Sequence[ActionCard],
        *,
        columns_min: str | None = None,
    ) -> str:
        deck = ActionDeck(
            cards=cards,
            columns_min=columns_min or self.action_columns_min,
            density=self.action_density,
            reveal=False,
        )
        return f"<div class='mission-flow-showcase__actions'>{deck.markup()}</div>"

    def _render_secondary_actions(self) -> str:
        if not self.secondary_actions:
            return ""
        columns_min = (
            self.secondary_action_columns_min
            if self.secondary_action_columns_min is not None
            else self.action_columns_min
        )
        return self._render_action_deck(
            self.secondary_actions,
            columns_min=columns_min,
        )

__all__ = [
    "BriefingCard",
    "TimelineMilestone",
    "TimelineHologram",
    "TimelineHologramItem",
    "TimelineHologramMetric",
    "HeroFlowStage",
    "TargetPresetMeta",
    "mission_briefing",
    "orbital_timeline",
    "guided_demo",
    "target_configurator",
    "render_card",
    "render_pill",
    "ChipRow",
    "TeslaHero", 
    "TeslaHeroBriefingScene",
    "MetricGalaxy",
    "MetricItem",
    "MetricSpec",
    "RankingCockpit",
    "MissionMetric",
    "MissionMetrics",
    "MissionFlowShowcase",
    "CarouselItem",
    "CarouselRail",
    "ActionCard",
    "ActionDeck",
    "GlassStack",
    "GlassCard",
]
