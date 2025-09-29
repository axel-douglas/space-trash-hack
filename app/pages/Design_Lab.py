from __future__ import annotations

from textwrap import dedent

import streamlit as st

from app.modules import ui_blocks

st.set_page_config(page_title="Design Lab", layout="wide")
ui_blocks.load_theme()

st.title("Design Lab")
st.caption("Previsualización de tokens antes de aplicarlos en el resto de la interfaz.")

color_scales = {
    "Primarios": "color-primary",
    "Neón": "color-neon",
    "Neblina": "color-mist",
}

with ui_blocks.surface(tone="raised"):
    st.subheader("Paletas cromáticas")
    for label, prefix in color_scales.items():
        st.markdown(f"#### {label}")
        swatches = []
        for step in (50, 100, 200, 300, 400, 500, 600, 700, 800, 900):
            swatches.append(
                f"""
                <div class="rex-token-swatch">
                    <div class="rex-token-swatch__visual" style="background: {ui_blocks.use_token(f'{prefix}-{step}')};"></div>
                    <div style="display:flex;flex-direction:column;gap:var(--space-3xs);">
                        <strong>{prefix.replace('color-', '').title()} {step}</strong>
                        <span class="text-small">{ui_blocks.use_token(f'{prefix}-{step}')}</span>
                    </div>
                </div>
                """
            )
        grid_html = "".join(swatches)
        st.markdown(f'<div class="rex-token-grid">{grid_html}</div>', unsafe_allow_html=True)

with ui_blocks.surface():
    st.subheader("Tipografía fluida")
    type_tokens = [
        ("Mega", "font-size-mega", "Hero headlines"),
        ("Display", "font-size-display", "Portadas y métricas clave"),
        ("Headline", "font-size-headline", "Secciones destacadas"),
        ("Title", "font-size-title", "Encabezados secundarios"),
        ("Body", "font-size-body", "Texto de párrafo"),
        ("Small", "font-size-small", "Metadatos y notas"),
    ]
    cols = st.columns(2)
    for idx, (label, token, usage) in enumerate(type_tokens):
        target = cols[idx % 2]
        target.markdown(
            f"""
            <div class="rex-type-stack">
                <h4 style="font-size:{ui_blocks.use_token(token)}; margin:0;">{label}</h4>
                <p>{usage}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with ui_blocks.surface(tone="sunken"):
    st.subheader("Ritmo de espaciado")
    spacings = ["px", "3xs", "2xs", "xs", "sm", "md", "lg", "xl", "2xl", "3xl"]
    spacing_blocks = []
    for key in spacings:
        spacing_blocks.append(
            f"""
            <div class="rex-token-swatch">
                <div style="display:flex;align-items:center;gap:var(--space-sm);">
                    <div style="width:calc({ui_blocks.use_token(f'space-{key}')} * 6);height:var(--space-sm);background:{ui_blocks.use_token('color-neon-500')};border-radius:999px;"></div>
                    <strong>space-{key}</strong>
                </div>
                <span class="text-small">{ui_blocks.use_token(f'space-{key}')}</span>
            </div>
            """
        )
    spacing_html = "".join(spacing_blocks)
    st.markdown(f'<div class="rex-token-grid">{spacing_html}</div>', unsafe_allow_html=True)

with ui_blocks.glass_card():
    st.subheader("Estados y superficies")
    st.markdown(
        dedent(
            f"""
            <div class="rex-token-sample">
                <button style="padding:var(--space-xs) var(--space-md);border-radius:999px;border:none;background:{ui_blocks.use_token('color-primary-500')};color:var(--surface-ink);box-shadow:var(--shadow-lift);transition:all 120ms ease;">
                    Hover / Press Demo
                </button>
                <div style="display:flex;gap:var(--space-md);flex-wrap:wrap;">
                    <div class="rex-surface" style="padding:var(--space-md);width:220px;">
                        <strong>Surface base</strong>
                        <p class="text-small">Usa var(--surface-bg) y var(--shadow-soft)</p>
                    </div>
                    <div class="rex-surface" data-tone="raised" style="padding:var(--space-md);width:220px;">
                        <strong>Surface raised</strong>
                        <p class="text-small">Varía con data-tone="raised"</p>
                    </div>
                    <div class="rex-glass" style="padding:var(--space-md);width:220px;">
                        <strong>Glass card</strong>
                        <p class="text-small">Basada en tokens glass-*</p>
                    </div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
