# Sistema de diseño

Este documento se genera desde `scripts/build_theme.py` y describe los tokens disponibles
tras compilar `app/static/design_tokens.scss`. Actualiza el SCSS y vuelve a ejecutar el
script para refrescar las tablas.

## Decisiones de tema

- `mars-minimal` se convierte en el preset predeterminado con altos contrastes y superficies planas para reducir coste de render.
- Conservamos `dark` y `dark-high-contrast` como variantes de compatibilidad para usuarios que ya dependen de esos valores.
- El bundle servido por defecto es `theme.min.css`; `theme.css` queda para depuración y la generación de esta documentación.

## Tokens
### Accent

| Token | Valor |
| --- | --- |
| `--accent` | `#ff6a3d` |
| `--accent-soft` | `#ff9d7a` |
| `--accent` | `#ff6a3d` |
| `--accent-soft` | `#ff9d7a` |
| `--accent` | `#5aa9ff` |
| `--accent-soft` | `#93c5fd` |
| `--accent` | `#1ea7ff` |
| `--accent-soft` | `#82d3ff` |
| `--accent` | `#0f7fb0` |
| `--accent-soft` | `#5cb6d8` |

### Lienzo
Fondo base de la aplicación.

| Token | Valor |
| --- | --- |
| `--app-bg` | `radial-gradient(1200px 460px at 20% -20%, rgba(77, 108, 255, 0.18), transparent) #05070f` |

### Badge

| Token | Valor |
| --- | --- |
| `--badge-ok` | `#31d0aa` |
| `--badge-warn` | `#ff8a3d` |
| `--badge-risk` | `#ff5e72` |
| `--badge-ok` | `#31d0aa` |
| `--badge-warn` | `#ff8a3d` |
| `--badge-risk` | `#ff5e72` |
| `--badge-ok` | `#2dd4bf` |
| `--badge-warn` | `#f59e0b` |
| `--badge-risk` | `#fb7185` |
| `--badge-ok` | `#34d399` |
| `--badge-warn` | `#f59e0b` |
| `--badge-risk` | `#fb7185` |
| `--badge-ok` | `#20897a` |
| `--badge-warn` | `#d9822b` |
| `--badge-risk` | `#c4476d` |

### Bd

| Token | Valor |
| --- | --- |
| `--bd` | `#512c2f` |
| `--bd` | `#512c2f` |
| `--bd` | `#243042` |
| `--bd` | `#3b82f6` |

### Bg

| Token | Valor |
| --- | --- |
| `--bg` | `#14090a` |
| `--bg` | `#14090a` |
| `--bg` | `#0b0d12` |
| `--bg` | `#010409` |

### Border

| Token | Valor |
| --- | --- |
| `--border-soft` | `#512c2f` |
| `--border-strong` | `#ff8a50` |
| `--border-soft` | `#512c2f` |
| `--border-strong` | `#ff8a50` |
| `--border-soft` | `#243042` |
| `--border-strong` | `#31405a` |
| `--border-soft` | `#3b82f6` |
| `--border-strong` | `#60a5fa` |

### Card

| Token | Valor |
| --- | --- |
| `--card` | `#201113` |
| `--card` | `#201113` |
| `--card` | `#141c2a` |
| `--card` | `#0a1220` |

### Chip

| Token | Valor |
| --- | --- |
| `--chip-bg` | `#2d1417` |
| `--chip-border` | `#633233` |
| `--chip-ink` | `#ffe8dc` |
| `--chip-bg` | `#2d1417` |
| `--chip-border` | `#633233` |
| `--chip-ink` | `#ffe8dc` |
| `--chip-bg` | `#16243a` |
| `--chip-border` | `#243a5f` |
| `--chip-ink` | `#f1f5ff` |
| `--chip-bg` | `#16243a` |
| `--chip-border` | `#243a5f` |
| `--chip-ink` | `#f1f5ff` |

### Escala neblina
Neutros fríos para fondos y bordes.

| Token | Valor |
| --- | --- |
| `--color-mist-50` | `#f5f6fa` |
| `--color-mist-100` | `#e3e7f1` |
| `--color-mist-200` | `#ced3df` |
| `--color-mist-300` | `#b5bcc9` |
| `--color-mist-400` | `#9aa3b1` |
| `--color-mist-500` | `#7b8492` |
| `--color-mist-600` | `#606978` |
| `--color-mist-700` | `#4b5260` |
| `--color-mist-800` | `#3a3f4b` |
| `--color-mist-900` | `#292c35` |

### Escala neón
Acentos brillantes para estados y CTA.

| Token | Valor |
| --- | --- |
| `--color-neon-50` | `#effdf9` |
| `--color-neon-100` | `#c6ffef` |
| `--color-neon-200` | `#8fffe1` |
| `--color-neon-300` | `#52fbd4` |
| `--color-neon-400` | `#1feecc` |
| `--color-neon-500` | `#09d6b5` |
| `--color-neon-600` | `#00ab92` |
| `--color-neon-700` | `#008674` |
| `--color-neon-800` | `#06685a` |
| `--color-neon-900` | `#064f45` |

### Escala primaria
Azules base usados para capas principales.

| Token | Valor |
| --- | --- |
| `--color-primary-50` | `#f2f6ff` |
| `--color-primary-100` | `#dbe6ff` |
| `--color-primary-200` | `#bfcfff` |
| `--color-primary-300` | `#98b0ff` |
| `--color-primary-400` | `#6f8dff` |
| `--color-primary-500` | `#4d6cff` |
| `--color-primary-600` | `#3a53d6` |
| `--color-primary-700` | `#2d40a8` |
| `--color-primary-800` | `#233280` |
| `--color-primary-900` | `#1c2864` |

### Font

| Token | Valor |
| --- | --- |
| `--font-scale` | `1` |
| `--font-scale` | `1` |
| `--font-scale` | `1.12` |
| `--font-scale` | `1.24` |

### Tipografía fluida
Clamps responsivos por jerarquía.

| Token | Valor |
| --- | --- |
| `--font-size-mega` | `clamp(2.8rem, calc(2.1rem + 1.5vw), 4.2rem)` |
| `--font-size-display` | `clamp(2.1rem, calc(1.7rem + 1vw), 3.1rem)` |
| `--font-size-headline` | `clamp(1.6rem, calc(1.35rem + 0.65vw), 2.2rem)` |
| `--font-size-title` | `clamp(1.3rem, calc(1.12rem + 0.45vw), 1.7rem)` |
| `--font-size-body` | `clamp(1rem, calc(0.95rem + 0.2vw), 1.15rem)` |
| `--font-size-small` | `clamp(0.88rem, calc(0.82rem + 0.18vw), 0.98rem)` |

### Glassmorphism
Tokens para tarjetas translúcidas.

| Token | Valor |
| --- | --- |
| `--glass-bg` | `rgba(26, 42, 88, 0.35)` |
| `--glass-border` | `rgba(132, 156, 214, 0.28)` |

### Hero

| Token | Valor |
| --- | --- |
| `--hero-background` | `#1d0c0e` |
| `--hero-border` | `#ff6a3d` |
| `--hero-background` | `#1d0c0e` |
| `--hero-border` | `#ff6a3d` |
| `--hero-background` | `linear-gradient(135deg, rgba(59,130,246,0.22), rgba(59,130,246,0.05)), linear-gradient(180deg, rgba(15,23,42,0.9), rgba(15,23,42,0.72))` |
| `--hero-border` | `#3b82f6` |
| `--hero-background` | `linear-gradient(135deg, rgba(59,130,246,0.22), rgba(59,130,246,0.05)), linear-gradient(180deg, rgba(15,23,42,0.9), rgba(15,23,42,0.72))` |
| `--hero-border` | `#3b82f6` |

### Home

| Token | Valor |
| --- | --- |
| `--home-content-max` | `58rem` |
| `--home-stack-gap` | `clamp(2.5rem, 4vw, 3.5rem)` |
| `--home-section-gap` | `clamp(1.1rem, calc(1.2rem + 0.5vw), 1.8rem)` |
| `--home-header-gap` | `var(--space-sm)` |
| `--home-icon-size` | `2.5rem` |
| `--home-icon-radius` | `0.9rem` |
| `--home-icon-font-size` | `1.35rem` |
| `--home-text-max` | `62ch` |
| `--home-lead-size` | `clamp(1rem, calc(0.98rem + 0.28vw), 1.2rem)` |
| `--home-card-gap` | `var(--space-lg)` |
| `--home-card-padding` | `clamp(1.1rem, calc(0.9rem + 1vw), 1.9rem)` |
| `--home-card-radius` | `1.25rem` |
| `--home-card-shadow` | `var(--shadow-soft)` |
| `--home-heading-spacing` | `var(--space-sm)` |
| `--home-list-gap` | `var(--space-xs)` |
| `--home-list-indent` | `1.25rem` |
| `--home-board-gap` | `var(--space-md)` |
| `--home-muted` | `var(--muted)` |
| `--home-content-max` | `58rem` |
| `--home-stack-gap` | `clamp(2.5rem, 4vw, 3.5rem)` |
| `--home-section-gap` | `clamp(1.1rem, calc(1.2rem + 0.5vw), 1.8rem)` |
| `--home-header-gap` | `var(--space-sm)` |
| `--home-icon-size` | `2.5rem` |
| `--home-icon-radius` | `0.9rem` |
| `--home-icon-font-size` | `1.35rem` |
| `--home-text-max` | `62ch` |
| `--home-lead-size` | `clamp(1rem, calc(0.98rem + 0.28vw), 1.2rem)` |
| `--home-card-gap` | `var(--space-lg)` |
| `--home-card-padding` | `clamp(1.1rem, calc(0.9rem + 1vw), 1.9rem)` |
| `--home-card-radius` | `1.25rem` |
| `--home-card-shadow` | `var(--shadow-soft)` |
| `--home-heading-spacing` | `var(--space-sm)` |
| `--home-list-gap` | `var(--space-xs)` |
| `--home-list-indent` | `1.25rem` |
| `--home-board-gap` | `var(--space-md)` |
| `--home-muted` | `var(--muted)` |
| `--home-content-max` | `58rem` |
| `--home-stack-gap` | `clamp(2.5rem, 4vw, 3.5rem)` |
| `--home-section-gap` | `clamp(1.1rem, calc(1.2rem + 0.5vw), 1.8rem)` |
| `--home-header-gap` | `var(--space-sm)` |
| `--home-icon-size` | `2.5rem` |
| `--home-icon-radius` | `0.9rem` |
| `--home-icon-font-size` | `1.35rem` |
| `--home-text-max` | `62ch` |
| `--home-lead-size` | `clamp(1rem, calc(0.98rem + 0.28vw), 1.2rem)` |
| `--home-card-gap` | `var(--space-lg)` |
| `--home-card-padding` | `clamp(1.1rem, calc(0.9rem + 1vw), 1.9rem)` |
| `--home-card-radius` | `1.25rem` |
| `--home-card-shadow` | `var(--shadow-soft)` |
| `--home-heading-spacing` | `var(--space-sm)` |
| `--home-list-gap` | `var(--space-xs)` |
| `--home-list-indent` | `1.25rem` |
| `--home-board-gap` | `var(--space-md)` |
| `--home-muted` | `var(--muted)` |
| `--home-content-max` | `58rem` |
| `--home-stack-gap` | `clamp(2.5rem, 4vw, 3.5rem)` |
| `--home-section-gap` | `clamp(1.1rem, calc(1.2rem + 0.5vw), 1.8rem)` |
| `--home-header-gap` | `var(--space-sm)` |
| `--home-icon-size` | `2.5rem` |
| `--home-icon-radius` | `0.9rem` |
| `--home-icon-font-size` | `1.35rem` |
| `--home-text-max` | `62ch` |
| `--home-lead-size` | `clamp(1rem, calc(0.98rem + 0.28vw), 1.2rem)` |
| `--home-card-gap` | `var(--space-lg)` |
| `--home-card-padding` | `clamp(1.1rem, calc(0.9rem + 1vw), 1.9rem)` |
| `--home-card-radius` | `1.25rem` |
| `--home-card-shadow` | `var(--shadow-soft)` |
| `--home-heading-spacing` | `var(--space-sm)` |
| `--home-list-gap` | `var(--space-xs)` |
| `--home-list-indent` | `1.25rem` |
| `--home-board-gap` | `var(--space-md)` |
| `--home-muted` | `var(--muted)` |

### Ink

| Token | Valor |
| --- | --- |
| `--ink` | `#fff5ed` |
| `--ink` | `#fff5ed` |
| `--ink` | `#f8fafc` |
| `--ink` | `#ffffff` |

### Metric

| Token | Valor |
| --- | --- |
| `--metric-bg` | `#1b0b0d` |
| `--metric-border` | `#633233` |
| `--metric-shadow` | `none` |
| `--metric-bg` | `#1b0b0d` |
| `--metric-border` | `#633233` |
| `--metric-shadow` | `none` |
| `--metric-bg` | `#101a2a` |
| `--metric-border` | `#223551` |
| `--metric-shadow` | `0 18px 32px rgba(4, 8, 20, 0.45)` |
| `--metric-bg` | `#101a2a` |
| `--metric-border` | `#223551` |
| `--metric-shadow` | `none` |

### Muted

| Token | Valor |
| --- | --- |
| `--muted` | `#f5b79f` |
| `--muted` | `#f5b79f` |
| `--muted` | `#b8c3d5` |
| `--muted` | `#dbe6ff` |

### Pill

| Token | Valor |
| --- | --- |
| `--pill-bg` | `transparent` |
| `--pill-bg` | `transparent` |
| `--pill-bg` | `transparent` |
| `--pill-bg` | `transparent` |

### Sombras multicapa
Capas apiladas para profundidad.

| Token | Valor |
| --- | --- |
| `--shadow-card` | `none` |
| `--shadow-flat` | `none` |
| `--shadow-card` | `none` |
| `--shadow-flat` | `none` |
| `--shadow-card` | `0 24px 48px rgba(4, 8, 20, 0.45)` |
| `--shadow-flat` | `none` |
| `--shadow-card` | `none` |
| `--shadow-flat` | `none` |
| `--shadow-soft` | `0 2px 6px -2px rgba(12, 18, 37, 0.4), 0 1px 0 rgba(255, 255, 255, 0.04)` |
| `--shadow-lift` | `0 8px 24px -6px rgba(24, 44, 88, 0.42), 0 1px 0 rgba(255, 255, 255, 0.06)` |
| `--shadow-float` | `0 18px 45px -10px rgba(10, 18, 40, 0.48), 0 4px 18px -6px rgba(46, 115, 255, 0.3)` |

### Ritmo de espaciado
Escalas en rem para layout.

| Token | Valor |
| --- | --- |
| `--space-px` | `1px` |
| `--space-3xs` | `0.125rem` |
| `--space-2xs` | `0.25rem` |
| `--space-xs` | `0.5rem` |
| `--space-sm` | `0.75rem` |
| `--space-md` | `1rem` |
| `--space-lg` | `1.5rem` |
| `--space-xl` | `2rem` |
| `--space-2xl` | `3rem` |
| `--space-3xl` | `4rem` |

### Tokens de estado
Transformaciones para hover/press/focus.

| Token | Valor |
| --- | --- |
| `--state-hover-filter` | `brightness(1.05) saturate(1.1)` |
| `--state-press-filter` | `brightness(0.95) saturate(0.95)` |
| `--state-focus-ring` | `0 0 0 3px rgba(111, 141, 255, 0.4)` |

### Step

| Token | Valor |
| --- | --- |
| `--step-bg` | `#1b0b0d` |
| `--step-border` | `#633233` |
| `--step-icon-bg` | `rgba(255, 106, 61, 0.24)` |
| `--step-bg` | `#1b0b0d` |
| `--step-border` | `#633233` |
| `--step-icon-bg` | `rgba(255, 106, 61, 0.24)` |
| `--step-bg` | `#101a2a` |
| `--step-border` | `#223551` |
| `--step-icon-bg` | `rgba(90, 169, 255, 0.24)` |
| `--step-bg` | `#101a2a` |
| `--step-border` | `#223551` |
| `--step-icon-bg` | `rgba(90, 169, 255, 0.24)` |

### Stroke

| Token | Valor |
| --- | --- |
| `--stroke` | `#402325` |
| `--stroke` | `#402325` |
| `--stroke` | `#1f2a3c` |
| `--stroke` | `#3b82f6` |

### Superficies
Valores base para superficies oscuras.

| Token | Valor |
| --- | --- |
| `--surface-panel` | `#1c0d0f` |
| `--surface-card` | `#201113` |
| `--surface-ghost` | `#2a1417` |
| `--surface-panel` | `#1c0d0f` |
| `--surface-card` | `#201113` |
| `--surface-ghost` | `#2a1417` |
| `--surface-panel` | `#101724` |
| `--surface-card` | `#141c2a` |
| `--surface-ghost` | `#1c2636` |
| `--surface-panel` | `#060b14` |
| `--surface-card` | `#0a1220` |
| `--surface-ghost` | `#0f1b2c` |
| `--surface-bg` | `rgba(12, 16, 28, 0.9)` |
| `--surface-border` | `rgba(126, 144, 184, 0.18)` |
| `--surface-ink` | `#f8fbff` |
| `--surface-muted` | `rgba(205, 215, 235, 0.75)` |

### Tone

| Token | Valor |
| --- | --- |
| `--tone-ok-bg` | `#11271f` |
| `--tone-ok-fg` | `#7fffc2` |
| `--tone-ok-border` | `#2dd4bf` |
| `--tone-warn-bg` | `#2b160a` |
| `--tone-warn-fg` | `#ffdd99` |
| `--tone-warn-border` | `#ff8a3d` |
| `--tone-risk-bg` | `#301013` |
| `--tone-risk-fg` | `#ff9da5` |
| `--tone-risk-border` | `#ff5e72` |
| `--tone-ok-bg` | `#11271f` |
| `--tone-ok-fg` | `#7fffc2` |
| `--tone-ok-border` | `#2dd4bf` |
| `--tone-warn-bg` | `#2b160a` |
| `--tone-warn-fg` | `#ffdd99` |
| `--tone-warn-border` | `#ff8a3d` |
| `--tone-risk-bg` | `#301013` |
| `--tone-risk-fg` | `#ff9da5` |
| `--tone-risk-border` | `#ff5e72` |
| `--tone-ok-bg` | `#123829` |
| `--tone-ok-fg` | `#63f0a5` |
| `--tone-ok-border` | `#1f6141` |
| `--tone-warn-bg` | `#3f2a0f` |
| `--tone-warn-fg` | `#f2c057` |
| `--tone-warn-border` | `#7a5b1f` |
| `--tone-risk-bg` | `#3d1518` |
| `--tone-risk-fg` | `#f18882` |
| `--tone-risk-border` | `#7a2b36` |
| `--tone-ok-bg` | `#083d2b` |
| `--tone-ok-fg` | `#7fffd4` |
| `--tone-ok-border` | `#34d399` |
| `--tone-warn-bg` | `#4a2a00` |
| `--tone-warn-fg` | `#fbbf24` |
| `--tone-warn-border` | `#f59e0b` |
| `--tone-risk-bg` | `#470011` |
| `--tone-risk-fg` | `#ff7b89` |
| `--tone-risk-border` | `#fb7185` |
| `--tone-ok-bg` | `#1b4d3a` |
| `--tone-ok-fg` | `#a4e9c0` |
| `--tone-ok-border` | `#2c7a58` |
| `--tone-warn-bg` | `#4a3b16` |
| `--tone-warn-fg` | `#f2c14e` |
| `--tone-warn-border` | `#b98020` |
| `--tone-risk-bg` | `#4a2633` |
| `--tone-risk-fg` | `#f29cb5` |
| `--tone-risk-border` | `#a8516e` |

## Uso

```bash
python scripts/build_theme.py
```

El comando anterior recompila el CSS y regenera esta documentación.
