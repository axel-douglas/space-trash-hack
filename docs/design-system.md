# Sistema de diseño

Este documento se genera desde `scripts/build_theme.py` y describe los tokens disponibles
tras compilar `app/static/design_tokens.scss`. Actualiza el SCSS y vuelve a ejecutar el
script para refrescar las tablas.

## Tokens
### Accent

| Token | Valor |
| --- | --- |
| `--accent` | `#5aa9ff` |
| `--accent-soft` | `#93c5fd` |
| `--accent` | `#5aa9ff` |
| `--accent-soft` | `#93c5fd` |
| `--accent` | `#1ea7ff` |
| `--accent-soft` | `#82d3ff` |
| `--accent` | `#2563eb` |
| `--accent-soft` | `#93c5fd` |
| `--accent` | `#0f4cd6` |
| `--accent-soft` | `#5f8dff` |
| `--accent` | `#268bd2` |
| `--accent-soft` | `#2aa198` |
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
| `--badge-ok` | `#2dd4bf` |
| `--badge-warn` | `#f59e0b` |
| `--badge-risk` | `#fb7185` |
| `--badge-ok` | `#2dd4bf` |
| `--badge-warn` | `#f59e0b` |
| `--badge-risk` | `#fb7185` |
| `--badge-ok` | `#34d399` |
| `--badge-warn` | `#f59e0b` |
| `--badge-risk` | `#fb7185` |
| `--badge-ok` | `#047857` |
| `--badge-warn` | `#c2410c` |
| `--badge-risk` | `#dc2626` |
| `--badge-ok` | `#15803d` |
| `--badge-warn` | `#b45309` |
| `--badge-risk` | `#b91c1c` |
| `--badge-ok` | `#2aa198` |
| `--badge-warn` | `#cb4b16` |
| `--badge-risk` | `#dc322f` |
| `--badge-ok` | `#20897a` |
| `--badge-warn` | `#d9822b` |
| `--badge-risk` | `#c4476d` |

### Bd

| Token | Valor |
| --- | --- |
| `--bd` | `#243042` |
| `--bd` | `#243042` |
| `--bd` | `#3b82f6` |
| `--bd` | `#cfd7ef` |
| `--bd` | `#1d4ed8` |
| `--bd` | `#d9caa6` |

### Bg

| Token | Valor |
| --- | --- |
| `--bg` | `#0b0d12` |
| `--bg` | `#0b0d12` |
| `--bg` | `#010409` |
| `--bg` | `#f4f7fb` |
| `--bg` | `#ffffff` |
| `--bg` | `#fdf6e3` |

### Border

| Token | Valor |
| --- | --- |
| `--border-soft` | `#243042` |
| `--border-strong` | `#31405a` |
| `--border-soft` | `#243042` |
| `--border-strong` | `#31405a` |
| `--border-soft` | `#3b82f6` |
| `--border-strong` | `#60a5fa` |
| `--border-soft` | `#cfd7ef` |
| `--border-strong` | `#94a3b8` |
| `--border-soft` | `#1d4ed8` |
| `--border-strong` | `#1d4ed8` |
| `--border-soft` | `#d9caa6` |
| `--border-strong` | `#b5a273` |

### Card

| Token | Valor |
| --- | --- |
| `--card` | `#141c2a` |
| `--card` | `#141c2a` |
| `--card` | `#0a1220` |
| `--card` | `#ffffff` |
| `--card` | `#f5faff` |
| `--card` | `#fefcf6` |

### Chip

| Token | Valor |
| --- | --- |
| `--chip-bg` | `#16243a` |
| `--chip-border` | `#243a5f` |
| `--chip-ink` | `#f1f5ff` |
| `--chip-bg` | `#16243a` |
| `--chip-border` | `#243a5f` |
| `--chip-ink` | `#f1f5ff` |
| `--chip-bg` | `#16243a` |
| `--chip-border` | `#243a5f` |
| `--chip-ink` | `#f1f5ff` |
| `--chip-bg` | `#e2e8f9` |
| `--chip-border` | `#c3d4fb` |
| `--chip-ink` | `#1e293b` |
| `--chip-bg` | `#e0ecff` |
| `--chip-border` | `#1d4ed8` |
| `--chip-ink` | `#0f172a` |
| `--chip-bg` | `#ece3c5` |
| `--chip-border` | `#d7c69a` |
| `--chip-ink` | `#073642` |

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
| `--hero-background` | `linear-gradient(135deg, rgba(59,130,246,0.22), rgba(59,130,246,0.05)), linear-gradient(180deg, rgba(15,23,42,0.9), rgba(15,23,42,0.72))` |
| `--hero-border` | `#3b82f6` |
| `--hero-background` | `linear-gradient(135deg, rgba(59,130,246,0.22), rgba(59,130,246,0.05)), linear-gradient(180deg, rgba(15,23,42,0.9), rgba(15,23,42,0.72))` |
| `--hero-border` | `#3b82f6` |
| `--hero-background` | `linear-gradient(135deg, rgba(59,130,246,0.22), rgba(59,130,246,0.05)), linear-gradient(180deg, rgba(15,23,42,0.9), rgba(15,23,42,0.72))` |
| `--hero-border` | `#3b82f6` |
| `--hero-background` | `linear-gradient(135deg, rgba(96,165,250,0.25), rgba(96,165,250,0.05)), linear-gradient(180deg, rgba(255,255,255,0.9), rgba(236,244,255,0.78))` |
| `--hero-border` | `#60a5fa` |
| `--hero-background` | `linear-gradient(135deg, rgba(96,165,250,0.25), rgba(96,165,250,0.05)), linear-gradient(180deg, rgba(255,255,255,0.9), rgba(236,244,255,0.78))` |
| `--hero-border` | `#60a5fa` |
| `--hero-background` | `linear-gradient(135deg, rgba(38,139,210,0.2), rgba(38,139,210,0.05)), linear-gradient(180deg, rgba(253,246,227,0.92), rgba(249,240,213,0.78))` |
| `--hero-border` | `#268bd2` |

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
| `--ink` | `#f8fafc` |
| `--ink` | `#f8fafc` |
| `--ink` | `#ffffff` |
| `--ink` | `#0f172a` |
| `--ink` | `#0b1526` |
| `--ink` | `#073642` |

### Metric

| Token | Valor |
| --- | --- |
| `--metric-bg` | `#101a2a` |
| `--metric-border` | `#223551` |
| `--metric-shadow` | `0 18px 32px rgba(4, 8, 20, 0.45)` |
| `--metric-bg` | `#101a2a` |
| `--metric-border` | `#223551` |
| `--metric-shadow` | `0 18px 32px rgba(4, 8, 20, 0.45)` |
| `--metric-bg` | `#101a2a` |
| `--metric-border` | `#223551` |
| `--metric-shadow` | `none` |
| `--metric-bg` | `#f1f4fc` |
| `--metric-border` | `#cbd5f5` |
| `--metric-shadow` | `0 16px 32px rgba(15, 23, 42, 0.1)` |
| `--metric-bg` | `#f1f4fc` |
| `--metric-border` | `#cbd5f5` |
| `--metric-shadow` | `none` |
| `--metric-bg` | `#f5ebd3` |
| `--metric-border` | `#d7c69a` |
| `--metric-shadow` | `0 12px 24px rgba(7, 54, 66, 0.12)` |

### Muted

| Token | Valor |
| --- | --- |
| `--muted` | `#b8c3d5` |
| `--muted` | `#b8c3d5` |
| `--muted` | `#dbe6ff` |
| `--muted` | `#4a5a7d` |
| `--muted` | `#1f2937` |
| `--muted` | `#596a72` |

### Pill

| Token | Valor |
| --- | --- |
| `--pill-bg` | `transparent` |
| `--pill-bg` | `transparent` |
| `--pill-bg` | `transparent` |
| `--pill-bg` | `transparent` |
| `--pill-bg` | `transparent` |
| `--pill-bg` | `transparent` |

### Sombras multicapa
Capas apiladas para profundidad.

| Token | Valor |
| --- | --- |
| `--shadow-card` | `0 24px 48px rgba(4, 8, 20, 0.45)` |
| `--shadow-flat` | `none` |
| `--shadow-card` | `0 24px 48px rgba(4, 8, 20, 0.45)` |
| `--shadow-flat` | `none` |
| `--shadow-card` | `none` |
| `--shadow-flat` | `none` |
| `--shadow-card` | `0 24px 48px rgba(15, 23, 42, 0.12)` |
| `--shadow-flat` | `none` |
| `--shadow-card` | `none` |
| `--shadow-flat` | `none` |
| `--shadow-card` | `0 20px 36px rgba(7, 54, 66, 0.18)` |
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
| `--step-bg` | `#101a2a` |
| `--step-border` | `#223551` |
| `--step-icon-bg` | `rgba(90, 169, 255, 0.24)` |
| `--step-bg` | `#101a2a` |
| `--step-border` | `#223551` |
| `--step-icon-bg` | `rgba(90, 169, 255, 0.24)` |
| `--step-bg` | `#101a2a` |
| `--step-border` | `#223551` |
| `--step-icon-bg` | `rgba(90, 169, 255, 0.24)` |
| `--step-bg` | `#e9effd` |
| `--step-border` | `#c3d4fb` |
| `--step-icon-bg` | `rgba(37, 99, 235, 0.18)` |
| `--step-bg` | `#e9effd` |
| `--step-border` | `#c3d4fb` |
| `--step-icon-bg` | `rgba(15, 76, 214, 0.16)` |
| `--step-bg` | `#f2e7c9` |
| `--step-border` | `#d7c69a` |
| `--step-icon-bg` | `rgba(38, 139, 210, 0.18)` |

### Stroke

| Token | Valor |
| --- | --- |
| `--stroke` | `#1f2a3c` |
| `--stroke` | `#1f2a3c` |
| `--stroke` | `#3b82f6` |
| `--stroke` | `#d7def1` |
| `--stroke` | `#1e40af` |
| `--stroke` | `#e7dcbf` |

### Superficies
Valores base para superficies oscuras.

| Token | Valor |
| --- | --- |
| `--surface-panel` | `#101724` |
| `--surface-card` | `#141c2a` |
| `--surface-ghost` | `#1c2636` |
| `--surface-panel` | `#101724` |
| `--surface-card` | `#141c2a` |
| `--surface-ghost` | `#1c2636` |
| `--surface-panel` | `#060b14` |
| `--surface-card` | `#0a1220` |
| `--surface-ghost` | `#0f1b2c` |
| `--surface-panel` | `#ffffff` |
| `--surface-card` | `#ffffff` |
| `--surface-ghost` | `#eef2ff` |
| `--surface-panel` | `#f5faff` |
| `--surface-card` | `#f5faff` |
| `--surface-ghost` | `#e2ecff` |
| `--surface-panel` | `#fefbf3` |
| `--surface-card` | `#fefcf6` |
| `--surface-ghost` | `#f2e5c7` |
| `--surface-bg` | `rgba(12, 16, 28, 0.9)` |
| `--surface-border` | `rgba(126, 144, 184, 0.18)` |
| `--surface-ink` | `#f8fbff` |
| `--surface-muted` | `rgba(205, 215, 235, 0.75)` |

### Tone

| Token | Valor |
| --- | --- |
| `--tone-ok-bg` | `#123829` |
| `--tone-ok-fg` | `#63f0a5` |
| `--tone-ok-border` | `#1f6141` |
| `--tone-warn-bg` | `#3f2a0f` |
| `--tone-warn-fg` | `#f2c057` |
| `--tone-warn-border` | `#7a5b1f` |
| `--tone-risk-bg` | `#3d1518` |
| `--tone-risk-fg` | `#f18882` |
| `--tone-risk-border` | `#7a2b36` |
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
| `--tone-ok-bg` | `#c6f6d5` |
| `--tone-ok-fg` | `#1f5136` |
| `--tone-ok-border` | `#38a169` |
| `--tone-warn-bg` | `#fef3c7` |
| `--tone-warn-fg` | `#854d0e` |
| `--tone-warn-border` | `#f59e0b` |
| `--tone-risk-bg` | `#fee2e2` |
| `--tone-risk-fg` | `#991b1b` |
| `--tone-risk-border` | `#f87171` |
| `--tone-ok-bg` | `#bbf7d0` |
| `--tone-ok-fg` | `#166534` |
| `--tone-ok-border` | `#15803d` |
| `--tone-warn-bg` | `#fde68a` |
| `--tone-warn-fg` | `#92400e` |
| `--tone-warn-border` | `#b45309` |
| `--tone-risk-bg` | `#fecaca` |
| `--tone-risk-fg` | `#7f1d1d` |
| `--tone-risk-border` | `#b91c1c` |
| `--tone-ok-bg` | `#c8f1dc` |
| `--tone-ok-fg` | `#1d5c3a` |
| `--tone-ok-border` | `#3a9d6a` |
| `--tone-warn-bg` | `#fee6b4` |
| `--tone-warn-fg` | `#7c5416` |
| `--tone-warn-border` | `#d48b1f` |
| `--tone-risk-bg` | `#f9c0c0` |
| `--tone-risk-fg` | `#8f1d2d` |
| `--tone-risk-border` | `#d2565d` |
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
