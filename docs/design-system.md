# Sistema de diseño

Este documento se genera desde `scripts/build_theme.py` y describe los tokens disponibles
tras compilar `app/static/design_tokens.scss`. Actualiza el SCSS y vuelve a ejecutar el
script para refrescar las tablas.

## Tokens
### Lienzo
Fondo base de la aplicación.

| Token | Valor |
| --- | --- |
| `--app-bg` | `radial-gradient(1200px 460px at 20% -20%, rgba(77, 108, 255, 0.18), transparent) #05070f` |

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

### Sombras multicapa
Capas apiladas para profundidad.

| Token | Valor |
| --- | --- |
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

### Superficies
Valores base para superficies oscuras.

| Token | Valor |
| --- | --- |
| `--surface-bg` | `rgba(12, 16, 28, 0.9)` |
| `--surface-border` | `rgba(126, 144, 184, 0.18)` |
| `--surface-ink` | `#f8fbff` |
| `--surface-muted` | `rgba(205, 215, 235, 0.75)` |

## Uso

```bash
python scripts/build_theme.py
```

El comando anterior recompila el CSS y regenera esta documentación.
