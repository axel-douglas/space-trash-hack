# UI Components Library

La librería `app.modules.luxe_components` concentra los bloques visuales premium que usamos en la demo de Rex-AI. Cada componente inyecta automáticamente estilos (parallax, glassmorphism, brillo dinámico) y expone parámetros declarativos para adaptar el branding sin duplicar CSS.

## Componentes disponibles

### `TeslaHero`
Hero translúcido con capas parallax y chips animados.

```python
from app.modules.luxe_components import TeslaHero

TeslaHero(
    title="Generador asistido por IA",
    subtitle="Rex-AI explora combinaciones y explica cada predicción.",
    chips=[
        {"label": "RandomForest multisalida", "tone": "accent"},
        {"label": "Confianza 95%", "tone": "info"},
    ],
    icon="🤖",
    gradient="linear-gradient(135deg, rgba(59,130,246,0.24), rgba(14,165,233,0.08))",
    glow="rgba(56,189,248,0.45)",
    density="cozy",
    parallax_icons=[
        {"icon": "🛰️", "top": "20%", "left": "75%", "size": "4rem"},
        {"icon": "🧪", "top": "60%", "left": "82%", "size": "3.4rem"},
    ],
).render()
```

Parámetros clave:
- **`gradient`**: define el fondo principal (acepta cualquier expresión CSS `linear-gradient`/`radial-gradient`).
- **`glow`**: color de brillo dinámico aplicado al halo radial.
- **`density`**: `"compact"`, `"cozy"` o `"roomy"` para ajustar paddings.
- **`parallax_icons`**: lista de capas decorativas (emoji/SVG) con control de posición, tamaño y velocidad.

### `ChipRow`
Hilera de chips glassmórficos reutilizables.

```python
from app.modules.luxe_components import ChipRow

ChipRow([
    {"label": "Paso 1 — Explorar", "tone": "info"},
    {"label": "Paso 2 — Seleccionar", "tone": "info"},
    {"label": "Paso 3 — Exportar", "tone": "accent"},
])
```

- `tone` soporta `accent`, `info`, `positive`, `warning` o `None` para el estilo neutro.
- `size` (`"sm"`, `"md"`, `"lg"`) controla tipografía y padding.
- Con `render=False` devuelve el HTML para incrustarlo dentro de otro componente (por ejemplo en un `GlassCard`).

### `MetricGalaxy`
Cuadrícula de métricas con brillo dinámico.

```python
from app.modules.luxe_components import MetricGalaxy, MetricItem

MetricGalaxy(
    metrics=[
        MetricItem(label="Score máximo", value="0.92", icon="🌟"),
        MetricItem(label="Mín. Agua", value="12.5 L", icon="💧", caption="Heurística: 14.3 L", delta="Δ -1.8"),
    ],
    density="compact",
).render()
```

- Cada `MetricItem` acepta `label`, `value`, `icon`, `caption` y `delta`.
- El `delta` se renderiza en un subtítulo pequeño; usalo para diferencias heurística/ML.
- `min_width` controla el ancho mínimo de cada tarjeta dentro de la grilla.

### `MissionMetrics`
Panel pegajoso para KPIs de misión con opción de grilla responsiva.

```python
from app.modules.luxe_components import MissionMetrics

payload = [
    {
        "key": "status",
        "label": "Estado",
        "value": "✅ Modelo listo",
        "details": ["Modelo <code>rexai-rf-ensemble</code>"],
        "stage_key": "inventory",
    },
    {
        "key": "training",
        "label": "Entrenamiento",
        "value": "15 ene 2024",
        "details": ["Origen: dataset marciano", "Muestras: 1.2k"],
        "stage_key": "generator",
    },
]

mission_metrics = MissionMetrics.from_payload(payload)
mission_metrics.render(highlight_key="generator")

# Grilla compacta para resúmenes finales
mission_metrics.render(layout="grid", detail_limit=2, show_title=False)
```

- `highlight_key` resalta la métrica asociada al paso activo del flujo.
- `layout="grid"` reutiliza los mismos datos en formato tablero (ideal para secciones de resultados).
- Ajustá `panel_density`/`grid_density` a `"compact"`, `"cozy"` o `"roomy"` según el espacio disponible.

### `CarouselRail`
Hilera horizontal scrollable para resúmenes rápidos (categorías, materiales, etc.).

```python
from app.modules.luxe_components import CarouselItem, CarouselRail

CarouselRail(
    items=[
        CarouselItem(title="EVA scraps", value="320 kg", description="Volumen: 450 L"),
        CarouselItem(title="Metales", value="210 kg", description="Volumen: 180 L"),
    ],
    data_track="categorias",
    density="compact",
).render()
```

- `density` controla gap y padding de cada tarjeta.
- Podés dejar `data_track` vacío o usarlo como `data-*` para analytics.
- Si necesitás componer dentro de otra tarjeta, usá `CarouselRail(...).markup()` y embébelo manualmente.

### `ActionDeck`
Grilla declarativa de CTAs o pasos operativos.

```python
from app.modules.luxe_components import ActionCard, ActionDeck

ActionDeck(
    cards=[
        ActionCard(
            title="Exportar receta",
            body="Descargá Sankey + trazabilidad completa.",
            icon="📤",
        ),
        ActionCard(
            title="Simular objetivo",
            body="Proba presets de energía, crew y materiales.",
            icon="🧮",
        ),
    ],
    columns_min="16rem",
    density="cozy",
).render()
```

- Usa `ActionDeck(..., reveal=False)` si no necesitás la animación de entrada.
- `ActionCard.body` acepta HTML ligero (`<code>`, `<strong>`) para destacar datos clave.
- Podés encadenar varios `ActionDeck` para separar pasos y CTAs secundarios manteniendo el mismo estilo.

### `GlassStack`
Stack responsivo de tarjetas glassmórficas.

```python
from app.modules.luxe_components import GlassStack, GlassCard

GlassStack(
    cards=[
        GlassCard(
            title="Ruta de misión",
            body="Normalizá residuos, fijá límites y generá recetas.",
            icon="🧭",
            footer="Dataset NASA + crew flags",
        ),
        GlassCard(
            title="Export",
            body="Descargá JSON/CSV con trazabilidad completa.",
            icon="📦",
        ),
    ],
    columns_min="15rem",
    density="cozy",
).render()
```

- `columns_min` define el ancho mínimo de cada tarjeta (el layout salta de 1 a N columnas según el viewport).
- `density` comparte semántica con el resto de componentes (`compact`/`cozy`/`roomy`).

## Lineamientos de branding

- **Gradientes**: mantené combinaciones deep-space (`rgba(59,130,246,…)`, `rgba(14,165,233,…)`, `rgba(45,212,191,…)`) para conservar la identidad Rex-AI. Los héroes secundarios pueden usar matices turquesa/azules; reservá violetas (`rgba(99,102,241,…)`) para secciones de export o analítica.
- **Íconos**: priorizá emojis nativos relacionados con espacio/ingeniería (🛰️, 🧪, 🧑‍🚀). El hero puede mezclar iconos grandes en `parallax_icons` para storytelling; mantené 2‑3 capas máximo para evitar ruido visual.
- **Densidad**: usá `compact` para dashboards densos (Pareto), `cozy` para flujos de operación y `roomy` para la Home/hero principal.
- **Chips**: combiná tonos `accent` y `info` para destacar features; `positive`/`warning` quedan reservados a mensajes de estado (seguridad, alertas).
- **Accesibilidad**: los textos dentro de los componentes deben ser breves y contrastantes; evitá párrafos largos dentro de `GlassCard`. Para contenido extendido, usar `st.markdown` debajo del componente.

## Buenas prácticas

1. Reutilizá siempre los componentes antes de escribir HTML crudo. Si falta alguna variante, ampliá `luxe_components` para mantener consistencia.
2. Inyectá chips/metricas desde datos dinámicos: formateá strings antes de pasarlas al componente para que la lógica de presentación se mantenga declarativa.
3. Para combinar con controles de Streamlit (`st.button`, `st.slider`), renderizá el componente y luego añadí los widgets. El CSS ya incluye `backdrop-filter` y sombras, así que evitá anidar `st.container` con estilos propios.
