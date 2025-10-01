# UI Components Library

La librería `app.modules.luxe_components` concentra los bloques visuales premium que usamos en la demo de Rex-AI. Cada componente inyecta automáticamente estilos (parallax, glassmorphism, brillo dinámico) y expone parámetros declarativos para adaptar el branding sin duplicar CSS.

> **Copy minimalista**: cuando documentes o agregues componentes, alineá los textos con la [Guía de copy minimalista](./ux-copy-minimal.md). Usá títulos cortos, verbos de acción y evita metáforas para mantener una narrativa consistente en toda la interfaz.

## Componentes disponibles

### `TeslaHero`
Hero translúcido con capas parallax y chips animados. Ahora soporta variantes `cinematic` (por defecto) y `minimal` para adaptar la densidad visual según la escena.

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
    variant="minimal",
).render()
```

Parámetros clave:
- **`gradient`**: define el fondo principal (acepta cualquier expresión CSS `linear-gradient`/`radial-gradient`).
- **`glow`**: color de brillo dinámico aplicado al halo radial.
- **`density`**: `"compact"`, `"cozy"` o `"roomy"` para ajustar paddings.
- **`parallax_icons`**: lista de capas decorativas (emoji/SVG) con control de posición, tamaño y velocidad (solo visibles en modo `cinematic`).
- **`variant`**: `"cinematic"` activa el loop de video y las capas parallax (ideal para briefings o demos inmersivas). `"minimal"` elimina el video y los elementos flotantes, reduce padding/typografía y se usa en vistas operativas (Home, Generador, Resultados) para priorizar métricas.

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

### `TimelineHologram`
Timeline lateral con animaciones Framer Motion y estados accesibles.

```python
from app.modules.luxe_components import (
    TimelineHologram,
    TimelineHologramItem,
    TimelineHologramMetric,
)

hologram = TimelineHologram(
    items=[
        TimelineHologramItem(
            title="Moldeado orbital",
            subtitle="ID RX-01",
            score=0.82,
            icon="🛠️",
            badges=("🛡️ Seal ready", "♻️ Problemáticos"),
            metrics=(
                TimelineHologramMetric(label="Rigidez", value="0.76", tone="positive"),
                TimelineHologramMetric(
                    label="Agua",
                    value="0.38 L · 38% máx",
                    tone="info",
                    sr_label="Agua 0.38 litros, 38 por ciento del máximo permitido.",
                ),
            ),
        ),
        TimelineHologramItem(
            title="Impulso EVA",
            subtitle="ID RX-02",
            score=0.77,
            icon="🚀",
            metrics=(
                TimelineHologramMetric(label="Rigidez", value="0.70"),
                TimelineHologramMetric(label="Agua", value="0.55 L · 55% máx", tone="warning"),
            ),
        ),
    ],
    priority_label="Prioridad rigidez ↔ agua",
    priority_value=0.6,
    priority_detail="Valores altos favorecen rigidez; bajos priorizan agua.",
    caption="Orden sugerido según la ponderación elegida. Cada nodo resume score, rigidez y agua.",
)

hologram.render()
```

- **Accesibilidad**: cada `TimelineHologramItem` expone `aria_label`, foco con `tabindex` y marca el elemento recomendado con `aria-current`.
- **Animaciones**: se importa Framer Motion (vía CDN) una sola vez. Si el usuario tiene `prefers-reduced-motion`, la timeline evita las transiciones.
- **Badges y métricas**: pasá `badges` (tupla/lista de strings) y `TimelineHologramMetric` para mostrar valores secundarios con `tone` (`"neutral"`, `"info"`, `"positive"` o `"warning"`).
- **Prioridad**: `priority_label`, `priority_value` y `priority_detail` dibujan el chip contextual que explica la ponderación aplicada.

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

### `RankingCockpit`
Cabina de ranking con tarjetas comparativas, barras neumórficas y filtros interactivos.

```python
from app.modules.luxe_components import MetricSpec, RankingCockpit

cockpit = RankingCockpit(
    entries=[
        {
            "Rank": 1,
            "Score": 0.91,
            "Proceso": "P04 · Sinter EVA",
            "Rigidez": 12.4,
            "Estanqueidad": 0.88,
            "Energía (kWh)": 14.2,
            "Agua (L)": 8.1,
            "Crew (min)": 36.0,
            "Seal": "✅",
            "Riesgo": "Bajo",
        },
        {
            "Rank": 2,
            "Score": 0.86,
            "Proceso": "P02 · Laminado CTB",
            "Rigidez": 11.9,
            "Estanqueidad": 0.9,
            "Energía (kWh)": 18.5,
            "Agua (L)": 6.4,
            "Crew (min)": 42.0,
            "Seal": "⚠️",
            "Riesgo": "Medio",
        },
    ],
    metric_specs=[
        MetricSpec("Rigidez", "Rigidez", "{:.2f}"),
        MetricSpec("Estanqueidad", "Estanqueidad", "{:.2f}"),
        MetricSpec("Energía (kWh)", "Energía", "{:.1f}", unit="kWh", higher_is_better=False),
        MetricSpec("Agua (L)", "Agua", "{:.1f}", unit="L", higher_is_better=False),
        MetricSpec("Crew (min)", "Crew", "{:.0f}", unit="min", higher_is_better=False),
    ],
    key="demo_ranking",
    selection_label="📌 Foco del cockpit",
)

focused_entry = cockpit.render()
```

- `entries` es una lista de diccionarios: cada fila debe exponer las claves para score, riesgo/sellado y métricas.
- Configurá las barras con `MetricSpec`: la propiedad `higher_is_better=False` invierte la escala (útil para agua/energía/crew).
- El usuario puede ordenar por cualquier métrica, filtrar riesgos/sellos y elegir una tarjeta activa (la clase `selected` aplica un glow azul).
- El método `render()` devuelve el diccionario del candidato seleccionado para que puedas sincronizarlo con otros módulos (`st.session_state`, tabs, etc.).
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
