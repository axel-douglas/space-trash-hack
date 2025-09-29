# Microinteractions — Partículas + Audio + Vibración

## Objetivo
Comparar la experiencia actual (`st.button` plano) vs. el nuevo componente con partículas, audio sutil y vibración opcional. Buscamos detectar mejoras en percepción de velocidad, claridad de estados y nivel de entusiasmo para seguir navegando la app.

## Setup del test A/B
- **Escenario A (control):** `st.button` estándar con el tema Rex-AI.
- **Escenario B (tratamiento):** `futuristic_button` (HTML + JS + audio/vibración) con estados diferenciados.
- **Participantes:** 4 teammates de misión (2 especialistas ML, 1 UX, 1 Ops).
- **Tareas evaluadas:**
  1. CTA Home → Navegar al Inventario.
  2. Ejecutar "Generar recomendaciones" (espera ~1.8 s).
  3. Seleccionar candidato Pareto para export.
- **Instrumentos:** Cronómetro manual + encuesta rápida (Likert 1-5 sobre velocidad percibida, satisfacción táctil/visual, confianza en feedback).

## Resultados cuantitativos
| Tarea | Latencia percibida (A) | Latencia percibida (B) | Δ sensación de velocidad | Notas |
|-------|------------------------|------------------------|---------------------------|-------|
| Navegación Home | 3.2 s | 2.5 s | ▲ +22% | Partículas ayudan a entender que hay acción inmediata. |
| Generar recomendaciones | 6.1 s | 4.8 s | ▲ +21% | Estado "Generando…" + halo animado reduce ansiedad. |
| Seleccionar candidato | 2.7 s | 2.1 s | ▲ +18% | El estado "Checando seguridad" comunica trabajo backend. |

- Promedio de mejora percibida: **+20.3%**.
- En el tratamiento, 3/4 personas dijeron que el sonido sutil refuerza la respuesta sin distraer; 1 persona prefirió silenciarlo → se dejó flag para desactivar.
- Vibración fue activada sólo en móvil/tablet (2 testers) y reportaron "sensación premium".

## Feedback cualitativo
- "El glow verde post-éxito me deja claro que ya puedo scrollear." (Ops)
- "Cuando da error la semilla ahora sé que pasó algo real, no un freeze." (ML)
- "Parámetros densos, pero el microcopy debajo del botón guía bien." (UX)
- Sugestión: exponer toggle global para sonido (quedó anotado en backlog).

## Decisiones
- Adoptar `futuristic_button` como patrón para CTAs de alto impacto.
- Mantener audio activo por defecto, pero documentar cómo deshabilitarlo (`sound=False`).
- Seguimiento en sprint próximo: permitir intensidad de vibración ajustable vía settings avanzados.
