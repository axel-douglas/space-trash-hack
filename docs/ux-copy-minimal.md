# Guía de copy minimalista

## Principios de tono
- Escribí en forma directa y concreta. Cada frase debe indicar la acción o el dato clave sin rodeos.
- Evitá superlativos y exageraciones. Preferí cifras o hechos verificables.
- Usá verbos de acción en títulos y botones ("Configurá", "Exportá", "Revisá").
- Optá por títulos cortos (máx. 4 palabras). Si necesitás contexto adicional, agregalo en un subtítulo o caption.
- No uses metáforas ni analogías espaciales; describí lo que ocurre en la interfaz.
- Mantené la voz en segunda persona singular (vos) para ser consistente con el resto del producto.

## Plantillas sugeridas
- **Título de sección:** `Verbo + objeto directo` (ej.: `Analizá residuos`, `Generá lote`).
- **Subtítulo o lead:** `Contexto breve + resultado` (ej.: `Analizamos el inventario NASA y señalamos masas críticas`).
- **Botón principal:** `Verbo de acción en infinitivo o imperativo` (ej.: `Generar lote`, `Exportar reporte`).
- **Mensajes de estado:** `Verbo + estado actual` (ej.: `Procesando lote`, `Resultados listos`).
- **Avisos informativos:** `Acción recomendada + condición` (ej.: `Cargá el inventario antes de generar recetas`).

## Ejemplos antes / después
### Home (`app/Home.py`)
- **Antes:** "Rex-AI orquesta el reciclaje orbital y marciano".
- **Después:** "Rex-AI coordina el reciclaje orbital y marciano" (sustituye la metáfora por un verbo directo).

- **Antes:** "Radiografiamos el inventario NASA, destacamos masas críticas...".
- **Después:** "Analizamos el inventario NASA, destacamos masas críticas..." (elimina la metáfora médica y mantiene la acción).

### Generador (`app/pages/3_Generator.py`)
- **Antes:** "¿Qué hace el generador (en criollo)?".
- **Después:** "¿Cómo funciona el generador?" (pregunta directa, sin modismos).

- **Antes:** "Pistas visuales sobre cómo evoluciona el frente Pareto...".
- **Después:** "Visualizá cómo evoluciona el frente Pareto..." (usa un verbo de acción y elimina lenguaje figurado).

Seguí estas pautas al redactar nuevos componentes o revisar copy existente para conservar una narrativa minimalista y consistente.
