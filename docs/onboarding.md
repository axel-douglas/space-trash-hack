# Onboarding interactivo Rex-AI

Este documento resume los elementos clave del home de Rex-AI tras la renovación "Mission Briefing". Úsalo como guía rápida para preparar assets multimedia, explicar el flow a la tripulación y depurar la demo guiada.

## 1. Mission Briefing
- **Componente**: `mission_briefing` (`app/modules/luxe_components.py`).
- **Qué hace**: construye el hero con video loop y tarjetas animadas que resumen capacidades críticas (Crew Ops + IA, trazabilidad, seguridad).
- **Narrativa escalonada**: cada paso se describe en el stepper lateral. El copy puede ajustarse en `app/Home.py` dentro del array `steps`.
- **Asset de video**: coloca un MP4 ligero (~6-8 MB máximo) en `app/static/mission_briefing_loop.mp4`. Si el archivo no está presente el componente renderiza un fallback animado, por lo que la UI sigue siendo usable.

## 2. Timeline orbital 3D
- **Componente**: `orbital_timeline` (`app/modules/luxe_components.py`).
- **Uso**: reemplaza la lista `<ul>` tradicional con milestones orbitando en pseudo-3D. Cada hito admite `label`, `description` (HTML permitido para snippets de código) e `icon` (emoji o SVG inline).
- **Customización**: edita la lista `TimelineMilestone` al final de `app/Home.py` para sumar/quitar eventos o ajustar el orden.

## 3. Demo guiada automática
- **Componente**: `guided_demo` (`app/modules/luxe_components.py`).
- **Cómo funciona**:
  1. Al pulsar **“▶️ Activar demo guiada”** se setean `?demo=mission&step=0` vía `st.experimental_set_query_params`.
  2. Un snippet JS avanza los pasos cada ~6.5 s (configurable) modificando el querystring y recargando la vista.
  3. Cada paso muestra un overlay translúcido con icono, título y descripción. El estado actual puede usarse para resaltar métricas (ver `metric_blocks` en `app/Home.py`).
  4. **“⏹️ Detener demo”** limpia los parámetros y elimina el overlay.
- **Loop**: por defecto vuelve al paso 0 al llegar al final para mantener la demo en rotación continua durante showcases.

## 4. Flujo recomendado para tripulación
1. **Inventario NASA** – Subir o usar el CSV de ejemplo (`data/waste_inventory_sample.csv`), validar flags EVA/multilayer.
2. **Target marciano** – Seleccionar preset (container, utensil, tool, interior) o ajustar límites manuales de agua/energía/crew.
3. **Generador IA** – Ejecutar blending con explicabilidad (feature contributions + bandas 95%).
4. **Resultados & feedback** – Comparar heurística vs IA, exportar recetas y registrar feedback para retraining.

## 5. Assets y estilos
- **CSS compartido**: el módulo `luxe_components.py` inyecta estilos de tarjetas, timeline y overlay solo una vez por sesión.
- **Colores/acentos**: cada `BriefingCard` puede definir `accent` para variar el glow. Ajustar en `mission_briefing`.
- **Accesibilidad**: mantener contraste mínimo 4.5:1 al introducir nuevos colores. Evitar texto vital dentro del video loop.

## 6. Troubleshooting rápido
- Si el video no carga, confirmar ruta relativa y peso (<10 MB para no penalizar load en Streamlit Cloud).
- Los parámetros `demo` y `step` se preservan en el URL; recuerda limpiar antes de compartir deep links si no querés iniciar la demo automáticamente.
- Para ajustar el ritmo de la guía modifica `step_duration` en la llamada a `guided_demo`.
