# Tablero Sprint 0 — Rex-AI Predictive Pipeline

URL del tablero colaborativo: https://www.notion.so/space-trash-hack/Rex-AI-Sprint-0-e7d7d7c4e4e1430ea6ab4f9779e472a6

> **Nota**: El enlace corresponde a la página creada en Notion durante la sesión. Si se replica en otra herramienta (Jira/Trello), mantener la misma estructura de columnas y IDs de tarea.

## Configuración del tablero

| Columna | Descripción |
| --- | --- |
| `Backlog` | Tareas planificadas aún sin responsable asignado. |
| `En progreso` | Actividades con owner activo y fecha de entrega acordada. |
| `En revisión` | Entregables listos esperando validación (PR, documento, etc.). |
| `Hecho` | Entregables cerrados y subidos al repositorio/tablero correspondiente. |

## Snapshot inicial (24 Sep 2025)

| ID | Tarea | Owner | Estado | Notas |
| --- | --- | --- | --- | --- |
| S0.1 | Inventariar datasets existentes y mapear columnas a features/targets | Ana (DataOps) | Hecho | Resultado en `docs/data_inventory.md`. |
| S0.2 | Definir esquema Parquet unificado (`features.parquet`, `labels.parquet`) | Bruno (ML) | Hecho | Ver `datasets/schema.yaml`. |
| S0.3 | Configurar entorno y validar `model_training` | Carla (Infra) | Hecho | Log en `docs/environment_setup.md`; `requirements-lock.txt` generado. |
| S0.4 | Crear tablero de seguimiento y asignar owners | Diego (PM) | Hecho | Tablero publicado en Notion (enlace arriba). |

## Próximas acciones en el tablero

1. Crear vistas filtradas por owner (`DataOps`, `ML`, `Infra`, `PM`).
2. Sincronizar el tablero con Slack #rex-ai para notificaciones de cambios de estado.
3. Duplicar la plantilla para Sprint 1 antes de la daily inicial.
