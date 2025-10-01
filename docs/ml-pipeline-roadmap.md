# Rex-AI — Roadmap operativo del pipeline predictivo (snapshot wiki)

> Última sincronización con el tablero oficial: 24 Sep 2025 (daily post-demo).

## Fuente oficial
- **Herramienta primaria:** Linear — proyecto "ML Delivery" (tablero compartido con Operaciones y App). Acceso restringido a `@space-trash-hack`.
- **Mirror wiki:** este documento resume los ítems vigentes para consulta offline. Las actualizaciones relevantes se copian aquí cuando cambia el estado en Linear.

## Estado consolidado

### Sprint 0 — Preparativos (cerrado)
| ID | Estado | Responsable | Notas |
| --- | --- | --- | --- |
| S0.1 | ✅ Hecho | Ana (DataOps) | Inventario en `docs/data_inventory.md` verificado con los dueños de datasets. |
| S0.2 | ✅ Hecho | Bruno (ML) | Esquema Parquet acordado; `datasets/schema.yaml` sirve como contrato. |
| S0.3 | ✅ Hecho | Carla (Infra) | `requirements-lock.txt` actualizado tras la validación de `model_training`. |
| S0.4 | ✅ Hecho | Diego (PM) | Tablero de gobierno activo; notificaciones sincronizadas con Slack `#rex-ai`. |

### Sprint 1 — Pipeline de datos + modelo base (en curso)
| ID | Estado en Linear | Responsable | Comentarios |
| --- | --- | --- | --- |
| S1.1 | 🟠 En progreso | Bruno (ML) | ETL `scripts/build_gold_dataset.py` reescrito; pendiente validación de columnas derivadas. |
| S1.2 | 🟠 En progreso | Camila (ML) | Extensión de `compute_feature_vector()` lista en rama `feature/vector-upgrade`; review en curso. |
| S1.3 | 🔴 Bloqueado | Bruno (ML) | Espera salida de S1.1 para entrenar targets adicionales (`rigidez`, `estanqueidad`, `energy_kwh`, `water_l`, `crew_min`). |
| S1.4 | 🟡 Preparado | Equipo ML | CI95 se implementará tras consolidar métricas en `metadata.json`. |
| S1.5 | 🟡 Preparado | Equipo ML | Registro de `feature_importances_` alineado a output de entrenamiento; falta wiring en `model_registry.py`. |
| S1.6 | 🟢 En cola | Data/ML | Notebook de validación planificado tras cierre de S1.3. |

### Sprint 1 — Integración en la app (dependiente)
- **S1.7 (UI + ModelRegistry):** queda en columna "Bloqueado" hasta que `metadata.json` exponga CI y `label_source` definitivos.
- **S1.8 (Selector de modo):** kickoff previsto una vez que QA valide el fallback híbrido.
- **S1.9 (Explicabilidad en UI):** se mantiene como enhancement; UX prepara mocks en paralelo.

### Sprint 2 — Generación y ranking (pre-planificado)
Los ítems permanecen en la columna "Backlog" de Linear hasta completar Sprint 1. Se conservaron las dependencias originales:
- **S2.1:** generador combinatorio (`scripts/generate_candidates.py`).
- **S2.2:** scoring multiobjetivo configurable.
- **S2.3:** clasificadores auxiliares para flags de seguridad.
- **S2.4:** pipeline de ranking con top-N candidatos.
- **S2.5:** UI con tabla de candidatos.

### Parking lot / Opcionales (archivo de Linear)
Las extensiones de Sprint 4 y experimentos de optimización se movieron al archivo "Opcionales" del proyecto. Se reabrirán cuando el equipo confirme capacidad extra:
- Modelos XGBoost por target.
- Autoencoder tabular para duplicados.
- Optimización en espacio latente.
- Workflow CI/CD para empaquetado de modelos.

## Próximos pasos acordados
1. Cerrar S1.1 y S1.2 para liberar entrenamiento multisalida.
2. Actualizar `metadata.json` con intervalos y procedencia antes de activar la UI (S1.7).
3. Revisar necesidades de recursos para retomar los opcionales del parking lot.
