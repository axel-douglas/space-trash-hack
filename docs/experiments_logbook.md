# Registro de experimentos físicos / simulador

Seguimiento breve del estado de los experimentos físicos y del simulador. La
propuesta detallada vive en
[`docs/proposals/experiments_feedback_flow.md`](proposals/experiments_feedback_flow.md)
mientras operaciones confirma su implementación.

## Estado actual

- El circuito activo continúa siendo el de `docs/feedback_loop.md` (feedback
  registrado vía app o Parquet manuales).
- Cualquier ensayo físico debe documentarse en el esquema `datasets/feedback_schema.yaml`.
- Las corridas con simulador quedan en pausa hasta contar con hardware y turnos
dedicados.

## Próximos pasos sugeridos

1. Revisar la propuesta externa y acordar un piloto con operaciones.
2. Definir formato final de `experiments_logbook.parquet` para integrarlo al
   pipeline gold.
3. Integrar notificaciones en el Mission Planner para señalar cuando haya
   experimentos pendientes de ingestión.
