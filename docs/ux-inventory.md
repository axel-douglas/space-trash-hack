# UX Inventory Builder

Descripción del workspace de inventario integrado en la Home de Rex-AI. Combina
layouts de Streamlit con AG Grid para que astronautas y analistas editen datos
en vivo sin perder contexto.

## Experiencia híbrida

- **Validación por color**: columnas de masa/volumen usan un gradiente
  teal→rojo para destacar valores anómalos.
- **Agrupación por categoría**: el panel de AG Grid agrupa por `category` para
  expandir/colapsar familias rápidamente.
- **Flag chips**: las banderas se representan con pills en la grilla y en la
  vista lateral para facilitar escaneo.
- **Layout dual**: la grilla editable queda a la izquierda; la derecha muestra
  resumen contextual, chips y acciones masivas.

## Sidebar analítica

- Métricas en vivo (masa, volumen, elementos problemáticos) con variaciones.
- Recomendaciones rápidas: flags problemáticos frecuentes para priorizar
  segregación.
- Filtros rápidos como chips persistentes (`st.session_state`).

## Edición por lotes

1. Seleccioná filas con los checkboxes de AG Grid.
2. Revisá la vista contextual y los chips en el panel derecho.
3. Ajustá valores en el formulario de batch edit (incluye tooltips de ayuda).
4. Confirmá para aplicar cambios; aparece un toast de éxito.

## Persistencia de estado

- Dataframe, filtros y configuración de AG Grid (orden, sorting, grouping)
  viven en `st.session_state`.
- Al editar o filtrar, el estado se fusiona de vuelta en `inventory_data` para
  mantener la configuración tras reruns o guardados.
- El guardado del inventario escribe el dataframe limpio y preserva la
  configuración de la grilla.

## Guardrails

- Inputs numéricos negativos se clampan a 0 durante ediciones masivas.
- Items problemáticos resaltan con borde rojo y aparecen en la sidebar.
- Tooltips de AG Grid recuerdan la semántica de cada flag editable.
