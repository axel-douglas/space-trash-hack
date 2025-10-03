# UX Inventory Builder

## Home inventory workspace
The inventory builder now lives on the Home flow and combines Streamlit layout primitives with an AG Grid-based editor to provide a hybrid workspace optimised for astronaut operations.

## Hybrid grid experience
- **Tesla-style validation colours**: mass and volume columns display a teal-to-red gradient that highlights invalid or negative values instantly.
- **Row grouping**: inventory items are grouped by `category` via the AG Grid group panel, enabling quick collapsing/expansion by type.
- **Flag chips**: flags render as pill-shaped chips both in-grid and in the lateral detail view for fast scanning of multi-flag items.
- **Side-by-side layout**: the editable grid occupies the left pane, while the right pane surfaces contextual previews, chip summaries and batch actions.

## Sidebar analytics
- The "Análisis instantáneo" sidebar exposes live metrics (mass, volume, problematics) with deltas to communicate trends.
- A lightweight recommendation engine surfaces the most frequent problematic flags to prioritise segregation.
- Quick filters appear as toggleable chips styled after MUI pills; selections persist thanks to `st.session_state`.

## Batch editing flow
1. Select one or more rows using the AG Grid checkboxes.
2. Review the contextual preview and flag chips that appear on the right pane.
3. Use the batch edit form to adjust mass (with tooltip guidance) and append shared flags in one action.
4. Submit to apply updates to all selected rows; a toast confirms completion.

## Persistence & state management
- The underlying dataframe, quick-filter selections and AG Grid view state (column order, sorting, grouping) are saved in `st.session_state`.
- When the user toggles filters or edits the grid, the state is merged back into `inventory_data`, ensuring the same configuration after reruns or saves.
- Saving the inventory writes the cleaned dataframe while preserving the in-session grid configuration for continuity.

## Guardrails & validation cues
- Negative numeric inputs are automatically clamped to zero during batch updates to prevent invalid states.
- Problematic items receive a red border accent in the grid and appear prominently in the sidebar metrics.
- Hover tooltips on editable columns leverage AG Grid defaults to remind users about shared flag semantics.

