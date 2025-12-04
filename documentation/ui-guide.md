## UI Guide

How the Streamlit UI is structured and how it connects to core logic.

### Layout
- Entry point: `app.py` sets up tabs and routes to tab modules under `ui/`.
- Alignment tab: `alignment_ui.py` renders recommendation/response alignment using helpers from `daphne_core.alignment_engine`.
- Display helpers: `ui/display_search.py`, `ui/display_shared.py`, and `ui/alignment_display.py` format search and alignment outputs.
- Search orchestration: `ui/search_logic.py` selects search modes and shapes result dicts; heavy lifting delegated to core helpers.

### Principles
- UI should orchestrate and render; business logic belongs in `daphne_core`.
- Keep Streamlit state handling (session_state) in UI; core functions stay stateless/pure.
- Reuse display helpers for consistent formatting of highlights and context.

### Adding UI features
- New tab: add a module under `ui/`, register in `app.py`, and pull core services instead of duplicating logic.
- New visualization: extend display helpers or add a focused helper file; avoid embedding data munging in the view code.
- Inputs/controls: keep parameter defaults close to the tab module; validate before calling core functions.
