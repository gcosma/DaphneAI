## UI Guide

How the Streamlit UI is structured and how it connects to core logic.

### Layout
- Entry point: `app.py` sets up tabs and routes to tab modules under `ui/`.
- Recommendations tab: implemented in `app.py`; supports a v1/v2 toggle for extraction.
- Alignment tab: `ui/alignment_ui.py` renders recommendation/response alignment with a v1/v2 toggle:
  - v1 uses `daphne_core.recommendation_extractor` + `daphne_core.alignment_engine`.
  - v2 uses `daphne_core.v2.*` (layout-aware; requires original PDF paths).
- Meeting comparison UI: `pfd_compare_app.py` is a focused Streamlit app for screenshare that compares:
  - **Action Verbs** (v1 recommendations, run over **v2-preprocessed** text),
  - **Full concerns** (v2 PFD concerns as blocks), and
  - **Extended Action Verbs** (v2 PFD concerns atomised into sentences).
- Display-only formatting toggles:
  - Recommendations and Alignment tabs include a “Display: single paragraph” toggle.
  - This is presentation-only; extraction/matching still uses the underlying sentence/layout structures.
- Display helpers: `ui/display_search.py`, `ui/display_shared.py`, and `ui/alignment_display.py` format search and alignment outputs.
- Search orchestration: `ui/search_logic.py` selects search modes and shapes result dicts; heavy lifting delegated to core helpers.

### Principles
- UI should orchestrate and render; business logic belongs in `daphne_core`.
- Keep Streamlit state handling (session_state) in UI; core functions stay stateless/pure.
- Reuse display helpers for consistent formatting of highlights and context.
- v2 pipelines require original PDFs: the upload flow persists PDFs to `output/uploads/` and stores `pdf_path` in each document dict.

### Meeting workflow (PFD comparison)
- Run: `streamlit run pfd_compare_app.py`
- Notes:
  - Saves uploaded PDFs to `output/uploads/pfd_compare/` (gitignored).
  - Uses v2 preprocessing as the common baseline so differences are attributable to extraction rules, not PDF ingestion.

### Adding UI features
- New tab: add a module under `ui/`, register in `app.py`, and pull core services instead of duplicating logic.
- New visualization: extend display helpers or add a focused helper file; avoid embedding data munging in the view code.
- Inputs/controls: keep parameter defaults close to the tab module; validate before calling core functions.
