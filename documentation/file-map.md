## File Map (quick reference)

Concise guide to key files and what they do.

### Entry point
- `app.py` — Streamlit app bootstrap; defines tabs and wires UI modules.
- `pfd_compare_app.py` — focused Streamlit meeting UI: side-by-side PFD extraction comparison (Action Verbs vs Full concerns vs Extended Action Verbs).

### Core logic (`daphne_core/`)
- `alignment_engine.py` — alignment heuristics and similarity helpers used by the alignment tab.
- `document_processor.py` — ingestion/preprocessing of PDFs/text into document dicts.
- `recommendation_extractor.py` — recommendation parsing and categorization (strict extractor).
- `v2/` — v2 pipeline (layout-aware preprocess, typed extractors, PFD helpers):
  - `v2/preprocess.py` — unstructured + syntok preprocessing into `PreprocessedText`.
  - `v2/recommendations.py` — v2 recommendation extraction (numbered headings + action-verb + PFD profile).
  - `v2/responses.py` — v2 response extraction (structured headers + fallback).
  - `v2/alignment.py` — v2 alignment strategy (label/number-first).
  - `v2/pfd_alignment.py` — PFD response-block segmentation + scoped matching helpers.
- `search_utils.py` — core search utilities (stop words, meaningful-word filtering, helpers).
- `search_engine.py` — experimental/legacy semantic search engine (dataclasses, embeddings).
- `core_utils.py` — shared utility functions (e.g., caching, generic helpers).
- `integration_helper.py` — wiring utilities for combining core pieces; app init logging.
- `__init__.py` — package exports and logging setup.

### UI (`ui/`)
- `alignment_ui.py` — alignment tab UI; calls core alignment.
- `alignment_display.py` — alignment-specific rendering helpers.
- `alignment_workflows.py` — workflow helpers used by alignment UI.
- `search_logic.py` — orchestrates search modes and shapes match results.
- `search_components.py` — legacy/compat search UI components (kept for reference).
- `display_search.py`, `display_shared.py`, `display_utils.py` — rendering/highlight helpers for search and shared UI bits.
- `__init__.py` — UI package exports.

### Documentation (`documentation/`)
- `architecture.md`, `alignment.md`, `pfd_alignment.md`, `pfd_rule_inventory.md`, `search.md`, `ui-guide.md`, `document_types.md`, `evaluation.md`, `README.md`, `file-map.md`, `DECISIONS.md` — guides and references for architecture, alignment (including PFD), UI, document types, evaluation, and decision log.

### Other
- `PLAN.md` — current work plan.
- `README.md` — project overview.
