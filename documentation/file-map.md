## File Map (quick reference)

Concise guide to key files and what they do.

### Entry point
- `app.py` — Streamlit app bootstrap; defines tabs and wires UI modules.

### Core logic (`daphne_core/`)
- `alignment_engine.py` — alignment heuristics and similarity helpers used by the alignment tab.
- `document_processor.py` — ingestion/preprocessing of PDFs/text into document dicts.
- `recommendation_extractor.py` — recommendation parsing and categorization (strict extractor).
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
- `architecture.md`, `alignment.md`, `search.md`, `ui-guide.md`, `document_types.md`, `evaluation.md`, `README.md`, `file-map.md` — guides and references for architecture, search/alignment logic, UI, document types, and evaluation placeholder.

### Other
- `PLAN.md` — current work plan.
- `README.md` — project overview.