## Architecture

High-level layout of DaphneAI with emphasis on where to extend or inspect for behavior.

### Components and flow
- **Ingestion/Preprocessing**: `daphne_core.document_processor` parses PDF/text, normalizes pages, extracts raw text + metadata. Output: list of document dicts (`text`, `pages`, `metadata`).
- **Search**: `daphne_core.search_utils` (core helpers) and `ui/search_logic.py` (orchestrates mode selection and result shaping). Outputs ranked matches with context and metadata.
- **Recommendation Extraction**: `daphne_core.recommendation_extractor` (strict extractor in current build) identifies recommendations and classifies type/urgency.
- **Alignment**: `daphne_core.alignment_engine` houses alignment heuristics (pattern matching, similarity scoring, status classification). UI invokes this and renders results.
- **UI**: Streamlit app in `app.py` plus tab modules under `ui/` (e.g., `alignment_ui.py`, `display_*` files). UI is thin; business logic stays in `daphne_core`.

### Data contracts
- **Document structure**: dict with `text` (full string), optional `pages` (list of page text), `metadata` (filename, source).
- **Search result**: dict with `matched_text`, `context`, `position`, `score`, `match_type`, `page_number`, `percentage_through`; some modes add `similarity`, `word_matches`, etc.
- **Recommendation**: dict with extracted text, ids, category/type, urgency, source location.
- **Alignment row**: combines recommendation, matched response snippet, similarity score, status classification, and context.

### Extension points
- Add search modes: implement in `daphne_core.search_utils` and register in `ui/search_logic.py` mode selection.
- Adjust extraction: extend/replace extractor in `daphne_core.recommendation_extractor`; keep outputs consistent (text + metadata).
- Improve alignment: modify heuristics in `daphne_core.alignment_engine` (see `alignment.md` for guardrails).
- UI changes: add tabs/components in `ui/`; avoid embedding business logic—call core helpers instead.

### Guardrails
- Keep core logic in `daphne_core`; UI should only orchestrate and render.
- Preserve result schemas to avoid breaking downstream tabs.
- Prefer pure functions in core; side effects (session state) belong in UI.
