## Architecture

High-level layout of DaphneAI with emphasis on where to extend or inspect for behavior.

### Components and flow
- **Ingestion/Preprocessing (v1)**: `daphne_core.document_processor` parses PDF/text, normalizes text, and produces a document dict (`text`, metadata). This is the baseline path used by v1 extraction/alignment.
- **Ingestion/Preprocessing (v2)**: `daphne_core.v2.preprocess.extract_text` runs `unstructured.partition_pdf` + `syntok` to produce `PreprocessedText` (flat text + sentence/page spans). v2 requires access to the original PDF file path.
- **Search**: `daphne_core.search_utils` (core helpers) and `ui/search_logic.py` (orchestrates mode selection and result shaping). Outputs ranked matches with context and metadata.
- **Recommendation Extraction**:
  - v1: `daphne_core.recommendation_extractor` (`StrictRecommendationExtractor`) for numbered “Recommendation N …” blocks and sentence-based inference fallback.
  - v2: `daphne_core.v2.recommendations.RecommendationExtractorV2` (numbered headings + action-verb channel + PFD profile support).
- **Response Extraction**:
  - v1: `daphne_core.alignment_engine.extract_response_sentences` (structured gov headers + filtered sentence-level extras).
  - v2: `daphne_core.v2.responses.ResponseExtractorV2` (structured gov headers with explicit spans; fallback as needed).
- **Alignment**:
  - v1: `daphne_core.alignment_engine.RecommendationResponseMatcher` (semantic embeddings when available, keyword fallback).
  - v2: `daphne_core.v2.alignment.AlignmentStrategyV2` (label/number-first keyword matching, optional 1:1).
- **UI**: Streamlit app in `app.py` plus tab modules under `ui/` (e.g., `alignment_ui.py`, `display_*` files). UI is thin; business logic stays in `daphne_core`.

### Data contracts
- **Document structure (UI/session)**: dict with `text`, `filename`, and (for PDFs) `pdf_path`. The upload pipeline persists PDFs into `output/uploads/` so v2 can re-read them.
- **v2 preprocessed structure**: `PreprocessedText(text, sentence_spans, page_spans)`.
- **Search result**: dict with `matched_text`, `context`, `position`, `score`, `match_type`, `page_number`, `percentage_through`; some modes add `similarity`, `word_matches`, etc.
- **Recommendation (v1)**: dict with `text`, `method`, `confidence`, optional `rec_number`.
- **Recommendation/Response (v2)**: typed objects in `daphne_core.v2.types` with explicit spans and IDs.

### Extension points
- Add search modes: implement in `daphne_core.search_utils` and register in `ui/search_logic.py` mode selection.
- Adjust extraction:
  - v1 is treated as a baseline; change cautiously and keep `OBSERVATIONS.md` + tests in sync.
  - v2 is the main generalisation surface: prefer adding profiles/rules under `daphne_core/v2/*`.
- Improve alignment: modify v1 heuristics in `daphne_core.alignment_engine` only when preserving the reference baseline, and implement new behaviour primarily in `daphne_core/v2/*` (see `documentation/alignment.md`).
- UI changes: add tabs/components in `ui/`; avoid embedding business logic—call core helpers instead.

### Guardrails
- Keep core logic in `daphne_core`; UI should only orchestrate and render.
- Preserve result schemas to avoid breaking downstream tabs.
- Prefer pure functions in core; side effects (session state) belong in UI.
- For extraction/alignment subtleties, treat `OBSERVATIONS.md` and `OBSERVATIONSv2.md` as part of the working contract.
