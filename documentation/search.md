## Search Modes and Helpers

Reference for the search layer, how modes differ, and where to extend.

### Modes (implemented in `ui/search_logic.py` using `daphne_core.search_utils`)
- **Smart**: token overlap on meaningful words with stop-word filtering; sentence-level scoring with mild boosts for multiple hits.
- **Exact**: substring search; returns exact span with surrounding context.
- **Fuzzy**: per-token similarity via `difflib`; good for typos/variants.
- **AI Semantic**: embedding-based similarity (SentenceTransformers); falls back to lexical if the model is unavailable.
- **Hybrid**: combines smart + semantic to balance precision/recall.

### Outputs
All modes return match dicts with position, matched_text, context, score, match_type, page_number, percentage_through, and mode-specific metadata (e.g., similarity, word_matches).

### Extension points
- Add a new mode by implementing a helper and wiring it into mode selection in `execute_search_with_ai`.
- Adjust stop-word handling in `daphne_core.search_utils` if domain vocabulary warrants it.
- Swap/upgrade the embedding model in AI Semantic mode; keep device handling and caching in mind.

### Guardrails
- Keep match dict schema consistent across modes to avoid UI breakage.
- Avoid UI-only logic for ranking/thresholds; keep reusable pieces in core helpers.
- Preserve deterministic fallbacks when AI components are unavailable.
