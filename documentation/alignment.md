## Alignment (Recommendations ↔ Responses)

Overview of how recommendations are paired with responses and where to extend the logic.

### Pipeline
- Canonical UI paths (current app):
  - **v1 (baseline)**:
    - Recommendations: `extract_recommendations` from `daphne_core.recommendation_extractor` (strict extractor).
    - Responses: `extract_response_sentences` from `daphne_core.alignment_engine` (government response headers + sentence-level filters).
    - Alignment: `RecommendationResponseMatcher.find_best_matches` (semantic model if available, keyword fallback).
  - **v2 (experimental)**:
    - Preprocess: `daphne_core.v2.preprocess.extract_text` (layout-aware; requires original PDF path).
    - Recommendations: `daphne_core.v2.recommendations.RecommendationExtractorV2`.
    - Responses: `daphne_core.v2.responses.ResponseExtractorV2`.
    - Alignment: `daphne_core.v2.alignment.AlignmentStrategyV2`.
- Lightweight alignment helpers (legacy/secondary path):
  - Pattern matching: `find_pattern_matches` scans responses for explicit references (ids/keywords).
  - Similarity scoring: `calculate_simple_similarity` uses meaningful-word Jaccard over recommendations vs. response snippets.
  - Status classification: `determine_alignment_status` tags each pair (e.g., matched/partial/none) based on thresholds and pattern hits.
  - Assembly: `align_recommendations_with_responses` produces alignment rows for simpler UIs/tools.

> Note: The Streamlit alignment tab (`ui/alignment_ui.py`) supports both v1 and v2 via a toggle. The older `ui/alignment_workflows` path is kept as a legacy/secondary interface and may be simplified or removed once v2 paths are fully covered by tests.

### Assumptions and limits
- Relies on cleaned text; noisy OCR can depress similarity.
- Heuristics favor lexical overlap; semantically similar but lexically divergent responses may be under-matched.
- Pattern rules are tailored to current recommendation formats (ids/verbs); changes in extraction format require updating patterns.

### Extension guidance
- Prefer improving **recommendation extraction** first: alignment quality is bounded by upstream units. In particular:
  - v2 `action_verb` is intentionally aligned to v1 semantics (same rule-based confidence and filtering).
  - For PFD reports, “null explicit recommendation” cases are common; response-required concerns often drive the response structure.
- Add semantic similarity: plug in embedding-based scoring alongside lexical Jaccard; combine with weighted fusion.
- Tune thresholds: adjust in `determine_alignment_status` with small, documented constants (remember v1 confidence is rule-based, not calibrated).
- Expand patterns: broaden `find_pattern_matches` to capture new id/label formats; keep them precise to avoid false positives.
- Context windowing: consider widening/narrowing response snippets before scoring to balance precision/recall.
- PFD (Regulation 28) alignment: keep PFD-specific semantics profile-scoped (scoped matching by addressee, response block segmentation, and response-required concerns vs directives). See `documentation/pfd_alignment.md` and `documentation/pfd_rule_inventory.md`.

### Guardrails
- Keep output schema stable (status, score, matched_text/context, linked recommendation).
- Avoid UI-side alignment logic; keep heuristics in `daphne_core.alignment_engine`.
- Document any new thresholds or model dependencies in this file to aid reproducibility and evaluation.
