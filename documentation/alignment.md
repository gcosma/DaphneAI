## Alignment (Recommendations ↔ Responses)

Overview of how recommendations are paired with responses and where to extend the logic.

### Pipeline
- Inputs: list of extracted recommendations (text + metadata) and candidate response text/snippets from search.
- Pattern matching: `find_pattern_matches` scans responses for explicit references (ids/keywords).
- Similarity scoring: `calculate_simple_similarity` uses meaningful-word Jaccard over recommendations vs. response snippets.
- Status classification: `determine_alignment_status` tags each pair (e.g., matched/partial/none) based on thresholds and pattern hits.
- Assembly: `align_recommendations_with_responses` produces alignment rows consumed by the UI.

### Assumptions and limits
- Relies on cleaned text; noisy OCR can depress similarity.
- Heuristics favor lexical overlap; semantically similar but lexically divergent responses may be under-matched.
- Pattern rules are tailored to current recommendation formats (ids/verbs); changes in extraction format require updating patterns.

### Extension guidance
- Add semantic similarity: plug in embedding-based scoring alongside lexical Jaccard; combine with weighted fusion.
- Tune thresholds: adjust in `determine_alignment_status` with small, documented constants.
- Expand patterns: broaden `find_pattern_matches` to capture new id/label formats; keep them precise to avoid false positives.
- Context windowing: consider widening/narrowing response snippets before scoring to balance precision/recall.

### Guardrails
- Keep output schema stable (status, score, matched_text/context, linked recommendation).
- Avoid UI-side alignment logic; keep heuristics in `daphne_core.alignment_engine`.
- Document any new thresholds or model dependencies in this file to aid reproducibility and evaluation.
