"""
v2 alignment strategies.

Aligns v2 `Recommendation` and `Response` objects using configurable
strategies (e.g. number‑first with semantic scoring, or stricter 1:1
matching).
"""

import logging
import re
from typing import Iterable, List, Optional

try:
    from daphne_core.search_utils import STOP_WORDS
except Exception:  # pragma: no cover - defensive
    STOP_WORDS = set()

from .types import AlignmentResult, Recommendation, Response


logger = logging.getLogger(__name__)


class AlignmentStrategyV2:
    """
    Alignment strategy for the v2 pipeline.

    Configuration is kept simple initially: we can add flags for 1:1 matching,
    similarity backends, or weighting later as needed.
    """

    def __init__(
        self,
        enforce_one_to_one: bool = False,
        use_embeddings: bool = True,
        semantic_matcher: Optional[object] = None,
    ):
        """
        Parameters
        ----------
        enforce_one_to_one:
            If True, enforce a 1:1 matching between recommendations and
            responses. If False, allow response reuse (v1‑like behaviour).
        use_embeddings:
            Whether to use semantic embeddings when available, falling back
            to keyword similarity otherwise.
        semantic_matcher:
            Optional semantic matcher instance (e.g. v1 `RecommendationResponseMatcher`)
            used to compute similarity when `use_embeddings=True`. This avoids
            re-loading large embedding models inside v2 core code.
        """
        self.enforce_one_to_one = enforce_one_to_one
        self.use_embeddings = use_embeddings
        self._semantic_matcher = (
            semantic_matcher
            if use_embeddings and bool(getattr(semantic_matcher, "use_transformer", False))
            else None
        )

    def align(
        self,
        recommendations: Iterable[Recommendation],
        responses: Iterable[Response],
    ) -> List[AlignmentResult]:
        """
        Align recommendations with responses.

        Parameters
        ----------
        recommendations:
            Iterable of v2 `Recommendation` objects.
        responses:
            Iterable of v2 `Response` objects.

        Returns
        -------
        List[AlignmentResult]
            One alignment result per recommendation; 1:1 constraints and
            reuse behaviour depend on configuration.
        """
        recs = list(recommendations)
        resps = list(responses)

        def score(rec_text: str, resp_text: str) -> float:
            if self._semantic_matcher is not None:
                try:
                    v = float(self._semantic_matcher.calculate_similarity(rec_text, resp_text))
                    return max(0.0, v)
                except Exception:  # pragma: no cover - defensive
                    return self._keyword_similarity(rec_text, resp_text)
            return self._keyword_similarity(rec_text, resp_text)

        # Index responses by rec_id and rec_number for number/label-first matching.
        responses_by_id: dict[str, List[Response]] = {}
        responses_by_number: dict[int, List[Response]] = {}
        for resp in resps:
            if resp.rec_id:
                responses_by_id.setdefault(resp.rec_id, []).append(resp)
            if resp.rec_number is not None:
                responses_by_number.setdefault(resp.rec_number, []).append(resp)

        used_responses: set[int] = set()
        alignments: List[AlignmentResult] = []

        for rec in recs:
            best_resp: Optional[Response] = None
            best_score: float = 0.0
            match_method = "none"

            # Prefer responses with matching rec_id (label) when available.
            if rec.rec_id and rec.rec_id in responses_by_id:
                candidates = responses_by_id[rec.rec_id]
                for resp in candidates:
                    if self.enforce_one_to_one and id(resp) in used_responses:
                        continue
                    s = score(rec.text, resp.text)
                    # Mirror v1: strong prior when a label match exists.
                    s = min(s + 0.4, 1.0)
                    if s > best_score:
                        best_score = s
                        best_resp = resp
                        match_method = "label_first_semantic" if self._semantic_matcher is not None else "label_first"

            # Next, fall back to numeric rec_number if both sides provide one.
            if best_resp is None and rec.rec_number is not None and rec.rec_number in responses_by_number:
                candidates = responses_by_number[rec.rec_number]
                for resp in candidates:
                    if self.enforce_one_to_one and id(resp) in used_responses:
                        continue
                    s = score(rec.text, resp.text)
                    # Mirror v1: strong prior when a numeric match exists.
                    s = min(s + 0.4, 1.0)
                    if s > best_score:
                        best_score = s
                        best_resp = resp
                        match_method = (
                            "number_first_semantic" if self._semantic_matcher is not None else "number_first"
                        )

            # Fall back to similarity across all responses if needed.
            if best_resp is None:
                for resp_idx, resp in enumerate(resps):
                    if self.enforce_one_to_one and id(resp) in used_responses:
                        continue
                    s = score(rec.text, resp.text)
                    if s > best_score:
                        best_score = s
                        best_resp = resp
                        match_method = "semantic" if self._semantic_matcher is not None else "keyword"

            # Apply lightweight thresholds so low-similarity matches show as "No Response".
            if best_resp is not None:
                threshold = 0.4 if self._semantic_matcher is not None else 0.25
                if best_score < threshold:
                    best_resp = None
                    best_score = 0.0
                    match_method = "none"

            if best_resp is not None and self.enforce_one_to_one:
                used_responses.add(id(best_resp))

            alignments.append(
                AlignmentResult(
                    recommendation=rec,
                    response=best_resp,
                    similarity=best_score if best_resp is not None else None,
                    match_method=match_method,
                )
            )

        return alignments

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Simple keyword-based similarity between two texts.

        This mirrors the spirit of the v1 keyword matcher without pulling in
        additional dependencies. It is intentionally lightweight and can be
        replaced or augmented with embeddings later if needed.
        """
        words1 = set(re.findall(r"\b\w+\b", (text1 or "").lower())) - STOP_WORDS
        words2 = set(re.findall(r"\b\w+\b", (text2 or "").lower())) - STOP_WORDS
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        if union == 0:
            return 0.0
        return intersection / union
