"""
v2 alignment strategies.

Aligns v2 `Recommendation` and `Response` objects using configurable
strategies (e.g. number‑first with semantic scoring, or stricter 1:1
matching).
"""

import logging
import re
from typing import Iterable, List, Optional

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
        """
        self.enforce_one_to_one = enforce_one_to_one
        self.use_embeddings = use_embeddings

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

        # Index responses by rec_number for number-first matching.
        responses_by_number: dict[int, List[Response]] = {}
        for resp in resps:
            if resp.rec_number is None:
                continue
            responses_by_number.setdefault(resp.rec_number, []).append(resp)

        used_responses: set[int] = set()
        alignments: List[AlignmentResult] = []

        for rec in recs:
            best_resp: Optional[Response] = None
            best_score: float = 0.0
            match_method = "none"

            # Prefer responses with matching rec_number when available.
            if rec.rec_number is not None and rec.rec_number in responses_by_number:
                candidates = responses_by_number[rec.rec_number]
                for resp in candidates:
                    if self.enforce_one_to_one and id(resp) in used_responses:
                        continue
                    score = self._keyword_similarity(rec.text, resp.text)
                    if score > best_score:
                        best_score = score
                        best_resp = resp
                        match_method = "number_first"

            # Fall back to keyword similarity across all responses if needed.
            if best_resp is None:
                for resp in resps:
                    if self.enforce_one_to_one and id(resp) in used_responses:
                        continue
                    score = self._keyword_similarity(rec.text, resp.text)
                    if score > best_score:
                        best_score = score
                        best_resp = resp
                        match_method = "keyword"

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
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        if union == 0:
            return 0.0
        return intersection / union



