"""
v2 alignment strategies.

Aligns v2 `Recommendation` and `Response` objects using configurable
strategies (e.g. number‑first with semantic scoring, or stricter 1:1
matching).
"""

from typing import Iterable, List, Optional

from .types import AlignmentResult, Recommendation, Response


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
        raise NotImplementedError("AlignmentStrategyV2.align is not implemented yet")


