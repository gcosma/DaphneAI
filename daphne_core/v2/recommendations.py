"""
v2 recommendation extraction.

Takes `PreprocessedText` and produces structured `Recommendation` objects.
"""

from typing import List, Optional

from .types import PreprocessedText, Recommendation


class RecommendationExtractorV2:
    """
    Generalisable recommendation extractor for the v2 pipeline.

    This class is intended to support different strategies via configuration,
    e.g. a GOV.UK‑style numbered heading profile vs more generic patterns.
    """

    def __init__(self, profile: Optional[str] = None):
        """
        Parameters
        ----------
        profile:
            Optional profile name (e.g. "gov_uk_report") controlling
            document‑specific heuristics.
        """
        self.profile = profile

    def extract(self, preprocessed: PreprocessedText, source_document: str) -> List[Recommendation]:
        """
        Extract recommendations from preprocessed text.

        Parameters
        ----------
        preprocessed:
            Output from `v2.preprocess.extract_text`.
        source_document:
            Identifier for the originating document (filename or logical id).

        Returns
        -------
        List[Recommendation]
            Structured recommendations with explicit spans and, when
            available, `rec_number`.
        """
        raise NotImplementedError("RecommendationExtractorV2.extract is not implemented yet")


