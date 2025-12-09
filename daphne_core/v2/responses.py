"""
v2 response extraction.

Takes `PreprocessedText` and produces structured `Response` objects, with
support for both structured sections (e.g. government responses) and
scattered response sentences.
"""

from typing import List, Optional

from .types import PreprocessedText, Response


class ResponseExtractorV2:
    """
    Generalisable response extractor for the v2 pipeline.

    Like the recommendation extractor, behaviour is controlled via a
    profile/config to keep document‑specific logic isolated.
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

    def extract(self, preprocessed: PreprocessedText, source_document: str) -> List[Response]:
        """
        Extract responses from preprocessed text.

        Parameters
        ----------
        preprocessed:
            Output from `v2.preprocess.extract_text`.
        source_document:
            Identifier for the originating document (filename or logical id).

        Returns
        -------
        List[Response]
            Structured responses with explicit spans and, when available,
            `rec_number`. The `response_type` indicates whether a response
            comes from a structured section or scattered sentence.
        """
        raise NotImplementedError("ResponseExtractorV2.extract is not implemented yet")


