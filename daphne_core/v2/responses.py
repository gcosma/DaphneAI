"""
v2 response extraction.

Takes `PreprocessedText` and produces structured `Response` objects, with
support for both structured sections (e.g. government responses) and
scattered response sentences.
"""

import logging
import re
from typing import List, Optional

from .types import PreprocessedText, Response


logger = logging.getLogger(__name__)


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
        text = preprocessed.text or ""
        if not text.strip():
            logger.warning("Empty text passed to ResponseExtractorV2.extract")
            return []

        responses: List[Response] = []

        # ------------------------------------------------------------------ #
        # Phase 1: structured responses ("Government response to recommendation N")
        # ------------------------------------------------------------------ #
        header_pattern = re.compile(
            r"Government\s+response\s+to\s+recommendation\s+(\d+)",
            re.IGNORECASE,
        )

        headers = list(header_pattern.finditer(text))
        logger.info("v2: Found %d structured response headers", len(headers))

        for idx, match in enumerate(headers):
            rec_num_str = match.group(1)
            try:
                rec_number = int(rec_num_str)
            except (TypeError, ValueError):
                rec_number = None

            start_body = match.end()
            end_block = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
            span = (start_body, end_block)
            body_text = text[start_body:end_block].strip()
            if not body_text:
                continue

            responses.append(
                Response(
                    text=body_text,
                    span=span,
                    rec_number=rec_number,
                    source_document=source_document,
                    response_type="structured",
                )
            )

        # ------------------------------------------------------------------ #
        # Phase 2: scattered responses (fallback only)
        # ------------------------------------------------------------------ #
        if not responses and preprocessed.sentence_spans:
            for sent_start, sent_end in preprocessed.sentence_spans:
                sentence = text[sent_start:sent_end].strip()
                if len(sentence) < 40 or len(sentence) > 500:
                    continue
                if self._is_scattered_response(sentence):
                    responses.append(
                        Response(
                            text=sentence,
                            span=(sent_start, sent_end),
                            rec_number=self._infer_rec_number(sentence),
                            source_document=source_document,
                            response_type="scattered",
                        )
                    )

        return responses

    def _is_scattered_response(self, sentence: str) -> bool:
        """
        Heuristic for identifying scattered response sentences.

        For now this is intentionally simple and biased towards government
        responses: we look for language like "The government supports / accepts
        / agrees / notes / rejects" or similar phrases.
        """
        lower = sentence.lower()
        if "the government supports" in lower:
            return True
        if "the government accepts" in lower:
            return True
        if "the government agrees" in lower:
            return True
        if "the government notes" in lower:
            return True
        if "the government rejects" in lower:
            return True
        return False

    def _infer_rec_number(self, sentence: str) -> Optional[int]:
        """Best-effort rec_number inference from a scattered response sentence."""
        lower = sentence.lower()
        m = re.search(r"recommendation\s+(\d+)", lower)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None
