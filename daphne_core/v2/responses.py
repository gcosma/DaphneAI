"""
v2 response extraction.

Takes `PreprocessedText` and produces structured `Response` objects, with
support for both structured sections (e.g. government responses) and
action-verb response sentences.
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
            comes from a structured section or an action-verb sentence.
        """
        text = preprocessed.text or ""
        if not text.strip():
            logger.warning("Empty text passed to ResponseExtractorV2.extract")
            return []

        responses: List[Response] = []

        # ------------------------------------------------------------------ #
        # Phase 1: structured responses ("Government response to recommendation X")
        # ------------------------------------------------------------------ #
        header_pattern = re.compile(
            r"Government\s+response\s+to\s+recommendation\s+([^\s:]+)",
            re.IGNORECASE,
        )

        headers = list(header_pattern.finditer(text))
        logger.info("v2: Found %d structured response headers", len(headers))

        # Some response documents include embedded "Recommendation N ..." excerpts
        # between response blocks. To avoid pulling recommendation excerpts into the
        # response content, stop the response block at the first heading-like
        # "Recommendation N" marker after the header.
        #
        # We intentionally treat these as *headings* by requiring start-of-line,
        # which avoids matching inline references like "see the response to
        # recommendation 13".
        rec_marker_re = re.compile(
            r"(?m)^\s*(?:Recommendations?\s+)?Recommendation\s+(\d{1,3})(?:\s+([A-Z]))?\b"
        )
        rec_markers: List[re.Match[str]] = list(rec_marker_re.finditer(text))

        for idx, match in enumerate(headers):
            rec_id = match.group(1)
            rec_number: Optional[int] = None
            if rec_id and rec_id.isdigit():
                try:
                    rec_number = int(rec_id)
                except (TypeError, ValueError):
                    rec_number = None

            start_body = match.end()
            end_block = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)

            # v1-style delimiter: stop at the first "Recommendation N <letter>" marker.
            for rec_m in rec_markers:
                pos = rec_m.start()
                if pos <= start_body:
                    continue
                if pos < end_block:
                    end_block = pos
                    break

            span = (start_body, end_block)
            body_text = text[start_body:end_block].strip()
            if not body_text:
                continue

            responses.append(
                Response(
                    text=body_text,
                    span=span,
                    source_document=source_document,
                    response_type="structured",
                    rec_id=rec_id,
                    rec_number=rec_number,
                )
            )

        # ------------------------------------------------------------------ #
        # Phase 2: action-verb responses (fallback only)
        # ------------------------------------------------------------------ #
        if not responses and preprocessed.sentence_spans:
            for sent_start, sent_end in preprocessed.sentence_spans:
                sentence = text[sent_start:sent_end].strip()
                if len(sentence) < 40 or len(sentence) > 500:
                    continue
                if self._is_scattered_response(sentence):
                    rec_id, rec_number = self._infer_rec_identity(sentence)
                    responses.append(
                        Response(
                            text=sentence,
                            span=(sent_start, sent_end),
                            source_document=source_document,
                            response_type="action_verb",
                            rec_id=rec_id,
                            rec_number=rec_number,
                        )
                    )

        return responses

    def _is_scattered_response(self, sentence: str) -> bool:
        """
        Heuristic for identifying action-verb response sentences.

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

    def _infer_rec_identity(self, sentence: str) -> tuple[Optional[str], Optional[int]]:
        """Best-effort rec_id / rec_number inference from an action-verb response sentence."""
        lower = sentence.lower()
        m = re.search(r"recommendation\s+([^\s:]+)", lower)
        if not m:
            return None, None
        rec_id = m.group(1)
        rec_number: Optional[int] = None
        if rec_id.isdigit():
            try:
                rec_number = int(rec_id)
            except ValueError:
                rec_number = None
        return rec_id, rec_number
