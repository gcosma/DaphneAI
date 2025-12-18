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

from daphne_core.alignment_engine import clean_pdf_artifacts, is_genuine_response, is_recommendation_text


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
        # Phase 2: scattered/unstructured responses (fallback only)
        # ------------------------------------------------------------------ #
        if not responses:
            # 2a) Paragraph-style blocks (common in letter-style or narrative responses).
            for start, end in self._paragraph_spans(text):
                block = text[start:end].strip()
                if not block:
                    continue
                cleaned = clean_pdf_artifacts(block)
                if len(cleaned) < 80 or len(cleaned) > 4000:
                    continue
                if is_recommendation_text(cleaned):
                    continue
                # Avoid matching embedded recommendation excerpts.
                if rec_marker_re.match(block):
                    continue
                # Prefer blocks that reference a recommendation or have strong response language.
                if "recommendation" not in cleaned.lower() and not is_genuine_response(cleaned):
                    continue

                rec_id, rec_number = self._infer_rec_identity(cleaned)
                responses.append(
                    Response(
                        text=cleaned,
                        span=(start, end),
                        source_document=source_document,
                        response_type="paragraph",
                        rec_id=rec_id,
                        rec_number=rec_number,
                    )
                )

        if not responses:
            # 2b) Sentence-level candidates for semantic/keyword matching when we
            # can't find any structured markers.
            for sent_start, sent_end in self._v1_sentence_spans(text):
                sentence_raw = text[sent_start:sent_end].strip()
                if not sentence_raw:
                    continue
                cleaned = clean_pdf_artifacts(sentence_raw)
                if len(cleaned) < 40 or len(cleaned) > 800:
                    continue
                if is_recommendation_text(cleaned):
                    continue
                if rec_marker_re.match(sentence_raw):
                    continue

                rec_id, rec_number = self._infer_rec_identity(cleaned)
                resp_type = "action_verb" if self._is_scattered_response(cleaned) else "sentence"
                responses.append(
                    Response(
                        text=cleaned,
                        span=(sent_start, sent_end),
                        source_document=source_document,
                        response_type=resp_type,
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
        return is_genuine_response(sentence)

    @staticmethod
    def _paragraph_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        if not text:
            return spans
        start = 0
        for m in re.finditer(r"\n{2,}", text):
            end = m.start()
            if end > start:
                spans.append((start, end))
            start = m.end()
        if start < len(text):
            spans.append((start, len(text)))
        return spans

    @staticmethod
    def _v1_sentence_spans(text: str) -> list[tuple[int, int]]:
        """
        Approximate v1 sentence splitting using a simple boundary regex.

        Mirrors the v1 alignment engine approach so the fallback behaviour
        is closer to the legacy pipeline.
        """
        spans: list[tuple[int, int]] = []
        if not text:
            return spans
        start = 0
        for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", text):
            end = match.end()
            spans.append((start, end))
            start = end
        if start < len(text):
            spans.append((start, len(text)))
        return spans

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
