"""
v2 recommendation extraction.

Takes `PreprocessedText` and produces structured `Recommendation` objects.
"""

import logging
import re
from typing import List, Optional

from .types import PreprocessedText, Recommendation


logger = logging.getLogger(__name__)


ACTION_VERBS = {
    "establish", "implement", "develop", "create", "improve", "enhance",
    "strengthen", "expand", "increase", "reduce", "review", "assess",
    "evaluate", "consider", "adopt", "introduce", "maintain", "monitor",
    "provide", "support", "enable", "facilitate", "promote", "encourage",
    "prioritise", "prioritize", "address", "tackle", "resolve", "prevent",
    "ensure", "commission", "consult", "update", "clarify", "publish",
    "engage", "deliver", "conduct", "undertake", "initiate", "collaborate",
    "coordinate", "oversee", "regulate", "enforce", "mandate", "allocate",
    "fund", "resource", "train", "educate", "inform", "report", "audit",
    "inspect", "investigate", "examine", "reform", "revise", "amend",
    "streamline", "simplify", "standardise", "standardize", "integrate",
    "consolidate", "extend", "limit", "restrict", "remove", "set", "define",
    "specify", "determine", "approve", "authorise", "authorize", "build",
    "design", "plan", "prepare", "bring", "make", "take",
}


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
        text = preprocessed.text or ""
        if not text.strip():
            logger.warning("Empty text passed to RecommendationExtractorV2.extract")
            return []

        # Initial v2 implementation: focus on "Recommendation ..." headings,
        # using heading-based segmentation similar to the v1 strict extractor
        # but applied to the centralised v2 preprocessed text. We treat the
        # token following "Recommendation" as a label (rec_id), which may be
        # a simple integer ("1") or a more complex code ("2018/007").
        heading_pattern = re.compile(r"(?:Recommendations?\s+)?Recommendation\b", re.IGNORECASE)

        # Only treat matches that look like true headings, i.e. those that
        # start at the beginning of the text or immediately after a newline.
        all_matches = list(heading_pattern.finditer(text))
        matches: List[re.Match[str]] = []
        for m in all_matches:
            start = m.start()
            if start == 0 or text[start - 1] == "\n":
                matches.append(m)

        logger.info("v2: Found %d numbered recommendation headings", len(matches))

        recommendations: List[Recommendation] = []

        for idx, match in enumerate(matches):
            # Derive rec_id from the token(s) immediately following the
            # "Recommendation" keyword.
            after = text[match.end() :]
            # Skip whitespace after "Recommendation".
            m_ws = re.match(r"\s+", after)
            offset = m_ws.end() if m_ws else 0
            after = after[offset:]

            rec_id: Optional[str] = None
            rec_number: Optional[int] = None

            # Case 1: code followed by colon, e.g. "2018/007:"
            m_label_colon = re.match(r"([^\s:]+):", after)
            if m_label_colon:
                rec_id = m_label_colon.group(1)
            else:
                # Case 2: handle spaced digits like "1 1" -> "11".
                m_spaced_digits = re.match(r"(\d{1,2})\s+(\d)\b", after)
                if m_spaced_digits:
                    rec_id = f"{m_spaced_digits.group(1)}{m_spaced_digits.group(2)}"
                else:
                    # Case 3: simple integer label ("1", "12").
                    m_int = re.match(r"(\d{1,3})\b", after)
                    if m_int:
                        rec_id = m_int.group(1)

            if rec_id and rec_id.isdigit():
                try:
                    rec_number = int(rec_id)
                except ValueError:
                    rec_number = None

            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            span = (start, end)
            raw_block = text[start:end]

            # Light normalisation to avoid duplicated section titles and simple
            # numbered artefacts, without reintroducing the heavy v1 cleaning
            # stack. Further domain-specific tweaks can live behind the
            # `profile` parameter once we know what generalisation cannot do.
            cleaned_block = raw_block
            cleaned_block = re.sub(
                r"\bRecommendations\s+Recommendation\s+(\d+)\b",
                r"Recommendation \1",
                cleaned_block,
                flags=re.IGNORECASE,
            )
            cleaned_block = re.sub(
                r"\bRecommendation\s+1\s+1\b",
                "Recommendation 11",
                cleaned_block,
                flags=re.IGNORECASE,
            )

            recommendations.append(
                Recommendation(
                    text=cleaned_block.strip(),
                    span=span,
                    source_document=source_document,
                    rec_id=rec_id,
                    rec_number=rec_number,
                )
            )

        # ------------------------------------------------------------------ #
        # Phase 2: scattered / non-numbered recommendations
        # ------------------------------------------------------------------ #
        if not recommendations and preprocessed.sentence_spans:
            numbered_spans = [rec.span for rec in recommendations]

            def in_numbered_span(start: int) -> bool:
                for s, e in numbered_spans:
                    if s <= start < e:
                        return True
                return False

            for sent_start, sent_end in preprocessed.sentence_spans:
                if in_numbered_span(sent_start):
                    continue
                sent_text = text[sent_start:sent_end].strip()
                if len(sent_text) < 40 or len(sent_text) > 500:
                    continue

                if self._is_scattered_recommendation(sent_text):
                    recommendations.append(
                        Recommendation(
                            text=sent_text,
                            span=(sent_start, sent_end),
                            source_document=source_document,
                            rec_id=None,
                            rec_number=None,
                        )
                    )

        return recommendations

    def _is_scattered_recommendation(self, sentence: str) -> bool:
        """
        Heuristic for identifying non-numbered / scattered recommendations.

        This is intentionally simple and general:
        - "we recommend" anywhere in the sentence; or
        - presence of a strong modal ("should", "must", "shall") together with
          at least one action verb from ACTION_VERBS.
        """
        lower = sentence.lower()
        if "we recommend" in lower:
            return True

        if re.search(r"\b(should|must|shall)\b", lower):
            if any(f" {verb} " in f" {lower} " for verb in ACTION_VERBS):
                return True

        return False
