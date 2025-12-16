"""
v2 recommendation extraction.

Takes `PreprocessedText` and produces structured `Recommendation` objects.
"""

import logging
import re
from typing import List, Optional

from .types import PreprocessedText, Recommendation


logger = logging.getLogger(__name__)

EXPLICIT_RECS_PROFILE = "explicit_recs"
PFD_REPORT_PROFILE = "pfd_report"


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
            Optional profile name (e.g. "explicit_recs", "pfd_report")
            controlling document‑specific heuristics.
        """
        self.profile = profile or EXPLICIT_RECS_PROFILE

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

        recommendations: List[Recommendation] = []

        if self.profile == PFD_REPORT_PROFILE:
            # PFD (Regulation 28) style reports:
            # - Primary units: numbered "MATTERS OF CONCERN" items when present.
            # - For long-form / narrative reports where those items are not a
            #   clean list, also extract explicit directive sentences ("I recommend…",
            #   "I request…", "It is vital that…", etc.).
            recommendations.extend(self._extract_pfd_concerns(text, source_document))
            recommendations.extend(self._extract_pfd_directives(preprocessed, source_document))
        else:
            # Default v2 implementation: focus on "Recommendation ..." headings,
            # using heading-based segmentation similar to the v1 strict
            # extractor but applied to the centralised v2 preprocessed text.
            # We treat the token following "Recommendation" as a label (rec_id),
            # which may be a simple integer ("1") or a more complex code ("2018/007").
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

                # Light normalisation to avoid duplicated section titles and
                # simple numbered artefacts, without reintroducing the heavy v1
                # cleaning stack. Further domain-specific tweaks can live behind
                # the `profile` parameter once we know what generalisation
                # cannot do.
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
                        rec_type="numbered",
                        detection_method="heading",
                    )
                )

        # ------------------------------------------------------------------ #
        # Phase 2: action-verb / non-numbered recommendations (all profiles)
        # ------------------------------------------------------------------ #
        if preprocessed.sentence_spans:
            primary_spans = [rec.span for rec in recommendations]

            def in_primary_span(start: int) -> bool:
                for s, e in primary_spans:
                    if s <= start < e:
                        return True
                return False

            for sent_start, sent_end in preprocessed.sentence_spans:
                if in_primary_span(sent_start):
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
                            rec_type="action_verb",
                            detection_method="verb_based",
                        )
                    )

        return recommendations

    def _is_scattered_recommendation(self, sentence: str) -> bool:
        """
        Heuristic for identifying non-numbered / action-verb recommendations.

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

    def _extract_pfd_concerns(self, text: str, source_document: str) -> List[Recommendation]:
        """
        Extract recommendation-like units from a Prevention of Future Deaths
        (Regulation 28) report.

        We treat the numbered items under "MATTERS OF CONCERN" as primary
        concerns, typically written as "(1)", "(2)", "(3)" etc.
        """
        concerns: List[Recommendation] = []

        anchor = re.search(r"MATTERS\s+OF\s+CONCERN", text, re.IGNORECASE)
        if not anchor:
            logger.info("v2: PFD profile selected but no 'MATTERS OF CONCERN' anchor found")
            return concerns

        start_search = anchor.end()
        end_match = re.search(r"ACTION\s+SHOULD\s+BE\s+TAKEN", text[start_search:], re.IGNORECASE)
        end_pos = start_search + end_match.start() if end_match else len(text)

        # Numbered concerns such as "(1)", "(2)", "(3)".
        numbered_pattern = re.compile(r"\(\s*(\d+)\s*\)")
        matches = list(numbered_pattern.finditer(text, start_search, end_pos))

        if not matches:
            logger.info(
                "v2: PFD profile – no numbered concerns found under 'MATTERS OF CONCERN'; "
                "skipping concerns list extraction",
            )
            return concerns

        for idx, m in enumerate(matches):
            concern_num_str = m.group(1)
            try:
                rec_number = int(concern_num_str)
            except ValueError:
                rec_number = None

            start = m.start()
            if idx + 1 < len(matches):
                end = matches[idx + 1].start()
            else:
                end = end_pos

            span = (start, end)
            raw_block = text[start:end]

            concerns.append(
                Recommendation(
                    text=raw_block.strip(),
                    span=span,
                    source_document=source_document,
                    rec_id=f"concern_{concern_num_str}",
                    rec_number=rec_number,
                    rec_type="pfd_concern",
                    detection_method="pfd_matters_of_concern",
                )
            )

        logger.info("v2: Extracted %d PFD concerns", len(concerns))
        return concerns

    def _extract_pfd_directives(self, preprocessed: PreprocessedText, source_document: str) -> List[Recommendation]:
        """
        Extract explicit directive sentences from a PFD-style document.

        This is designed for long-form/narrative reports where the actionable
        recommendations appear as embedded directives rather than as a clean
        numbered "MATTERS OF CONCERN" list.
        """
        text = preprocessed.text or ""
        if not text.strip() or not preprocessed.sentence_spans:
            return []

        directives: List[Recommendation] = []

        first_person_patterns = [
            r"\bi\s+(?:strongly\s+)?recommend\b",
            r"\bi\s+request\b",
            r"\bi\s+suggest\b",
            r"\bi\s+encourage\b",
        ]
        importance_patterns = [
            r"\bit\s+is\s+(?:vital|critical|important)\s+that\b",
        ]
        first_person_re = re.compile("|".join(first_person_patterns), re.IGNORECASE)
        importance_re = re.compile("|".join(importance_patterns), re.IGNORECASE)

        # In long-form PFD reports, some directives are written as "The <addressee> should <action> ...".
        # We keep this narrow to avoid pulling in third-party statements like "Mr X agreed that ..."
        # and generic boilerplate like "action should be taken".
        addressee_re = re.compile(
            r"\b("
            r"secretary\s+of\s+state|home\s+department|home\s+office|"
            r"chief\s+constable|police|trust|nhs|department|minister|"
            r"prison|probation|hmpps|mappa|pathfinder|prevent"
            r")\b",
            re.IGNORECASE,
        )

        boilerplate_re = re.compile(
            r"\b(in\s+my\s+opinion\s+)?action\s+should\s+be\s+taken\s+to\s+prevent\s+future\s+deaths\b|"
            r"\bduty\s+of\s+those\s+receiving\s+this\s+report\b.*\baction\s+that\s+should\s+be\s+taken\b|"
            r"\byou\s+are\s+under\s+a\s+duty\s+to\s+respond\b",
            re.IGNORECASE,
        )

        reported_speech_re = re.compile(
            r"^(mr|ms|mrs|dr|professor|sir)\s+\w+.*\b(agreed|said|stated|accepted|explained|"
            r"confirmed|indicated|noted|recognised|recognized)\b.*\bshould\b",
            re.IGNORECASE,
        )

        narrative_conclusion_re = re.compile(
            r"\bmy\s+overall\s+conclusions\b|\bfactual\s+findings\b",
            re.IGNORECASE,
        )

        for sent_start, sent_end in preprocessed.sentence_spans:
            sent_text = text[sent_start:sent_end].strip()
            if len(sent_text) < 40 or len(sent_text) > 700:
                continue

            lower = sent_text.lower()
            if boilerplate_re.search(sent_text):
                continue
            if reported_speech_re.search(sent_text):
                continue

            # Drop long narrative/citation sentences that are primarily quoting or summarising findings.
            if narrative_conclusion_re.search(sent_text) and ":" in sent_text[:200]:
                continue

            is_first_person = bool(first_person_re.search(sent_text))
            is_importance = bool(importance_re.search(sent_text))
            is_should_addressee = False
            if not (is_first_person or is_importance):
                if " should " in f" {lower} " and addressee_re.search(sent_text):
                    if any(f" {verb} " in f" {lower} " for verb in ACTION_VERBS):
                        is_should_addressee = True

            is_directive = is_first_person or is_importance or is_should_addressee

            if not is_directive:
                continue

            directives.append(
                Recommendation(
                    text=sent_text,
                    span=(sent_start, sent_end),
                    source_document=source_document,
                    rec_id=f"directive_{len(directives) + 1}",
                    rec_number=None,
                    rec_type="pfd_directive",
                    detection_method="pfd_directive_sentence",
                )
            )

        if directives:
            logger.info("v2: Extracted %d PFD directive sentences", len(directives))
        return directives
