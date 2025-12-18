"""
v2 recommendation extraction.

Takes `PreprocessedText` and produces structured `Recommendation` objects.
"""

import logging
import re
from functools import lru_cache
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

    def __init__(
        self,
        profile: Optional[str] = None,
        action_verb_min_confidence: float = 0.75,
        enable_pfd_directives: bool = False,
        pfd_atomize_concerns: bool = False,
    ):
        """
        Parameters
        ----------
        profile:
            Optional profile name (e.g. "explicit_recs", "pfd_report")
            controlling document‑specific heuristics.
        action_verb_min_confidence:
            Minimum confidence threshold for v1-style action-verb inference.
            This is deliberately aligned with the v1 extractor default so that
            v2's "action_verb" channel behaves like the legacy implementation.
        enable_pfd_directives:
            Whether to extract `rec_type="pfd_directive"` units for PFD reports.
            This is kept opt-in so the default behaviour remains close to the
            legacy action-verb approach (supervisor continuity).
        pfd_atomize_concerns:
            When `profile="pfd_report"`, controls whether numbered concerns are
            returned as block spans (legacy) or split into sentence-level units
            for review/tuning.
        """
        self.profile = profile or EXPLICIT_RECS_PROFILE
        self.action_verb_min_confidence = action_verb_min_confidence
        self.enable_pfd_directives = enable_pfd_directives
        self.pfd_atomize_concerns = pfd_atomize_concerns

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
            if self.enable_pfd_directives:
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
                        confidence=None,
                    )
                )

        # ------------------------------------------------------------------ #
        # Phase 2: action-verb / non-numbered recommendations (all profiles)
        # ------------------------------------------------------------------ #
        # To keep behaviour aligned with v1, we intentionally use the same
        # simple regex sentence boundaries for the action-verb channel rather
        # than syntok spans. v2 still uses syntok for other profile-specific
        # logic (e.g. PFD directives), but "action_verb" extraction should
        # match the legacy v1 heuristic semantics as closely as possible.
        sentence_spans = self._v1_sentence_spans(text)
        if sentence_spans:
            primary_spans = [rec.span for rec in recommendations]

            def in_primary_span(start: int) -> bool:
                for s, e in primary_spans:
                    if s <= start < e:
                        return True
                return False

            for sent_start, sent_end in sentence_spans:
                if in_primary_span(sent_start):
                    continue
                sent_text = text[sent_start:sent_end].strip()
                is_rec, confidence, method, cleaned_text = self._v1_style_action_verb(sent_text)
                if not is_rec or confidence < self.action_verb_min_confidence:
                    continue

                recommendations.append(
                    Recommendation(
                        text=cleaned_text,
                        span=(sent_start, sent_end),
                        source_document=source_document,
                        rec_id=None,
                        rec_number=None,
                        rec_type="action_verb",
                        detection_method=method,
                        confidence=confidence,
                    )
                )

        return recommendations

    @staticmethod
    def _v1_sentence_spans(text: str) -> list[tuple[int, int]]:
        """
        Approximate v1 sentence splitting using the same boundary regex.

        v1 uses:
          re.split(r'(?<=[.!?])\\s+(?=[A-Z])', text)

        Here we produce spans so we can preserve offsets and apply span-based
        exclusion against primary recommendation blocks.
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

    @staticmethod
    @lru_cache(maxsize=1)
    def _v1_extractor():  # type: ignore[no-untyped-def]
        """
        Cached v1 extractor instance used to keep v2 action-verb logic
        behaviorally identical to the legacy implementation.
        """
        from daphne_core.recommendation_extractor import StrictRecommendationExtractor

        return StrictRecommendationExtractor()

    def _v1_style_action_verb(self, sentence: str) -> tuple[bool, float, str, str]:
        """
        Run the legacy v1 inference logic over a single sentence.

        This intentionally mirrors the v1 fallback (non-numbered) path:
        - aggressive cleaning
        - garbage/meta filtering
        - pattern-based recommendation detection returning (is_rec, confidence, method)
        """
        extractor = self._v1_extractor()
        cleaned = extractor.clean_text(sentence)

        is_garbage, _reason = extractor.is_garbage(cleaned, is_numbered_rec=False)
        if is_garbage:
            return False, 0.0, "garbage", cleaned

        if extractor.is_meta_recommendation(cleaned):
            return False, 0.0, "meta", cleaned

        is_rec, confidence, method, _verb = extractor.is_genuine_recommendation(cleaned, is_numbered_rec=False)
        return is_rec, float(confidence), method, cleaned

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
            if self.pfd_atomize_concerns:
                return self._extract_pfd_concerns_fallback(text, source_document)
            return concerns

        start_search = anchor.end()
        end_match = re.search(r"ACTION\s+SHOULD\s+BE\s+TAKEN", text[start_search:], re.IGNORECASE)
        end_pos = start_search + end_match.start() if end_match else len(text)

        # Numbered concerns such as "(1)", "(2)", "(3)".
        # Anchor to start-of-line to avoid matching incidental parentheticals like "(and/or)".
        numbered_pattern = re.compile(r"(?m)^\s*\(\s*(\d+)\s*\)")
        matches = list(numbered_pattern.finditer(text, start_search, end_pos))

        if not matches:
            logger.info(
                "v2: PFD profile – no numbered concerns found under 'MATTERS OF CONCERN'; "
                "skipping concerns list extraction",
            )
            if self.pfd_atomize_concerns:
                # First try the MATTERS-of-concern window; if it yields no hits
                # (common when the window mostly contains narrative context),
                # fall back to scanning the broader document like v1.
                window_hits = self._extract_pfd_concerns_fallback(text[start_search:end_pos], source_document)
                if window_hits:
                    return window_hits
                return self._extract_pfd_concerns_fallback(text, source_document)
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

            if not self.pfd_atomize_concerns:
                concerns.append(
                    Recommendation(
                        text=raw_block.strip(),
                        span=span,
                        source_document=source_document,
                        rec_id=f"concern_{concern_num_str}",
                        rec_number=rec_number,
                        rec_type="pfd_concern",
                        detection_method="pfd_matters_of_concern",
                        confidence=None,
                    )
                )
                continue

            sentence_spans = self._v1_sentence_spans(raw_block)
            for sent_idx, (sent_start, sent_end) in enumerate(sentence_spans, 1):
                global_start = start + sent_start
                global_end = start + sent_end
                sent_text = text[global_start:global_end].strip()
                if not sent_text:
                    continue

                if self._pfd_is_boilerplate_sentence(sent_text):
                    continue

                # Baseline guarantee: if the legacy v1 action-verb logic would
                # extract this sentence as a recommendation, include it in the
                # atomised PFD output.
                is_rec, confidence, method, cleaned_text = self._v1_style_action_verb(sent_text)
                if is_rec and confidence >= self.action_verb_min_confidence:
                    concerns.append(
                        Recommendation(
                            text=cleaned_text,
                            span=(global_start, global_end),
                            source_document=source_document,
                            rec_id=f"concern_{concern_num_str}_s{sent_idx}",
                            rec_number=rec_number,
                            rec_type="pfd_concern",
                            detection_method=f"pfd_matters_of_concern_atomized_v1:{method}",
                            confidence=confidence,
                        )
                    )
                    continue

                trigger = self._pfd_atomized_trigger(sent_text, allow_weak=True)
                if not trigger:
                    continue

                concerns.append(
                    Recommendation(
                        text=sent_text,
                        span=(global_start, global_end),
                        source_document=source_document,
                        rec_id=f"concern_{concern_num_str}_s{sent_idx}",
                        rec_number=rec_number,
                        rec_type="pfd_concern",
                        detection_method=f"pfd_matters_of_concern_atomized:{trigger}",
                        confidence=None,
                    )
                )

        if self.pfd_atomize_concerns:
            # Ensure v1 is a subset of the "extended" output by always adding
            # the v1 baseline extractions (ported rules) as additional items.
            concerns = self._add_v1_baseline_pfd_concerns(text, source_document, concerns)

        logger.info("v2: Extracted %d PFD concerns", len(concerns))
        if self.pfd_atomize_concerns and not concerns:
            # If we entered the structured PFD path but ended up emitting nothing
            # (e.g., numbering matched unexpectedly), fall back to the broader scan.
            return self._extract_pfd_concerns_fallback(text, source_document)
        return concerns

    @staticmethod
    def _pfd_atomized_trigger(sentence: str, *, allow_weak: bool) -> Optional[str]:
        """
        Returns a short trigger label if the sentence looks like a response-required
        concern (Target B); otherwise returns None.

        This is intentionally high-precision and is used to filter out pure
        context/timeline sentences when `pfd_atomize_concerns=True`.
        """
        cleaned = sentence.strip()
        cleaned = re.sub(r"^(?:[-–•\s]+)", "", cleaned)
        cleaned = re.sub(r"^(?:Part\s+\d+\s*)?(?:\(\d+\)|\d+\.)\s*", "", cleaned, flags=re.IGNORECASE)

        starters = [
            ("There was no", re.compile(r"(?i)^there\s+was\s+no\b")),
            ("There was a lack of", re.compile(r"(?i)^there\s+was\s+a\s+lack\s+of\b")),
            ("There was a failure to", re.compile(r"(?i)^there\s+was\s+a\s+failure\s+to\b")),
            ("No action was taken", re.compile(r"(?i)^no\s+action\s+was\s+taken\b")),
            ("I am concerned", re.compile(r"(?i)^i\s+am\s+concerned\b")),
            ("I remain concerned", re.compile(r"(?i)^i\s+remain\s+concerned\b")),
            ("I am alarmed", re.compile(r"(?i)^i\s+am\s+alarmed\b")),
            ("There remains no", re.compile(r"(?i)^there\s+remains\b.*\bno\b")),
            ("There are no", re.compile(r"(?i)^there\s+are\s+no\b")),
        ]
        if allow_weak:
            starters.extend(
                [
                    ("was not", re.compile(r"(?i)^.*\bwas\s+not\b")),
                    ("were not", re.compile(r"(?i)^.*\bwere\s+not\b")),
                ]
            )
        for label, pattern in starters:
            if pattern.search(cleaned):
                return label

        m = re.search(r"(?i)\bdid\s+not\s+(\w+)\b", cleaned)
        if m:
            return f"did not {m.group(1).lower()}"

        m = re.search(r"(?i)\bfailed\s+to\s+(\w+)\b", cleaned)
        if m:
            return f"failed to {m.group(1).lower()}"

        m = re.search(r"(?i)\bunable\s+to\s+(\w+)\b", cleaned)
        if m:
            return f"unable to {m.group(1).lower()}"

        if re.search(r"(?i)\bgives?\s+rise\s+to\s+(?:a\s+)?risk\b", cleaned):
            return "gives rise to risk"

        return None

    @staticmethod
    def _pfd_is_boilerplate_sentence(sentence: str) -> bool:
        text = " ".join(sentence.strip().split())
        if not text:
            return True
        return bool(
            re.search(
                r"(?i)\b("
                r"in\s+my\s+opinion\s+there\s+is\s+a\s+risk\s+that\s+future\s+deaths?"
                r"|in\s+my\s+opinion\s+action\s+should\s+be\s+taken\s+to\s+prevent\s+future\s+deaths"
                r"|you\s+are\s+under\s+a\s+duty\s+to\s+respond"
                r"|within\s+56\s+days"
                r"|your\s+response\s+must\s+contain"
                r"|otherwise\s+you\s+must\s+explain\s+why\s+no\s+action\s+is\s+proposed"
                r"|copies\s+and\s+publication"
                r")\b",
                text,
            )
        )

    def _extract_pfd_concerns_fallback(self, text: str, source_document: str) -> List[Recommendation]:
        """
        Fallback for PFD atomised concerns when no clean numbered MATTERS OF CONCERN
        list is found.

        Strategy:
        - Prefer the "CORONER'S CONCERNS" → "ACTION SHOULD BE TAKEN" window when present.
        - Otherwise scan the entire document.
        - Emit only sentences matching the high-precision trigger set.
        """
        if not text or not text.strip():
            return []

        def extract_from_region(region: str, region_start: int, *, allow_weak: bool) -> List[Recommendation]:
            out_local: List[Recommendation] = []
            for sent_idx, (s_start, s_end) in enumerate(self._v1_sentence_spans(region), 1):
                sent_text = region[s_start:s_end].strip()
                if not sent_text:
                    continue
                if self._pfd_is_boilerplate_sentence(sent_text):
                    continue

                is_rec, confidence, method, cleaned_text = self._v1_style_action_verb(sent_text)
                if is_rec and confidence >= self.action_verb_min_confidence:
                    out_local.append(
                        Recommendation(
                            text=cleaned_text,
                            span=(region_start + s_start, region_start + s_end),
                            source_document=source_document,
                            rec_id=f"concern_fallback_s{sent_idx}",
                            rec_number=None,
                            rec_type="pfd_concern",
                            detection_method=f"pfd_concerns_fallback_atomized_v1:{method}",
                            confidence=confidence,
                        )
                    )
                    continue

                trigger = self._pfd_atomized_trigger(sent_text, allow_weak=allow_weak)
                if not trigger:
                    continue

                out_local.append(
                    Recommendation(
                        text=sent_text,
                        span=(region_start + s_start, region_start + s_end),
                        source_document=source_document,
                        rec_id=f"concern_fallback_s{sent_idx}",
                        rec_number=None,
                        rec_type="pfd_concern",
                        detection_method=f"pfd_concerns_fallback_atomized:{trigger}",
                        confidence=None,
                    )
                )
            return out_local

        # Pass 1: prefer the coroner concerns window if present (reduces noise).
        start = 0
        m_start = re.search(r"CORONER[’']?S\s+CONCERNS", text, re.IGNORECASE)
        if m_start:
            start = m_start.end()
            m_ws = re.match(r"\s+", text[start:])
            if m_ws:
                start += m_ws.end()

        end = len(text)
        m_end = re.search(r"ACTION\s+SHOULD\s+BE\s+TAKEN", text[start:], re.IGNORECASE)
        if m_end:
            end = start + m_end.start()

        window_region = text[start:end]
        out = extract_from_region(window_region, start, allow_weak=False)

        # Pass 2: if the windowed scan yields nothing but v1 would have returned
        # results elsewhere, fall back to scanning the full document (v1-style).
        if not out and start != 0:
            out = extract_from_region(text, 0, allow_weak=False)

        if self.pfd_atomize_concerns:
            out = self._add_v1_baseline_pfd_concerns(text, source_document, out)

        logger.info("v2: PFD fallback atomised concerns extracted %d items", len(out))
        return out

    def _add_v1_baseline_pfd_concerns(
        self,
        text: str,
        source_document: str,
        existing: List[Recommendation],
    ) -> List[Recommendation]:
        """
        Guarantee: when `pfd_atomize_concerns=True`, v1 extractions are always a
        subset of the returned v2 items.

        We do this by running the legacy v1 action-verb logic across the full
        document (over v1 sentence spans) and adding any hits that are not already
        represented in the atomised output.
        """
        if not text or not text.strip():
            return existing

        def norm(s: str) -> str:
            txt = " ".join((s or "").strip().lower().split())
            # Strip common enumeration prefixes so that "(2) The Trust..." and
            # "The Trust..." are treated as the same unit for deduplication.
            txt = re.sub(r"^(?:part\s+\d+\s+)?(?:\(\d+\)|\d+\.)\s*", "", txt)
            return txt

        seen: set[str] = set()
        for r in existing:
            seen.add(norm(getattr(r, "text", "") or ""))

        out = list(existing)
        for idx, (s_start, s_end) in enumerate(self._v1_sentence_spans(text), 1):
            sent_text = text[s_start:s_end].strip()
            if not sent_text:
                continue
            if self._pfd_is_boilerplate_sentence(sent_text):
                continue

            is_rec, confidence, method, cleaned_text = self._v1_style_action_verb(sent_text)
            if not is_rec or confidence < self.action_verb_min_confidence:
                continue

            key = norm(cleaned_text)
            if not key or key in seen:
                continue
            seen.add(key)

            out.append(
                Recommendation(
                    text=cleaned_text,
                    span=(s_start, s_end),
                    source_document=source_document,
                    rec_id=f"concern_v1_s{idx}",
                    rec_number=None,
                    rec_type="pfd_concern",
                    detection_method=f"pfd_v1_baseline:{method}",
                    confidence=confidence,
                )
            )

        return out

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
                    confidence=None,
                )
            )

        if directives:
            logger.info("v2: Extracted %d PFD directive sentences", len(directives))
        return directives
