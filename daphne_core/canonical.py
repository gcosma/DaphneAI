"""
Canonical (single) pipeline entrypoints.

This module exists to support the “one canonical implementation” phase:
- v2 preprocessing as baseline ingestion
- v2 structured extraction (profiles)
- v1-parity action-verb inference as an additional channel (second pass)

UI and CLIs should prefer these helpers instead of directly wiring together
multiple v1/v2 codepaths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from daphne_core.v2.preprocess import extract_text as extract_text_v2
from daphne_core.v2.recommendations import (
    EXPLICIT_RECS_PROFILE,
    PFD_REPORT_PROFILE,
    RecommendationExtractorV2,
)
from daphne_core.v2.types import PreprocessedText, Recommendation


@dataclass(frozen=True)
class CanonicalRecsResult:
    preprocessed: PreprocessedText
    recommendations: List[Recommendation]


def extract_recommendations_from_pdf(
    pdf_path: Path,
    *,
    source_document: Optional[str] = None,
    profile: str = EXPLICIT_RECS_PROFILE,
    action_verb_min_confidence: float = 0.75,
    pfd_atomize_concerns: bool = False,
    enable_pfd_directives: bool = False,
    dedupe_action_verbs: bool = True,
) -> CanonicalRecsResult:
    """
    Canonical recommendation extraction from a PDF.

    Notes
    -----
    - Uses v2 preprocessing.
    - Uses v2 structured extraction for the selected profile.
    - Uses v1-parity action-verb inference as a second pass (implemented inside v2).
    """
    pre = extract_text_v2(pdf_path)

    extractor = RecommendationExtractorV2(
        profile=profile,
        action_verb_min_confidence=action_verb_min_confidence,
        enable_pfd_directives=enable_pfd_directives,
        pfd_atomize_concerns=pfd_atomize_concerns,
    )
    src = source_document or pdf_path.name
    recs = extractor.extract(pre, source_document=src)

    if dedupe_action_verbs:
        recs = _dedupe_action_verbs(recs)

    return CanonicalRecsResult(preprocessed=pre, recommendations=recs)


def infer_profile_from_filename(filename: str) -> str:
    """
    Cheap heuristic to select a default document profile from a filename.
    """
    name = (filename or "").lower()
    if "prevention-of-future-deaths" in name or "pfd" in name:
        return PFD_REPORT_PROFILE
    return EXPLICIT_RECS_PROFILE


def _dedupe_action_verbs(recs: Iterable[Recommendation]) -> List[Recommendation]:
    """
    Drop duplicate action-verb recommendations by normalized text.

    Rationale: v1 de-duplicates its sentence-level hits; canonical output
    should avoid repeating the same sentence multiple times even if it occurs
    more than once in the preprocessed text.
    """

    def norm(s: str) -> str:
        return " ".join((s or "").strip().lower().split())

    kept: List[Recommendation] = []
    seen: set[str] = set()

    for r in recs:
        if (r.rec_type or "") != "action_verb":
            kept.append(r)
            continue

        key = norm(r.text)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(r)

    return kept

