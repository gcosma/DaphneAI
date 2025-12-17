from __future__ import annotations

import warnings
from pathlib import Path

import pytest

def _assert_count_with_warning(name: str, actual: int, expected: int) -> None:
    """
    Enforce a regression baseline:
    - fail if `actual` < `expected` (regression)
    - warn (but pass) if `actual` > `expected` (new items picked up)
    """
    if actual < expected:
        pytest.fail(f"{name} regressed: expected >= {expected}, got {actual}")
    if actual > expected:
        warnings.warn(
            f"{name} increased: expected {expected}, got {actual} (review whether this is desired)",
            UserWarning,
        )


def test_v2_reference_pair_regression() -> None:
    """
    v2 regression check for the canonical reference PDFs.

    Semantics:
    - red (fail): missing expected items or broken numbered alignment
    - yellow (warn): additional items are picked up vs the current baseline

    This is intended to be a quick "did we regress?" gate while v2 evolves.
    """
    # Keep the signal of our regression warnings high by suppressing known noisy dependency warnings.
    # This needs to run before importing `daphne_core`, which may import PyPDF2 at import time.
    warnings.filterwarnings("ignore", message=r"PyPDF2 is deprecated.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=r".*joblib will operate in serial mode.*", category=UserWarning)

    from daphne_core.v2.alignment import AlignmentStrategyV2
    from daphne_core.v2.preprocess import extract_text
    from daphne_core.v2.recommendations import EXPLICIT_RECS_PROFILE, RecommendationExtractorV2
    from daphne_core.v2.responses import ResponseExtractorV2

    recs_pdf = Path("samplepdfs/Report1.pdf")
    resps_pdf = Path("samplepdfs/Report1Response.pdf")
    assert recs_pdf.exists(), f"Missing reference PDF: {recs_pdf}"
    assert resps_pdf.exists(), f"Missing reference PDF: {resps_pdf}"

    try:
        recs_pre = extract_text(recs_pdf)
        resps_pre = extract_text(resps_pdf)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    rec_extractor = RecommendationExtractorV2(profile=EXPLICIT_RECS_PROFILE)
    resp_extractor = ResponseExtractorV2()

    recs = rec_extractor.extract(recs_pre, source_document=recs_pdf.name)
    resps = resp_extractor.extract(resps_pre, source_document=resps_pdf.name)
    alignments = AlignmentStrategyV2(enforce_one_to_one=False).align(recs, resps)

    numbered_recs = [r for r in recs if getattr(r, "rec_type", None) == "numbered"]
    action_verb_recs = [r for r in recs if getattr(r, "rec_type", None) == "action_verb"]
    structured_resps = [r for r in resps if r.response_type == "structured"]
    action_verb_resps = [r for r in resps if r.response_type == "action_verb"]

    expected = {
        "recs_total": 13,
        "recs_numbered": 13,
        "recs_action_verb": 0,
        "resps_total": 13,
        "resps_structured": 13,
        "resps_action_verb": 0,
        "alignments_total": 13,
    }

    _assert_count_with_warning("v2 recommendations (total)", len(recs), expected["recs_total"])
    _assert_count_with_warning("v2 recommendations (numbered)", len(numbered_recs), expected["recs_numbered"])
    _assert_count_with_warning("v2 recommendations (action-verb)", len(action_verb_recs), expected["recs_action_verb"])
    _assert_count_with_warning("v2 responses (total)", len(resps), expected["resps_total"])
    _assert_count_with_warning("v2 responses (structured)", len(structured_resps), expected["resps_structured"])
    _assert_count_with_warning("v2 responses (action-verb)", len(action_verb_resps), expected["resps_action_verb"])
    _assert_count_with_warning("v2 alignments (total)", len(alignments), expected["alignments_total"])

    expected_numbers = set(range(1, 14))
    numbered_rec_numbers = {r.rec_number for r in numbered_recs if r.rec_number is not None}
    structured_resp_numbers = {r.rec_number for r in structured_resps if r.rec_number is not None}

    missing_rec_numbers = expected_numbers - numbered_rec_numbers
    missing_resp_numbers = expected_numbers - structured_resp_numbers
    if missing_rec_numbers:
        pytest.fail(f"Missing numbered recommendations for: {sorted(missing_rec_numbers)}")
    if missing_resp_numbers:
        pytest.fail(f"Missing structured responses for: {sorted(missing_resp_numbers)}")

    extra_rec_numbers = numbered_rec_numbers - expected_numbers
    extra_resp_numbers = structured_resp_numbers - expected_numbers
    if extra_rec_numbers:
        warnings.warn(
            f"Extra numbered recommendation IDs found: {sorted(extra_rec_numbers)} (review whether desired)",
            UserWarning,
        )
    if extra_resp_numbers:
        warnings.warn(
            f"Extra structured response IDs found: {sorted(extra_resp_numbers)} (review whether desired)",
            UserWarning,
        )

    # Alignment contract for the reference pair: each numbered recommendation 1..13
    # should align to the correspondingly numbered structured response.
    for rec in numbered_recs:
        if rec.rec_number is None:
            continue
        alignment = next((a for a in alignments if a.recommendation is rec), None)
        if alignment is None or alignment.response is None:
            pytest.fail(f"Missing alignment response for numbered recommendation {rec.rec_number}")
        if alignment.response.rec_number != rec.rec_number:
            pytest.fail(
                "Misaligned numbered recommendation "
                f"{rec.rec_number}: got response rec_number={alignment.response.rec_number}"
            )
