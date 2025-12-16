from pathlib import Path

import pytest

from tools.align_cli import run_alignment_pipeline


def test_v1_reference_report_alignment():
    """
    Regression test for the v1 pipeline on the original reference documents.

    Treats current behaviour as a golden baseline:
    - 13 recommendations from Report1.pdf (numbered 1..13)
    - 14 responses from Report1Response.pdf (13 structured + 1 sentence-level)
    - 13 alignments, all "Accepted" under keyword matching.
    """
    recs_pdf = Path("samplepdfs/Report1.pdf")
    resps_pdf = Path("samplepdfs/Report1Response.pdf")

    assert recs_pdf.exists(), f"Missing reference PDF: {recs_pdf}"
    assert resps_pdf.exists(), f"Missing reference PDF: {resps_pdf}"

    # Ensure the v1 test stays deterministic and does not attempt to fetch
    # embedding models over the network (keyword matching is sufficient for
    # this golden baseline).
    from daphne_core import alignment_engine as v1_alignment

    def _disable_transformer(self):  # type: ignore[no-untyped-def]
        self.model = None
        self.use_transformer = False

    v1_alignment.RecommendationResponseMatcher._initialize_model = _disable_transformer  # type: ignore[method-assign]

    results = run_alignment_pipeline(recs_pdf, resps_pdf, min_confidence=0.75)

    recommendations = results["recommendations"]
    responses = results["responses"]
    alignments = results["alignments"]

    # Golden counts based on current, validated behaviour.
    assert len(recommendations) == 13, "Unexpected number of recommendations for v1 reference report"
    assert len(responses) == 14, "Unexpected number of responses for v1 reference report"
    assert len(alignments) == len(recommendations) == 13, "Alignments should be one per recommendation"

    # Status distribution (keyword baseline): all Accepted.
    accepted = 0
    partial = 0
    for alignment in alignments:
        if not alignment.get("has_response"):
            continue
        resp = alignment.get("response") or {}
        status = resp.get("status", "Unclear")
        if status == "Accepted":
            accepted += 1
        elif status == "Partial":
            partial += 1

    assert accepted == 13, f"Expected 13 Accepted alignments, got {accepted}"
    assert partial == 0, f"Expected 0 Partial alignments, got {partial}"
