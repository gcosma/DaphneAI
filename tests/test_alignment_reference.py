from pathlib import Path

from tools.align_cli import run_alignment_pipeline


def test_reference_report_alignment():
    """
    Regression test for the original reference documents.

    Treats current behaviour as a golden baseline:
    - 10 recommendations from Report1.pdf
    - 26 responses from Report1Response.pdf
    - 10 alignments using the semantic matcher, with 9 Accepted and 1 Partial.
    """
    recs_pdf = Path("samplepdfs/Report1.pdf")
    resps_pdf = Path("samplepdfs/Report1Response.pdf")

    assert recs_pdf.exists(), f"Missing reference PDF: {recs_pdf}"
    assert resps_pdf.exists(), f"Missing reference PDF: {resps_pdf}"

    results = run_alignment_pipeline(recs_pdf, resps_pdf, min_confidence=0.75)

    recommendations = results["recommendations"]
    responses = results["responses"]
    alignments = results["alignments"]

    # Golden counts based on current, validated behaviour
    assert len(recommendations) == 10, "Unexpected number of recommendations for reference report"
    assert len(responses) == 26, "Unexpected number of responses for reference report"
    assert len(alignments) == len(recommendations) == 10, "Alignments should be one per recommendation"

    # Status distribution: 9 Accepted, 1 Partial (no Rejected for this pair)
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

    assert accepted == 9, f"Expected 9 Accepted alignments, got {accepted}"
    assert partial == 1, f"Expected 1 Partial alignment, got {partial}"
