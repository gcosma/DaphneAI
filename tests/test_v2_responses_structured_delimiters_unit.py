from __future__ import annotations

from daphne_core.v2.responses import ResponseExtractorV2
from daphne_core.v2.types import PreprocessedText


def test_v2_structured_response_stops_at_embedded_recommendation_excerpt() -> None:
    text = (
        "Government response to recommendation 1 The government supports the intent of this recommendation.\n"
        "We will take action accordingly.\n"
        "\n"
        "Recommendation 2 Every provider and commissioner of NHS-funded care should have access to digital platforms.\n"
        "\n"
        "Government response to recommendation 2 The government notes this recommendation.\n"
        "We will consider it.\n"
    )
    pre = PreprocessedText(text=text, sentence_spans=[], page_spans=[])
    resps = ResponseExtractorV2().extract(pre, source_document="Report1Response.pdf")

    assert len(resps) == 2
    assert resps[0].rec_id == "1"
    assert resps[0].rec_number == 1
    assert "Recommendation 2" not in resps[0].text
    assert "We will take action accordingly." in resps[0].text
