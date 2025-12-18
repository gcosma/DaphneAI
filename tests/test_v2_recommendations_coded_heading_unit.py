from daphne_core.v2.recommendations import EXPLICIT_RECS_PROFILE, RecommendationExtractorV2
from daphne_core.v2.types import PreprocessedText


def test_v2_extracts_coded_heading_blocks_like_safety_recommendation() -> None:
    text = (
        "Intro paragraph.\n\n"
        "Safety recommendation R/2024/025:\n"
        "The organisation should review its processes.\n\n"
        "Safety recommendation R/2024/026:\n"
        "The regulator should publish guidance.\n"
    )
    pre = PreprocessedText(text=text, sentence_spans=[], page_spans=[(0, len(text))])
    recs = RecommendationExtractorV2(profile=EXPLICIT_RECS_PROFILE).extract(pre, source_document="doc.pdf")

    structured = [r for r in recs if getattr(r, "rec_type", None) == "numbered"]
    assert len(structured) == 2
    assert structured[0].rec_id == "R/2024/025"
    assert structured[0].rec_number is None
    assert structured[1].rec_id == "R/2024/026"
    assert structured[1].rec_number is None

