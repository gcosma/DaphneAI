from __future__ import annotations

from daphne_core.v2.recommendations import PFD_REPORT_PROFILE, RecommendationExtractorV2
from daphne_core.v2.types import PreprocessedText


def _make_preprocessed(sentences: list[str]) -> PreprocessedText:
    text = ""
    spans = []
    for s in sentences:
        if text:
            text += "\n"
        start = len(text)
        text += s
        end = len(text)
        spans.append((start, end))
    return PreprocessedText(text=text, sentence_spans=spans, page_spans=[(0, len(text))])


def test_pfd_directives_filter_boilerplate_and_reported_speech() -> None:
    pre = _make_preprocessed(
        [
            "Mr Stewart agreed that mental health should never be a reason not to make a Prevent referral.",
            "I am encouraged that these issues are being addressed but it is vital that the Secretary of State for the Home Department ensures there is effective monitoring.",
            "The Secretary of State for the Home Department should direct that the new systems are adequately monitored and evaluated to ensure the problems have been addressed.",
            "In my opinion action should be taken to prevent future deaths.",
            "However, it is the duty of those receiving this Report to identify the action that should be taken to address the risk.",
        ]
    )

    recs = RecommendationExtractorV2(profile=PFD_REPORT_PROFILE, enable_pfd_directives=True).extract(
        pre,
        source_document="synthetic.pdf",
    )
    directives = [r for r in recs if r.rec_type == "pfd_directive"]

    assert len(directives) == 2
    texts = "\n".join(r.text for r in directives).lower()
    assert "vital that the secretary of state" in texts
    assert "should direct" in texts
    assert "action should be taken" not in texts
    assert "mr stewart agreed" not in texts
