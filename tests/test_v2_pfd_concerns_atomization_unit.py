from daphne_core.v2.recommendations import PFD_REPORT_PROFILE, RecommendationExtractorV2
from daphne_core.v2.types import PreprocessedText


def test_pfd_concerns_atomization_toggle() -> None:
    text = (
        "CORONER'S CONCERNS\n"
        "The MATTERS OF CONCERN are as follows:\n"
        "(1) There was no care plan. There was a lack of structure.\n"
        "(2) The Trust should review its missing person procedure promptly. On 1 January 2020 a friend called. No action was taken.\n"
        "ACTION SHOULD BE TAKEN\n"
        "In my opinion action should be taken to prevent future deaths.\n"
    )
    preprocessed = PreprocessedText(text=text, sentence_spans=[], page_spans=[])

    block_recs = RecommendationExtractorV2(profile=PFD_REPORT_PROFILE, pfd_atomize_concerns=False).extract(
        preprocessed, source_document="test.pdf"
    )
    assert len([r for r in block_recs if r.rec_type == "pfd_concern"]) == 2
    assert {r.rec_id for r in block_recs if r.rec_type == "pfd_concern"} == {"concern_1", "concern_2"}

    atom_recs = RecommendationExtractorV2(profile=PFD_REPORT_PROFILE, pfd_atomize_concerns=True).extract(
        preprocessed, source_document="test.pdf"
    )
    pfd_concerns = [r for r in atom_recs if r.rec_type == "pfd_concern"]
    # Exact counts can vary because v1-style sentence splitting may merge across
    # "(2)" boundaries; the key contract is that atomisation yields per-sentence
    # concerns and also includes the v1 baseline extractions.
    assert len(pfd_concerns) >= 4
    assert any((r.detection_method or "").startswith("pfd_v1_baseline:") for r in pfd_concerns)
    assert any(
        (r.detection_method or "").startswith(("pfd_matters_of_concern_atomized:", "pfd_matters_of_concern_atomized_v1:"))
        for r in pfd_concerns
    )
    rec_numbers = {r.rec_number for r in pfd_concerns}
    assert {1, 2}.issubset(rec_numbers)


def test_pfd_concerns_atomization_falls_back_without_matters_window() -> None:
    text = (
        "CIRCUMSTANCES OF THE DEATH\n"
        "On 1 January 2020 a friend called.\n"
        "CORONER'S CONCERNS\n"
        "I am concerned that contact numbers are not answered.\n"
        "ACTION SHOULD BE TAKEN\n"
        "In my opinion action should be taken to prevent future deaths.\n"
    )
    preprocessed = PreprocessedText(text=text, sentence_spans=[], page_spans=[])
    atom_recs = RecommendationExtractorV2(profile=PFD_REPORT_PROFILE, pfd_atomize_concerns=True).extract(
        preprocessed, source_document="test.pdf"
    )
    pfd_concerns = [r for r in atom_recs if r.rec_type == "pfd_concern"]
    assert len(pfd_concerns) == 1
    assert (pfd_concerns[0].detection_method or "").startswith("pfd_concerns_fallback_atomized:")


def test_pfd_concerns_atomization_falls_back_to_full_doc_when_window_empty() -> None:
    text = (
        "CORONER'S CONCERNS\n"
        "The MATTERS OF CONCERN are as follows:\n"
        "This section contains background but no triggered sentences.\n"
        "ACTION SHOULD BE TAKEN\n"
        "Annex\n"
        "The Trust should review its procedures.\n"
    )
    preprocessed = PreprocessedText(text=text, sentence_spans=[], page_spans=[])
    atom_recs = RecommendationExtractorV2(profile=PFD_REPORT_PROFILE, pfd_atomize_concerns=True).extract(
        preprocessed, source_document="test.pdf"
    )
    pfd_concerns = [r for r in atom_recs if r.rec_type == "pfd_concern"]
    assert any((r.detection_method or "").startswith("pfd_concerns_fallback_atomized_v1:") for r in pfd_concerns)


def test_pfd_atomized_always_includes_v1_baseline_outside_window() -> None:
    text = (
        "CORONER'S CONCERNS\n"
        "The MATTERS OF CONCERN are as follows:\n"
        "(1) There was no care plan.\n"
        "ACTION SHOULD BE TAKEN\n"
        "Appendix\n"
        "The Trust should review its procedure.\n"
    )
    preprocessed = PreprocessedText(text=text, sentence_spans=[], page_spans=[])
    atom_recs = RecommendationExtractorV2(
        profile=PFD_REPORT_PROFILE, pfd_atomize_concerns=True, action_verb_min_confidence=0.75
    ).extract(preprocessed, source_document="test.pdf")
    pfd_concerns = [r for r in atom_recs if r.rec_type == "pfd_concern"]
    assert any((r.detection_method or "").startswith("pfd_v1_baseline:") for r in pfd_concerns)
