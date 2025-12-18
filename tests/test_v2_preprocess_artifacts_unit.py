from daphne_core.v2.preprocess import _clean_page_break_artifacts, _repair_broken_urls


def test_clean_page_break_artifacts_removes_digit_runs_and_line_end_markers() -> None:
    raw = (
        "There was no meaningful contact with the family to report the concerning lack 2\n"
        "6 7 8 9 of contact with Mr Fraser, until the 29 July 2024.\n"
    )
    cleaned = _clean_page_break_artifacts(raw)
    assert "6 7 8 9" not in cleaned
    assert "lack 2\nof contact" not in cleaned
    assert "lack\nof contact" in cleaned


def test_clean_page_break_artifacts_preserves_legitimate_section_reference() -> None:
    raw = "Section 2\nof the Act applies.\n"
    cleaned = _clean_page_break_artifacts(raw)
    assert cleaned == raw


def test_repair_broken_urls_removes_whitespace_after_hyphen_when_next_delimiter_is_hyphen() -> None:
    raw = (
        "See https://www.england.nhs.uk/publication/national- guidance-on-quality-risk-response-and-escalation-in-integrated-care-systems/ for details."
    )
    cleaned = _repair_broken_urls(raw)
    assert "national- guidance-on" not in cleaned
    assert "national-guidance-on-quality" in cleaned


def test_repair_broken_urls_removes_whitespace_after_hyphen_when_next_delimiter_is_slash() -> None:
    raw = "See https://www.england.nhs.uk/ourwork/part- rel/nqb/experience-of-care-framework/ for details."
    cleaned = _repair_broken_urls(raw)
    assert "/part- rel/" not in cleaned
    assert "/part-rel/nqb/" in cleaned


def test_repair_broken_urls_does_not_change_normal_text_hyphen_spacing() -> None:
    raw = "We discussed risk-based guidance on quality and safety."
    cleaned = _repair_broken_urls(raw)
    assert cleaned == raw
