from daphne_core.v2.preprocess import _clean_page_break_artifacts


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

