from daphne_core.text_utils import compact_urls_markdown, format_display_markdown


def test_compact_urls_markdown_ends_at_whitespace() -> None:
    raw = "See https://example.com/path for details."
    out = compact_urls_markdown(raw)
    assert out == "See [link](https://example.com/path) for details."


def test_compact_urls_markdown_ends_before_closing_paren() -> None:
    raw = "See (https://example.com/path) for details."
    out = compact_urls_markdown(raw)
    assert out == "See ([link](https://example.com/path)) for details."


def test_compact_urls_markdown_numbers_links() -> None:
    raw = "A https://a/b and https://c/d"
    out = compact_urls_markdown(raw)
    assert out == "A [link](https://a/b) and [link2](https://c/d)"


def test_compact_urls_markdown_does_not_double_wrap_existing_markdown_link() -> None:
    raw = "See [this](https://example.com/path) now."
    out = compact_urls_markdown(raw)
    assert out == raw


def test_format_display_markdown_single_paragraph_collapses_newlines() -> None:
    raw = "Line 1\n\nLine 2 https://example.com/x\nLine 3"
    out = format_display_markdown(raw, single_paragraph=True)
    assert "\n" not in out
    assert "[link](https://example.com/x)" in out
