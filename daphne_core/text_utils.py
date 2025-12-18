from __future__ import annotations

import re


def compact_urls_markdown(text: str, *, label: str = "link") -> str:
    """
    Replace bare URLs with a compact Markdown hyperlink label.

    URL detection is intentionally simple and tuned to our PDFs:
    - URL starts with "http://" or "https://"
    - URL ends at the first whitespace or ')' (exclusive)

    Examples
    --------
    "(https://example.com/path)" -> "([link](https://example.com/path))"
    "See https://a/b for more" -> "See [link](https://a/b) for more"
    """
    if not text:
        return ""

    pattern = re.compile(r"https?://[^\s)]+")

    parts: list[str] = []
    last = 0
    n = 0

    for m in pattern.finditer(text):
        url = m.group(0)

        # Avoid rewriting URLs that are already part of markdown links: "](" + URL + ")"
        if m.start() >= 2 and text[m.start() - 2 : m.start()] == "](":
            continue

        parts.append(text[last : m.start()])
        n += 1
        link_label = label if n == 1 else f"{label}{n}"
        parts.append(f"[{link_label}]({url})")
        last = m.end()

    parts.append(text[last:])
    return "".join(parts)


def collapse_whitespace(text: str) -> str:
    """
    Collapse all whitespace (including newlines) into single spaces.

    This is intended for display only; extraction/matching should continue to
    operate over the original text/spans.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def format_display_markdown(
    text: str,
    *,
    single_paragraph: bool = False,
    url_label: str = "link",
) -> str:
    """
    Display-only formatting for Streamlit markdown blocks.

    - Compacts raw URLs into `[link](...)`
    - Optionally collapses whitespace into one paragraph
    """
    formatted = compact_urls_markdown(text or "", label=url_label)
    if single_paragraph:
        formatted = collapse_whitespace(formatted)
    return formatted
