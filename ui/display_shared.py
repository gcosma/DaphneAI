"""Shared display helpers for highlighting and text formatting."""

from __future__ import annotations

import re
from typing import List, Tuple

import streamlit as st

from daphne_core.search_utils import STOP_WORDS, get_meaningful_words

# --------------------------------------------------------------------------- #
# Text cleaning and formatting
# --------------------------------------------------------------------------- #

def clean_html_artifacts(text: str) -> str:
    """Remove HTML artifacts and clean up text."""
    if not text:
        return "No content available"
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    return " ".join(text.split()).strip()


def format_text_as_clean_paragraphs(text: str) -> str:
    """Format text as readable paragraphs."""
    if not text:
        return "*No content available*"

    clean_text = clean_html_artifacts(text)
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return clean_text

    paragraphs, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if len(current) >= 3 and any(
            indicator in sentence.lower()
            for indicator in ["however", "furthermore", "additionally", "therefore", "moreover", "meanwhile"]
        ):
            paragraphs.append(" ".join(current))
            current = []
        if len(current) >= 4:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    formatted = "\n\n".join(f"> {paragraph}" for paragraph in paragraphs if paragraph.strip())
    return formatted or f"> {clean_text}"


def format_as_beautiful_paragraphs(text: str) -> str:
    """Format text as spaced paragraphs (legacy helper)."""
    if not text:
        return "No content available"

    text = clean_html_artifacts(text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text

    paragraphs, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if len(current) >= 3 and any(
            indicator in sentence.lower()
            for indicator in ["however", "furthermore", "additionally", "in conclusion", "therefore", "moreover"]
        ):
            paragraphs.append(" ".join(current))
            current = []
        elif len(current) >= 4:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    formatted_paragraphs = [f"> {paragraph.strip()}" for paragraph in paragraphs if paragraph.strip()]
    return "\n\n".join(formatted_paragraphs) if formatted_paragraphs else text


# --------------------------------------------------------------------------- #
# Highlighting helpers
# --------------------------------------------------------------------------- #

def highlight_with_html_method(text: str, words_to_highlight: List[str]) -> str:
    if not words_to_highlight:
        return text
    highlighted = text
    for word in sorted(words_to_highlight, key=len, reverse=True):
        if len(word) <= 2:
            continue
        try:
            pattern = re.compile(r"\b" + re.escape(word) + r"\w*", re.IGNORECASE)

            def replace(match):
                token = match.group()
                return (
                    '<mark style="background-color: #FFEB3B; padding: 2px; border-radius: 2px; '
                    f'font-weight: bold; color: #000;">{token}</mark>'
                )

            highlighted = pattern.sub(replace, highlighted)
        except re.error:
            continue
    return highlighted


def highlight_with_bold_method(text: str, words_to_highlight: List[str]) -> str:
    if not words_to_highlight:
        return text
    highlighted = text
    for word in sorted(words_to_highlight, key=len, reverse=True):
        if len(word) <= 2:
            continue
        try:
            pattern = re.compile(r"\b" + re.escape(word) + r"\w*", re.IGNORECASE)
            highlighted = pattern.sub(lambda m: f"**{m.group()}**", highlighted)
        except re.error:
            continue
    return highlighted


def highlight_with_capitalization_method(text: str, words_to_highlight: List[str]) -> str:
    if not words_to_highlight:
        return text
    highlighted = text
    for word in sorted(words_to_highlight, key=len, reverse=True):
        if len(word) <= 2:
            continue
        try:
            pattern = re.compile(r"\b" + re.escape(word) + r"\w*", re.IGNORECASE)
            highlighted = pattern.sub(lambda m: f"**{m.group().upper()}**", highlighted)
        except re.error:
            continue
    return highlighted


def highlight_with_brackets_method(text: str, words_to_highlight: List[str]) -> str:
    if not words_to_highlight:
        return text
    highlighted = text
    for word in sorted(words_to_highlight, key=len, reverse=True):
        if len(word) <= 2:
            continue
        try:
            pattern = re.compile(r"\b" + re.escape(word) + r"\w*", re.IGNORECASE)
            highlighted = pattern.sub(lambda m: f"[{m.group()}]", highlighted)
        except re.error:
            continue
    return highlighted


def smart_highlight_text(text: str, query: str, method_preference: str = "auto") -> Tuple[str, str]:
    if not text or not query:
        return text, "none"
    meaningful_words = get_meaningful_words(query)
    if not meaningful_words:
        return text, "none"

    if method_preference in ("html", "auto"):
        highlighted = highlight_with_html_method(text, meaningful_words)
        if highlighted != text:
            return highlighted, "html"
    if method_preference in ("bold", "auto"):
        highlighted = highlight_with_bold_method(text, meaningful_words)
        if highlighted != text:
            return highlighted, "bold"
    if method_preference in ("caps", "auto"):
        highlighted = highlight_with_capitalization_method(text, meaningful_words)
        if highlighted != text:
            return highlighted, "caps"
    highlighted = highlight_with_brackets_method(text, meaningful_words)
    return highlighted, "brackets"


def highlight_government_terms(text: str) -> Tuple[str, str]:
    gov_terms = [
        "recommend",
        "recommendation",
        "recommendations",
        "suggest",
        "advise",
        "propose",
        "accept",
        "reject",
        "agree",
        "disagree",
        "implement",
        "implementation",
        "consider",
        "approved",
        "declined",
        "response",
        "reply",
        "answer",
        "policy",
        "framework",
        "guideline",
        "protocol",
        "strategy",
        "committee",
        "department",
        "ministry",
        "government",
        "authority",
        "urgent",
        "immediate",
        "critical",
        "priority",
        "essential",
        "budget",
        "funding",
        "financial",
        "cost",
        "expenditure",
        "review",
        "analysis",
        "assessment",
        "evaluation",
        "inquiry",
    ]
    highlighted = highlight_with_html_method(text, gov_terms)
    if highlighted != text:
        return highlighted, "html"
    highlighted = highlight_with_bold_method(text, gov_terms)
    if highlighted != text:
        return highlighted, "bold"
    highlighted = highlight_with_brackets_method(text, gov_terms)
    return highlighted, "brackets"


def highlight_with_color_codes(text: str, words_to_highlight: List[str], color_scheme: str = "default") -> str:
    if not words_to_highlight:
        return text
    color_schemes = {
        "default": {"bg": "#FFEB3B", "text": "#000"},
        "blue": {"bg": "#2196F3", "text": "#FFF"},
        "green": {"bg": "#4CAF50", "text": "#FFF"},
        "red": {"bg": "#F44336", "text": "#FFF"},
        "purple": {"bg": "#9C27B0", "text": "#FFF"},
    }
    colors = color_schemes.get(color_scheme, color_schemes["default"])

    highlighted = text
    for word in sorted(words_to_highlight, key=len, reverse=True):
        if len(word) <= 2:
            continue
        try:
            pattern = re.compile(r"\b" + re.escape(word) + r"\w*", re.IGNORECASE)

            def replace(match):
                token = match.group()
                return (
                    f'<mark style="background-color: {colors["bg"]}; color: {colors["text"]}; '
                    f'padding: 2px; border-radius: 2px; font-weight: bold;">{token}</mark>'
                )

            highlighted = pattern.sub(replace, highlighted)
        except re.error:
            continue
    return highlighted


def create_highlighting_legend(words_highlighted: List[str], method_used: str) -> str:
    if not words_highlighted:
        return ""
    legend = "💡 **Highlighting Legend:** "
    if method_used == "html":
        legend += "Highlighted words are shown with colored background"
    elif method_used == "bold":
        legend += "**Highlighted words are shown in bold**"
    elif method_used == "caps":
        legend += "**HIGHLIGHTED WORDS ARE SHOWN IN CAPITALS**"
    elif method_used == "brackets":
        legend += "Highlighted words are shown in [brackets]"
    else:
        legend += "Highlighted words are emphasized"
    legend += f" | Words found: {', '.join(words_highlighted[:5])}{'...' if len(words_highlighted) > 5 else ''}"
    return legend


def highlight_with_context_awareness(text: str, query: str, context_type: str = "general") -> Tuple[str, str, List[str]]:
    meaningful_words = get_meaningful_words(query)
    if not meaningful_words:
        return text, "none", []

    if context_type == "recommendation":
        highlighted = highlight_with_color_codes(text, meaningful_words, "green")
        if highlighted != text:
            return highlighted, "html_green", meaningful_words
    elif context_type == "response":
        highlighted = highlight_with_color_codes(text, meaningful_words, "blue")
        if highlighted != text:
            return highlighted, "html_blue", meaningful_words
    elif context_type == "policy":
        highlighted = highlight_with_color_codes(text, meaningful_words, "purple")
        if highlighted != text:
            return highlighted, "html_purple", meaningful_words

    highlighted, method_used = smart_highlight_text(text, query, "auto")
    return highlighted, method_used, meaningful_words


def render_highlighted_content_with_legend(content: str, query: str, context_type: str = "general", show_legend: bool = True) -> None:
    highlighted_content, method_used, words_found = highlight_with_context_awareness(content, query, context_type)
    if method_used.startswith("html"):
        st.markdown(highlighted_content, unsafe_allow_html=True)
    elif method_used in ["bold", "caps", "brackets"]:
        st.markdown(highlighted_content)
    else:
        st.markdown(content)

    if show_legend and words_found and method_used != "none":
        st.caption(create_highlighting_legend(words_found, method_used))


# --------------------------------------------------------------------------- #
# Debug/testing helpers (kept for parity)
# --------------------------------------------------------------------------- #

def test_highlighting_methods() -> None:
    st.markdown("### 🧪 Highlighting Method Test")
    test_text = "The committee recommends implementing new policies for government response."
    test_query = "recommend policy response"
    st.markdown("**Original text:**")
    st.code(test_text)
    st.markdown("**Query:**")
    st.code(test_query)
    meaningful_words = get_meaningful_words(test_query)
    st.markdown(f"**Meaningful words:** {meaningful_words}")

    st.markdown("### 🎨 Testing Different Highlighting Methods")
    st.markdown("**Method 1: HTML Highlighting**")
    html_result = highlight_with_html_method(test_text, meaningful_words)
    st.markdown(html_result, unsafe_allow_html=True)
    st.code(f"Result: {html_result}")

    st.markdown("**Method 2: Bold Highlighting**")
    bold_result = highlight_with_bold_method(test_text, meaningful_words)
    st.markdown(bold_result)
    st.code(f"Result: {bold_result}")

    st.markdown("**Method 3: Capitalization Highlighting**")
    caps_result = highlight_with_capitalization_method(test_text, meaningful_words)
    st.markdown(caps_result)
    st.code(f"Result: {caps_result}")

    st.markdown("**Method 4: Bracket Highlighting**")
    bracket_result = highlight_with_brackets_method(test_text, meaningful_words)
    st.markdown(bracket_result)
    st.code(f"Result: {bracket_result}")

    st.markdown("**Smart Method (Auto-Select):**")
    smart_result, method_used = smart_highlight_text(test_text, test_query, "auto")
    st.markdown(f"**Method used:** {method_used}")
    if method_used == "html":
        st.markdown(smart_result, unsafe_allow_html=True)
    else:
        st.markdown(smart_result)
    st.code(f"Result: {smart_result}")


def debug_highlighting_issue(problematic_text: str, query: str) -> None:
    st.markdown("### 🔧 Debugging Highlighting Issue")
    st.markdown("**Problematic Text:**")
    st.code(problematic_text)
    st.markdown("**Query:**")
    st.code(query)

    meaningful_words = get_meaningful_words(query)
    st.markdown(f"**Meaningful words extracted:** {meaningful_words}")
    if not meaningful_words:
        st.error("❌ No meaningful words found in query!")
        return

    st.markdown("### Step-by-Step Analysis")
    for i, word in enumerate(meaningful_words, 1):
        st.markdown(f"**Step {i}: Testing word '{word}'**")
        if word.lower() in problematic_text.lower():
            st.success(f"✅ Word '{word}' found in text")
            html_test = highlight_with_html_method(problematic_text, [word])
            bold_test = highlight_with_bold_method(problematic_text, [word])
            bracket_test = highlight_with_brackets_method(problematic_text, [word])
            st.code(f"HTML result: {html_test}")
            st.code(f"Bold result: {bold_test}")
            st.code(f"Bracket result: {bracket_test}")
        else:
            st.warning(f"⚠️ Word '{word}' not found in text")

    st.markdown("### Final Smart Highlighting Test")
    final_result, method_used = smart_highlight_text(problematic_text, query, "auto")
    st.markdown(f"**Method selected:** {method_used}")
    if method_used == "html":
        st.markdown(final_result, unsafe_allow_html=True)
    else:
        st.markdown(final_result)
    st.code(f"Raw result: {final_result}")
    if final_result == problematic_text:
        st.error("❌ No highlighting was applied!")


__all__ = [
    "STOP_WORDS",
    "get_meaningful_words",
    "clean_html_artifacts",
    "format_text_as_clean_paragraphs",
    "format_as_beautiful_paragraphs",
    "highlight_with_html_method",
    "highlight_with_bold_method",
    "highlight_with_capitalization_method",
    "highlight_with_brackets_method",
    "smart_highlight_text",
    "highlight_government_terms",
    "highlight_with_color_codes",
    "create_highlighting_legend",
    "highlight_with_context_awareness",
    "render_highlighted_content_with_legend",
    "test_highlighting_methods",
    "debug_highlighting_issue",
]
