"""Alignment-specific display helpers (extracted from the legacy display_utils)."""

from typing import Dict, List, Tuple

import re
import streamlit as st

from daphne_core.search_utils import get_meaningful_words


# -----------------------------------------------------------------------------
# Shared text helpers (minimal copy to avoid circular imports while refactoring)
# -----------------------------------------------------------------------------

def clean_html_artifacts(text: str) -> str:
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
    if not text:
        return "*No content available*"
    clean_text = clean_html_artifacts(text)
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return clean_text

    paragraphs: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        current.append(sentence)
        if len(" ".join(current)) > 400:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


# -----------------------------------------------------------------------------
# Highlighting utilities
# -----------------------------------------------------------------------------

def highlight_with_html_method(text: str, words: List[str]) -> str:
    if not text or not words:
        return text
    highlighted = text
    for word in sorted(words, key=len, reverse=True):
        highlighted = re.sub(
            rf"\b{re.escape(word)}\b",
            f'<span style="background: #fff59d; padding: 2px 4px; border-radius: 4px;">{word}</span>',
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted


def highlight_with_bold_method(text: str, words: List[str]) -> str:
    if not text or not words:
        return text
    highlighted = text
    for word in sorted(words, key=len, reverse=True):
        highlighted = re.sub(rf"\b{re.escape(word)}\b", f"**{word}**", highlighted, flags=re.IGNORECASE)
    return highlighted


def highlight_with_capitalization_method(text: str, words: List[str]) -> str:
    if not text or not words:
        return text
    highlighted = text
    for word in sorted(words, key=len, reverse=True):
        highlighted = re.sub(rf"\b{re.escape(word)}\b", lambda m: m.group(0).upper(), highlighted, flags=re.IGNORECASE)
    return highlighted


def highlight_with_brackets_method(text: str, words: List[str]) -> str:
    if not text or not words:
        return text
    highlighted = text
    for word in sorted(words, key=len, reverse=True):
        highlighted = re.sub(rf"\b{re.escape(word)}\b", f"[{word}]", highlighted, flags=re.IGNORECASE)
    return highlighted


def smart_highlight_text(text: str, query: str, method: str = "auto") -> Tuple[str, str]:
    if not text or not query:
        return text, "none"
    words = get_meaningful_words(query)
    if not words:
        return text, "none"

    if method == "html":
        return highlight_with_html_method(text, words), "html"
    if method == "bold":
        return highlight_with_bold_method(text, words), "bold"
    if method == "caps":
        return highlight_with_capitalization_method(text, words), "caps"
    if method == "brackets":
        return highlight_with_brackets_method(text, words), "brackets"

    highlighted = highlight_with_html_method(text, words)
    if highlighted != text:
        return highlighted, "html"
    highlighted = highlight_with_bold_method(text, words)
    if highlighted != text:
        return highlighted, "bold"
    highlighted = highlight_with_capitalization_method(text, words)
    if highlighted != text:
        return highlighted, "caps"
    highlighted = highlight_with_brackets_method(text, words)
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
    return highlighted, "brackets" if highlighted != text else "none"


# -----------------------------------------------------------------------------
# Alignment display
# -----------------------------------------------------------------------------

def display_alignment_results_beautiful(alignments: List[Dict], show_ai_summaries: bool):
    if not alignments:
        st.warning("No recommendations found in the uploaded documents")
        return

    st.markdown("### 📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Recommendations", len(alignments))
    aligned_count = sum(1 for a in alignments if a.get("responses", []))
    col2.metric("Recommendations with Responses", aligned_count)
    avg_confidence = sum(a.get("alignment_confidence", 0) for a in alignments) / len(alignments)
    col3.metric("Avg Alignment Confidence", f"{avg_confidence:.2f}")
    high_confidence = sum(1 for a in alignments if a.get("alignment_confidence", 0) > 0.7)
    col4.metric("High Confidence Alignments", high_confidence)

    st.markdown("### 🔗 Recommendation-Response Alignments")
    for i, alignment in enumerate(alignments, 1):
        display_single_alignment_beautiful(alignment, i, show_ai_summaries)


def display_single_alignment_beautiful(alignment: Dict, index: int, show_ai_summaries: bool):
    rec = alignment.get("recommendation", {})
    responses = alignment.get("responses", [])
    confidence = alignment.get("alignment_confidence", 0)

    confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
    rec_type = rec.get("recommendation_type", "General")

    with st.expander(f"{confidence_color} Recommendation {index} - {rec_type} (Confidence: {confidence:.2f})", expanded=index <= 3):
        st.markdown("### 📋 Complete Extract")
        st.markdown("#### 🎯 Recommendation")

        doc_name = rec.get("document", {}).get("filename", "Unknown Document")
        page_num = rec.get("page_number", 1)
        st.info(f"📄 **Document:** {doc_name} | **Page:** {page_num}")

        full_sentence = rec.get("sentence", "No sentence available")
        clean_sentence = clean_html_artifacts(full_sentence)
        st.markdown("**📝 Full Recommendation:**")

        highlighted_sentence, highlight_method = highlight_government_terms(clean_sentence)
        if highlight_method == "html":
            st.markdown(f"> {highlighted_sentence}", unsafe_allow_html=True)
        elif highlight_method in {"bold", "brackets"}:
            st.markdown(f"> {highlighted_sentence}")
            if highlight_method == "brackets":
                st.caption("💡 Government terms are shown in [brackets]")
        else:
            st.markdown(f"> {clean_sentence}")

        st.markdown("")

        full_context = rec.get("context", "No context available")
        if full_context and full_context != full_sentence:
            st.markdown("#### 📖 Complete Context")
            clean_context = clean_html_artifacts(full_context)
            highlighted_context, context_method = highlight_government_terms(clean_context)
            formatted_context = format_text_as_clean_paragraphs(highlighted_context)
            if context_method == "html":
                st.markdown(formatted_context, unsafe_allow_html=True)
            else:
                st.markdown(formatted_context)
                if context_method == "brackets":
                    st.caption("💡 Government terms are shown in [brackets]")

        if responses:
            st.markdown("#### ↩️ Related Responses")
            for j, resp_match in enumerate(responses, 1):
                resp = resp_match.get("response", {})
                similarity = resp_match.get("combined_score", 0)
                if similarity > 0.7:
                    confidence_text = "High Confidence"
                elif similarity > 0.5:
                    confidence_text = "Medium Confidence"
                else:
                    confidence_text = "Lower Confidence"

                resp_doc_name = resp.get("document", {}).get("filename", "Unknown Document")
                resp_page_num = resp.get("page_number", 1)

                full_resp_sentence = resp.get("sentence", "No sentence available")
                full_resp_context = resp.get("context", "No context available")

                clean_resp_sentence = clean_html_artifacts(full_resp_sentence)
                clean_resp_context = clean_html_artifacts(full_resp_context)

                st.markdown(f"**📄 Response {j} - {confidence_text} ({similarity:.2f})**")
                st.info(f"📄 **Document:** {resp_doc_name} | **Page:** {resp_page_num}")
                st.markdown("**📝 Full Response:**")

                rec_sentence = str(rec.get("sentence", ""))
                highlighted_response, resp_method = smart_highlight_text(clean_resp_sentence, rec_sentence, "auto")
                if resp_method == "html":
                    st.markdown(f"> {highlighted_response}", unsafe_allow_html=True)
                elif resp_method in {"bold", "caps"}:
                    st.markdown(f"> {highlighted_response}")
                elif resp_method == "brackets":
                    st.markdown(f"> {highlighted_response}")
                    st.caption("💡 Related terms are shown in [brackets]")
                else:
                    st.markdown(f"> {clean_resp_sentence}")

                if clean_resp_context and clean_resp_context != clean_resp_sentence:
                    st.markdown("**📖 Complete Context:**")
                    formatted_context = format_text_as_clean_paragraphs(clean_resp_context)
                    st.markdown(formatted_context)

                if j < len(responses):
                    st.markdown("---")
        else:
            st.markdown(
                """
            <div style="
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                border-left: 4px solid #dc3545;
                padding: 20px;
                border-radius: 8px;
                margin: 10px 0;
                text-align: center;
            ">
                <strong>❌ No matching responses found for this recommendation</strong><br>
                <small>This recommendation may be awaiting a response or responses may be in separate documents</small>
            </div>
            """,
                unsafe_allow_html=True,
            )


def show_alignment_feature_info_beautiful():
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    ">
        <h2 style="margin: 0 0 20px 0; text-align: center;">
            🎯 Recommendation-Response Alignment
        </h2>
        <p style="margin: 0; font-size: 18px; text-align: center; opacity: 0.9;">
            Automatically discover and align government recommendations with their corresponding responses
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        <div style="
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 10px 0;
            height: 300px;
        ">
            <h3 style="margin: 0 0 15px 0;">🔍 What It Finds</h3>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>All recommendations in your documents</li>
                <li>Corresponding responses to those recommendations</li>
                <li>Alignment confidence scores</li>
                <li>Content relationships and patterns</li>
                <li>Cross-document connections</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div style="
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 10px 0;
            height: 300px;
        ">
            <h3 style="margin: 0 0 15px 0;">📊 What You Get</h3>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>Side-by-side recommendation + response view</li>
                <li>Pattern analysis and statistics</li>
                <li>Confidence scores for each alignment</li>
                <li>Export options for further analysis</li>
                <li>Document relationship insights</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


__all__ = [
    "display_alignment_results_beautiful",
    "display_single_alignment_beautiful",
    "show_alignment_feature_info_beautiful",
    "highlight_government_terms",
    "smart_highlight_text",
    "clean_html_artifacts",
    "format_text_as_clean_paragraphs",
]
