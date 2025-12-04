"""Search result display components."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from .display_shared import (
    clean_html_artifacts,
    format_text_as_clean_paragraphs,
    get_meaningful_words,
    smart_highlight_text,
)


def display_search_results_beautiful(results: List[Dict], query: str, search_time: float, show_context: bool, highlight_matches: bool):
    """Display search results with highlighting and summary."""
    if not results:
        st.warning(f"No results found for '{query}'")
        return

    doc_groups: Dict[str, List[Dict]] = {}
    for result in results:
        doc_name = result["document"]["filename"]
        doc_groups.setdefault(doc_name, []).append(result)

    meaningful_words = get_meaningful_words(query)
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0 0 10px 0;">🎯 Search Results</h3>
        <p style="margin: 0; font-size: 18px;">
            Found <strong>{len(results)}</strong> results in <strong>{len(doc_groups)}</strong> documents for <strong>"{query}"</strong>
        </p>
        <small style="opacity: 0.9;">Search completed in {search_time:.3f} seconds</small>
        {f'<br><small style="opacity: 0.8;">Focused on meaningful words: {", ".join(meaningful_words[:5])}{"..." if len(meaningful_words) > 5 else ""}</small>' if meaningful_words else ''}
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy All Results"):
            copy_results_beautiful(results, query)
    with col2:
        if st.button("📊 Export to CSV"):
            export_results_csv_beautiful(results, query)

    for doc_name, doc_results in doc_groups.items():
        best_score = max(r.get("score", 0) for r in doc_results)
        doc = doc_results[0].get("document", {})

        with st.expander(f"📄 {doc_name} ({len(doc_results)} matches, best score: {best_score:.1f})", expanded=True):
            word_count = len(doc.get("text", "").split()) if doc.get("text") else 0
            char_count = len(doc.get("text", "")) if doc.get("text") else 0
            est_pages = max(1, char_count // 2000)

            st.markdown(
                f"""
            <div style="
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #007bff;
            ">
                <strong>📊 Document Statistics:</strong><br>
                <strong>Words:</strong> {word_count:,} | <strong>Characters:</strong> {char_count:,} | <strong>Est. Pages:</strong> {est_pages}
            </div>
            """,
                unsafe_allow_html=True,
            )

            for i, result in enumerate(doc_results, 1):
                display_single_search_result_beautiful(result, i, query, show_context, highlight_matches)


def display_single_search_result_beautiful(result: Dict, index: int, query: str, show_context: bool, highlight_matches: bool):
    """Display a single search result with highlighting."""
    method = result.get("match_type", "unknown")
    score = result.get("score", 0)
    page = result.get("page_number", 1)
    position = result.get("position", 0)

    method_info = {
        "exact": {"icon": "🎯", "color": "#28a745", "name": "Exact Match"},
        "smart": {"icon": "🧠", "color": "#007bff", "name": "Smart Search"},
        "fuzzy": {"icon": "🌀", "color": "#ffc107", "name": "Fuzzy Match"},
        "semantic": {"icon": "🤖", "color": "#6f42c1", "name": "Semantic Match"},
        "hybrid": {"icon": "🔄", "color": "#17a2b8", "name": "Hybrid Search"},
    }
    info = method_info.get(method, {"icon": "🔍", "color": "#6c757d", "name": "Search"})

    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, {info['color']} 0%, {info['color']}dd 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    ">
        <h4 style="margin: 0 0 5px 0;">
            {info['icon']} Match {index} - {info['name']}
        </h4>
        <div style="font-size: 14px; opacity: 0.9;">
            <strong>Score:</strong> {score:.1f} | <strong>Page:</strong> {page} | <strong>Position:</strong> {position:,}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if method == "smart" and "meaningful_words_found" in result:
        meaningful_found = result["meaningful_words_found"]
        total_meaningful = result.get("total_meaningful_words", 0)
        if meaningful_found:
            st.markdown(
                f"""
            <div style="
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 12px;
                border-radius: 6px;
                margin: 10px 0;
                font-size: 14px;
            ">
                <strong>🎯 Meaningful Words Found:</strong> {', '.join(meaningful_found)} 
                ({len(meaningful_found)}/{total_meaningful} words matched)
            </div>
            """,
                unsafe_allow_html=True,
            )

    if show_context:
        full_context = result.get("context", "")
        if full_context:
            st.markdown("**📖 Complete Context:**")
            if highlight_matches:
                highlighted_context, method_used = smart_highlight_text(full_context, query, "auto")
                if method_used == "html":
                    st.markdown(highlighted_context, unsafe_allow_html=True)
                elif method_used in ("bold", "caps"):
                    st.markdown(format_text_as_clean_paragraphs(highlighted_context))
                elif method_used == "brackets":
                    st.markdown(format_text_as_clean_paragraphs(highlighted_context))
                    st.caption("💡 Highlighted words are shown in [brackets]")
                else:
                    clean_content = clean_html_artifacts(full_context)
                    st.markdown(format_text_as_clean_paragraphs(clean_content))
            else:
                clean_content = clean_html_artifacts(full_context)
                st.markdown(format_text_as_clean_paragraphs(clean_content))
            percentage = result.get("percentage_through", 0)
            st.caption(f"💡 This content appears around page {page}, {percentage:.1f}% through the document")


def display_manual_search_results_beautiful(matches: List[Dict], target_sentence: str, search_time: float, show_scores: bool, search_mode: str):
    """Display manual search results with highlighting."""
    if not matches:
        st.markdown(
            """
        <div style="
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left: 4px solid #dc3545;
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
        ">
            <h4 style="margin: 0 0 10px 0; color: #721c24;">No matches found</h4>
            <p style="margin: 0; color: #721c24;">
                No similar content found for your sentence.<br>
                <small>💡 Try lowering the similarity threshold or changing the search type.</small>
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    meaningful_words = get_meaningful_words(target_sentence)
    mode_text = {"recommendations": "recommendations", "responses": "responses", "similar content": "sentences"}.get(
        search_mode, "items"
    )

    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0 0 10px 0;">🎯 Similar Content Found</h3>
        <p style="margin: 0; font-size: 18px;">
            Found <strong>{len(matches)}</strong> similar {mode_text} in <strong>{search_time:.3f}</strong> seconds
        </p>
        {f'<br><small style="opacity: 0.8;">Based on meaningful words: {", ".join(meaningful_words[:5])}{"..." if len(meaningful_words) > 5 else ""}</small>' if meaningful_words else ''}
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📝 Your Original Sentence")
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        font-size: 16px;
        line-height: 1.8;
        font-style: italic;
    ">
        "{target_sentence}"
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔍 Similar Content Matches")
    for i, match in enumerate(matches, 1):
        similarity = match.get("similarity_score", 0)
        if similarity > 0.8:
            confidence_text, confidence_icon = "Very High Similarity", "🟢"
        elif similarity > 0.6:
            confidence_text, confidence_icon = "High Similarity", "🟡"
        elif similarity > 0.4:
            confidence_text, confidence_icon = "Medium Similarity", "🟠"
        else:
            confidence_text, confidence_icon = "Lower Similarity", "🔴"

        score_text = f" (Score: {similarity:.3f})" if show_scores else ""
        content_type = match.get("content_type", "General")

        with st.expander(f"{confidence_icon} Match {i} - {content_type} - {confidence_text}{score_text}", expanded=i <= 3):
            if "matched_meaningful_words" in match:
                matched_words = match["matched_meaningful_words"]
                total_words = match.get("total_meaningful_words", 0)
                if matched_words:
                    st.markdown(
                        f"""
                    <div style="
                        background: #e8f5e8;
                        border-left: 4px solid #4caf50;
                        padding: 12px;
                        border-radius: 6px;
                        margin: 10px 0;
                        font-size: 14px;
                    ">
                        <strong>🎯 Meaningful Words Matched:</strong> {', '.join(matched_words)} 
                        ({len(matched_words)}/{total_words} words)
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            doc_name = match.get("document", {}).get("filename", "Unknown")
            page_num = match.get("page_number", 1)
            st.info(f"📄 **File:** {doc_name} | **Page:** {page_num} | **Type:** {content_type}")

            full_sentence = match.get("sentence", "No sentence available")
            st.markdown("#### 📄 Complete Found Content")
            if show_scores:
                highlighted_sentence, sentence_method = smart_highlight_text(full_sentence, target_sentence, "auto")
                if sentence_method == "html":
                    st.markdown(highlighted_sentence, unsafe_allow_html=True)
                elif sentence_method in ["bold", "caps", "brackets"]:
                    st.markdown(highlighted_sentence)
                    if sentence_method == "brackets":
                        st.caption("💡 Matched terms are shown in [brackets]")
                else:
                    st.markdown(f"> {full_sentence}")
            else:
                st.markdown(f"> {full_sentence}")

            full_context = match.get("context", "")
            if full_context and full_context != full_sentence:
                st.markdown("#### 📖 Complete Context")
                if show_scores:
                    highlighted_context, context_method = smart_highlight_text(full_context, target_sentence, "auto")
                    clean_context = clean_html_artifacts(highlighted_context)
                    formatted_context = format_text_as_clean_paragraphs(clean_context)
                    st.markdown(formatted_context, unsafe_allow_html=context_method == "html")
                    if context_method == "brackets":
                        st.caption("💡 Matched terms are shown in [brackets]")
                else:
                    clean_context = clean_html_artifacts(full_context)
                    st.markdown(format_text_as_clean_paragraphs(clean_context))


def copy_results_beautiful(results: List[Dict], query: str):
    """Show a copyable text report of results."""
    meaningful_words = get_meaningful_words(query)
    output = f"""
DAPHNE AI - SEARCH RESULTS REPORT
==================================

Search Query: "{query}"
Meaningful Words: {', '.join(meaningful_words) if meaningful_words else 'None'}
Total Results: {len(results)}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

    doc_groups: Dict[str, List[Dict]] = {}
    for result in results:
        doc_name = result["document"]["filename"]
        doc_groups.setdefault(doc_name, []).append(result)

    for doc_name, doc_results in doc_groups.items():
        output += f"\n📄 DOCUMENT: {doc_name}\n"
        output += f"{'=' * (len(doc_name) + 15)}\n"
        output += f"Total Matches: {len(doc_results)}\n\n"
        for i, result in enumerate(doc_results, 1):
            score = result.get("score", 0)
            method = result.get("match_type", "unknown").title()
            page = result.get("page_number", 1)
            context = result.get("context", "")
            output += f"Match {i} - {method} (Score: {score:.1f}) - Page {page}\n"
            output += f"{'-' * 50}\n"
            clean_context = clean_html_artifacts(context)
            output += f"Content:\n{clean_context}\n\n"

    st.markdown("### 📋 Complete Results Report")
    st.code(output, language="text")
    st.success("✅ Complete results displayed above! Use Ctrl+A, Ctrl+C to copy to clipboard")

    safe_query = "".join(c for c in query if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_query = safe_query.replace(" ", "_")[:20]
    st.download_button(
        label="📥 Download Complete Report",
        data=output,
        file_name=f"daphne_search_results_{safe_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


def export_results_csv_beautiful(results: List[Dict], query: str):
    """Export results to CSV."""
    csv_data = []
    meaningful_words = get_meaningful_words(query)

    for i, result in enumerate(results, 1):
        context = result.get("context", "")
        clean_context = clean_html_artifacts(context)
        csv_data.append(
            {
                "Match_Number": i,
                "Query": query,
                "Meaningful_Words": ", ".join(meaningful_words),
                "Document": result["document"]["filename"],
                "Match_Type": result.get("match_type", "Unknown"),
                "Score": result.get("score", 0),
                "Page_Number": result.get("page_number", 1),
                "Position": result.get("position", 0),
                "Percentage_Through": result.get("percentage_through", 0),
                "Complete_Context": clean_context,
                "Word_Count": len(clean_context.split()),
                "Character_Count": len(clean_context),
            }
        )

    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    safe_query = "".join(c for c in query if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_query = safe_query.replace(" ", "_")[:20]
    filename = f"daphne_results_{safe_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    st.download_button(label="📥 Download Complete CSV", data=csv, file_name=filename, mime="text/csv")
    st.success(f"✅ CSV ready for download with complete content! ({len(results)} results)")

    with st.expander("📊 CSV Preview"):
        st.dataframe(df[["Match_Number", "Document", "Score", "Page_Number", "Word_Count"]].head())


__all__ = [
    "display_search_results_beautiful",
    "display_single_search_result_beautiful",
    "display_manual_search_results_beautiful",
    "copy_results_beautiful",
    "export_results_csv_beautiful",
]
