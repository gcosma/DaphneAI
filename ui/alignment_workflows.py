"""Alignment-related Streamlit workflows extracted from search_components."""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from .alignment_display import display_alignment_results_beautiful, show_alignment_feature_info_beautiful
from .display_utils import display_manual_search_results_beautiful
from .search_logic import (
    align_recommendations_with_responses,
    classify_content_type,
    find_pattern_matches,
    find_similar_content_filtered,
    get_meaningful_words,
)

logger = logging.getLogger(__name__)


def render_auto_alignment_with_extractor(documents: List[Dict[str, Any]]):
    """Automatic alignment using the advanced recommendation extractor."""
    st.markdown("### 🔄 Advanced Recommendation-Response Alignment")
    st.markdown("*Uses AI-powered recommendation detection + semantic response matching*")

    try:
        from daphne_core.recommendation_extractor import extract_recommendations
        extractor_available = True
    except ImportError:
        extractor_available = False
        st.error("Advanced recommendation extractor not available")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Recommendation Detection**")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for recommendations",
        )

        detection_methods = st.multiselect(
            "Detection methods",
            ["gerund", "imperative", "modal", "explicit_high", "explicit_medium"],
            default=["gerund", "imperative", "modal"],
            help="Methods to use for finding recommendations",
        )

    with col2:
        st.markdown("**↩️ Response Detection**")
        resp_patterns = st.multiselect(
            "Response keywords",
            ["accept", "reject", "agree", "disagree", "implement", "consider", "approved", "declined", "support"],
            default=["accept", "reject", "agree", "implement"],
        )

        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.2,
            max_value=0.8,
            value=0.3,
            step=0.1,
            help="Minimum similarity for alignment",
        )

    doc_names = [doc["filename"] for doc in documents]
    selected_docs = st.multiselect(
        "📄 Select documents to analyse:",
        doc_names,
        default=doc_names[: min(2, len(doc_names))],
        help="Select one or more documents",
    )

    if not selected_docs:
        st.warning("Please select at least one document")
        return

    if st.button("🔍 Extract & Align Recommendations", type="primary"):
        with st.spinner("🔍 Analysing documents with advanced extraction..."):
            try:
                st.info("Step 1/3: Extracting recommendations using AI methods...")

                all_recommendations = []
                for doc in documents:
                    if doc["filename"] not in selected_docs:
                        continue

                    text = doc.get("text", "")
                    if not text:
                        continue

                    recs = extract_recommendations(text, min_confidence=confidence_threshold)
                    if detection_methods:
                        recs = [r for r in recs if r["method"] in detection_methods]

                    for rec in recs:
                        rec["document"] = doc
                        rec["sentence"] = rec["text"]
                        rec["id"] = f"rec_{len(all_recommendations) + 1}"
                        rec["pattern"] = rec["verb"]
                        rec["recommendation_type"] = classify_content_type(rec["text"])

                    all_recommendations.extend(recs)

                st.success(f"✅ Found {len(all_recommendations)} recommendations")

                st.info("Step 2/3: Finding responses...")
                responses = find_pattern_matches(documents, resp_patterns, "response")
                responses = [r for r in responses if r["document"]["filename"] in selected_docs]
                st.success(f"✅ Found {len(responses)} responses")

                st.info("Step 3/3: Aligning recommendations with responses...")
                alignments = align_recommendations_with_responses(all_recommendations, responses, similarity_threshold)
                st.success(f"✅ Created {len(alignments)} alignments")

                display_alignment_results_beautiful(alignments, show_ai_summaries=False)
                if alignments:
                    export_alignments_to_csv(alignments, selected_docs)

            except Exception as exc:  # pragma: no cover - UI only
                logger.error(f"Advanced alignment error: {exc}")
                st.error(f"❌ Alignment error: {exc}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())


def display_advanced_alignment_results(alignments: List[Dict[str, Any]]):
    """Display advanced alignment results using the reusable display helper."""
    display_alignment_results_beautiful(alignments, show_ai_summaries=False)


def export_alignments_to_csv(alignments: List[Dict[str, Any]], selected_docs: List[str]):
    """Export alignments to CSV."""
    st.markdown("---")
    st.markdown("### 📥 Export Results")

    export_data = []
    for idx, alignment in enumerate(alignments, 1):
        rec = alignment["recommendation"]
        responses = alignment["responses"]

        row = {
            "ID": idx,
            "Recommendation": rec["sentence"],
            "Action Verb": alignment["action_verb"],
            "Detection Method": alignment["detection_method"],
            "Detection Confidence": f"{alignment['detection_confidence']:.0%}",
            "Document": rec["document"]["filename"],
            "Alignment Status": alignment["alignment_status"],
            "Response Count": len(responses),
        }

        if responses:
            top_response = responses[0]["response"]
            row["Top Response"] = top_response["sentence"]
            row["Response Document"] = top_response["document"]["filename"]
            row["Alignment Confidence"] = f"{responses[0]['combined_score']:.0%}"
        else:
            row["Top Response"] = "No response found"
            row["Response Document"] = ""
            row["Alignment Confidence"] = "0%"

        export_data.append(row)

    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)

    docs_str = "_".join(selected_docs)[:50]
    filename = f"alignments_{docs_str}_{datetime.now().strftime('%Y%m%d')}.csv"

    st.download_button(
        label="📥 Download Alignments as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        help="Download all recommendation-response alignments",
    )


def render_recommendation_alignment_interface(documents: List[Dict[str, Any]]):
    """Recommendation-response alignment interface."""
    st.header("🏛️ Recommendation-Response Alignment")
    st.markdown("*Automatically find recommendations and their corresponding responses*")

    if not documents:
        st.warning("📁 Please upload documents first")
        show_alignment_feature_info_beautiful()
        return

    tab_selection = st.radio("Choose alignment mode:", ["🔄 Auto Alignment", "🔍 Manual Search"], horizontal=True)
    if tab_selection == "🔄 Auto Alignment":
        render_auto_alignment_fixed(documents)
    else:
        render_manual_search_fixed(documents)


def render_auto_alignment_fixed(documents: List[Dict[str, Any]]):
    """Automatic alignment using advanced recommendation extractor."""
    st.markdown("### 🔄 Advanced Recommendation-Response Alignment")
    st.markdown("*Uses AI-powered recommendation detection + semantic response matching*")

    try:
        from ..extractors.recommendation_extractor import extract_recommendations
        extractor_available = True
    except ImportError:
        extractor_available = False
        st.warning("⚠️ Advanced extractor not available - using basic keyword matching")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Recommendation Detection**")

        if extractor_available:
            confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum confidence for recommendations",
            )

            detection_methods = st.multiselect(
                "Detection methods",
                ["gerund", "imperative", "modal", "explicit_high", "explicit_medium"],
                default=["gerund", "imperative", "modal"],
                help="AI methods: gerund (Establishing...), imperative (Implement...), modal (should/must)",
            )
        else:
            rec_patterns = st.multiselect(
                "Recommendation Keywords",
                ["recommend", "suggest", "advise", "propose", "urge", "should", "must"],
                default=["recommend", "suggest", "advise"],
            )

    with col2:
        st.markdown("**↩️ Response Detection**")
        resp_patterns = st.multiselect(
            "Response keywords",
            ["accept", "reject", "agree", "disagree", "implement", "consider", "approved", "declined", "support"],
            default=["accept", "reject", "agree", "implement"],
        )

        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.2,
            max_value=0.8,
            value=0.3,
            step=0.1,
            help="Minimum similarity for alignment",
        )

    doc_names = [doc["filename"] for doc in documents]
    selected_docs = st.multiselect(
        "📄 Select documents to analyse:",
        doc_names,
        default=doc_names[: min(2, len(doc_names))],
        help="Select one or more documents",
    )

    if not selected_docs:
        st.warning("Please select at least one document")
        return

    if st.button("🔍 Extract & Align Recommendations", type="primary"):
        with st.spinner("🔍 Analysing documents..."):
            try:
                if extractor_available:
                    st.info("Step 1/3: Extracting recommendations...")
                    all_recommendations = []
                    for doc in documents:
                        if doc["filename"] not in selected_docs:
                            continue

                        text = doc.get("text", "")
                        if not text:
                            continue

                        recs = extract_recommendations(text, min_confidence=confidence_threshold)
                        if detection_methods:
                            recs = [r for r in recs if r["method"] in detection_methods]

                        for rec in recs:
                            rec["document"] = doc
                            rec["sentence"] = rec.get("text", rec.get("sentence", ""))
                            rec["pattern"] = rec.get("verb", "")
                            rec["recommendation_type"] = classify_content_type(rec["sentence"])

                        all_recommendations.extend(recs)
                else:
                    st.info("Step 1/3: Finding recommendations with basic keywords...")
                    all_recommendations = find_pattern_matches(documents, rec_patterns, "recommendation")
                    all_recommendations = [rec for rec in all_recommendations if rec["document"]["filename"] in selected_docs]

                st.success(f"✅ Found {len(all_recommendations)} recommendations")

                st.info("Step 2/3: Finding responses...")
                responses = find_pattern_matches(documents, resp_patterns, "response")
                responses = [r for r in responses if r["document"]["filename"] in selected_docs]
                st.success(f"✅ Found {len(responses)} responses")

                st.info("Step 3/3: Aligning recommendations with responses...")
                if extractor_available:
                    alignments = align_recommendations_with_responses(
                        all_recommendations,
                        responses,
                        similarity_threshold,
                    )
                else:
                    alignments = create_simple_alignments(all_recommendations, responses)

                st.success(f"✅ Created {len(alignments)} alignments")

                if extractor_available:
                    display_advanced_alignment_results(alignments)
                else:
                    display_alignment_results_beautiful(alignments, show_ai_summaries=False)

                if alignments:
                    export_alignments_to_csv(alignments, selected_docs)

            except Exception as exc:  # pragma: no cover - UI only
                logger.error(f"Alignment analysis error: {exc}")
                st.error(f"❌ Analysis error: {exc}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

                if not extractor_available:
                    show_basic_pattern_analysis(documents, rec_patterns, resp_patterns)


def create_simple_alignments(recommendations: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback alignment when advanced extractor is unavailable."""
    return align_recommendations_with_responses(recommendations, responses, similarity_threshold=0.3)


def show_basic_pattern_analysis(documents: List[Dict[str, Any]], rec_patterns: List[str], resp_patterns: List[str]):
    """Display simple pattern-based matches as a lightweight fallback."""
    st.markdown("### 🔎 Basic Pattern Analysis (Fallback)")
    recs = find_pattern_matches(documents, rec_patterns, "recommendation")
    resps = find_pattern_matches(documents, resp_patterns, "response")

    st.info(f"Found {len(recs)} recommendations and {len(resps)} responses via keyword patterns.")
    if recs:
        st.write("**Sample recommendations**")
        for rec in recs[:5]:
            st.write(f"- {rec['sentence']} ({rec['document'].get('filename', 'unknown')})")
    if resps:
        st.write("**Sample responses**")
        for resp in resps[:5]:
            st.write(f"- {resp['sentence']} ({resp['document'].get('filename', 'unknown')})")
    st.caption("These results use simple keyword spotting and may be less accurate than the AI extractor.")


def render_manual_search_fixed(documents: List[Dict[str, Any]]):
    """Manual search with filtering."""
    st.markdown("### 🔍 Manual Sentence Search")

    search_sentence = st.text_area(
        "📝 Paste your sentence here:",
        placeholder="e.g., 'The committee recommends implementing new security protocols'",
        help="Similarity matching will focus on meaningful words",
        height=100,
    )

    col1, col2 = st.columns(2)

    with col1:
        search_type = st.selectbox("Search for:", ["Similar content", "Recommendations", "Responses"])
        similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.3, 0.1)

    with col2:
        max_matches = st.selectbox("Max matches", [5, 10, 20, 50])
        show_scores = st.checkbox("Show similarity scores", True)

    if search_sentence.strip():
        meaningful_words = get_meaningful_words(search_sentence)
        if meaningful_words:
            display_words = ", ".join(meaningful_words[:10])
            suffix = "..." if len(meaningful_words) > 10 else ""
            st.info(f"🔍 **Focusing on meaningful words:** {display_words}{suffix}")
        else:
            st.warning("⚠️ No meaningful words found in your sentence")

    if st.button("🔎 Find Matches", type="primary") and search_sentence.strip():
        search_start = time.time()
        with st.spinner("🔍 Searching for similar content..."):
            try:
                matches = find_similar_content_filtered(
                    documents,
                    search_sentence,
                    search_type,
                    similarity_threshold,
                    max_matches,
                )

                search_time = time.time() - search_start
                display_manual_search_results_beautiful(
                    matches, search_sentence, search_time, show_scores, search_type.lower()
                )
            except Exception as exc:  # pragma: no cover - UI only
                logger.error(f"Manual search error: {exc}")
                st.error(f"Search error: {exc}")


__all__ = [
    "render_recommendation_alignment_interface",
    "render_auto_alignment_fixed",
    "render_auto_alignment_with_extractor",
    "render_manual_search_fixed",
]
