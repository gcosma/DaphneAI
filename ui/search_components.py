"""Thin wrappers for search and alignment Streamlit components."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import streamlit as st

from .alignment_workflows import render_recommendation_alignment_interface
from .display_utils import display_search_results_beautiful
from .search_logic import STOP_WORDS, check_rag_availability, execute_search_with_ai, filter_stop_words

logger = logging.getLogger(__name__)


def render_search_interface(documents: List[Dict[str, Any]]):
    """Main search interface with all methods including AI."""
    st.header("🔍 Advanced Document Search")
    st.markdown("*Search with intelligent filtering and AI-powered semantic understanding*")

    if not documents:
        st.warning("📁 Please upload documents first")
        return

    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., committee recommends, department policy, budget allocation",
        help="Search will focus on meaningful words and filter out common words like 'the', 'and', 'a'",
    )

    st.markdown("### 🎯 Search Method")
    search_method = st.radio(
        "Choose your search approach:",
        [
            "🧠 Smart Search - Enhanced keyword matching",
            "🎯 Exact Match - Find exact phrases",
            "🌀 Fuzzy Search - Handle typos and misspellings",
            "🤖 AI Semantic - AI finds related concepts",
            "🔄 Hybrid Search - Combines Smart + AI for best results",
        ],
        index=0,
        help="Smart search filters out common words for better results. AI methods understand meaning and context.",
    )

    col1, col2 = st.columns(2)
    with col1:
        max_results = st.selectbox(
            "Max results per document",
            [5, 10, 15, 20, "All"],
            index=1,
            help="Limit results or show all matches",
        )
        case_sensitive = st.checkbox("Case sensitive search", value=False)

    with col2:
        show_context = st.checkbox("Show context around matches", value=True)
        highlight_matches = st.checkbox("Highlight search terms", value=True)

    ai_available = check_rag_availability()
    is_streamlit_cloud = (
        os.getenv("STREAMLIT_SHARING_MODE")
        or "streamlit.app" in os.getenv("HOSTNAME", "")
        or "/mount/src/" in os.getcwd()
    )

    if search_method in ["🤖 AI Semantic", "🔄 Hybrid Search"]:
        if is_streamlit_cloud:
            st.info("🌐 **Streamlit Cloud Detected** - Using optimized semantic search for government documents")
            with st.expander("ℹ️ Enhanced Semantic Search on Streamlit Cloud"):
                st.markdown(
                    """
                **Streamlit Cloud Optimization:**
                - ✅ **Government-tuned** - Specialized for policy documents  
                - ✅ **Faster performance** - No model loading delays
                - ✅ **Better results** - Domain-specific semantic matching

                **Semantic Features:**
                - Word groups: recommend → suggest → advise → propose
                - Government terms: department → ministry → agency
                - Policy vocabulary: framework → protocol → guideline
                - Response patterns: accept → agree → approve → implement
                """
                )
        elif ai_available:
            st.info("🤖 **Full AI semantic search available** - Using sentence transformers")
        else:
            st.info("🤖 **Enhanced semantic search active** - Using government-optimized matching")
            if st.button("💡 Install Full AI for Local Development"):
                st.code("pip install sentence-transformers torch huggingface-hub")

    if query:
        if "Smart" in search_method or "AI" in search_method or "Hybrid" in search_method:
            filtered_words = filter_stop_words(query)
            if filtered_words != query:
                st.info(f"🔍 **Searching for meaningful words:** {filtered_words}")
                st.caption(f"Filtered out: {', '.join(set(query.lower().split()) & STOP_WORDS)}")
            else:
                st.info(f"🔍 **Searching for:** {query}")
        else:
            st.info(f"🔍 **Exact search for:** {query}")

    if st.button("🔍 Search Documents", type="primary") and query:
        start_time = time.time()
        with st.spinner("🔍 Searching..."):
            results = execute_search_with_ai(
                documents=documents,
                query=query,
                method=search_method,
                max_results=max_results if max_results != "All" else None,
                case_sensitive=case_sensitive,
            )

            search_time = time.time() - start_time
            display_search_results_beautiful(
                results=results,
                query=query,
                search_time=search_time,
                show_context=show_context,
                highlight_matches=highlight_matches,
            )


__all__ = [
    "render_search_interface",
    "render_recommendation_alignment_interface",
    "check_rag_availability",
    "filter_stop_words",
    "STOP_WORDS",
]
