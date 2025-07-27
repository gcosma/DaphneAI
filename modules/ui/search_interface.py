# modules/ui/search_interface.py - Main Search Interface
import streamlit as st
import time
from typing import Dict, List, Any
from .search_methods import execute_search
from .result_display import display_results_grouped
from .search_utils import check_rag_availability

def render_search_interface(documents: List[Dict[str, Any]]):
    """Enhanced search interface with clear options and descriptions"""
    
    st.header("🔍 Advanced Document Search")
    st.markdown("*Search through your documents with multiple AI-powered methods*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        return
    
    # Search input
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., recommendations, policy changes, budget allocation",
        help="Enter keywords, phrases, or concepts to search for"
    )
    
    # Search method selection with clear descriptions
    st.markdown("### 🎯 Search Method")
    
    search_method = st.radio(
        "Choose your search approach:",
        [
            "🧠 Smart Search - Finds keywords and phrases (recommended for most searches)",
            "🎯 Exact Match - Finds exact words only (fastest, most precise)",
            "🌀 Fuzzy Search - Handles typos and misspellings", 
            "🤖 AI Semantic - AI finds related concepts (needs AI libraries)",
            "🔄 Hybrid - Combines Smart + AI for best results (needs AI libraries)"
        ],
        index=0,
        help="Different methods find different types of matches"
    )
    
    # Extract search method key
    method_mapping = {
        "🧠 Smart Search": "smart",
        "🎯 Exact Match": "exact", 
        "🌀 Fuzzy Search": "fuzzy",
        "🤖 AI Semantic": "semantic",
        "🔄 Hybrid": "hybrid"
    }
    
    method_key = next(key for desc, key in method_mapping.items() if desc in search_method)
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        max_results = st.slider("Max results per document", 1, 20, 5)
        case_sensitive = st.checkbox("Case sensitive search", value=False)
    
    with col2:
        show_context = st.checkbox("Show context around matches", value=True)
        highlight_matches = st.checkbox("Highlight search terms", value=True)
    
    # AI availability check
    ai_available = check_rag_availability()
    if method_key in ["semantic", "hybrid"] and not ai_available:
        st.warning("🤖 AI search requires: `pip install sentence-transformers torch`")
        if st.button("💡 Show Installation Instructions"):
            st.code("pip install sentence-transformers torch scikit-learn")
        return
    
    # Search execution
    if st.button("🔍 Search Documents", type="primary") and query:
        
        start_time = time.time()
        
        with st.spinner(f"🔍 Searching with {search_method.split(' - ')[0]}..."):
            
            # Execute search based on method
            results = execute_search(
                documents=documents,
                query=query,
                method=method_key,
                max_results=max_results,
                case_sensitive=case_sensitive
            )
            
            search_time = time.time() - start_time
            
            # Display results
            display_results_grouped(
                results=results,
                query=query,
                search_time=search_time,
                show_context=show_context,
                highlight_matches=highlight_matches,
                search_method=search_method.split(' - ')[0]
            )
