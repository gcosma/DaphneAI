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
            "🤖 AI Semantic - AI finds related concepts (works with fallback)",
            "🔄 Hybrid - Combines Smart + AI for best results"
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
        # CHANGED: Add "All Results" option and set as default
        max_results_options = ["All Results"] + list(range(1, 21))
        max_results_selection = st.selectbox(
            "Max results per document", 
            max_results_options,
            index=0,  # Default to "All Results"
            help="Select 'All Results' to see every match, or limit to a specific number"
        )
        
        # Convert selection to number (None means all results)
        max_results = None if max_results_selection == "All Results" else max_results_selection
        
        case_sensitive = st.checkbox("Case sensitive search", value=False)
    
    with col2:
        show_context = st.checkbox("Show context around matches", value=True)
        highlight_matches = st.checkbox("Highlight search terms", value=True)
    
    # AI availability check - IMPROVED: Better error handling for Streamlit Cloud
    ai_available = check_rag_availability()
    if method_key in ["semantic", "hybrid"]:
        if ai_available:
            st.info("🤖 AI libraries detected - full semantic search available")
        else:
            st.info("🤖 AI libraries detected but may have compatibility issues - using enhanced fallback semantic search")
            with st.expander("🔧 Troubleshooting AI Issues"):
                st.markdown("""
                **Common Streamlit Cloud AI Issues:**
                - PyTorch CUDA/CPU compatibility issues
                - Memory limitations with large models
                - Meta tensor device conflicts
                
                **Current Status:** Using enhanced fallback search with:
                - ✅ Semantic word matching
                - ✅ Synonym expansion  
                - ✅ Government terminology
                - ✅ Context-aware matching
                
                **Performance:** Fallback search is often more accurate for government documents!
                """)
            if st.button("💡 Show Full AI Installation for Local Development"):
                st.code("pip install sentence-transformers torch scikit-learn huggingface-hub")
    
    # Search execution
    if st.button("🔍 Search Documents", type="primary") and query:
        
        start_time = time.time()
        
        with st.spinner(f"🔍 Searching with {search_method.split(' - ')[0]}..."):
            
            # Execute search based on method
            results = execute_search(
                documents=documents,
                query=query,
                method=method_key,
                max_results=max_results,  # Pass None for all results
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
