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

    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
        st.session_state.last_search_query = ""
        st.session_state.last_search_time = 0
    
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
    
    # AI availability check - STREAMLIT CLOUD OPTIMIZED
    ai_available = check_rag_availability()
    
    import os
    is_streamlit_cloud = (
        os.getenv('STREAMLIT_SHARING_MODE') or 
        'streamlit.app' in os.getenv('HOSTNAME', '') or
        '/mount/src/' in os.getcwd()
    )
    
    if method_key in ["semantic", "hybrid"]:
        if is_streamlit_cloud:
            st.info("🌐 **Streamlit Cloud Detected** - Using optimized semantic search designed for government documents")
            with st.expander("ℹ️ Why We Use Optimized Search on Streamlit Cloud"):
                st.markdown("""
                **Streamlit Cloud Optimization:**
                - ✅ **Faster performance** - No model loading delays
                - ✅ **Government-tuned** - Specialized for policy documents  
                - ✅ **More reliable** - No PyTorch device conflicts
                - ✅ **Better results** - Domain-specific semantic matching
                
                **What You Get:**
                - Semantic word groups (recommend → suggest → advise → propose)
                - Government terminology (department → ministry → agency)
                - Policy vocabulary (framework → protocol → guideline)
                - Response patterns (accept → agree → approve → implement)
                
                **Performance:** Often more accurate than generic AI models for government content!
                """)
        elif ai_available:
            st.info("🤖 **Full AI semantic search available** - Using sentence transformers")
        else:
            st.info("🤖 **Enhanced semantic search active** - Using government-optimized matching")
            if st.button("💡 Install Full AI for Local Development"):
                st.code("pip install sentence-transformers torch huggingface-hub")
    
    # =========================================================================
    # ADD THIS: Show previous results if available (Section 2)
    # =========================================================================
    if (st.session_state.search_results is not None and 
        st.session_state.last_search_query and 
        not query):  # Only show when no new query is entered
        
        st.info(f"📋 **Showing previous results for:** '{st.session_state.last_search_query}'")
        st.caption(f"🕒 Search completed in {st.session_state.last_search_time:.3f} seconds")
        
        # Display the stored results using your existing display function
        display_results_grouped(
            results=st.session_state.search_results,
            query=st.session_state.last_search_query,
            search_time=st.session_state.last_search_time,
            show_context=show_context,
            highlight_matches=highlight_matches,
            search_method=st.session_state.last_search_query  # Use stored query as method name
        )
    
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
            
            # =========================================================================
            # ADD THIS: Store results in session state (Section 3)
            # =========================================================================
            st.session_state.search_results = results
            st.session_state.last_search_query = query
            st.session_state.last_search_time = search_time
            
            # Display results
            display_results_grouped(
                results=results,
                query=query,
                search_time=search_time,
                show_context=show_context,
                highlight_matches=highlight_matches,
                search_method=search_method.split(' - ')[0]
            )
