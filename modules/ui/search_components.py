# modules/ui/search_components.py
# Search interface components without charts

import streamlit as st
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Import search engines
try:
    from ..search.rag_search import RAGSearchEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG search not available")

try:
    from ..search.smart_search import SmartPatternSearch
    SMART_SEARCH_AVAILABLE = True
except ImportError:
    SMART_SEARCH_AVAILABLE = False
    logging.warning("Smart search not available")

try:
    from ..search.fuzzy_search import FuzzySearchEngine
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("Fuzzy search not available")

from ..core_utils import log_action

class SearchInterface:
    """Main search interface class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize search engines
        self.rag_engine = RAGSearchEngine() if RAG_AVAILABLE else None
        self.smart_engine = SmartPatternSearch() if SMART_SEARCH_AVAILABLE else None
        self.fuzzy_engine = FuzzySearchEngine() if FUZZY_AVAILABLE else None
        
        # Search modes
        self.search_modes = {
            'rag': 'RAG Semantic Search',
            'smart': 'Smart Pattern Search', 
            'hybrid': 'Hybrid (RAG + Smart)',
            'fuzzy': 'Fuzzy Search'
        }

def render_search_interface(documents: List[Dict[str, Any]]):
    """Render the main search interface"""
    search_ui = SearchInterface()
    
    st.header("🔍 Advanced Search")
    
    # Search input
    query = st.text_input(
        "Search Query", 
        placeholder="Enter your search terms...",
        help="Search across all uploaded documents"
    )
    
    # Search mode selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        available_modes = {}
        if RAG_AVAILABLE:
            available_modes['rag'] = 'RAG Semantic Search'
            available_modes['hybrid'] = 'Hybrid (RAG + Smart)'
        if SMART_SEARCH_AVAILABLE:
            available_modes['smart'] = 'Smart Pattern Search'
        if FUZZY_AVAILABLE:
            available_modes['fuzzy'] = 'Fuzzy Search'
        
        if not available_modes:
            available_modes['basic'] = 'Basic Search'
        
        search_mode = st.selectbox(
            "Search Mode",
            options=list(available_modes.keys()),
            format_func=lambda x: available_modes[x],
            help="Choose search algorithm"
        )
    
    with col2:
        max_results = st.number_input(
            "Max Results", 
            min_value=1, 
            max_value=50, 
            value=10,
            help="Maximum number of results to show"
        )
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_type_filter = st.selectbox(
                "File Type",
                options=['All'] + list(set(doc.get('file_type', 'unknown') for doc in documents)),
                help="Filter by file type"
            )
        
        with col2:
            min_words = st.number_input(
                "Min Words",
                min_value=0,
                value=0,
                help="Minimum word count"
            )
        
        with col3:
            max_words = st.number_input(
                "Max Words", 
                min_value=0,
                value=0,
                help="Maximum word count (0 = no limit)"
            )
    
    # Search execution
    if query:
        search_button = st.button("🔍 Search", type="primary")
        
        if search_button or query:
            start_time = time.time()
            
            # Apply filters
            filtered_docs = apply_filters(documents, {
                'file_type': file_type_filter if file_type_filter != 'All' else None,
                'min_words': min_words if min_words > 0 else None,
                'max_words': max_words if max_words > 0 else None
            })
            
            # Perform search
            results = perform_search(search_ui, query, filtered_docs, search_mode, max_results)
            
            search_time = time.time() - start_time
            
            # Log search
            log_action("search_performed", {
                "query": query,
                "mode": search_mode,
                "results_count": len(results),
                "search_time": search_time
            })
            
            # Display results
            display_search_results(results, query, search_time, search_mode)

def apply_filters(documents: List[Dict], filters: Dict) -> List[Dict]:
    """Apply search filters to documents"""
    filtered = []
    
    for doc in documents:
        # File type filter
        if filters.get('file_type') and doc.get('file_type') != filters['file_type']:
            continue
        
        # Word count filters
        word_count = doc.get('word_count', 0)
        if filters.get('min_words') and word_count < filters['min_words']:
            continue
        if filters.get('max_words') and word_count > filters['max_words']:
            continue
        
        filtered.append(doc)
    
    return filtered

def perform_search(search_ui: SearchInterface, query: str, documents: List[Dict], mode: str, max_results: int) -> List[Dict]:
    """Perform search using selected mode"""
    
    try:
        if mode == 'rag' and search_ui.rag_engine:
            return search_ui.rag_engine.search(query, documents, max_results)
        
        elif mode == 'smart' and search_ui.smart_engine:
            return search_ui.smart_engine.search(query, documents, max_results)
        
        elif mode == 'hybrid' and search_ui.rag_engine and search_ui.smart_engine:
            # Combine RAG and Smart search
            rag_results = search_ui.rag_engine.search(query, documents, max_results // 2)
            smart_results = search_ui.smart_engine.search(query, documents, max_results // 2)
            
            # Merge and re-rank results
            return merge_hybrid_results(rag_results, smart_results, max_results)
        
        elif mode == 'fuzzy' and search_ui.fuzzy_engine:
            return search_ui.fuzzy_engine.search(query, documents, max_results)
        
        else:
            # Fallback to basic search
            return basic_search(query, documents, max_results)
    
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def basic_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Basic text search fallback"""
    results = []
    query_lower = query.lower()
    
    for doc in documents:
        if 'text' not in doc:
            continue
        
        text_lower = doc['text'].lower()
        if query_lower in text_lower:
            score = text_lower.count(query_lower)
            snippet = extract_snippet(doc['text'], query)
            
            results.append({
                'document': doc,
                'score': score,
                'snippet': snippet,
                'match_type': 'basic'
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def extract_snippet(text: str, query: str, max_length: int = 200) -> str:
    """Extract relevant snippet from text"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    index = text_lower.find(query_lower)
    if index == -1:
        return text[:max_length] + "..."
    
    start = max(0, index - max_length // 2)
    end = min(len(text), index + len(query) + max_length // 2)
    
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet

def merge_hybrid_results(rag_results: List[Dict], smart_results: List[Dict], max_results: int) -> List[Dict]:
    """Merge RAG and Smart search results"""
    merged = {}
    
    # Add RAG results with boost
    for result in rag_results:
        doc_id = result['document']['filename']
        result['final_score'] = result.get('similarity', 0) * 1.2
        merged[doc_id] = result
    
    # Add Smart results
    for result in smart_results:
        doc_id = result['document']['filename']
        if doc_id in merged:
            # Combine scores
            merged[doc_id]['final_score'] += result.get('score', 0) * 0.8
        else:
            result['final_score'] = result.get('score', 0) * 0.8
            merged[doc_id] = result
    
    # Sort by final score
    final_results = list(merged.values())
    final_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    return final_results[:max_results]

def display_search_results(results: List[Dict], query: str, search_time: float, search_mode: str):
    """Display search results without charts"""
    
    if not results:
        st.warning("No results found.")
        return
    
    # Results header
    st.markdown(f"### 📊 Search Results")
    st.write(f"Found **{len(results)}** results for '{query}' in {search_time:.2f} seconds using {search_mode} search")
    
    # Results list
    for i, result in enumerate(results, 1):
        doc = result['document']
        score = result.get('score', result.get('similarity', result.get('final_score', 0)))
        
        # Result container
        with st.container():
            st.markdown("---")
            
            # Header row
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{i}. {doc['filename']}**")
            
            with col2:
                match_quality = classify_match_quality(score)
                st.markdown(f"**{match_quality}**")
            
            with col3:
                st.markdown(f"**Score: {score:.2f}**")
            
            # Document info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Type: {doc.get('file_type', 'unknown').upper()}")
            with col2:
                st.caption(f"Words: {doc.get('word_count', 0):,}")
            with col3:
                st.caption(f"Size: {doc.get('file_size_mb', 0):.1f} MB")
            
            # Snippet
            snippet = result.get('snippet', '')
            if snippet:
                st.markdown("**Relevant excerpt:**")
                # Highlight query terms
                highlighted_snippet = highlight_terms(snippet, query)
                st.markdown(highlighted_snippet)
            
            # Show more details option
            if st.button(f"📄 View Full Text", key=f"view_{i}"):
                with st.expander(f"Full text of {doc['filename']}", expanded=True):
                    st.text(doc.get('text', 'No text available'))

def classify_match_quality(score: float) -> str:
    """Classify match quality based on score"""
    if score >= 2.5:
        return "🎯 Excellent"
    elif score >= 1.5:
        return "✅ Very Good"
    elif score >= 1.0:
        return "👍 Good"
    elif score >= 0.5:
        return "🔍 Fair"
    else:
        return "❓ Weak"

def highlight_terms(text: str, query: str) -> str:
    """Highlight search terms in text"""
    if not query or not text:
        return text
    
    # Split query into words
    query_words = query.lower().split()
    
    # Highlight each word
    highlighted = text
    for word in query_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(f"**{word}**", highlighted)
    
    return highlighted
