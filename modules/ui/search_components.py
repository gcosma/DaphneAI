# modules/ui/search_components.py
# Simple, working search components - restores your original functionality

import streamlit as st
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Simple search that works with your uploaded documents
def render_search_interface(documents: List[Dict[str, Any]]):
    """Main search interface - simple and functional"""
    st.header("🔍 Document Search")
    
    if not documents:
        st.info("No documents uploaded. Please upload documents first.")
        return
    
    # Search input
    query = st.text_input(
        "Search Documents", 
        placeholder="Enter your search terms...",
        help="Search across all uploaded documents"
    )
    
    # Search options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_mode = st.selectbox(
            "Search Mode",
            ["Smart Search", "Exact Match", "Fuzzy Search"],
            help="Choose search algorithm"
        )
    
    with col2:
        max_results = st.number_input(
            "Max Results", 
            min_value=1, 
            max_value=50, 
            value=10
        )
    
    # Perform search
    if query:
        start_time = time.time()
        
        if search_mode == "Smart Search":
            results = smart_search(query, documents, max_results)
        elif search_mode == "Exact Match":
            results = exact_search(query, documents, max_results)
        else:  # Fuzzy Search
            results = fuzzy_search(query, documents, max_results)
        
        search_time = time.time() - start_time
        
        # Display results
        display_results(results, query, search_time)

def smart_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Smart search with relevance scoring"""
    results = []
    query_lower = query.lower()
    query_words = query_lower.split()
    
    for doc in documents:
        if 'text' not in doc:
            continue
        
        text = doc['text']
        text_lower = text.lower()
        score = 0
        
        # Exact phrase bonus
        if query_lower in text_lower:
            score += 10
        
        # Word matching
        word_matches = sum(1 for word in query_words if word in text_lower)
        score += word_matches * 2
        
        # Frequency bonus
        for word in query_words:
            score += text_lower.count(word) * 0.5
        
        # Title relevance
        filename = doc.get('filename', '').lower()
        if any(word in filename for word in query_words):
            score += 3
        
        if score > 0:
            snippet = extract_snippet(text, query)
            results.append({
                'document': doc,
                'score': score,
                'snippet': snippet
            })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def exact_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Exact phrase matching"""
    results = []
    query_lower = query.lower()
    
    for doc in documents:
        if 'text' not in doc:
            continue
        
        text = doc['text']
        if query_lower in text.lower():
            count = text.lower().count(query_lower)
            snippet = extract_snippet(text, query)
            
            results.append({
                'document': doc,
                'score': count,
                'snippet': snippet
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def fuzzy_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Simple fuzzy search"""
    results = []
    query_words = query.lower().split()
    
    for doc in documents:
        if 'text' not in doc:
            continue
        
        text = doc['text']
        text_lower = text.lower()
        score = 0
        
        # Check for similar words
        for query_word in query_words:
            for text_word in text_lower.split():
                if query_word in text_word or text_word in query_word:
                    score += 1
                elif len(query_word) > 3 and len(text_word) > 3:
                    # Simple similarity check
                    common_chars = len(set(query_word) & set(text_word))
                    if common_chars >= len(query_word) * 0.6:
                        score += 0.5
        
        if score > 0:
            snippet = extract_snippet(text, query)
            results.append({
                'document': doc,
                'score': score,
                'snippet': snippet
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def extract_snippet(text: str, query: str, max_length: int = 300) -> str:
    """Extract relevant snippet from text"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Find query in text
    index = text_lower.find(query_lower)
    if index == -1:
        # If exact phrase not found, find first word
        words = query_lower.split()
        for word in words:
            index = text_lower.find(word)
            if index != -1:
                break
    
    if index == -1:
        return text[:max_length] + "..."
    
    # Extract snippet around found text
    start = max(0, index - max_length // 2)
    end = min(len(text), index + len(query) + max_length // 2)
    snippet = text[start:end]
    
    # Highlight query
    highlighted = highlight_query(snippet, query)
    
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    
    return prefix + highlighted + suffix

def highlight_query(text: str, query: str) -> str:
    """Highlight query terms in text"""
    # Highlight exact phrase
    highlighted = re.sub(
        f'({re.escape(query)})', 
        r'**\1**', 
        text, 
        flags=re.IGNORECASE
    )
    
    # If no exact phrase, highlight individual words
    if '**' not in highlighted:
        words = query.split()
        for word in words:
            highlighted = re.sub(
                f'({re.escape(word)})', 
                r'**\1**', 
                highlighted, 
                flags=re.IGNORECASE
            )
    
    return highlighted

def display_results(results: List[Dict], query: str, search_time: float):
    """Display search results"""
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    st.success(f"Found {len(results)} results for '{query}' in {search_time:.2f} seconds")
    
    for i, result in enumerate(results, 1):
        doc = result['document']
        score = result['score']
        snippet = result['snippet']
        
        with st.expander(f"{i}. {doc['filename']} (Score: {score:.1f})"):
            # Document info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Type: {doc.get('file_type', 'unknown').upper()}")
            with col2:
                st.caption(f"Words: {doc.get('word_count', 0):,}")
            with col3:
                st.caption(f"Size: {doc.get('file_size_mb', 0):.1f} MB")
            
            # Snippet
            st.markdown("**Relevant excerpt:**")
            st.markdown(snippet)
            
            # View full text option
            if st.button(f"📄 View Full Text", key=f"view_full_{i}"):
                with st.expander(f"Full content of {doc['filename']}", expanded=True):
                    st.text_area(
                        "Full Text", 
                        doc.get('text', 'No text available'),
                        height=400,
                        key=f"full_text_{i}"
                    )

# Required functions for compatibility
def render_search_tab():
    """Main search tab - simple implementation"""
    st.title("🔍 Document Search")
    
    # Get documents from session state
    documents = st.session_state.get('documents', [])
    
    if documents:
        render_search_interface(documents)
    else:
        st.info("📁 Please upload documents first using the Upload tab.")
        
        # Sample data for testing
        if st.button("🎯 Load Sample Data"):
            sample_docs = [
                {
                    'filename': 'sample_policy.txt',
                    'text': 'The government recommends implementing new healthcare policies to improve patient access and reduce waiting times. This policy should be considered for immediate implementation.',
                    'word_count': 25,
                    'file_type': 'txt'
                },
                {
                    'filename': 'sample_response.txt', 
                    'text': 'The department accepts the recommendation for healthcare reform. We will establish a task force to oversee the implementation of these critical changes.',
                    'word_count': 24,
                    'file_type': 'txt'
                }
            ]
            st.session_state.documents = sample_docs
            st.rerun()

def render_smart_search_tab():
    """Alias for render_search_tab"""
    return render_search_tab()

def check_search_availability():
    """Check if search is available"""
    return True  # Always available with simple search

# Export functions
__all__ = [
    'render_search_interface',
    'render_search_tab', 
    'render_smart_search_tab',
    'check_search_availability'
]
