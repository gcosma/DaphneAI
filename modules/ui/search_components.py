# modules/ui/search_components.py
# Complete working search components

import streamlit as st
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

def render_search_interface(documents: List[Dict[str, Any]]):
    """Main search interface - simple and functional"""
    st.header("🔍 Document Search")
    
    if not documents:
        st.info("No documents uploaded. Please upload documents first.")
        # Add sample data button for testing
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
        
        # Exact phrase match (highest score)
        if query_lower in text_lower:
            score += 100
        
        # Individual word matches
        for word in query_words:
            word_count = text_lower.count(word)
            score += word_count * 10
        
        # Context scoring (words close together)
        for i in range(len(query_words) - 1):
            word1, word2 = query_words[i], query_words[i + 1]
            if word1 in text_lower and word2 in text_lower:
                score += 20
        
        if score > 0:
            results.append({
                'document': doc,
                'score': score,
                'filename': doc.get('filename', 'Unknown'),
                'snippet': extract_snippet(text, query, 200)
            })
    
    # Sort by score and limit results
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
            results.append({
                'document': doc,
                'score': 100,
                'filename': doc.get('filename', 'Unknown'),
                'snippet': extract_snippet(text, query, 200)
            })
    
    return results[:max_results]

def fuzzy_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Fuzzy search using word overlap"""
    results = []
    query_words = set(query.lower().split())
    
    for doc in documents:
        if 'text' not in doc:
            continue
            
        text = doc['text']
        text_words = set(text.lower().split())
        
        # Calculate overlap ratio
        overlap = len(query_words.intersection(text_words))
        if overlap > 0:
            score = (overlap / len(query_words)) * 100
            results.append({
                'document': doc,
                'score': score,
                'filename': doc.get('filename', 'Unknown'),
                'snippet': extract_snippet(text, query, 200)
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def extract_snippet(text: str, query: str, max_length: int = 200) -> str:
    """Extract relevant snippet around query"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Find first occurrence of query
    pos = text_lower.find(query_lower)
    if pos == -1:
        # If exact query not found, use beginning of text
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # Extract snippet around the query
    start = max(0, pos - max_length // 2)
    end = min(len(text), pos + len(query) + max_length // 2)
    
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet

def display_results(results: List[Dict], query: str, search_time: float):
    """Display search results"""
    if not results:
        st.info("No results found.")
        return
    
    # Results summary
    st.success(f"Found {len(results)} results in {search_time:.3f} seconds")
    
    # Display each result
    for i, result in enumerate(results):
        with st.expander(f"📄 {result['filename']} (Score: {result['score']:.1f})"):
            doc = result['document']
            
            # Document metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Type", doc.get('file_type', 'unknown').upper())
            with col2:
                st.metric("Word Count", doc.get('word_count', 0))
            with col3:
                st.metric("File Size", f"{doc.get('file_size_mb', 0)} MB")
            
            # Snippet with highlighting
            snippet = result['snippet']
            highlighted_snippet = highlight_query_in_text(snippet, query)
            st.markdown("**Relevant Content:**")
            st.markdown(highlighted_snippet, unsafe_allow_html=True)
            
            # Show full content button
            if st.button(f"Show Full Content", key=f"show_full_{i}"):
                st.text_area("Full Document Content", doc['text'], height=300)

def highlight_query_in_text(text: str, query: str) -> str:
    """Highlight query terms in text"""
    if not query:
        return text
    
    # Escape special regex characters in query
    escaped_query = re.escape(query)
    
    # Highlight exact phrase
    highlighted = re.sub(
        f'({escaped_query})', 
        r'<mark style="background-color: yellow">\1</mark>', 
        text, 
        flags=re.IGNORECASE
    )
    
    # Highlight individual words
    for word in query.split():
        if len(word) > 2:  # Only highlight words longer than 2 characters
            escaped_word = re.escape(word)
            highlighted = re.sub(
                f'\\b({escaped_word})\\b', 
                r'<mark style="background-color: lightblue">\1</mark>', 
                highlighted, 
                flags=re.IGNORECASE
            )
    
    return highlighted

# Alias functions for compatibility
def render_search_tab():
    """Render search tab (compatibility alias)"""
    documents = st.session_state.get('documents', [])
    render_search_interface(documents)

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
