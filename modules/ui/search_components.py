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
        
        text_lower = doc['text'].lower()
        score = 0
        matches = []
        
        # Exact phrase matching (highest score)
        if query_lower in text_lower:
            score += 100
            matches.append(f"Exact phrase: '{query}'")
        
        # Individual word matching
        word_scores = []
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                word_count = text_lower.count(word)
                if word_count > 0:
                    word_score = word_count * 10
                    score += word_score
                    word_scores.append(f"{word}({word_count})")
        
        if word_scores:
            matches.append(f"Words: {', '.join(word_scores)}")
        
        # Length bonus (prefer longer documents with matches)
        if score > 0:
            length_bonus = min(len(doc['text']) / 1000, 10)
            score += length_bonus
            
            # Find context around matches
            context = extract_context(doc['text'], query, 150)
            
            results.append({
                'document': doc,
                'score': score,
                'matches': matches,
                'context': context,
                'search_type': 'smart'
            })
    
    # Sort by score and limit results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def exact_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Exact keyword matching search"""
    results = []
    query_lower = query.lower()
    
    for doc in documents:
        if 'text' not in doc:
            continue
        
        text_lower = doc['text'].lower()
        
        if query_lower in text_lower:
            # Count occurrences
            count = text_lower.count(query_lower)
            context = extract_context(doc['text'], query, 150)
            
            results.append({
                'document': doc,
                'score': count * 100,  # Simple scoring by occurrence count
                'matches': [f"Exact matches: {count}"],
                'context': context,
                'search_type': 'exact'
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def fuzzy_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Fuzzy search for typo tolerance"""
    results = []
    query_words = query.lower().split()
    
    for doc in documents:
        if 'text' not in doc:
            continue
        
        text_lower = doc['text'].lower()
        text_words = re.findall(r'\b\w+\b', text_lower)
        
        score = 0
        matches = []
        
        for query_word in query_words:
            if len(query_word) < 3:  # Skip very short words
                continue
                
            # Look for similar words
            for text_word in text_words:
                similarity = calculate_similarity(query_word, text_word)
                if similarity > 0.7:  # 70% similarity threshold
                    score += similarity * 10
                    if similarity < 1.0:  # Not exact match
                        matches.append(f"{query_word}→{text_word}({similarity:.2f})")
        
        if score > 0:
            context = extract_context(doc['text'], query, 150)
            results.append({
                'document': doc,
                'score': score,
                'matches': matches,
                'context': context,
                'search_type': 'fuzzy'
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def calculate_similarity(word1: str, word2: str) -> float:
    """Calculate simple string similarity (Levenshtein-based)"""
    if word1 == word2:
        return 1.0
    
    # Simple character-based similarity
    set1, set2 = set(word1), set(word2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def extract_context(text: str, query: str, context_length: int = 150) -> str:
    """Extract text context around search matches"""
    text_lower = text.lower()
    query_lower = query.lower()
    
    # Find first occurrence
    index = text_lower.find(query_lower)
    if index == -1:
        # If exact query not found, try first word
        words = query_lower.split()
        if words:
            index = text_lower.find(words[0])
    
    if index == -1:
        # Return beginning of text if no match found
        return text[:context_length] + "..." if len(text) > context_length else text
    
    # Extract context around the match
    start = max(0, index - context_length // 2)
    end = min(len(text), index + len(query) + context_length // 2)
    
    context = text[start:end]
    
    # Add ellipsis if truncated
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."
    
    return context

def display_results(results: List[Dict], query: str, search_time: float):
    """Display search results in a user-friendly format"""
    
    if not results:
        st.warning(f"No results found for '{query}'")
        st.info("Try different search terms or use a different search mode.")
        return
    
    # Search summary
    st.success(f"Found {len(results)} result(s) for '{query}' in {search_time:.3f} seconds")
    
    # Display each result
    for i, result in enumerate(results, 1):
        doc = result['document']
        
        with st.expander(f"📄 {i}. {doc['filename']} (Score: {result['score']:.1f})", expanded=(i <= 3)):
            
            # Document metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Type", doc.get('file_type', 'unknown').upper())
            with col2:
                st.metric("Words", f"{doc.get('word_count', 0):,}")
            with col3:
                st.metric("Size", f"{doc.get('file_size_mb', 0):.1f} MB")
            
            # Match information
            if result['matches']:
                st.info("🎯 " + " | ".join(result['matches']))
            
            # Context preview
            st.markdown("**📖 Context:**")
            # Highlight query terms in context
            context = result['context']
            highlighted_context = highlight_terms(context, query)
            st.markdown(highlighted_context, unsafe_allow_html=True)
            
            # View full document button
            if st.button(f"📖 View Full Document", key=f"view_{i}"):
                st.text_area(
                    f"Full content of {doc['filename']}:",
                    doc.get('text', 'No content available'),
                    height=300,
                    key=f"full_text_{i}"
                )

def highlight_terms(text: str, query: str) -> str:
    """Highlight search terms in text"""
    words = query.lower().split()
    highlighted = text
    
    for word in words:
        if len(word) > 2:  # Skip very short words
            # Case-insensitive replacement with highlighting
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(f'<mark style="background-color: yellow">{word}</mark>', highlighted)
    
    return highlighted

def get_search_stats() -> Dict[str, Any]:
    """Get search statistics from session state"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    history = st.session_state.search_history
    
    return {
        'total_searches': len(history),
        'unique_queries': len(set(h.get('query', '') for h in history)),
        'avg_results': sum(h.get('result_count', 0) for h in history) / max(len(history), 1),
        'recent_queries': [h.get('query', '') for h in history[-5:]]
    }

def log_search(query: str, results: List[Dict], search_time: float, search_type: str):
    """Log search activity for analytics"""
    if 'search_history' not in st.session_state:
        st.session_state
