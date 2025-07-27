# modules/ui/search_components.py
# BEST SOLUTION - Complete working search components with multiple matches per document

import streamlit as st
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

def render_search_interface(documents: List[Dict[str, Any]]):
    """Main search interface - enhanced to show multiple matches per document"""
    st.header("🔍 Document Search")
    
    if not documents:
        st.info("No documents uploaded. Please upload documents first.")
        # Add sample data button for testing
        if st.button("🎯 Load Sample Data"):
            sample_docs = [
                {
                    'filename': 'sample_policy.txt',
                    'text': 'The government recommends implementing new healthcare policies to improve patient access and reduce waiting times. This policy should be considered for immediate implementation. We recommend further consultation.',
                    'word_count': 28,
                    'file_type': 'txt'
                },
                {
                    'filename': 'sample_response.txt', 
                    'text': 'The department accepts the recommendation for healthcare reform. We will establish a task force to oversee the implementation of these critical changes. The committee recommends monthly reviews.',
                    'word_count': 30,
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
        help="Search across all uploaded documents - shows multiple matches per document"
    )
    
    # Search options
    col1, col2, col3 = st.columns([2, 1, 1])
    
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
            max_value=100, 
            value=20,
            help="Maximum total results to show"
        )
    
    with col3:
        group_by_doc = st.checkbox(
            "Group by Document", 
            value=True,
            help="Group multiple matches from same document"
        )
    
    # Perform search
    if query:
        start_time = time.time()
        
        if search_mode == "Smart Search":
            results = smart_search_enhanced(query, documents, max_results)
        elif search_mode == "Exact Match":
            results = exact_search_enhanced(query, documents, max_results)
        else:  # Fuzzy Search
            results = fuzzy_search_enhanced(query, documents, max_results)
        
        search_time = time.time() - start_time
        
        # Log search for analytics
        log_search(query, results, search_time, search_mode.lower().replace(' ', '_'))
        
        # Display results
        if group_by_doc:
            display_results_grouped(results, query, search_time)
        else:
            display_results_flat(results, query, search_time)

def smart_search_enhanced(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Enhanced smart search that finds multiple matches per document"""
    all_results = []
    query_lower = query.lower()
    query_words = [word for word in query_lower.split() if len(word) > 2]
    
    if not query_words:
        return []
    
    for doc in documents:
        if 'text' not in doc or not doc['text']:
            continue
        
        text = doc['text']
        text_lower = text.lower()
        
        # Split text into meaningful chunks (sentences/paragraphs)
        chunks = split_text_into_chunks(text)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = 0
            matches = []
            
            # Exact phrase matching (highest score)
            if query_lower in chunk_lower:
                phrase_count = chunk_lower.count(query_lower)
                score += phrase_count * 100
                matches.append(f"Exact phrase '{query}': {phrase_count}x")
            
            # Individual word matching
            word_scores = []
            for word in query_words:
                word_count = chunk_lower.count(word)
                if word_count > 0:
                    word_score = word_count * 10
                    score += word_score
                    word_scores.append(f"{word}({word_count})")
            
            if word_scores:
                matches.append(f"Words: {', '.join(word_scores)}")
            
            # Position bonus (earlier matches score higher)
            if score > 0:
                position_bonus = max(5 - (chunk_idx * 0.5), 0)
                score += position_bonus
                
                # Extract context around best match
                context = extract_context_from_chunk(chunk, query, 200)
                
                all_results.append({
                    'document': doc,
                    'score': score,
                    'matches': matches,
                    'context': context,
                    'search_type': 'smart',
                    'chunk_index': chunk_idx,
                    'match_id': f"{doc['filename']}_{chunk_idx}"
                })
    
    # Sort by score and limit results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:max_results]

def exact_search_enhanced(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Enhanced exact search that finds all occurrences"""
    all_results = []
    query_lower = query.lower()
    
    for doc in documents:
        if 'text' not in doc or not doc['text']:
            continue
        
        text = doc['text']
        text_lower = text.lower()
        
        # Find all occurrences
        start = 0
        occurrence = 0
        
        while True:
            index = text_lower.find(query_lower, start)
            if index == -1:
                break
            
            occurrence += 1
            
            # Extract context around this match
            context = extract_context_at_position(text, index, query, 200)
            
            all_results.append({
                'document': doc,
                'score': 100 + occurrence,  # Higher score for multiple occurrences
                'matches': [f"Exact match #{occurrence} at position {index}"],
                'context': context,
                'search_type': 'exact',
                'position': index,
                'match_id': f"{doc['filename']}_pos_{index}"
            })
            
            start = index + 1
    
    # Sort by score and limit
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:max_results]

def fuzzy_search_enhanced(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Enhanced fuzzy search with multiple matches per document"""
    all_results = []
    query_words = [word.lower() for word in query.split() if len(word) > 2]
    
    if not query_words:
        return []
    
    for doc in documents:
        if 'text' not in doc or not doc['text']:
            continue
        
        text = doc['text']
        chunks = split_text_into_chunks(text)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            chunk_words = re.findall(r'\b\w+\b', chunk_lower)
            
            score = 0
            matches = []
            
            for query_word in query_words:
                best_similarity = 0
                best_match = ""
                
                for chunk_word in chunk_words:
                    similarity = calculate_similarity(query_word, chunk_word)
                    if similarity > best_similarity and similarity >= 0.7:
                        best_similarity = similarity
                        best_match = chunk_word
                
                if best_similarity > 0:
                    score += best_similarity * 20
                    if best_similarity < 1.0:
                        matches.append(f"{query_word}→{best_match}({best_similarity:.2f})")
                    else:
                        matches.append(f"{query_word}(exact)")
            
            if score > 10:  # Minimum threshold for fuzzy matches
                context = extract_context_from_chunk(chunk, query, 200)
                
                all_results.append({
                    'document': doc,
                    'score': score,
                    'matches': matches,
                    'context': context,
                    'search_type': 'fuzzy',
                    'chunk_index': chunk_idx,
                    'match_id': f"{doc['filename']}_fuzzy_{chunk_idx}"
                })
    
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:max_results]

def split_text_into_chunks(text: str, max_chunk_size: int = 500) -> List[str]:
    """Split text into meaningful chunks (sentences/paragraphs)"""
    # First try to split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
        else:
            # Split long paragraphs by sentences
            sentences = re.split(r'[.!?]+', paragraph)
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk + sentence) <= max_chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def extract_context_from_chunk(chunk: str, query: str, context_length: int = 200) -> str:
    """Extract context from a text chunk"""
    if len(chunk) <= context_length:
        return chunk
    
    query_lower = query.lower()
    chunk_lower = chunk.lower()
    
    # Find best position for query
    index = chunk_lower.find(query_lower)
    if index == -1:
        # If exact query not found, find first query word
        for word in query_lower.split():
            index = chunk_lower.find(word)
            if index != -1:
                break
    
    if index == -1:
        return chunk[:context_length] + "..."
    
    # Extract context around the match
    start = max(0, index - context_length // 2)
    end = min(len(chunk), index + len(query) + context_length // 2)
    
    context = chunk[start:end]
    
    # Add ellipsis if truncated
    if start > 0:
        context = "..." + context
    if end < len(chunk):
        context = context + "..."
    
    return context

def extract_context_at_position(text: str, position: int, query: str, context_length: int = 200) -> str:
    """Extract context around a specific position in text"""
    start = max(0, position - context_length // 2)
    end = min(len(text), position + len(query) + context_length // 2)
    
    context = text[start:end]
    
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."
    
    return context

def calculate_similarity(word1: str, word2: str) -> float:
    """Calculate similarity between two words"""
    if word1 == word2:
        return 1.0
    
    if len(word1) < 3 or len(word2) < 3:
        return 0.0
    
    # Simple character-based similarity
    set1, set2 = set(word1), set(word2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def display_results_grouped(results: List[Dict], query: str, search_time: float):
    """Display results grouped by document"""
    if not results:
        st.warning(f"No results found for '{query}'")
        st.info("Try different search terms or use a different search mode.")
        return
    
    # Group results by document
    doc_groups = {}
    for result in results:
        filename = result['document']['filename']
        if filename not in doc_groups:
            doc_groups[filename] = []
        doc_groups[filename].append(result)
    
    # Sort groups by highest scoring result in each group
    sorted_groups = sorted(doc_groups.items(), 
                          key=lambda x: max(r['score'] for r in x[1]), 
                          reverse=True)
    
    st.success(f"Found {len(results)} result(s) in {len(doc_groups)} document(s) for '{query}' in {search_time:.3f} seconds")
    
    # Display each document group
    for doc_filename, doc_results in sorted_groups:
        doc = doc_results[0]['document']
        highest_score = max(r['score'] for r in doc_results)
        
        with st.expander(f"📄 {doc_filename} ({len(doc_results)} matches, best score: {highest_score:.1f})", 
                        expanded=(len(doc_groups) <= 3)):
            
            # Document metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Type", doc.get('file_type', 'unknown').upper())
            with col2:
                st.metric("Words", f"{doc.get('word_count', 0):,}")
            with col3:
                st.metric("Size", f"{doc.get('file_size_mb', 0):.1f} MB")
            
            # Sort matches within document by score
            doc_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Show all matches from this document
            for i, result in enumerate(doc_results, 1):
                st.markdown(f"**🎯 Match {i} (Score: {result['score']:.1f})**")
                
                if result['matches']:
                    st.info(" | ".join(result['matches']))
                
                # Context with highlighting
                highlighted_context = highlight_terms(result['context'], query)
                st.markdown(highlighted_context, unsafe_allow_html=True)
                
                if i < len(doc_results):
                    st.markdown("---")
            
            # View full document option
            if st.button(f"📖 View Full Document: {doc_filename}", key=f"view_full_{doc_filename}"):
                st.text_area(
                    f"Full content of {doc_filename}:",
                    doc.get('text', 'No content available'),
                    height=400,
                    key=f"full_text_{doc_filename}"
                )

def display_results_flat(results: List[Dict], query: str, search_time: float):
    """Display results in flat list (not grouped)"""
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    st.success(f"Found {len(results)} result(s) for '{query}' in {search_time:.3f} seconds")
    
    for i, result in enumerate(results, 1):
        doc = result['document']
        
        with st.expander(f"📄 {i}. {doc['filename']} (Score: {result['score']:.1f})", 
                        expanded=(i <= 5)):
            
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
            
            # Context with highlighting
            highlighted_context = highlight_terms(result['context'], query)
            st.markdown("**📖 Context:**")
            st.markdown(highlighted_context, unsafe_allow_html=True)

def highlight_terms(text: str, query: str) -> str:
    """Highlight search terms in text"""
    if not text or not query:
        return text
    
    words = query.lower().split()
    highlighted = text
    
    for word in words:
        if len(word) > 2:  # Skip very short words
            # Create case-insensitive pattern that preserves original case
            pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
            highlighted = pattern.sub(r'<mark style="background-color: yellow; padding: 2px;">\1</mark>', highlighted)
    
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
        'avg_search_time': sum(h.get('search_time', 0) for h in history) / max(len(history), 1),
        'recent_queries': [h.get('query', '') for h in history[-10:]]
    }

def log_search(query: str, results: List[Dict], search_time: float, search_type: str):
    """Log search activity for analytics"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    search_entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'result_count': len(results),
        'search_time': search_time,
        'search_type': search_type,
        'unique_documents': len(set(r['document']['filename'] for r in results))
    }
    
    st.session_state.search_history.append(search_entry)
    
    # Keep only last 100 searches
    if len(st.session_state.search_history) > 100:
        st.session_state.search_history = st.session_state.search_history[-100:]

# Backward compatibility functions
def smart_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Backward compatibility wrapper"""
    return smart_search_enhanced(query, documents, max_results)

def exact_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Backward compatibility wrapper"""
    return exact_search_enhanced(query, documents, max_results)

def fuzzy_search(query: str, documents: List[Dict], max_results: int) -> List[Dict]:
    """Backward compatibility wrapper"""
    return fuzzy_search_enhanced(query, documents, max_results)

def display_results(results: List[Dict], query: str, search_time: float):
    """Backward compatibility wrapper"""
    return display_results_grouped(results, query, search_time)

# Export all functions
__all__ = [
    'render_search_interface',
    'smart_search_enhanced',
    'exact_search_enhanced',
    'fuzzy_search_enhanced',
    'display_results_grouped',
    'display_results_flat',
    'get_search_stats',
    'log_search',
    # Backward compatibility
    'smart_search',
    'exact_search',
    'fuzzy_search',
    'display_results'
]
