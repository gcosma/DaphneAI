# modules/ui/search_components.py - Main Interface File
"""
Main search and alignment interfaces for DaphneAI
This file contains the primary user interfaces and core functionality
"""

import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime
from typing import Dict, List, Any
import logging
import difflib

# Import the beautiful display functions from the separate file
from .beautiful_display import (
    display_search_results_beautiful,
    display_alignment_results_beautiful,
    display_manual_search_results_beautiful,
    show_alignment_feature_info_beautiful,
    format_as_beautiful_paragraphs
)

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================

def render_search_interface(documents: List[Dict[str, Any]]):
    """Main search interface with error handling"""
    
    st.header("🔍 Advanced Document Search")
    st.markdown("*Search through your documents with multiple methods*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        return
    
    # Search input
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., recommendations, policy changes, budget allocation",
        help="Enter keywords, phrases, or concepts to search for"
    )
    
    # Search method selection
    st.markdown("### 🎯 Search Method")
    
    search_method = st.radio(
        "Choose your search approach:",
        [
            "🧠 Smart Search - Enhanced keyword matching",
            "🎯 Exact Match - Find exact phrases",
            "🌀 Fuzzy Search - Handle typos and misspellings"
        ],
        index=0,
        help="Different methods find different types of matches"
    )
    
    # Search options
    col1, col2 = st.columns(2)
    
    with col1:
        max_results = st.selectbox(
            "Max results per document", 
            [5, 10, 15, 20, "All"],
            index=1,
            help="Limit results or show all matches"
        )
        
        case_sensitive = st.checkbox("Case sensitive search", value=False)
    
    with col2:
        show_context = st.checkbox("Show context around matches", value=True)
        highlight_matches = st.checkbox("Highlight search terms", value=True)
    
    # Search execution
    if st.button("🔍 Search Documents", type="primary") and query:
        
        start_time = time.time()
        
        with st.spinner(f"🔍 Searching..."):
            
            # Execute search
            results = execute_simple_search(
                documents=documents,
                query=query,
                method=search_method,
                max_results=max_results if max_results != "All" else None,
                case_sensitive=case_sensitive
            )
            
            search_time = time.time() - start_time
            
            # Display results with beautiful formatting
            display_search_results_beautiful(
                results=results,
                query=query,
                search_time=search_time,
                show_context=show_context,
                highlight_matches=highlight_matches
            )

def render_recommendation_alignment_interface(documents: List[Dict[str, Any]]):
    """Fixed recommendation-response alignment interface"""
    
    st.header("🏛️ Recommendation-Response Alignment")
    st.markdown("*Automatically find recommendations and their corresponding responses*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        show_alignment_feature_info_beautiful()
        return
    
    # Simple tab structure to avoid unpacking errors
    tab_selection = st.radio(
        "Choose alignment mode:",
        ["🔄 Auto Alignment", "🔍 Manual Search"],
        horizontal=True
    )
    
    if tab_selection == "🔄 Auto Alignment":
        render_auto_alignment_fixed(documents)
    else:
        render_manual_search_fixed(documents)

def render_auto_alignment_fixed(documents: List[Dict[str, Any]]):
    """Fixed automatic alignment"""
    
    st.markdown("### 🔄 Automatic Recommendation-Response Alignment")
    
    # Configuration
    st.markdown("**📋 Search Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rec_patterns = st.multiselect(
            "Recommendation Keywords",
            ["recommend", "suggest", "advise", "propose", "urge", "should", "must"],
            default=["recommend", "suggest", "advise"],
        )
    
    with col2:
        resp_patterns = st.multiselect(
            "Response Keywords", 
            ["accept", "reject", "agree", "disagree", "implement", "consider", "approved", "declined"],
            default=["accept", "reject", "agree", "implement"],
        )
    
    # Analysis button
    if st.button("🔍 Find & Align Recommendations", type="primary"):
        with st.spinner("🔍 Analyzing documents..."):
            
            try:
                # Find recommendations
                recommendations = find_pattern_matches(documents, rec_patterns, "recommendation")
                
                # Find responses
                responses = find_pattern_matches(documents, resp_patterns, "response")
                
                # Create simple alignments
                alignments = create_simple_alignments(recommendations, responses)
                
                # Display results with beautiful formatting
                display_alignment_results_beautiful(alignments, show_ai_summaries=False)
                
            except Exception as e:
                logger.error(f"Alignment analysis error: {e}")
                st.error(f"Analysis error: {str(e)}")
                
                # Fallback display
                show_basic_pattern_analysis(documents, rec_patterns, resp_patterns)

def render_manual_search_fixed(documents: List[Dict[str, Any]]):
    """Fixed manual search"""
    
    st.markdown("### 🔍 Manual Sentence Search")
    
    # Search input
    search_sentence = st.text_area(
        "📝 Paste your sentence here:",
        placeholder="e.g., 'We recommend implementing new security protocols'",
        height=100
    )
    
    # Search options
    col1, col2 = st.columns(2)
    
    with col1:
        search_type = st.selectbox(
            "Search for:",
            ["Similar content", "Recommendations", "Responses"]
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold", 0.1, 1.0, 0.3, 0.1
        )
    
    with col2:
        max_matches = st.selectbox("Max matches", [5, 10, 20, 50])
        show_scores = st.checkbox("Show similarity scores", True)
    
    # Search execution
    if st.button("🔎 Find Matches", type="primary") and search_sentence.strip():
        
        with st.spinner("🔍 Searching..."):
            
            try:
                matches = find_similar_content(
                    documents, search_sentence, search_type, 
                    similarity_threshold, max_matches
                )
                
                display_manual_search_results_beautiful(
                    matches, search_sentence, 0.1, show_scores, search_type.lower()
                )
                
            except Exception as e:
                logger.error(f"Manual search error: {e}")
                st.error(f"Search error: {str(e)}")

# =============================================================================
# SEARCH EXECUTION FUNCTIONS
# =============================================================================

def execute_simple_search(documents: List[Dict], query: str, method: str, 
                         max_results: int = None, case_sensitive: bool = False) -> List[Dict]:
    """Simple search implementation"""
    
    results = []
    
    # Process query
    search_query = query if case_sensitive else query.lower()
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        search_text = text if case_sensitive else text.lower()
        
        # Find matches based on method
        if "Smart" in method:
            matches = smart_search_simple(text, search_text, query, search_query)
        elif "Exact" in method:
            matches = exact_search_simple(text, search_text, query, search_query)
        elif "Fuzzy" in method:
            matches = fuzzy_search_simple(text, search_text, query, search_query)
        else:
            matches = smart_search_simple(text, search_text, query, search_query)
        
        # Limit results if specified
        if max_results:
            matches = matches[:max_results]
        
        # Add document info
        for match in matches:
            match['document'] = doc
            results.append(match)
    
    # Sort by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return results

def smart_search_simple(text: str, search_text: str, query: str, search_query: str) -> List[Dict]:
    """Simple smart search"""
    
    matches = []
    
    # Split into words
    query_words = [w for w in search_query.split() if len(w) > 2]
    
    if not query_words:
        return matches
    
    # Find sentences containing query words
    sentences = re.split(r'[.!?]+', text)
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        sentence_lower = sentence.lower()
        word_matches = sum(1 for word in query_words if word in sentence_lower)
        
        if word_matches > 0:
            # Calculate score
            score = (word_matches / len(query_words)) * 100
            
            # Find position
            pos = text.find(sentence.strip())
            if pos == -1:
                pos = i * 100  # Estimate
            
            # Create match
            match = {
                'position': pos,
                'matched_text': sentence.strip(),
                'context': get_context_simple(sentences, i),
                'score': score,
                'match_type': 'smart',
                'page_number': max(1, pos // 2000 + 1),
                'word_matches': word_matches,
                'total_words': len(query_words),
                'percentage_through': (pos / len(text)) * 100 if text else 0
            }
            
            matches.append(match)
    
    return matches

def exact_search_simple(text: str, search_text: str, query: str, search_query: str) -> List[Dict]:
    """Simple exact search"""
    
    matches = []
    start = 0
    
    while True:
        pos = search_text.find(search_query, start)
        if pos == -1:
            break
        
        # Extract context
        context_start = max(0, pos - 100)
        context_end = min(len(text), pos + len(query) + 100)
        context = text[context_start:context_end]
        
        match = {
            'position': pos,
            'matched_text': text[pos:pos + len(query)],
            'context': context,
            'score': 100.0,
            'match_type': 'exact',
            'page_number': max(1, pos // 2000 + 1),
            'percentage_through': (pos / len(text)) * 100 if text else 0
        }
        
        matches.append(match)
        start = pos + 1
    
    return matches

def fuzzy_search_simple(text: str, search_text: str, query: str, search_query: str) -> List[Dict]:
    """Simple fuzzy search"""
    
    matches = []
    words = text.split()
    query_words = search_query.split()
    
    for query_word in query_words:
        if len(query_word) < 3:
            continue
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Simple similarity check
            if query_word in word_lower or word_lower in query_word:
                similarity = len(set(query_word) & set(word_lower)) / len(set(query_word) | set(word_lower))
                
                if similarity > 0.5:
                    # Find position
                    pos = len(' '.join(words[:i]))
                    if i > 0:
                        pos += 1
                    
                    # Extract context
                    context_start = max(0, i - 10)
                    context_end = min(len(words), i + 10)
                    context = ' '.join(words[context_start:context_end])
                    
                    match = {
                        'position': pos,
                        'matched_text': word,
                        'context': context,
                        'score': similarity * 100,
                        'match_type': 'fuzzy',
                        'page_number': max(1, pos // 2000 + 1),
                        'similarity': similarity,
                        'percentage_through': (pos / len(text)) * 100 if text else 0
                    }
                    
                    matches.append(match)
    
    # Remove duplicates and sort
    unique_matches = []
    seen_positions = set()
    
    for match in sorted(matches, key=lambda x: x['score'], reverse=True):
        pos = match['position']
        if pos not in seen_positions:
            unique_matches.append(match)
            seen_positions.add(pos)
    
    return unique_matches

# =============================================================================
# ALIGNMENT FUNCTIONS
# =============================================================================

def find_pattern_matches(documents: List[Dict], patterns: List[str], match_type: str) -> List[Dict]:
    """Find pattern matches in documents"""
    
    matches = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        text_lower = text.lower()
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            
            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(pattern_lower, start)
                if pos == -1:
                    break
                
                # Extract sentence containing the pattern
                sentence_start = max(0, text.rfind('.', 0, pos) + 1)
                sentence_end = text.find('.', pos)
                if sentence_end == -1:
                    sentence_end = len(text)
                
                sentence = text[sentence_start:sentence_end].strip()
                
                # Get context
                sentences = re.split(r'[.!?]+', text)
                sentence_index = -1
                for i, sent in enumerate(sentences):
                    if sentence in sent:
                        sentence_index = i
                        break
                
                context = get_context_simple(sentences, sentence_index, 2) if sentence_index != -1 else sentence
                
                match = {
                    'id': f"{match_type}_{len(matches) + 1}",
                    'document': doc,
                    'pattern': pattern,
                    'sentence': sentence,
                    'context': context,
                    'position': pos,
                    'page_number': max(1, pos // 2000 + 1),
                    'match_type': match_type,
                    'recommendation_type': classify_content_type(sentence),
                    'response_type': classify_content_type(sentence)
                }
                
                matches.append(match)
                start = pos + 1
    
    return matches

def create_simple_alignments(recommendations: List[Dict], responses: List[Dict]) -> List[Dict]:
    """Create simple alignments between recommendations and responses"""
    
    alignments = []
    
    for rec in recommendations:
        # Find responses in the same document
        rec_doc = rec['document']['filename']
        related_responses = [r for r in responses if r['document']['filename'] == rec_doc]
        
        # Simple similarity scoring
        best_responses = []
        for resp in related_responses:
            similarity = calculate_simple_similarity(rec['sentence'], resp['sentence'])
            if similarity > 0.2:
                best_responses.append({
                    'response': resp,
                    'combined_score': similarity,
                    'similarity_score': similarity,
                    'topic_similarity': similarity
                })
        
        # Sort by similarity
        best_responses.sort(key=lambda x: x['combined_score'], reverse=True)
        
        alignment = {
            'recommendation': rec,
            'responses': best_responses[:3],  # Top 3
            'alignment_confidence': best_responses[0]['combined_score'] if best_responses else 0,
            'alignment_status': determine_alignment_status(best_responses)
        }
        
        alignments.append(alignment)
    
    return alignments

def find_similar_content(documents: List[Dict], target_sentence: str, search_type: str, 
                        threshold: float, max_matches: int) -> List[Dict]:
    """Find similar content to target sentence"""
    
    matches = []
    target_words = set(target_sentence.lower().split())
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip() or len(sentence.strip()) < 20:
                continue
            
            sentence_words = set(sentence.lower().split())
            
            # Calculate similarity
            intersection = len(target_words & sentence_words)
            union = len(target_words | sentence_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                
                # Filter by type if specified
                if search_type == "Recommendations":
                    if not any(word in sentence.lower() for word in ['recommend', 'suggest', 'advise']):
                        continue
                elif search_type == "Responses":
                    if not any(word in sentence.lower() for word in ['accept', 'reject', 'agree', 'implement']):
                        continue
                
                # Get context
                context = get_context_simple(sentences, i, 2)
                
                match = {
                    'sentence': sentence.strip(),
                    'context': context,
                    'similarity_score': similarity,
                    'document': doc,
                    'position': text.find(sentence),
                    'page_number': max(1, text.find(sentence) // 2000 + 1) if sentence in text else 1,
                    'content_type': classify_content_type(sentence),
                    'matched_patterns': []
                }
                
                matches.append(match)
    
    # Sort by similarity and limit
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matches[:max_matches]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_context_simple(sentences: List[str], index: int, window: int = 1) -> str:
    """Get context around a sentence"""
    
    start = max(0, index - window)
    end = min(len(sentences), index + window + 1)
    
    context_sentences = [s.strip() for s in sentences[start:end] if s.strip()]
    return ' '.join(context_sentences)

def calculate_simple_similarity(text1: str, text2: str) -> float:
    """Calculate simple word overlap similarity"""
    
    words1 = set(w.lower() for w in text1.split() if len(w) > 2)
    words2 = set(w.lower() for w in text2.split() if len(w) > 2)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def determine_alignment_status(responses: List[Dict]) -> str:
    """Determine alignment status"""
    
    if not responses:
        return "No Response Found"
    
    best_score = responses[0]['combined_score']
    
    if best_score > 0.7:
        return "Strong Alignment"
    elif best_score > 0.5:
        return "Good Alignment"
    elif best_score > 0.3:
        return "Weak Alignment"
    else:
        return "Poor Alignment"

def classify_content_type(sentence: str) -> str:
    """Classify content type"""
    
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['urgent', 'immediate', 'critical']):
        return 'Urgent'
    elif any(word in sentence_lower for word in ['policy', 'framework', 'guideline']):
        return 'Policy'
    elif any(word in sentence_lower for word in ['financial', 'budget', 'cost']):
        return 'Financial'
    else:
        return 'General'

def show_basic_pattern_analysis(documents: List[Dict], rec_patterns: List[str], resp_patterns: List[str]):
    """Show basic pattern analysis as fallback"""
    
    st.markdown("### 📊 Basic Pattern Analysis")
    
    total_rec = 0
    total_resp = 0
    
    for doc in documents:
        text = doc.get('text', '').lower()
        
        for pattern in rec_patterns:
            total_rec += text.count(pattern.lower())
        
        for pattern in resp_patterns:
            total_resp += text.count(pattern.lower())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Recommendation Pattern Matches", total_rec)
        
        st.markdown("**Patterns Found:**")
        for pattern in rec_patterns:
            count = sum(doc.get('text', '').lower().count(pattern.lower()) for doc in documents)
            if count > 0:
                st.write(f"• '{pattern}': {count}")
    
    with col2:
        st.metric("Response Pattern Matches", total_resp)
        
        st.markdown("**Patterns Found:**")
        for pattern in resp_patterns:
            count = sum(doc.get('text', '').lower().count(pattern.lower()) for doc in documents)
            if count > 0:
                st.write(f"• '{pattern}': {count}")

# =============================================================================
# COMPATIBILITY FUNCTIONS
# =============================================================================

def check_rag_availability() -> bool:
    """Check if AI features are available"""
    try:
        import sentence_transformers
        import torch
        return True
    except ImportError:
        return False

def filter_stop_words(query: str) -> str:
    """Basic stop word filtering"""
    stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                  'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                  'to', 'was', 'will', 'with', 'but', 'they', 'have', 'had', 'what'}
    
    words = query.lower().split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    return ' '.join(filtered_words) if filtered_words else query

# Compatibility exports
STOP_WORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
              'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
              'to', 'was', 'will', 'with', 'but', 'they', 'have', 'had', 'what'}

# Export all functions for compatibility
__all__ = [
    'render_search_interface',
    'render_recommendation_alignment_interface',
    'check_rag_availability',
    'filter_stop_words',
    'STOP_WORDS'
]
