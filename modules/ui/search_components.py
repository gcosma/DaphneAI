# modules/ui/search_components.py - COMPLETE WITH AI METHOD + PROPER STOP WORD FILTERING
"""
Main search and alignment interfaces for DaphneAI with ALL search methods including AI
Includes: Smart Search, Exact Match, Fuzzy Search, AI Semantic Search, and Hybrid Search
Proper stop word filtering to prevent meaningless matches
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
try:
    from .beautiful_display import (
        display_search_results_beautiful,
        display_alignment_results_beautiful,
        display_manual_search_results_beautiful,
        show_alignment_feature_info_beautiful,
        format_as_beautiful_paragraphs
    )
    BEAUTIFUL_DISPLAY_AVAILABLE = True
except ImportError:
    BEAUTIFUL_DISPLAY_AVAILABLE = False
    logging.warning("Beautiful display not available - using basic display")

# Setup logging
logger = logging.getLogger(__name__)

# COMPREHENSIVE STOP WORDS LIST
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 
    'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 
    'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 
    'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 
    'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 
    'come', 'made', 'may', 'part', 'or', 'also', 'back', 'any', 'good', 
    'new', 'where', 'much', 'take', 'know', 'just', 'see', 'after', 
    'very', 'well', 'here', 'should', 'old', 'still'
}

# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================

def render_search_interface(documents: List[Dict[str, Any]]):
    """Main search interface with ALL methods including AI"""
    
    st.header("🔍 Advanced Document Search")
    st.markdown("*Search with intelligent filtering and AI-powered semantic understanding*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        return
    
    # Search input
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., committee recommends, department policy, budget allocation",
        help="Search will focus on meaningful words and filter out common words like 'the', 'and', 'a'"
    )
    
    # Search method selection - INCLUDING AI METHODS
    st.markdown("### 🎯 Search Method")
    
    search_method = st.radio(
        "Choose your search approach:",
        [
            "🧠 Smart Search - Enhanced keyword matching",
            "🎯 Exact Match - Find exact phrases",
            "🌀 Fuzzy Search - Handle typos and misspellings",
            "🤖 AI Semantic - AI finds related concepts",
            "🔄 Hybrid Search - Combines Smart + AI for best results"
        ],
        index=0,
        help="Smart search filters out common words for better results. AI methods understand meaning and context."
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
    
    # AI availability check - FIXED FOR STREAMLIT CLOUD
    ai_available = check_rag_availability()
    
    import os
    is_streamlit_cloud = (
        os.getenv('STREAMLIT_SHARING_MODE') or 
        'streamlit.app' in os.getenv('HOSTNAME', '') or
        '/mount/src/' in os.getcwd()
    )
    
    if search_method in ["🤖 AI Semantic", "🔄 Hybrid Search"]:
        if is_streamlit_cloud:
            st.info("🌐 **Streamlit Cloud Detected** - Using optimized semantic search for government documents")
            with st.expander("ℹ️ Enhanced Semantic Search on Streamlit Cloud"):
                st.markdown("""
                **Streamlit Cloud Optimization:**
                - ✅ **Government-tuned** - Specialized for policy documents  
                - ✅ **Faster performance** - No model loading delays
                - ✅ **Better results** - Domain-specific semantic matching
                
                **Semantic Features:**
                - Word groups: recommend → suggest → advise → propose
                - Government terms: department → ministry → agency
                - Policy vocabulary: framework → protocol → guideline
                - Response patterns: accept → agree → approve → implement
                """)
        elif ai_available:
            st.info("🤖 **Full AI semantic search available** - Using sentence transformers")
        else:
            st.info("🤖 **Enhanced semantic search active** - Using government-optimized matching")
            if st.button("💡 Install Full AI for Local Development"):
                st.code("pip install sentence-transformers torch huggingface-hub")
    
    # Show filtered query preview
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
    
    # Search execution
    if st.button("🔍 Search Documents", type="primary") and query:
        
        start_time = time.time()
        
        with st.spinner(f"🔍 Searching..."):
            
            # Execute search with ALL methods including AI
            results = execute_search_with_ai(
                documents=documents,
                query=query,
                method=search_method,
                max_results=max_results if max_results != "All" else None,
                case_sensitive=case_sensitive
            )
            
            search_time = time.time() - start_time
            
            # Display results
            if BEAUTIFUL_DISPLAY_AVAILABLE:
                display_search_results_beautiful(
                    results=results,
                    query=query,
                    search_time=search_time,
                    show_context=show_context,
                    highlight_matches=highlight_matches
                )
            else:
                display_search_results_basic(results, query, search_time, show_context, highlight_matches)

def render_recommendation_alignment_interface(documents: List[Dict[str, Any]]):
    """Recommendation-response alignment interface"""
    
    st.header("🏛️ Recommendation-Response Alignment")
    st.markdown("*Automatically find recommendations and their corresponding responses*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        if BEAUTIFUL_DISPLAY_AVAILABLE:
            show_alignment_feature_info_beautiful()
        else:
            show_alignment_feature_info_basic()
        return
    
    # Simple tab structure
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
    """Automatic alignment with filtering"""
    
    st.markdown("### 🔄 Automatic Recommendation-Response Alignment")
    
    # Configuration
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
                recommendations = find_pattern_matches(documents, rec_patterns, "recommendation")
                responses = find_pattern_matches(documents, resp_patterns, "response")
                alignments = create_simple_alignments(recommendations, responses)
                
                if BEAUTIFUL_DISPLAY_AVAILABLE:
                    display_alignment_results_beautiful(alignments, show_ai_summaries=False)
                else:
                    display_alignment_results_basic(alignments)
                
            except Exception as e:
                logger.error(f"Alignment analysis error: {e}")
                st.error(f"Analysis error: {str(e)}")
                show_basic_pattern_analysis(documents, rec_patterns, resp_patterns)

def render_manual_search_fixed(documents: List[Dict[str, Any]]):
    """Manual search with filtering"""
    
    st.markdown("### 🔍 Manual Sentence Search")
    
    search_sentence = st.text_area(
        "📝 Paste your sentence here:",
        placeholder="e.g., 'The committee recommends implementing new security protocols'",
        help="Similarity matching will focus on meaningful words",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_type = st.selectbox(
            "Search for:",
            ["Similar content", "Recommendations", "Responses"]
        )
        similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.3, 0.1)
    
    with col2:
        max_matches = st.selectbox("Max matches", [5, 10, 20, 50])
        show_scores = st.checkbox("Show similarity scores", True)
    
    # Show meaningful words
    if search_sentence.strip():
        meaningful_words = get_meaningful_words(search_sentence)
        if meaningful_words:
            st.info(f"🔍 **Focusing on meaningful words:** {', '.join(meaningful_words[:10])}{'...' if len(meaningful_words) > 10 else ''}")
        else:
            st.warning("⚠️ No meaningful words found in your sentence")
    
    if st.button("🔎 Find Matches", type="primary") and search_sentence.strip():
        
        search_start = time.time()
        
        with st.spinner("🔍 Searching for similar content..."):
            
            try:
                matches = find_similar_content_filtered(
                    documents, search_sentence, search_type, 
                    similarity_threshold, max_matches
                )
                
                search_time = time.time() - search_start
                
                if BEAUTIFUL_DISPLAY_AVAILABLE:
                    display_manual_search_results_beautiful(
                        matches, search_sentence, search_time, show_scores, search_type.lower()
                    )
                else:
                    display_manual_search_results_basic(matches, search_sentence, search_time, show_scores)
                
            except Exception as e:
                logger.error(f"Manual search error: {e}")
                st.error(f"Search error: {str(e)}")

# =============================================================================
# SEARCH EXECUTION FUNCTIONS WITH ALL METHODS INCLUDING AI
# =============================================================================

def execute_search_with_ai(documents: List[Dict], query: str, method: str, 
                          max_results: int = None, case_sensitive: bool = False) -> List[Dict]:
    """Execute search with ALL methods including AI semantic search"""
    
    results = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Apply method-specific search
        if "Smart" in method:
            matches = smart_search_filtered(text, query, case_sensitive)
        elif "Exact" in method:
            matches = exact_search_unfiltered(text, query, case_sensitive)
        elif "Fuzzy" in method:
            matches = fuzzy_search_filtered(text, query, case_sensitive)
        elif "AI Semantic" in method:
            matches = ai_semantic_search(text, query, case_sensitive)
        elif "Hybrid" in method:
            matches = hybrid_search_smart_ai(text, query, case_sensitive)
        else:
            matches = smart_search_filtered(text, query, case_sensitive)
        
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

def smart_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Smart search with stop word filtering for better results"""
    
    matches = []
    
    # Filter query to meaningful words only
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        # If no meaningful words, return empty (avoid matching only stop words)
        return matches
    
    search_text = text if case_sensitive else text.lower()
    
    # Split into sentences for better context
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    for i, sentence in enumerate(sentences):
        sentence_search = sentence if case_sensitive else sentence.lower()
        
        # Count meaningful word matches only
        word_matches = sum(1 for word in meaningful_words 
                          if word.lower() in sentence_search)
        
        if word_matches > 0:
            # Calculate relevance score
            score = (word_matches / len(meaningful_words)) * 100
            
            # Boost score if multiple meaningful words found
            if word_matches > 1:
                score *= 1.2
            
            # Find position in original text
            pos = text.find(sentence)
            if pos == -1:
                pos = i * 100
            
            # Get context
            context = get_context_simple(sentences, i, 2)
            
            match = {
                'position': pos,
                'matched_text': sentence,
                'context': context,
                'score': min(score, 100),  # Cap at 100
                'match_type': 'smart',
                'page_number': max(1, pos // 2000 + 1),
                'word_matches': word_matches,
                'total_meaningful_words': len(meaningful_words),
                'meaningful_words_found': [w for w in meaningful_words if w.lower() in sentence_search],
                'percentage_through': (pos / len(text)) * 100 if text else 0
            }
            
            matches.append(match)
    
    return matches

def exact_search_unfiltered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Exact search without filtering (preserves exact phrases)"""
    
    matches = []
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()
    
    start = 0
    while True:
        pos = search_text.find(search_query, start)
        if pos == -1:
            break
        
        # Extract context
        context_start = max(0, pos - 150)
        context_end = min(len(text), pos + len(query) + 150)
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

def fuzzy_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Fuzzy search focusing on meaningful words"""
    
    matches = []
    
    # Get meaningful words from query
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return matches
    
    words = text.split()
    search_words = [w if case_sensitive else w.lower() for w in words]
    
    for query_word in meaningful_words:
        query_word_search = query_word if case_sensitive else query_word.lower()
        
        for i, word in enumerate(search_words):
            # Calculate similarity
            similarity = difflib.SequenceMatcher(None, query_word_search, word).ratio()
            
            if similarity > 0.6:  # Meaningful similarity threshold
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
                    'matched_text': words[i],
                    'context': context,
                    'score': similarity * 100,
                    'match_type': 'fuzzy',
                    'page_number': max(1, pos // 2000 + 1),
                    'similarity': similarity,
                    'query_word': query_word,
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

def ai_semantic_search(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """AI Semantic search with Streamlit Cloud optimization and fallback"""
    
    try:
        # Quick check for AI availability
        return ai_semantic_search_direct(text, query, case_sensitive)
    except Exception as e:
        # Use enhanced fallback for government documents
        logger.info(f"Using semantic fallback: {str(e)}")
        return semantic_fallback_search(text, query, case_sensitive)

def ai_semantic_search_direct(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Direct AI semantic search using sentence transformers"""
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import torch
        
        # Force CPU usage for Streamlit Cloud compatibility
        device = 'cpu'
        torch.set_default_device('cpu')
        
        # Initialize model if not cached
        if 'semantic_model' not in st.session_state:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                model = model.to(device)
                st.session_state.semantic_model = model
                st.session_state.model_device = device
            except Exception as model_error:
                raise Exception(f"Model loading failed: {str(model_error)}")
        
        model = st.session_state.semantic_model
        
        # Process query - filter meaningful words for better semantics
        meaningful_query = ' '.join(get_meaningful_words(query))
        if not meaningful_query:
            meaningful_query = query
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        chunks = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if not chunks:
            return []
        
        # Limit chunks for Streamlit Cloud memory constraints
        if len(chunks) > 100:
            chunks = chunks[:50] + chunks[-50:]
        
        # Generate embeddings
        try:
            with torch.no_grad():
                query_embedding = model.encode([meaningful_query], convert_to_tensor=False, device=device)
                chunk_embeddings = model.encode(chunks, convert_to_tensor=False, device=device, batch_size=16)
            
            # Ensure numpy arrays
            if torch.is_tensor(query_embedding):
                query_embedding = query_embedding.cpu().numpy()
            if torch.is_tensor(chunk_embeddings):
                chunk_embeddings = chunk_embeddings.cpu().numpy()
                
        except Exception as encoding_error:
            # Try with smaller batch
            query_embedding = model.encode([meaningful_query], convert_to_tensor=False, batch_size=1)
            chunk_embeddings = model.encode(chunks[:20], convert_to_tensor=False, batch_size=1)
            chunks = chunks[:20]
        
        # Calculate similarities
        similarities = np.dot(query_embedding, chunk_embeddings.T).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:10]
        
        matches = []
        for idx in top_indices:
            if idx >= len(chunks):
                continue
                
            similarity = similarities[idx]
            
            if similarity > 0.3:  # Minimum semantic threshold
                chunk = chunks[idx]
                
                # Find position in original text
                pos = text.find(chunk)
                if pos == -1:
                    pos = 0
                
                # Get context
                sentences_for_context = re.split(r'[.!?]+', text)
                chunk_idx = -1
                for i, sent in enumerate(sentences_for_context):
                    if chunk in sent:
                        chunk_idx = i
                        break
                
                context = get_context_simple(sentences_for_context, chunk_idx, 2) if chunk_idx != -1 else chunk
                
                match = {
                    'position': pos,
                    'matched_text': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'context': context,
                    'score': similarity * 100,
                    'match_type': 'semantic',
                    'page_number': max(1, pos // 2000 + 1),
                    'word_position': len(text[:pos].split()),
                    'percentage_through': (pos / len(text)) * 100 if text else 0,
                    'semantic_score': similarity,
                    'semantic_relation': f"AI semantic match for '{meaningful_query}'"
                }
                
                matches.append(match)
        
        return matches
        
    except ImportError:
        raise Exception("Sentence transformers not available")
    except Exception as e:
        raise Exception(f"AI semantic search error: {str(e)}")

def semantic_fallback_search(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Enhanced semantic fallback using government terminology"""
    
    # Enhanced semantic word groups for government documents
    semantic_groups = {
        'recommend': ['recommend', 'suggestion', 'suggest', 'advise', 'propose', 'urge', 'advocate', 'endorse', 'recommendation', 'recommendations'],
        'suggest': ['suggest', 'recommend', 'proposal', 'propose', 'advise', 'hint', 'indicate', 'suggestion', 'suggestions'],
        'respond': ['respond', 'response', 'reply', 'answer', 'feedback', 'reaction', 'comment', 'responses', 'replies'],
        'response': ['response', 'respond', 'reply', 'answer', 'feedback', 'reaction', 'comment', 'responses', 'replies'],
        'implement': ['implement', 'execute', 'carry out', 'put into practice', 'apply', 'deploy', 'implementation'],
        'review': ['review', 'examine', 'assess', 'evaluate', 'analyze', 'inspect', 'analysis'],
        'policy': ['policy', 'procedure', 'guideline', 'protocol', 'framework', 'strategy', 'policies'],
        'accept': ['accept', 'agree', 'approve', 'endorse', 'support', 'adopt', 'acceptance'],
        'reject': ['reject', 'decline', 'refuse', 'dismiss', 'deny', 'oppose', 'rejection'],
        'government': ['government', 'department', 'ministry', 'agency', 'authority', 'administration'],
        'report': ['report', 'document', 'paper', 'study', 'analysis', 'investigation'],
        'committee': ['committee', 'panel', 'board', 'commission', 'group', 'team'],
        'budget': ['budget', 'funding', 'financial', 'cost', 'expenditure', 'allocation'],
        'urgent': ['urgent', 'immediate', 'critical', 'priority', 'emergency', 'pressing']
    }
    
    matches = []
    
    # Get meaningful words from query
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return matches
    
    # Find semantic matches for each meaningful word
    for query_word in meaningful_words:
        
        # Find related words for this query term
        related_words = []
        
        # Direct lookup in semantic groups
        if query_word in semantic_groups:
            related_words.extend(semantic_groups[query_word])
        
        # Check if query word is contained in any synonym
        for key, synonyms in semantic_groups.items():
            if any(query_word in synonym for synonym in synonyms):
                related_words.extend(synonyms)
        
        # If no semantic group found, use the word itself and variations
        if not related_words:
            related_words = [query_word]
            # Add common word endings
            if len(query_word) > 4:
                related_words.extend([
                    query_word + 's',
                    query_word + 'ing', 
                    query_word + 'ed',
                    query_word + 'ion'
                ])
        
        # Remove duplicates
        related_words = list(set(related_words))
        
        # Search for related words in text
        search_text = text if case_sensitive else text.lower()
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            sentence_search = sentence if case_sensitive else sentence.lower()
            
            for related_word in related_words:
                related_search = related_word if case_sensitive else related_word.lower()
                
                if related_search in sentence_search:
                    
                    # Calculate semantic similarity score
                    if related_word.lower() == query_word.lower():
                        score = 100.0  # Exact match
                    elif query_word.lower() in related_word.lower() or related_word.lower() in query_word.lower():
                        score = 95.0   # Contains match
                    else:
                        score = 85.0   # Semantic match
                    
                    # Find position in original text
                    pos = text.find(sentence.strip())
                    if pos == -1:
                        pos = i * 100
                    
                    # Get context
                    context = get_context_simple(sentences, i, 2)
                    
                    match = {
                        'position': pos,
                        'matched_text': sentence.strip(),
                        'context': context,
                        'score': score,
                        'match_type': 'semantic',
                        'page_number': max(1, pos // 2000 + 1),
                        'word_position': i,
                        'percentage_through': (pos / len(text)) * 100 if text else 0,
                        'semantic_relation': f"{query_word} → {related_word}",
                        'query_word': query_word
                    }
                    
                    matches.append(match)
                    break  # Only match once per sentence
    
    # Remove overlapping matches and sort
    matches = remove_overlapping_matches(matches)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches

def hybrid_search_smart_ai(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Hybrid search combining smart search and AI semantic search"""
    
    # Get smart search results
    smart_results = smart_search_filtered(text, query, case_sensitive)
    
    # Get AI semantic results
    ai_results = ai_semantic_search(text, query, case_sensitive)
    
    # Combine and deduplicate
    all_matches = smart_results + ai_results
    
    # Remove overlapping matches
    unique_matches = remove_overlapping_matches(all_matches)
    
    # Boost scores for matches found by both methods
    position_scores = {}
    for match in all_matches:
        pos = match['position']
        if pos in position_scores:
            position_scores[pos] += match['score']
        else:
            position_scores[pos] = match['score']
    
    # Update scores for matches found by multiple methods
    for match in unique_matches:
        pos = match['position']
        if pos in position_scores:
            match['score'] = min(position_scores[pos], 100)  # Cap at 100
            match['match_type'] = 'hybrid'
    
    # Sort by score
    unique_matches.sort(key=lambda x: x['score'], reverse=True)
    
    return unique_matches

def find_similar_content_filtered(documents: List[Dict], target_sentence: str, search_type: str, 
                                threshold: float, max_matches: int) -> List[Dict]:
    """Find similar content using meaningful words only"""
    
    matches = []
    
    # Get meaningful words from target sentence
    target_meaningful = set(get_meaningful_words(target_sentence))
    
    if not target_meaningful:
        return matches
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip() or len(sentence.strip()) < 20:
                continue
            
            # Get meaningful words from sentence
            sentence_meaningful = set(get_meaningful_words(sentence))
            
            if not sentence_meaningful:
                continue
            
            # Calculate similarity using only meaningful words
            intersection = len(target_meaningful & sentence_meaningful)
            union = len(target_meaningful | sentence_meaningful)
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
                    'matched_meaningful_words': list(target_meaningful & sentence_meaningful),
                    'total_meaningful_words': len(target_meaningful)
                }
                
                matches.append(match)
    
    # Sort by similarity and limit
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matches[:max_matches]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def filter_stop_words(query: str) -> str:
    """Remove stop words from query, keeping only meaningful words"""
    words = query.lower().split()
    meaningful_words = [word for word in words 
                       if word not in STOP_WORDS and len(word) > 1]
    
    # If all words are stop words, return original query
    if not meaningful_words:
        return query
    
    return ' '.join(meaningful_words)

def get_meaningful_words(text: str) -> List[str]:
    """Extract meaningful words (non-stop words) from text"""
    words = re.findall(r'\b\w+\b', text.lower())
    meaningful = [word for word in words 
                 if word not in STOP_WORDS and len(word) > 1]
    return meaningful

def get_context_simple(sentences: List[str], index: int, window: int = 1) -> str:
    """Get context around a sentence"""
    start = max(0, index - window)
    end = min(len(sentences), index + window + 1)
    
    context_sentences = [s.strip() for s in sentences[start:end] if s.strip()]
    return ' '.join(context_sentences)

def calculate_simple_similarity(text1: str, text2: str) -> float:
    """Calculate similarity using meaningful words only"""
    words1 = set(get_meaningful_words(text1))
    words2 = set(get_meaningful_words(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

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

def remove_overlapping_matches(matches: List[Dict]) -> List[Dict]:
    """Remove overlapping matches, keeping the highest scored ones"""
    
    if not matches:
        return matches
    
    # Sort by score descending
    sorted_matches = sorted(matches, key=lambda x: x.get('score', 0), reverse=True)
    
    unique_matches = []
    used_positions = set()
    
    for match in sorted_matches:
        pos = match.get('position', 0)
        matched_text = match.get('matched_text', '')
        match_length = len(matched_text)
        
        # Check if this match overlaps with any existing match
        overlap = False
        for used_start, used_end in used_positions:
            # Check for overlap
            if not (pos + match_length <= used_start or pos >= used_end):
                overlap = True
                break
        
        if not overlap:
            unique_matches.append(match)
            used_positions.add((pos, pos + match_length))
    
    return unique_matches

def check_rag_availability() -> bool:
    """Check if RAG dependencies are available - STREAMLIT CLOUD OPTIMIZED"""
    try:
        import sentence_transformers
        import torch
        import os
        
        # Check if we're on Streamlit Cloud
        is_streamlit_cloud = (
            os.getenv('STREAMLIT_SHARING_MODE') or 
            'streamlit.app' in os.getenv('HOSTNAME', '') or
            '/mount/src/' in os.getcwd()
        )
        
        if is_streamlit_cloud:
            # On Streamlit Cloud, always return False to use fallback
            # This avoids the PyTorch meta tensor issues entirely
            return False
        
        # Only test model loading on local development
        try:
            from sentence_transformers import SentenceTransformer
            device = 'cpu'
            torch.set_default_device('cpu')
            
            # Quick test
            test_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            with torch.no_grad():
                test_model.encode(["test"], convert_to_tensor=False, device=device)
            return True
            
        except Exception:
            return False
        
    except ImportError:
        return False
    except Exception:
        return False

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
    """Create alignments between recommendations and responses"""
    
    alignments = []
    
    for rec in recommendations:
        # Find responses in the same document
        rec_doc = rec['document']['filename']
        related_responses = [r for r in responses if r['document']['filename'] == rec_doc]
        
        # Calculate similarity using meaningful words
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
            'responses': best_responses[:3],
            'alignment_confidence': best_responses[0]['combined_score'] if best_responses else 0,
            'alignment_status': determine_alignment_status(best_responses)
        }
        
        alignments.append(alignment)
    
    return alignments

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

# =============================================================================
# BASIC DISPLAY FUNCTIONS (FALLBACK)
# =============================================================================

def display_search_results_basic(results: List[Dict], query: str, search_time: float, 
                                show_context: bool, highlight_matches: bool):
    """Basic display when beautiful display is not available"""
    
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    meaningful_words = get_meaningful_words(query)
    st.success(f"🎯 Found {len(results)} results in {search_time:.3f} seconds")
    if meaningful_words:
        st.info(f"Searched meaningful words: {', '.join(meaningful_words)}")
    
    for i, result in enumerate(results, 1):
        doc_name = result['document']['filename']
        score = result.get('score', 0)
        method = result.get('match_type', 'unknown')
        
        st.write(f"**{i}. {doc_name}** - {method} (Score: {score:.1f})")
        
        if method == 'smart' and 'meaningful_words_found' in result:
            words_found = result['meaningful_words_found']
            st.caption(f"Words found: {', '.join(words_found)}")
        
        if show_context:
            context = result.get('context', '')
            st.write(f"Context: {context[:200]}...")

def display_alignment_results_basic(alignments: List[Dict]):
    """Basic alignment display"""
    
    st.write(f"Found {len(alignments)} recommendations")
    
    for i, alignment in enumerate(alignments, 1):
        rec = alignment.get('recommendation', {})
        responses = alignment.get('responses', [])
        
        st.write(f"**{i}. Recommendation:** {rec.get('sentence', '')[:100]}...")
        st.write(f"**Responses:** {len(responses)}")

def display_manual_search_results_basic(matches: List[Dict], target: str, search_time: float, show_scores: bool):
    """Basic manual search display"""
    
    if not matches:
        st.warning("No matches found")
        return
    
    meaningful_words = get_meaningful_words(target)
    st.success(f"Found {len(matches)} matches in {search_time:.3f} seconds")
    if meaningful_words:
        st.info(f"Based on meaningful words: {', '.join(meaningful_words)}")
    
    for i, match in enumerate(matches, 1):
        similarity = match.get('similarity_score', 0)
        sentence = match.get('sentence', '')
        
        st.write(f"**{i}.** {sentence[:100]}...")
        if show_scores:
            st.write(f"Similarity: {similarity:.3f}")
        
        if 'matched_meaningful_words' in match:
            matched_words = match['matched_meaningful_words']
            if matched_words:
                st.caption(f"Matched words: {', '.join(matched_words)}")

def show_basic_pattern_analysis(documents: List[Dict], rec_patterns: List[str], resp_patterns: List[str]):
    """Basic pattern analysis"""
    
    total_rec = 0
    total_resp = 0
    
    for doc in documents:
        text = doc.get('text', '').lower()
        for pattern in rec_patterns:
            total_rec += text.count(pattern.lower())
        for pattern in resp_patterns:
            total_resp += text.count(pattern.lower())
    
    st.metric("Recommendation Patterns", total_rec)
    st.metric("Response Patterns", total_resp)

def show_alignment_feature_info_basic():
    """Basic alignment info"""
    st.info("Upload documents to use the alignment feature")

# Export all functions for compatibility
__all__ = [
    'render_search_interface',
    'render_recommendation_alignment_interface',
    'check_rag_availability',
    'filter_stop_words',
    'STOP_WORDS'
]
