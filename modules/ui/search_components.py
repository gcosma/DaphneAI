# Complete search_components.py with Enhanced Features and Recommendation-Response Alignment
import streamlit as st
import re
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import time
from collections import defaultdict

# ========== CORE SEARCH INTERFACE ==========

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

# ========== SEARCH EXECUTION ==========

def execute_search(documents: List[Dict], query: str, method: str, max_results: int = 5, case_sensitive: bool = False) -> List[Dict]:
    """Execute search using the specified method with stop word filtering"""
    return execute_search_with_preprocessing(documents, query, method, max_results, case_sensitive)

def execute_search_with_preprocessing(documents: List[Dict], query: str, method: str, max_results: int = 5, case_sensitive: bool = False) -> List[Dict]:
    """Execute search with query preprocessing"""
    
    # Preprocess query
    processed_query = preprocess_query(query, method)
    
    # Show query transformation if different
    if processed_query != query and method != "exact":
        st.info(f"🔍 Search terms after filtering stop words: '{processed_query}'")
    
    all_results = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Execute search based on method using processed query
        if method == "exact":
            matches = exact_search(text, query, case_sensitive)  # Use original query for exact
        elif method == "smart":
            matches = smart_search_filtered(text, processed_query, case_sensitive)
        elif method == "fuzzy":
            matches = fuzzy_search_filtered(text, processed_query, case_sensitive)
        elif method == "semantic":
            matches = semantic_search(text, processed_query)
        elif method == "hybrid":
            matches = hybrid_search_filtered(text, processed_query, case_sensitive)
        else:
            matches = smart_search_filtered(text, processed_query, case_sensitive)
        
        # Limit results per document
        matches = matches[:max_results]
        
        # Add document info to each match
        for match in matches:
            match['document'] = doc
            match['search_method'] = method
            match['original_query'] = query
            match['processed_query'] = processed_query
            all_results.append(match)
    
    # Sort by relevance score
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return all_results

def exact_search(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Find exact matches only"""
    
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()
    
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
            'score': 100.0,  # Exact matches get highest score
            'match_type': 'exact',
            'page_number': estimate_page_number(pos, text),
            'word_position': len(text[:pos].split()),
            'percentage_through': (pos / len(text)) * 100 if text else 0
        }
        
        matches.append(match)
        start = pos + 1
    
    return matches

def smart_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Smart search with stop word filtering - ENHANCED for word variations"""
    
    if not query.strip():
        return []
    
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()
    
    matches = []
    
    # Split query into meaningful words (stop words already filtered)
    query_words = [word for word in query.split() if len(word) > 1]
    
    if not query_words:
        return []
    
    # Find each meaningful word with better pattern matching
    for word in query_words:
        # ENHANCED: Create root word for better matching
        root_word = word
        if word.endswith(('ing', 'ed', 'er', 's', 'ly')):
            # Try to find root word
            if word.endswith('ing'):
                root_word = word[:-3]
            elif word.endswith('ed'):
                root_word = word[:-2]
            elif word.endswith(('er', 'ly')):
                root_word = word[:-2]
            elif word.endswith('s'):
                root_word = word[:-1]
        
        # ENHANCED: More flexible pattern to catch variations
        # This will match: recommend, recommending, recommended, recommendation, etc.
        patterns = [
            rf'\b{re.escape(word)}\w*',           # Original word + endings
            rf'\b{re.escape(root_word)}\w*',      # Root word + endings
        ]
        
        # Remove duplicates
        patterns = list(set(patterns))
        
        for pattern in patterns:
            try:
                for match in re.finditer(pattern, search_text):
                    pos = match.start()
                    matched_text = text[match.start():match.end()]
                    
                    # Calculate score based on match quality
                    matched_lower = match.group().lower()
                    word_lower = word.lower()
                    
                    if matched_lower == word_lower:
                        score = 100.0  # Exact word match
                    elif matched_lower.startswith(word_lower):
                        score = 95.0   # Word starts with query (recommend -> recommending)
                    elif matched_lower.startswith(root_word.lower()):
                        score = 90.0   # Root word match (recommendation -> recommend)
                    elif word_lower in matched_lower:
                        score = 85.0   # Query word contained in match
                    else:
                        score = 70.0   # Partial match
                    
                    # Bonus for common word variations
                    if any(matched_lower.endswith(suffix) for suffix in ['ing', 'ed', 'tion', 'ment']):
                        score += 5.0
                    
                    # Extract context
                    context_start = max(0, pos - 150)
                    context_end = min(len(text), pos + len(matched_text) + 150)
                    context = text[context_start:context_end]
                    
                    match_info = {
                        'position': pos,
                        'matched_text': matched_text,
                        'context': context,
                        'score': score,
                        'match_type': 'smart',
                        'page_number': estimate_page_number(pos, text),
                        'word_position': len(text[:pos].split()),
                        'percentage_through': (pos / len(text)) * 100 if text else 0,
                        'query_word': word,
                        'pattern_used': pattern
                    }
                    
                    matches.append(match_info)
            except re.error:
                # Skip invalid regex patterns
                continue
    
    # Remove overlapping matches and sort by score
    matches = remove_overlapping_matches(matches)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches

def fuzzy_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Fuzzy search with stop word filtering - FIXED highlighting"""
    
    if not query.strip():
        return []
    
    import difflib
    
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()
    
    # Split into meaningful words
    words = text.split()
    search_words = search_text.split()
    query_words = query.split()
    
    matches = []
    
    # Check each word against each query word
    for query_word in query_words:
        if len(query_word) < 2:
            continue
            
        for i, word in enumerate(search_words):
            similarity = difflib.SequenceMatcher(None, query_word, word).ratio()
            
            if similarity > 0.6:  # Lower threshold for better results (was 0.7)
                # Find position in original text
                pos = len(' '.join(words[:i]))
                if i > 0:
                    pos += 1  # Add space
                
                # Extract context with proper highlighting context
                context_start = max(0, pos - 150)
                context_end = min(len(text), pos + len(words[i]) + 150)
                context = text[context_start:context_end]
                
                # FIXED: Use original case word for matched_text
                matched_text = words[i]
                
                match = {
                    'position': pos,
                    'matched_text': matched_text,  # FIXED: Original case preserved
                    'context': context,
                    'score': similarity * 100,
                    'match_type': 'fuzzy',
                    'similarity': similarity,
                    'page_number': estimate_page_number(pos, text),
                    'word_position': i,
                    'percentage_through': (pos / len(text)) * 100 if text else 0,
                    'query_word': query_word
                }
                
                matches.append(match)
    
    # Remove overlapping matches and sort by similarity
    matches = remove_overlapping_matches(matches)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches

def semantic_search(text: str, query: str) -> List[Dict]:
    """AI semantic search with fallback when RAG not available"""
    
    try:
        # Try to use RAG engine first
        if 'rag_engine' not in st.session_state:
            from modules.search.rag_search import RAGSearchEngine
            st.session_state.rag_engine = RAGSearchEngine()
        
        rag_engine = st.session_state.rag_engine
        
        # Use RAG search
        rag_results = rag_engine.search(query, [{'text': text, 'filename': 'current_doc'}])
        
        matches = []
        for result in rag_results:
            # Convert RAG result to our format
            match = {
                'position': result.get('position', 0),
                'matched_text': result.get('chunk', ''),
                'context': result.get('chunk', ''),
                'score': result.get('score', 0) * 100,
                'match_type': 'semantic',
                'page_number': result.get('page_number', 1),
                'word_position': 0,
                'percentage_through': 0
            }
            matches.append(match)
        
        return matches
        
    except Exception as e:
        # FALLBACK: Use enhanced word similarity when RAG fails
        st.warning(f"🤖 AI semantic search not available. Using enhanced similarity matching instead.")
        return semantic_fallback_search(text, query)

def semantic_fallback_search(text: str, query: str) -> List[Dict]:
    """Fallback semantic search using word similarity and synonyms"""
    
    # Define semantic word groups (synonyms and related terms)
    semantic_groups = {
        'recommend': ['recommend', 'suggest', 'advise', 'propose', 'urge', 'advocate', 'endorse'],
        'suggest': ['suggest', 'recommend', 'propose', 'advise', 'hint', 'indicate'],
        'implement': ['implement', 'execute', 'carry out', 'put into practice', 'apply', 'deploy'],
        'review': ['review', 'examine', 'assess', 'evaluate', 'analyze', 'inspect'],
        'policy': ['policy', 'procedure', 'guideline', 'protocol', 'framework', 'strategy'],
        'response': ['response', 'reply', 'answer', 'feedback', 'reaction', 'comment'],
        'accept': ['accept', 'agree', 'approve', 'endorse', 'support', 'adopt'],
        'reject': ['reject', 'decline', 'refuse', 'dismiss', 'deny', 'oppose']
    }
    
    matches = []
    query_words = query.lower().split()
    
    # Find semantic matches
    for query_word in query_words:
        if len(query_word) < 3:
            continue
            
        # Find related words
        related_words = []
        for key, synonyms in semantic_groups.items():
            if query_word in synonyms or any(query_word in syn for syn in synonyms):
                related_words.extend(synonyms)
        
        # If no semantic group found, use the word itself
        if not related_words:
            related_words = [query_word]
        
        # Search for related words in text
        search_text = text.lower()
        words = text.split()
        search_words = search_text.split()
        
        for i, word in enumerate(search_words):
            for related_word in related_words:
                if related_word in word or word in related_word:
                    # Calculate semantic similarity score
                    if word == related_word:
                        score = 95.0  # Exact semantic match
                    elif word == query_word:
                        score = 100.0  # Exact query match
                    elif related_word in word:
                        score = 85.0  # Contains semantic word
                    else:
                        score = 70.0  # Partial semantic match
                    
                    # Find position in original text
                    pos = len(' '.join(words[:i]))
                    if i > 0:
                        pos += 1
                    
                    # Extract context
                    context_start = max(0, pos - 150)
                    context_end = min(len(text), pos + len(words[i]) + 150)
                    context = text[context_start:context_end]
                    
                    match = {
                        'position': pos,
                        'matched_text': words[i],
                        'context': context,
                        'score': score,
                        'match_type': 'semantic',
                        'page_number': estimate_page_number(pos, text),
                        'word_position': i,
                        'percentage_through': (pos / len(text)) * 100 if text else 0,
                        'semantic_relation': f"{query_word} → {related_word}"
                    }
                    
                    matches.append(match)
                    break  # Only match once per word
    
    # Remove overlapping matches and sort by score
    matches = remove_overlapping_matches(matches)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches

def hybrid_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Hybrid search with stop word filtering"""
    
    # Get smart search results with filtering
    smart_results = smart_search_filtered(text, query, case_sensitive)
    
    # Get semantic search results (semantic search handles its own processing)
    semantic_results = semantic_search(text, query)
    
    # Combine and deduplicate
    all_matches = smart_results + semantic_results
    
    # Remove overlapping matches
    unique_matches = remove_overlapping_matches(all_matches)
    
    # Sort by score
    unique_matches.sort(key=lambda x: x['score'], reverse=True)
    
    return unique_matches

def remove_overlapping_matches(matches: List[Dict]) -> List[Dict]:
    """Remove overlapping matches, keeping the highest scored ones"""
    
    if not matches:
        return matches
    
    # Sort by score descending
    sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)
    
    unique_matches = []
    used_positions = set()
    
    for match in sorted_matches:
        pos = match['position']
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

# ========== RESULT DISPLAY ==========

def display_results_grouped(results: List[Dict], query: str, search_time: float, 
                          show_context: bool = True, highlight_matches: bool = True,
                          search_method: str = "Search"):
    """Display search results grouped by document with enhanced information"""
    
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    # Group results by document
    doc_groups = defaultdict(list)
    for result in results:
        doc_name = result['document']['filename']
        doc_groups[doc_name].append(result)
    
    # Summary
    total_docs = len(doc_groups)
    total_matches = len(results)
    
    st.success(f"🎯 Found **{total_matches}** result(s) in **{total_docs}** document(s) for '**{query}**' in {search_time:.3f} seconds")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy All Results"):
            copy_all_results(results, query)
    with col2:
        if st.button("📊 Export to CSV"):
            export_results_csv(results, query)
    with col3:
        if st.button("📄 Generate Report"):
            generate_search_report(results, query, search_method)
    
    # Display each document group
    for doc_name, doc_results in doc_groups.items():
        
        doc = doc_results[0]['document']
        best_score = max(r['score'] for r in doc_results)
        
        # Document header with enhanced info
        with st.expander(f"📄 {doc_name} ({len(doc_results)} matches, best score: {best_score:.1f})", expanded=True):
            
            # Document statistics
            text = doc.get('text', '')
            word_count = len(text.split())
            char_count = len(text)
            est_pages = max(1, char_count // 2000)
            file_size_mb = char_count / (1024 * 1024)
            
            st.markdown(f"""
            **File Type:** {doc_name.split('.')[-1].upper()}  |  **Words:** {word_count:,}  |  **Size:** {file_size_mb:.1f} MB  |  **Est. Pages:** {est_pages}
            """)
            
            # Display each match in this document
            for i, result in enumerate(doc_results, 1):
                display_single_result(result, i, query, show_context, highlight_matches)

def display_single_result(result: Dict, index: int, query: str, show_context: bool, highlight_matches: bool):
    """Display a single search result with enhanced formatting - ENHANCED DEBUG INFO"""
    
    # Result header with score and method
    method_icons = {
        'exact': '🎯',
        'smart': '🧠', 
        'fuzzy': '🌀',
        'semantic': '🤖',
        'hybrid': '🔄'
    }
    
    method = result.get('match_type', 'unknown')
    icon = method_icons.get(method, '🔍')
    score = result.get('score', 0)
    
    st.markdown(f"""
    **{icon} {method.title()} Search - Match {index}**  
    Score: {score:.1f}
    """)
    
    # DEBUG: Show what was actually matched
    matched_text = result.get('matched_text', '')
    query_word = result.get('query_word', query)
    
    if method == 'fuzzy' and 'similarity' in result:
        similarity = result['similarity']
        st.caption(f"🔍 **Fuzzy Match:** '{matched_text}' ↔ '{query_word}' (similarity: {similarity:.2f})")
    elif method == 'semantic' and 'semantic_relation' in result:
        st.caption(f"🤖 **Semantic:** {result['semantic_relation']}")
    elif method == 'smart':
        pattern_used = result.get('pattern_used', 'standard')
        st.caption(f"🧠 **Smart Match:** Found '{matched_text}' using pattern search")
    
    # Position information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"📄 **Page {result.get('page_number', 1)}**")
    
    with col2:
        pos = result.get('position', 0)
        st.markdown(f"📍 **Position {pos:,}**")
    
    with col3:
        percentage = result.get('percentage_through', 0)
        st.markdown(f"📊 **{percentage:.0f}% through doc**")
    
    with col4:
        word_pos = result.get('word_position', 0)
        st.markdown(f"🔢 **Word {word_pos:,}**")
    
    # Match details with better information
    if matched_text:
        # For exact matches, show phrase counting
        if method == 'exact':
            exact_count = result.get('context', '').lower().count(query.lower())
            st.markdown(f"🎯 **Exact phrase '{query}':** Found {exact_count} time(s)")
        else:
            # For other methods, show word matching info (filtered)
            original_query = result.get('original_query', query)
            filtered_query = result.get('processed_query', filter_stop_words(query))
            
            # Show both original and filtered if different
            if filtered_query != original_query:
                st.markdown(f"🔍 **Original query:** '{original_query}' → **Filtered:** '{filtered_query}'")
            
            # Show which meaningful words were found
            query_words = filtered_query.lower().split() if filtered_query else []
            context_lower = result.get('context', '').lower()
            found_words = [word for word in query_words if word in context_lower]
            st.markdown(f"📝 **Meaningful words found:** {', '.join(found_words) if found_words else 'Related terms found'}")
    
    # Context display
    if show_context:
        context = result.get('context', '')
        if context:
            
            # Highlight matches if requested
            if highlight_matches:
                # For fuzzy search, also highlight the actual matched word
                highlight_query = query
                if method == 'fuzzy' and matched_text:
                    highlight_query = f"{query} {matched_text}"
                
                highlighted_context = highlight_search_terms(context, highlight_query)
                st.markdown(f"📖 **Context:** {highlighted_context}", unsafe_allow_html=True)
            else:
                st.markdown(f"📖 **Context:** {context}")
            
            # Position hint
            page_num = result.get('page_number', 1)
            percentage = result.get('percentage_through', 0)
            st.caption(f"💡 This appears around page {page_num}, {percentage:.1f}% through the document")
    
    # Additional info for special search types
    if method == 'fuzzy' and 'similarity' in result:
        similarity = result['similarity']
        st.caption(f"🌀 Fuzzy match similarity: {similarity:.2f} (threshold: 0.6)")
    
    elif method == 'semantic':
        st.caption(f"🤖 AI found this as semantically related to your query")
    
    st.markdown("---")

def highlight_search_terms(text: str, query: str) -> str:
    """Highlight search terms in text - FIXED to exclude stop words"""
    
    # Filter out stop words from highlighting (same as search filtering)
    filtered_query = filter_stop_words(query)
    query_words = filtered_query.split() if filtered_query else []
    
    highlighted = text
    
    # Sort words by length (longest first) to avoid partial highlighting conflicts
    query_words.sort(key=len, reverse=True)
    
    for word in query_words:
        if len(word) > 1 and word.lower() not in STOP_WORDS:  # Double-check stop words
            
            # Create multiple patterns for better matching
            patterns = [
                word,                    # Exact word
                word.rstrip('s'),        # Remove plural 's'
                word + 'ing',           # Add 'ing'
                word + 'ed',            # Add 'ed'
                word + 'tion',          # Add 'tion'
                word + 'ment',          # Add 'ment'
            ]
            
            # Also try root forms
            if word.endswith('ing'):
                patterns.append(word[:-3])  # Remove 'ing'
            if word.endswith('ed'):
                patterns.append(word[:-2])  # Remove 'ed'
            if word.endswith('tion'):
                patterns.append(word[:-4])  # Remove 'tion'
            
            # Remove short or duplicate patterns, and filter stop words again
            patterns = list(set([p for p in patterns if len(p) > 2 and p.lower() not in STOP_WORDS]))
            
            # Apply highlighting for each pattern
            for pattern in patterns:
                try:
                    # Case-insensitive word boundary matching
                    regex_pattern = rf'\b{re.escape(pattern)}\w*'
                    compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)
                    
                    def highlight_match(match):
                        matched_word = match.group()
                        # Double-check: don't highlight if it's a stop word
                        if matched_word.lower() in STOP_WORDS:
                            return matched_word
                        return f"<mark style='background-color: #FFEB3B; padding: 2px; border-radius: 2px; font-weight: bold;'>{matched_word}</mark>"
                    
                    highlighted = compiled_pattern.sub(highlight_match, highlighted)
                except re.error:
                    # Skip invalid patterns
                    continue
    
    return highlighted

# ========== UTILITY FUNCTIONS ==========

def estimate_page_number(char_position: int, text: str) -> int:
    """Estimate page number based on character position"""
    if char_position <= 0:
        return 1
    return max(1, char_position // 2000 + 1)  # ~2000 chars per page

def check_rag_availability() -> bool:
    """Check if RAG dependencies are available"""
    try:
        import sentence_transformers
        import torch
        return True
    except ImportError:
        return False

# ========== STOP WORDS AND QUERY PROCESSING ==========

# Comprehensive stop words list
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 
    'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 
    'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 
    'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 
    'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 
    'come', 'made', 'may', 'part'
}

def filter_stop_words(query: str) -> str:
    """Remove stop words from query"""
    words = query.lower().split()
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    
    # If all words are stop words, keep the original query
    if not filtered_words:
        return query
    
    return ' '.join(filtered_words)

def preprocess_query(query: str, method: str) -> str:
    """Preprocess query based on search method"""
    
    # For exact search, don't filter stop words
    if method == "exact":
        return query
    
    # For other methods, filter stop words
    filtered_query = filter_stop_words(query)
    
    return filtered_query

def copy_all_results(results: List[Dict], query: str):
    """Copy all results to clipboard"""
    
    output = f"Search Results for: {query}\n"
    output += "=" * 50 + "\n\n"
    
    doc_groups = defaultdict(list)
    for result in results:
        doc_name = result['document']['filename']
        doc_groups[doc_name].append(result)
    
    for doc_name, doc_results in doc_groups.items():
        output += f"Document: {doc_name}\n"
        output += f"Matches: {len(doc_results)}\n\n"
        
        for i, result in enumerate(doc_results, 1):
            output += f"Match {i}:\n"
            output += f"  Page: {result.get('page_number', 1)}\n"
            output += f"  Score: {result.get('score', 0):.1f}\n"
            output += f"  Context: {result.get('context', '')}\n\n"
        
        output += "-" * 30 + "\n\n"
    
    st.code(output)
    st.success("Results copied to display! Use Ctrl+A, Ctrl+C to copy to clipboard")

def export_results_csv(results: List[Dict], query: str):
    """Export search results to CSV"""
    
    csv_data = []
    
    for result in results:
        row = {
            'Query': query,
            'Document': result['document']['filename'],
            'Match_Type': result.get('match_type', ''),
            'Score': result.get('score', 0),
            'Page_Number': result.get('page_number', 1),
            'Position': result.get('position', 0),
            'Matched_Text': result.get('matched_text', ''),
            'Context': result.get('context', ''),
            'Percentage_Through': result.get('percentage_through', 0)
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"search_results_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def generate_search_report(results: List[Dict], query: str, search_method: str):
    """Generate a comprehensive search report"""
    
    # Group results by document
    doc_groups = defaultdict(list)
    for result in results:
        doc_name = result['document']['filename']
        doc_groups[doc_name].append(result)
    
    report = f"""# Search Report: "{query}"

**Search Method:** {search_method}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Results:** {len(results)} matches in {len(doc_groups)} documents

## Summary Statistics

| Metric | Value |
|--------|-------|
| Documents Searched | {len(doc_groups)} |
| Total Matches | {len(results)} |
| Average Score | {sum(r.get('score', 0) for r in results) / len(results):.2f} |
| Highest Score | {max(r.get('score', 0) for r in results):.2f} |

## Results by Document

"""
    
    for doc_name, doc_results in doc_groups.items():
        best_score = max(r.get('score', 0) for r in doc_results)
        avg_score = sum(r.get('score', 0) for r in doc_results) / len(doc_results)
        
        report += f"""### 📄 {doc_name}
- **Matches:** {len(doc_results)}
- **Best Score:** {best_score:.2f}
- **Average Score:** {avg_score:.2f}

"""
        
        for i, result in enumerate(doc_results, 1):
            report += f"""**Match {i}** (Score: {result.get('score', 0):.1f})  
*Page {result.get('page_number', 1)}, Position {result.get('position', 0):,}*

> {result.get('context', '')[:200]}...

---

"""
    
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"search_report_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

# ========== RECOMMENDATION-RESPONSE ALIGNMENT SYSTEM ==========

def render_recommendation_alignment_interface(documents: List[Dict[str, Any]]):
    """Specialized interface for aligning recommendations with responses"""
    
    st.header("🏛️ Recommendation-Response Alignment")
    st.markdown("*Automatically find recommendations and their corresponding responses*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        return
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Search Configuration:**")
        
        # Recommendation patterns
        rec_patterns = st.multiselect(
            "Recommendation Keywords",
            ["recommend", "suggest", "advise", "propose", "urge", "call for", "should", "must"],
            default=["recommend", "suggest", "advise"],
            help="Keywords that indicate recommendations"
        )
        
        # Response patterns  
        resp_patterns = st.multiselect(
            "Response Keywords", 
            ["accept", "reject", "agree", "disagree", "implement", "consider", "response", "reply", "approved", "declined"],
            default=["accept", "reject", "agree", "implement"],
            help="Keywords that indicate responses"
        )
    
    with col2:
        st.markdown("**🤖 AI Configuration:**")
        
        # AI options
        use_ai_summary = st.checkbox("Generate AI Summaries", value=True)
        ai_available = check_rag_availability()
        
        if not ai_available:
            st.warning("🤖 AI summaries require: pip install sentence-transformers torch")
            use_ai_summary = False
        
        # Summary length
        summary_length = st.selectbox(
            "Summary Length",
            ["Short (1-2 sentences)", "Medium (3-4 sentences)", "Long (5+ sentences)"],
            index=1
        )
    
    # Start analysis button
    if st.button("🔍 Find & Align Recommendations", type="primary"):
        with st.spinner("🔍 Analyzing documents for recommendations and responses..."):
            
            # Step 1: Find recommendations
            recommendations = find_recommendations(documents, rec_patterns)
            
            # Step 2: Find responses
            responses = find_responses(documents, resp_patterns)
            
            # Step 3: Align recommendations with responses
            alignments = align_recommendations_responses(recommendations, responses)
            
            # Step 4: Generate AI summaries if enabled
            if use_ai_summary and ai_available:
                alignments = add_ai_summaries(alignments, summary_length)
            
            # Step 5: Display results
            display_alignment_results(alignments, use_ai_summary)

# ========== RECOMMENDATION FINDING FUNCTIONS ==========

def find_recommendations(documents: List[Dict], patterns: List[str]) -> List[Dict]:
    """Find all recommendations in documents"""
    
    recommendations = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Split into sentences for better analysis
        sentences = split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if sentence contains recommendation patterns
            for pattern in patterns:
                if pattern.lower() in sentence_lower:
                    
                    # Extract context (previous and next sentence)
                    context_start = max(0, i - 1)
                    context_end = min(len(sentences), i + 2)
                    context = ' '.join(sentences[context_start:context_end])
                    
                    # Calculate position
                    char_position = text.find(sentence)
                    
                    recommendation = {
                        'id': f"rec_{len(recommendations) + 1}",
                        'document': doc,
                        'sentence': sentence,
                        'context': context,
                        'pattern_matched': pattern,
                        'sentence_index': i,
                        'char_position': char_position,
                        'page_number': estimate_page_number(char_position, text),
                        'recommendation_type': classify_recommendation_type(sentence)
                    }
                    
                    recommendations.append(recommendation)
                    break  # Only match one pattern per sentence
    
    return recommendations

def find_responses(documents: List[Dict], patterns: List[str]) -> List[Dict]:
    """Find all responses in documents"""
    
    responses = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        sentences = split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if sentence contains response patterns
            for pattern in patterns:
                if pattern.lower() in sentence_lower:
                    
                    # Extract context
                    context_start = max(0, i - 1)
                    context_end = min(len(sentences), i + 2)
                    context = ' '.join(sentences[context_start:context_end])
                    
                    # Calculate position
                    char_position = text.find(sentence)
                    
                    response = {
                        'id': f"resp_{len(responses) + 1}",
                        'document': doc,
                        'sentence': sentence,
                        'context': context,
                        'pattern_matched': pattern,
                        'sentence_index': i,
                        'char_position': char_position,
                        'page_number': estimate_page_number(char_position, text),
                        'response_type': classify_response_type(sentence)
                    }
                    
                    responses.append(response)
                    break
    
    return responses

def align_recommendations_responses(recommendations: List[Dict], responses: List[Dict]) -> List[Dict]:
    """Align recommendations with their corresponding responses using semantic similarity"""
    
    alignments = []
    
    for rec in recommendations:
        best_matches = []
        
        # Find potential response matches
        for resp in responses:
            similarity_score = calculate_semantic_similarity(
                rec['context'], 
                resp['context']
            )
            
            # Also check if they reference similar topics
            topic_similarity = calculate_topic_similarity(
                rec['sentence'],
                resp['sentence']
            )
            
            combined_score = (similarity_score * 0.7) + (topic_similarity * 0.3)
            
            if combined_score > 0.3:  # Threshold for potential match
                best_matches.append({
                    'response': resp,
                    'similarity_score': similarity_score,
                    'topic_similarity': topic_similarity,
                    'combined_score': combined_score
                })
        
        # Sort by combined score
        best_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Create alignment
        alignment = {
            'recommendation': rec,
            'responses': best_matches[:3],  # Top 3 matches
            'alignment_confidence': best_matches[0]['combined_score'] if best_matches else 0,
            'alignment_status': determine_alignment_status(best_matches)
        }
        
        alignments.append(alignment)
    
    return alignments

def add_ai_summaries(alignments: List[Dict], summary_length: str) -> List[Dict]:
    """Add AI-generated summaries to alignments"""
    
    try:
        # Determine summary parameters
        max_sentences = {
            "Short (1-2 sentences)": 2,
            "Medium (3-4 sentences)": 4,  
            "Long (5+ sentences)": 6
        }.get(summary_length, 4)
        
        for alignment in alignments:
            rec = alignment['recommendation']
            responses = alignment['responses']
            
            # Create summary prompt
            summary_text = f"Recommendation: {rec['sentence']}\n\n"
            
            if responses:
                summary_text += "Related Responses:\n"
                for i, resp_match in enumerate(responses[:2], 1):
                    resp = resp_match['response']
                    summary_text += f"{i}. {resp['sentence']}\n"
            
            # Generate AI summary
            summary = generate_ai_summary(summary_text, max_sentences)
            
            alignment['ai_summary'] = summary
            alignment['summary_confidence'] = len(responses) * 0.2  # Simple confidence
    
    except Exception as e:
        st.warning(f"AI summary generation failed: {str(e)}")
        # Add empty summaries
        for alignment in alignments:
            alignment['ai_summary'] = "AI summary not available"
            alignment['summary_confidence'] = 0
    
    return alignments

def display_alignment_results(alignments: List[Dict], show_ai_summaries: bool):
    """Display the recommendation-response alignments"""
    
    if not alignments:
        st.warning("No recommendations found in the uploaded documents")
        return
    
    # Summary statistics
    st.markdown("### 📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recommendations", len(alignments))
    
    with col2:
        aligned_count = sum(1 for a in alignments if a['responses'])
        st.metric("Recommendations with Responses", aligned_count)
    
    with col3:
        avg_confidence = sum(a['alignment_confidence'] for a in alignments) / len(alignments) if alignments else 0
        st.metric("Avg Alignment Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        high_confidence = sum(1 for a in alignments if a['alignment_confidence'] > 0.7)
        st.metric("High Confidence Alignments", high_confidence)
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Export Alignment Report"):
            export_alignment_report(alignments)
    
    with col2:
        if st.button("📋 Export to CSV"):
            export_alignment_csv(alignments)
    
    # Display individual alignments
    st.markdown("### 🔗 Recommendation-Response Alignments")
    
    for i, alignment in enumerate(alignments, 1):
        display_single_alignment(alignment, i, show_ai_summaries)

def display_single_alignment(alignment: Dict, index: int, show_ai_summaries: bool):
    """Display a single recommendation-response alignment"""
    
    rec = alignment['recommendation']
    responses = alignment['responses']
    confidence = alignment['alignment_confidence']
    
    # Confidence indicator
    confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
    
    with st.expander(f"{confidence_color} Recommendation {index} - {rec['recommendation_type']} (Confidence: {confidence:.2f})", 
                    expanded=index <= 3):
        
        # Two-column layout: Extract + Summary
        col1, col2 = st.columns([3, 2] if show_ai_summaries else [1])
        
        with col1:
            st.markdown("**📋 Original Extract:**")
            
            # Recommendation section
            st.markdown("**🎯 Recommendation:**")
            st.info(f"📄 {rec['document']['filename']} (Page {rec['page_number']})")
            
            # Highlight the recommendation
            highlighted_rec = f"<mark style='background-color: #FFEB3B; padding: 4px; border-radius: 4px;'>{rec['sentence']}</mark>"
            st.markdown(highlighted_rec, unsafe_allow_html=True)
            
            # Show context
            st.caption(f"📖 Context: {rec['context']}")
            
            # Responses section
            if responses:
                st.markdown("**↩️ Related Responses:**")
                
                for j, resp_match in enumerate(responses, 1):
                    resp = resp_match['response']
                    similarity = resp_match['combined_score']
                    
                    # Color code by similarity
                    resp_color = "#4CAF50" if similarity > 0.7 else "#FF9800" if similarity > 0.5 else "#F44336"
                    
                    st.info(f"📄 {resp['document']['filename']} (Page {resp['page_number']}) - Similarity: {similarity:.2f}")
                    
                    highlighted_resp = f"<mark style='background-color: {resp_color}; color: white; padding: 4px; border-radius: 4px;'>{resp['sentence']}</mark>"
                    st.markdown(highlighted_resp, unsafe_allow_html=True)
                    
                    st.caption(f"📖 Context: {resp['context']}")
                    
                    if j < len(responses):
                        st.markdown("---")
            else:
                st.warning("❌ No matching responses found")
        
        # AI Summary column
        if show_ai_summaries:
            with col2:
                st.markdown("**🤖 AI Summary:**")
                
                if 'ai_summary' in alignment and alignment['ai_summary']:
                    # Summary box
                    summary_confidence = alignment.get('summary_confidence', 0)
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 16px;
                        border-radius: 8px;
                        margin: 8px 0;
                    ">
                        <h4 style="margin: 0 0 12px 0; color: white;">📝 Summary</h4>
                        <p style="margin: 0; line-height: 1.6;">{alignment['ai_summary']}</p>
                        <small style="opacity: 0.8;">Confidence: {summary_confidence:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key insights
                    if responses:
                        st.markdown("**🔍 Key Insights:**")
                        insights = generate_key_insights(rec, responses)
                        for insight in insights:
                            st.markdown(f"• {insight}")
                else:
                    st.info("🤖 AI summary not available")
        
        # Detailed information
        with st.expander("🔍 Detailed Analysis"):
            
            # Recommendation details
            st.markdown("**📋 Recommendation Details:**")
            rec_details = [
                f"**Type:** {rec['recommendation_type']}",
                f"**Document:** {rec['document']['filename']}",
                f"**Page:** {rec['page_number']}",
                f"**Pattern Matched:** {rec['pattern_matched']}",
                f"**Position:** {rec['char_position']:,} characters"
            ]
            
            for detail in rec_details:
                st.markdown(f"• {detail}")
            
            # Response analysis
            if responses:
                st.markdown("**↩️ Response Analysis:**")
                
                response_types = [r['response']['response_type'] for r in responses]
                type_counts = {t: response_types.count(t) for t in set(response_types)}
                
                st.markdown(f"• **Response Types Found:** {', '.join(type_counts.keys())}")
                st.markdown(f"• **Documents with Responses:** {len(set(r['response']['document']['filename'] for r in responses))}")
                st.markdown(f"• **Average Similarity:** {sum(r['combined_score'] for r in responses) / len(responses):.3f}")

# ========== UTILITY FUNCTIONS FOR ALIGNMENT ==========

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Improved sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

def classify_recommendation_type(sentence: str) -> str:
    """Classify the type of recommendation"""
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['urgent', 'immediate', 'critical', 'emergency']):
        return 'Urgent'
    elif any(word in sentence_lower for word in ['consider', 'review', 'explore', 'examine']):
        return 'Consideration'
    elif any(word in sentence_lower for word in ['implement', 'establish', 'create', 'develop']):
        return 'Implementation'
    elif any(word in sentence_lower for word in ['policy', 'regulation', 'framework', 'legislation']):
        return 'Policy'
    elif any(word in sentence_lower for word in ['budget', 'funding', 'financial', 'cost']):
        return 'Financial'
    elif any(word in sentence_lower for word in ['training', 'education', 'skills', 'capacity']):
        return 'Training'
    else:
        return 'General'

def classify_response_type(sentence: str) -> str:
    """Classify the type of response"""
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['accept', 'agree', 'approve', 'endorse']):
        return 'Acceptance'
    elif any(word in sentence_lower for word in ['reject', 'decline', 'disagree', 'oppose']):
        return 'Rejection'
    elif any(word in sentence_lower for word in ['consider', 'review', 'evaluate', 'assess']):
        return 'Under Review'
    elif any(word in sentence_lower for word in ['implement', 'action', 'proceed', 'execute']):
        return 'Implementation'
    elif any(word in sentence_lower for word in ['partial', 'some', 'limited', 'qualified']):
        return 'Partial Acceptance'
    else:
        return 'General Response'

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    # Enhanced word overlap similarity
    words1 = set(w.lower() for w in text1.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
    words2 = set(w.lower() for w in text2.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def calculate_topic_similarity(sentence1: str, sentence2: str) -> float:
    """Calculate topic similarity based on key terms"""
    
    # Extract key topics/entities
    key_terms1 = extract_key_terms(sentence1)
    key_terms2 = extract_key_terms(sentence2)
    
    if not key_terms1 or not key_terms2:
        return 0.0
    
    intersection = len(set(key_terms1) & set(key_terms2))
    union = len(set(key_terms1) | set(key_terms2))
    
    return intersection / union if union > 0 else 0.0

def extract_key_terms(sentence: str) -> List[str]:
    """Extract key terms from a sentence"""
    # Enhanced key term extraction
    words = re.findall(r'\b\w+\b', sentence.lower())
    key_terms = [word for word in words if len(word) > 3 and word not in STOP_WORDS]
    
    # Add some domain-specific important terms even if short
    important_short_terms = {'ai', 'it', 'hr', 'ceo', 'cfo', 'uk', 'eu', 'us', 'gdp'}
    short_terms = [word for word in words if word in important_short_terms]
    
    return key_terms + short_terms

def determine_alignment_status(matches: List[Dict]) -> str:
    """Determine the status of recommendation-response alignment"""
    if not matches:
        return "No Response Found"
    
    best_score = matches[0]['combined_score']
    
    if best_score > 0.8:
        return "Strong Alignment"
    elif best_score > 0.6:
        return "Good Alignment"
    elif best_score > 0.4:
        return "Weak Alignment"
    else:
        return "Poor Alignment"

def generate_ai_summary(text: str, max_sentences: int) -> str:
    """Generate AI summary of recommendation-response pair"""
    
    # Split into lines and analyze
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    summary_points = []
    
    # Extract key information
    for line in lines:
        if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'accept', 'reject', 'implement', 'consider']):
            # Clean and simplify the sentence
            clean_line = line.strip()
            if clean_line and len(clean_line) > 15:
                # Remove redundant phrases
                clean_line = re.sub(r'\b(that|which|who|where|when)\b', '', clean_line)
                clean_line = re.sub(r'\s+', ' ', clean_line).strip()
                summary_points.append(clean_line)
    
    # Limit to max sentences
    summary_points = summary_points[:max_sentences]
    
    if not summary_points:
        return "This recommendation-response pair shows standard government consultation process."
    
    # Create coherent summary
    if len(summary_points) == 1:
        return summary_points[0]
    else:
        return '. '.join(summary_points) + '.'

def generate_key_insights(recommendation: Dict, responses: List[Dict]) -> List[str]:
    """Generate key insights from recommendation-response analysis"""
    insights = []
    
    if responses:
        # Analyze response types
        response_types = [r['response']['response_type'] for r in responses]
        
        if 'Acceptance' in response_types:
            insights.append("✅ At least one positive response found")
        
        if 'Rejection' in response_types:
            insights.append("❌ Some rejections or concerns identified")
        
        if 'Under Review' in response_types:
            insights.append("⏳ Response indicates ongoing consideration")
        
        if 'Partial Acceptance' in response_types:
            insights.append("🔄 Partial or qualified acceptance noted")
        
        # Document spread
        response_docs = set(r['response']['document']['filename'] for r in responses)
        if len(response_docs) > 1:
            insights.append(f"📄 Responses found across {len(response_docs)} documents")
        
        # Similarity analysis
        avg_similarity = sum(r['combined_score'] for r in responses) / len(responses)
        if avg_similarity > 0.7:
            insights.append("🎯 High confidence in recommendation-response matching")
        elif avg_similarity < 0.5:
            insights.append("⚠️ Low confidence - may need manual review")
        
        # Timeline analysis (basic)
        rec_type = recommendation['recommendation_type']
        if rec_type == 'Urgent' and 'Implementation' in response_types:
            insights.append("⚡ Urgent recommendation received implementation response")
        
    else:
        insights.append("❓ No corresponding responses identified")
        insights.append("📝 May require follow-up or be addressed elsewhere")
    
    return insights

def export_alignment_report(alignments: List[Dict]):
    """Export detailed alignment report"""
    
    report = f"""# Recommendation-Response Alignment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Recommendations:** {len(alignments)}
- **Recommendations with Responses:** {sum(1 for a in alignments if a['responses'])}
- **Average Alignment Confidence:** {sum(a['alignment_confidence'] for a in alignments) / len(alignments):.3f if alignments else 0}
- **High Confidence Alignments:** {sum(1 for a in alignments if a['alignment_confidence'] > 0.7)}

## Recommendation Types Analysis
"""
    
    # Analyze recommendation types
    rec_types = [a['recommendation']['recommendation_type'] for a in alignments]
    type_counts = {t: rec_types.count(t) for t in set(rec_types)}
    
    for rec_type, count in sorted(type_counts.items()):
        report += f"- **{rec_type}:** {count} recommendations\n"
    
    report += "\n## Detailed Analysis\n"
    
    for i, alignment in enumerate(alignments, 1):
        rec = alignment['recommendation']
        responses = alignment['responses']
        
        report += f"""
### {i}. {rec['recommendation_type']} Recommendation
**Document:** {rec['document']['filename']} (Page {rec['page_number']})  
**Confidence:** {alignment['alignment_confidence']:.3f} ({alignment['alignment_status']})

**Recommendation Text:**
> {rec['sentence']}

"""
        
        if responses:
            report += f"**{len(responses)} Related Response(s) Found:**\n\n"
            for j, resp_match in enumerate(responses, 1):
                resp = resp_match['response']
                report += f"**Response {j}:** {resp['response_type']} (Similarity: {resp_match['combined_score']:.3f})  \n"
                report += f"*Source: {resp['document']['filename']} (Page {resp['page_number']})*\n\n"
                report += f"> {resp['sentence']}\n\n"
        else:
            report += "**Status:** No corresponding responses found\n\n"
        
        report += "---\n\n"
    
    # Download button
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"recommendation_alignment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

def export_alignment_csv(alignments: List[Dict]):
    """Export alignments to CSV"""
    
    csv_data = []
    
    for i, alignment in enumerate(alignments, 1):
        rec = alignment['recommendation']
        
        base_row = {
            'Recommendation_ID': i,
            'Recommendation_Text': rec['sentence'],
            'Recommendation_Type': rec['recommendation_type'],
            'Recommendation_Document': rec['document']['filename'],
            'Recommendation_Page': rec['page_number'],
            'Alignment_Confidence': alignment['alignment_confidence'],
            'Alignment_Status': alignment['alignment_status'],
            'Pattern_Matched': rec['pattern_matched']
        }
        
        if alignment['responses']:
            for j, resp_match in enumerate(alignment['responses'], 1):
                resp = resp_match['response']
                row = base_row.copy()
                row.update({
                    'Response_Number': j,
                    'Response_Text': resp['sentence'],
                    'Response_Type': resp['response_type'],
                    'Response_Document': resp['document']['filename'],
                    'Response_Page': resp['page_number'],
                    'Response_Similarity': resp_match['combined_score'],
                    'Topic_Similarity': resp_match['topic_similarity'],
                    'Response_Pattern': resp['pattern_matched']
                })
                csv_data.append(row)
        else:
            base_row.update({
                'Response_Number': 0,
                'Response_Text': 'No response found',
                'Response_Type': 'None',
                'Response_Document': 'None',
                'Response_Page': 0,
                'Response_Similarity': 0,
                'Topic_Similarity': 0,
                'Response_Pattern': 'None'
            })
            csv_data.append(base_row)
    
    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"recommendation_alignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
