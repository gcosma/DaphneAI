# modules/ui/search_components.py
# COMPLETE FILE - Enhanced search with page numbers and detailed info

import streamlit as st
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

def render_search_interface(documents: List[Dict[str, Any]]):
    """Main search interface with clear, simple options"""
    st.header("🔍 Document Search")
    
    if not documents:
        st.info("No documents uploaded. Please upload documents first.")
        # Add sample data button for testing
        if st.button("🎯 Load Sample Data"):
            sample_docs = [
                {
                    'filename': 'sample_policy.txt',
                    'text': 'The government recommends implementing new healthcare policies to improve patient access and reduce waiting times. This policy should be considered for immediate implementation. We recommend further consultation with stakeholders.',
                    'word_count': 28,
                    'file_type': 'txt'
                },
                {
                    'filename': 'sample_response.txt', 
                    'text': 'The department accepts the recommendation for healthcare reform. We will establish a task force to oversee the implementation of these critical changes. The committee recommends monthly reviews and suggests immediate action.',
                    'word_count': 32,
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
    
    # Search method selection with clear explanations
    st.markdown("**Choose how to search:**")
    
    # Check if AI is available
    ai_available = check_rag_availability()
    
    # Create clear options with explanations
    search_options = [
        ("Smart Search", "🧠", "Finds keywords and phrases - recommended for most searches"),
        ("Exact Match", "🎯", "Finds exact words only - fastest, most precise"),
        ("Fuzzy Search", "🌀", "Handles typos and misspellings"),
    ]
    
    # Add AI options if available
    if ai_available:
        search_options.extend([
            ("AI Semantic", "🤖", "AI finds related concepts and meanings - slower but smarter"),
            ("Hybrid", "🔄", "Combines Smart + AI for best results - recommended for exploration")
        ])
    
    # Display as selectbox with clear labels
    search_labels = [f"{emoji} {name}" for name, emoji, desc in search_options]
    search_choice = st.selectbox(
        "Search Method:",
        options=search_labels,
        help="Choose the search method that fits your needs"
    )
    
    # Show explanation for selected method
    for name, emoji, desc in search_options:
        if f"{emoji} {name}" == search_choice:
            st.info(f"ℹ️ **{name}**: {desc}")
            break
    
    # Settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_results = st.number_input(
            "Max Results", 
            min_value=1, 
            max_value=100, 
            value=20,
            help="Maximum results to show"
        )
    
    with col2:
        group_by_doc = st.checkbox(
            "Group by Document", 
            value=True,
            help="Group multiple matches from same document"
        )
    
    with col3:
        if ai_available:
            min_similarity = st.slider(
                "AI Similarity", 
                0.0, 1.0, 0.1, 0.05,
                help="Minimum similarity for AI results"
            )
        else:
            min_similarity = 0.1
    
    # Show AI installation hint if needed
    if not ai_available:
        with st.expander("🤖 Want AI Search? (Optional)"):
            st.markdown("""
            **Install AI libraries for smarter search:**
            ```bash
            pip install sentence-transformers torch
            ```
            
            **AI search finds concepts, not just keywords:**
            - Searching "recommend" also finds "suggest", "advise", "propose"
            - Understands context and meaning
            - Great for exploring unfamiliar documents
            """)
    
    # Perform search
    if query:
        # Extract method name from selection
        method_name = search_choice.split(" ", 1)[1]  # Remove emoji
        
        start_time = time.time()
        
        try:
            # Route to appropriate search function
            if method_name == "Smart Search":
                results = smart_search_enhanced(query, documents, max_results)
                search_info = "Smart keyword matching"
                
            elif method_name == "Exact Match":
                results = exact_search_enhanced(query, documents, max_results)
                search_info = "Exact word matching"
                
            elif method_name == "Fuzzy Search":
                results = fuzzy_search_enhanced(query, documents, max_results)
                search_info = "Typo-tolerant search"
                
            elif method_name == "AI Semantic" and ai_available:
                with st.spinner("🤖 AI is analyzing document meanings..."):
                    results = rag_semantic_search(query, documents, max_results, min_similarity)
                search_info = "AI semantic analysis"
                
            elif method_name == "Hybrid" and ai_available:
                with st.spinner("🔄 Combining Smart + AI search..."):
                    results = hybrid_search(query, documents, max_results, min_similarity)
                search_info = "Hybrid (Smart + AI) search"
                
            else:
                # Fallback to smart search
                results = smart_search_enhanced(query, documents, max_results)
                search_info = "Smart search (fallback)"
            
            search_time = time.time() - start_time
            
            # Log search for analytics
            log_search(query, results, search_time, method_name.lower().replace(' ', '_'))
            
            # Display results
            display_search_results(results, query, search_time, method_name, search_info, group_by_doc)
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            st.info("💡 Try a different search method or check your query")
            logging.error(f"Search failed: {e}")

def display_search_results(results, query, search_time, method_name, search_info, group_by_doc):
    """Display search results with clear explanations"""
    
    if not results:
        st.warning(f"No results found for '{query}' using {method_name}")
        
        # Suggest alternatives
        st.markdown("**💡 Try these alternatives:**")
        if method_name == "Exact Match":
            st.info("• Use **Smart Search** for broader keyword matching")
        elif method_name == "Smart Search":
            st.info("• Try **Fuzzy Search** if there might be typos")
            if check_rag_availability():
                st.info("• Try **AI Semantic** to find related concepts")
        elif method_name == "Fuzzy Search":
            if check_rag_availability():
                st.info("• Try **AI Semantic** to find related concepts")
            else:
                st.info("• Try **Smart Search** for standard keyword matching")
        elif "AI" in method_name:
            st.info("• Try **Smart Search** for exact keyword matching")
        
        return
    
    # Success message
    doc_count = len(set(r['document']['filename'] for r in results))
    st.success(f"✅ {search_info} complete!")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Results", len(results))
    with col2:
        st.metric("Documents", doc_count)
    with col3:
        st.metric("Time", f"{search_time:.3f}s")
    
    # Show breakdown for hybrid search
    if method_name == "Hybrid":
        smart_results = [r for r in results if 'smart' in r.get('search_type', '')]
        ai_results = [r for r in results if 'rag' in r.get('search_type', '')]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🧠 Smart Matches", len(smart_results))
        with col2:
            st.metric("🤖 AI Matches", len(ai_results))
    
    # Display results
    if group_by_doc:
        display_results_grouped(results, query, search_time)
    else:
        display_results_flat(results, query, search_time)

# ========== RAG SEARCH FUNCTIONS ==========

def check_rag_availability() -> bool:
    """Check if RAG dependencies are available"""
    try:
        import sentence_transformers
        import torch
        return True
    except ImportError:
        return False

def rag_semantic_search(query: str, documents: List[Dict], max_results: int, min_similarity: float = 0.1) -> List[Dict]:
    """RAG semantic search using sentence transformers"""
    try:
        from modules.search.rag_search import RAGSearchEngine
        
        # Initialize RAG engine (cached)
        if 'rag_engine' not in st.session_state:
            st.session_state.rag_engine = RAGSearchEngine()
        
        rag_engine = st.session_state.rag_engine
        
        # Perform RAG search
        rag_results = rag_engine.search(query, documents, max_results)
        
        # Convert to standard format
        results = []
        for rag_result in rag_results:
            if rag_result.get('similarity', 0) >= min_similarity:
                results.append({
                    'document': rag_result['document'],
                    'score': rag_result['similarity'] * 100,  # Convert to 0-100 scale
                    'matches': [f"Semantic similarity: {rag_result['similarity']:.3f}"],
                    'context': rag_result.get('snippet', ''),
                    'search_type': 'rag_semantic',
                    'match_id': f"rag_{rag_result['document']['filename']}_{rag_result.get('chunk_info', {}).get('chunk_index', 0)}"
                })
        
        return results
        
    except ImportError:
        st.error("RAG search requires: pip install sentence-transformers torch")
        return []
    except Exception as e:
        st.error(f"RAG search error: {str(e)}")
        logging.error(f"RAG search failed: {e}")
        return []

def hybrid_search(query: str, documents: List[Dict], max_results: int, min_similarity: float = 0.1) -> List[Dict]:
    """
    HYBRID SEARCH - Combines Smart Search + AI for best results
    """
    
    # Get results from both methods
    smart_results = smart_search_enhanced(query, documents, max_results // 2)
    
    try:
        ai_results = rag_semantic_search(query, documents, max_results // 2, min_similarity)
    except:
        # If AI fails, just use smart search
        ai_results = []
    
    # Combine results intelligently
    combined = []
    seen_docs = set()
    
    # Add smart search results first (they're reliable)
    for result in smart_results:
        result['search_type'] = 'hybrid_smart'
        combined.append(result)
        seen_docs.add(result['document']['filename'])
    
    # Add AI results that don't duplicate smart results
    for result in ai_results:
        if result['document']['filename'] not in seen_docs:
            result['search_type'] = 'hybrid_rag'
            combined.append(result)
    
    # Sort by score and limit
    combined.sort(key=lambda x: x['score'], reverse=True)
    return combined[:max_results]

# ========== ENHANCED SEARCH FUNCTIONS ==========

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
        
        # Split text into meaningful chunks
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

# ========== UTILITY FUNCTIONS ==========

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

# ========== ENHANCED RESULT DISPLAY FUNCTIONS ==========

def display_results_grouped(results: List[Dict], query: str, search_time: float):
    """Display results grouped by document with detailed information"""
    if not results:
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
    
    # Display each document group
    for doc_filename, doc_results in sorted_groups:
        doc = doc_results[0]['document']
        highest_score = max(r['score'] for r in doc_results)
        search_types = list(set(r['search_type'] for r in doc_results))
        
        # Create display label for search types
        type_labels = {
            'smart': '🧠', 'exact': '🎯', 'fuzzy': '🌀', 
            'rag_semantic': '🤖', 'hybrid_smart': '🔄🧠', 'hybrid_rag': '🔄🤖'
        }
        type_display = ' '.join([type_labels.get(t, '🔍') for t in search_types])
        
        with st.expander(f"📄 {doc_filename} ({len(doc_results)} matches, best score: {highest_score:.1f}) {type_display}", 
                        expanded=(len(sorted_groups) <= 3)):
            
            # Enhanced document metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("File Type", doc.get('file_type', 'unknown').upper())
            with col2:
                st.metric("Words", f"{doc.get('word_count', 0):,}")
            with col3:
                st.metric("Size", f"{doc.get('file_size_mb', 0):.1f} MB")
            with col4:
                # Calculate estimated pages (roughly 250 words per page)
                estimated_pages = max(1, doc.get('word_count', 0) // 250)
                st.metric("Est. Pages", estimated_pages)
            
            # Sort matches within document by score
            doc_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Show all matches from this document with detailed info
            for i, result in enumerate(doc_results, 1):
                display_detailed_match(result, query, i, doc)

def display_detailed_match(result: Dict, query: str, match_number: int, doc: Dict):
    """Display a single match with detailed information"""
    
    search_type_name = {
        'smart': '🧠 Smart Search',
        'exact': '🎯 Exact Match', 
        'fuzzy': '🌀 Fuzzy Search',
        'rag_semantic': '🤖 AI Semantic',
        'hybrid_smart': '🔄🧠 Hybrid (Smart)',
        'hybrid_rag': '🔄🤖 Hybrid (AI)'
    }.get(result['search_type'], '🔍 Search')
    
    # Calculate detailed position info
    position_info = calculate_position_info(result, doc)
    
    # Match header with detailed info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{search_type_name} - Match {match_number}**")
    with col2:
        st.markdown(f"**Score: {result['score']:.1f}**")
    
    # Detailed position information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if position_info['page_number']:
            st.caption(f"📄 Page {position_info['page_number']}")
        else:
            st.caption("📄 Page N/A")
    
    with col2:
        st.caption(f"📍 Position {position_info['char_position']:,}")
    
    with col3:
        st.caption(f"📊 {position_info['percentage_through']:.1f}% through doc")
    
    with col4:
        if position_info['word_position']:
            st.caption(f"🔢 Word {position_info['word_position']:,}")
    
    # Match details
    if result['matches']:
        st.info("🎯 " + " | ".join(result['matches']))
    
    # Enhanced context with more detail
    display_enhanced_context(result, query, position_info)
    
    # Additional match information
    with st.expander(f"🔍 More details for Match {match_number}"):
        display_match_details(result, position_info, doc)

def calculate_position_info(result: Dict, doc: Dict) -> Dict:
    """Calculate detailed position information for a match"""
    text = doc.get('text', '')
    context = result.get('context', '')
    
    # Find the position of this match in the full document
    char_position = 0
    word_position = 0
    page_number = None
    percentage_through = 0
    
    try:
        # For exact matches with position
        if 'position' in result:
            char_position = result['position']
        
        # For chunk-based matches
        elif 'chunk_index' in result:
            chunk_idx = result['chunk_index']
            # Estimate position based on chunk
            if text:
                chunks = split_text_into_chunks(text)
                if chunk_idx < len(chunks):
                    # Find where this chunk starts in the original text
                    chunk_text = chunks[chunk_idx]
                    char_position = text.find(chunk_text)
                    if char_position == -1:
                        # Fallback: estimate based on chunk size
                        avg_chunk_size = len(text) // len(chunks)
                        char_position = chunk_idx * avg_chunk_size
        
        # For RAG matches with chunk info
        elif 'chunk_info' in result:
            chunk_info = result['chunk_info']
            if 'start_word' in chunk_info:
                words = text.split()
                word_position = chunk_info['start_word']
                # Convert word position to character position
                if word_position < len(words):
                    char_position = len(' '.join(words[:word_position]))
        
        # If we still don't have position, try to find context in text
        if char_position == 0 and context:
            # Remove ellipsis and find context in original text
            clean_context = context.replace('...', '').strip()
            if clean_context and len(clean_context) > 10:
                char_position = text.find(clean_context)
                if char_position == -1:
                    # Try with first 50 characters of context
                    short_context = clean_context[:50]
                    char_position = text.find(short_context)
        
        # Calculate percentage through document
        if text and char_position >= 0:
            percentage_through = (char_position / len(text)) * 100
        
        # Estimate page number (roughly 2000 characters per page for PDFs)
        if char_position > 0:
            page_number = max(1, char_position // 2000 + 1)
        
        # Calculate word position if not already set
        if word_position == 0 and char_position > 0:
            text_before = text[:char_position]
            word_position = len(text_before.split())
    
    except Exception as e:
        logging.warning(f"Error calculating position info: {e}")
    
    return {
        'char_position': max(0, char_position),
        'word_position': max(0, word_position),
        'page_number': page_number,
        'percentage_through': max(0, min(100, percentage_through))
    }

def display_enhanced_context(result: Dict, query: str, position_info: Dict):
    """Display context with enhanced highlighting and navigation"""
    
    context = result.get('context', '')
    if not context:
        st.warning("No context available for this match")
        return
    
    # Enhanced highlighting with different colors for different match types
    highlighted_context = enhance_highlighting(context, query, result['search_type'])
    
    # Context header with navigation info
    st.markdown("**📖 Context:**")
    
    # Show where this context appears in the document
    if position_info['page_number']:
        st.caption(f"💡 This appears around page {position_info['page_number']}, {position_info['percentage_through']:.1f}% through the document")
    
    # Display the highlighted context
    st.markdown(highlighted_context, unsafe_allow_html=True)
    
    # Add context length info
    st.caption(f"📏 Showing {len(context)} characters of context")

def enhance_highlighting(text: str, query: str, search_type: str) -> str:
    """Enhanced highlighting with different colors for different search types"""
    if not text or not query:
        return text
    
    # Choose highlight color based on search type
    highlight_colors = {
        'smart': '#FFEB3B',      # Yellow
        'exact': '#4CAF50',      # Green  
        'fuzzy': '#FF9800',      # Orange
        'rag_semantic': '#2196F3', # Blue
        'hybrid_smart': '#9C27B0', # Purple
        'hybrid_rag': '#E91E63'    # Pink
    }
    
    color = highlight_colors.get(search_type, '#FFEB3B')
    
    words = query.lower().split()
    highlighted = text
    
    for word in words:
        if len(word) > 2:  # Skip very short words
            # Create case-insensitive pattern that preserves original case
            pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<mark style="background-color: {color}; padding: 2px; border-radius: 3px; font-weight: bold;">\\1</mark>', 
                highlighted
            )
    
    return highlighted

def display_match_details(result: Dict, position_info: Dict, doc: Dict):
    """Display detailed information about the match"""
    
    # Match metadata
    st.markdown("**🔍 Match Information:**")
    
    match_details = []
    match_details.append(f"**Search Type:** {result['search_type']}")
    match_details.append(f"**Match Score:** {result['score']:.2f}")
    
    if 'match_id' in result:
        match_details.append(f"**Match ID:** {result['match_id']}")
    
    for detail in match_details:
        st.markdown(f"• {detail}")
    
    # Position details
    st.markdown("**📍 Position Details:**")
    position_details = [
        f"**Character Position:** {position_info['char_position']:,}",
        f"**Word Position:** {position_info['word_position']:,}",
        f"**Document Progress:** {position_info['percentage_through']:.2f}%"
    ]
    
    if position_info['page_number']:
        position_details.insert(0, f"**Estimated Page:** {position_info['page_number']}")
    
    for detail in position_details:
        st.markdown(f"• {detail}")
    
    # Context details
    context = result.get('context', '')
    if context:
        st.markdown("**📖 Context Details:**")
        context_details = [
            f"**Context Length:** {len(context)} characters",
            f"**Context Words:** {len(context.split())} words"
        ]
        
        # Check if context is truncated
        if context.startswith('...') or context.endswith('...'):
            context_details.append("**Note:** Context is truncated for display")
        
        for detail in context_details:
            st.markdown(f"• {detail}")
    
    # Advanced options for this match
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"📍 Show Surrounding Text", key=f"surrounding_{result.get('match_id', 'unknown')}"):
            show_surrounding_text(result, doc, position_info)
    
    with col2:
        if st.button(f"📋 Copy Match Info", key=f"copy_{result.get('match_id', 'unknown')}"):
            copy_match_info(result, position_info, doc)

def show_surrounding_text(result: Dict, doc: Dict, position_info: Dict):
    """Show more surrounding text around the match"""
    
    text = doc.get('text', '')
    char_pos = position_info['char_position']
    
    if not text or char_pos <= 0:
        st.warning("Cannot show surrounding text - position not available")
        return
    
    # Extract larger context (1000 characters around the match)
    context_size = 1000
    start = max(0, char_pos - context_size // 2)
    end = min(len(text), char_pos + context_size // 2)
    
    surrounding_text = text[start:end]
    
    # Add markers to show boundaries
    if start > 0:
        surrounding_text = "... " + surrounding_text
    if end < len(text):
        surrounding_text = surrounding_text + " ..."
    
    st.markdown("**📖 Extended Context (±500 characters):**")
    
    # Show with highlighting
    query_words = result.get('matches', [])
    if query_words:
        # Try to extract query from matches
        for match in query_words:
            if ':' in match:
                query = match.split(':')[0].strip().replace('Exact phrase', '').replace("'", "")
                break
    else:
        query = ""
    
    highlighted = enhance_highlighting(surrounding_text, query, result['search_type'])
    st.markdown(highlighted, unsafe_allow_html=True)

def copy_match_info(result: Dict, position_info: Dict, doc: Dict):
    """Display copyable match information"""
    
    match_info = f"""
Match Information:
- Document: {doc['filename']}
- Search Type: {result['search_type']}
- Score: {result['score']:.2f}
- Page: {position_info['page_number'] or 'N/A'}
- Position: {position_info['char_position']:,} chars, {position_info['word_position']:,} words
- Progress: {position_info['percentage_through']:.1f}% through document

Context:
{result.get('context', 'No context available')}
    """.strip()
    
    st.text_area(
        "📋 Match Information (ready to copy):",
        match_info,
        height=200,
        key=f"copy_area_{result.get('match_id', 'unknown')}"
    )

def display_results_flat(results: List[Dict], query: str, search_time: float):
    """Display results in flat list (not grouped)"""
    if not results:
        return
    
    for i, result in enumerate(results, 1):
        doc = result['document']
        search_type_name = {
            'smart': '🧠 Smart',
            'exact': '🎯 Exact', 
            'fuzzy': '🌀 Fuzzy',
            'rag_semantic': '🤖 AI',
            'hybrid_smart': '🔄🧠 Hybrid-Smart',
            'hybrid_rag': '🔄🤖 Hybrid-AI'
        }.get(result['search_type'], '🔍')
        
        # Calculate position info for flat display
        position_info = calculate_position_info(result, doc)
        
        with st.expander(f"{search_type_name} {i}. {doc['filename']} (Score: {result['score']:.1f}) Page {position_info['page_number'] or 'N/A'}", 
                        expanded=(i <= 5)):
            
            # Document metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Type", doc.get('file_type', 'unknown').upper())
            with col2:
                st.metric("Words", f"{doc.get('word_count', 0):,}")
            with col3:
                st.metric("Size", f"{doc.get('file_size_mb', 0):.1f} MB")
            
            # Position info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📄 Page {position_info['page_number'] or 'N/A'}")
            with col2:
                st.caption(f"📍 Position {position_info['char_position']:,}")
            with col3:
                st.caption(f"📊 {position_info['percentage_through']:.1f}% through")
            
            # Match information
            if result['matches']:
                st.info("🎯 " + " | ".join(result['matches']))
            
            # Context with highlighting
            highlighted_context = enhance_highlighting(result['context'], query, result['search_type'])
            st.markdown("**📖 Context:**")
            st.markdown(highlighted_context, unsafe_allow_html=True)

def export_results_to_csv(results: List[Dict], query: str):
    """Export search results to CSV format"""
    
    import pandas as pd
    
    # Prepare data for CSV
    export_data = []
    
    for i, result in enumerate(results, 1):
        doc = result['document']
        position_info = calculate_position_info(result, doc)
        
        export_data.append({
            'Match_Number': i,
            'Query': query,
            'Document': doc['filename'],
            'Search_Type': result['search_type'],
            'Score': result['score'],
            'Page_Number': position_info['page_number'],
            'Character_Position': position_info['char_position'],
            'Word_Position': position_info['word_position'],
            'Percentage_Through': position_info['percentage_through'],
            'Context': result.get('context', ''),
            'Matches': ' | '.join(result.get('matches', [])),
            'File_Type': doc.get('file_type', ''),
            'Word_Count': doc.get('word_count', 0),
            'File_Size_MB': doc.get('file_size_mb', 0)
        })
    
    # Create DataFrame and CSV
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)
    
    # Offer download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{query.replace(' ', '_')}_{timestamp}.csv"
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )
    
    st.success(f"✅ Ready to download {len(results)} results as CSV!")

# ========== ANALYTICS FUNCTIONS ==========

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

# ========== BACKWARD COMPATIBILITY ==========

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
    'rag_semantic_search',
    'hybrid_search',
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
