# modules/ui/search_components.py
# COMPLETE FILE - Clear search options + Keep existing + Add RAG + Hybrid

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
    
    This gives you:
    1. Fast, reliable keyword matches (Smart Search)
    2. Intelligent concept matches (AI Semantic)
    3. Best overall coverage
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

# ========== ENHANCED SEARCH FUNCTIONS (KEEP EXISTING) ==========

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

# ========== RESULT DISPLAY FUNCTIONS ==========

def display_results_grouped(results: List[Dict], query: str, search_time: float):
    """Display results grouped by document"""
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
        
        with st.expander(f"📄 {doc_filename} ({len(doc_results)} matches, score: {highest_score:.1f}) {type_display}", 
                        expanded=(len(sorted_groups) <= 3)):
            
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
                search_type_name = {
                    'smart': '🧠 Smart Search',
                    'exact': '🎯 Exact Match', 
                    'fuzzy': '🌀 Fuzzy Search',
                    'rag_semantic': '🤖 AI Semantic',
                    'hybrid_smart': '🔄🧠 Hybrid (Smart)',
                    'hybrid_rag': '🔄🤖 Hybrid (AI)'
                }.get(result['search_type'], '🔍 Search')
                
                st.markdown(f"**{search_type_name} - Match {i}** (Score: {result['score']:.1f})")
                
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
        
        with st.expander(f"{search_type_name} {i}. {doc['filename']} (Score: {result['score']:.1f})", 
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
