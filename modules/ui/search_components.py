# modules/ui/search_components.py - Fixed version with proper error handling
"""
Fixed search components with comprehensive error handling for the "not enough values to unpack" error.
This is a drop-in replacement that should resolve the alignment tab issues.
"""

import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

# Setup logging
logger = logging.getLogger(__name__)

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
            
            # Display results
            display_search_results(
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
        show_alignment_info()
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
                
                # Display results
                display_alignment_analysis(recommendations, responses, documents)
                
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
                
                display_manual_search_results(
                    matches, search_sentence, show_scores
                )
                
            except Exception as e:
                logger.error(f"Manual search error: {e}")
                st.error(f"Search error: {str(e)}")

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
                'total_words': len(query_words)
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
            'page_number': max(1, pos // 2000 + 1)
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
                        'similarity': similarity
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

def get_context_simple(sentences: List[str], index: int, window: int = 1) -> str:
    """Get context around a sentence"""
    
    start = max(0, index - window)
    end = min(len(sentences), index + window + 1)
    
    context_sentences = [s.strip() for s in sentences[start:end] if s.strip()]
    return ' '.join(context_sentences)

def display_search_results(results: List[Dict], query: str, search_time: float, 
                          show_context: bool, highlight_matches: bool):
    """Display search results"""
    
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    # Group by document
    doc_groups = {}
    for result in results:
        doc_name = result['document']['filename']
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(result)
    
    # Summary
    st.success(f"🎯 Found **{len(results)}** results in **{len(doc_groups)}** documents in {search_time:.3f} seconds")
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy Results"):
            copy_results_simple(results, query)
    with col2:
        if st.button("📊 Export CSV"):
            export_results_csv_simple(results, query)
    
    # Display results
    for doc_name, doc_results in doc_groups.items():
        with st.expander(f"📄 {doc_name} ({len(doc_results)} matches)", expanded=True):
            
            for i, result in enumerate(doc_results, 1):
                
                score = result.get('score', 0)
                method = result.get('match_type', 'unknown')
                page = result.get('page_number', 1)
                
                st.markdown(f"**Match {i}** - {method.title()} (Score: {score:.1f}) - Page {page}")
                
                if show_context:
                    context = result.get('context', '')
                    if highlight_matches and context:
                        # Simple highlighting
                        highlighted = highlight_text_simple(context, query)
                        st.markdown(highlighted, unsafe_allow_html=True)
                    else:
                        st.markdown(f"📖 {context}")
                
                st.markdown("---")

def highlight_text_simple(text: str, query: str) -> str:
    """Simple text highlighting"""
    
    highlighted = text
    query_words = query.split()
    
    for word in query_words:
        if len(word) > 2:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                f"<mark style='background-color: #FFEB3B; padding: 2px;'>{word}</mark>", 
                highlighted
            )
    
    return highlighted

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
                
                match = {
                    'document': doc,
                    'pattern': pattern,
                    'sentence': sentence,
                    'position': pos,
                    'page_number': max(1, pos // 2000 + 1),
                    'match_type': match_type
                }
                
                matches.append(match)
                start = pos + 1
    
    return matches

def display_alignment_analysis(recommendations: List[Dict], responses: List[Dict], documents: List[Dict]):
    """Display alignment analysis results"""
    
    st.markdown("### 📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recommendations", len(recommendations))
    
    with col2:
        st.metric("Total Responses", len(responses))
    
    with col3:
        rec_docs = len(set(r['document']['filename'] for r in recommendations))
        st.metric("Documents with Recommendations", rec_docs)
    
    with col4:
        resp_docs = len(set(r['document']['filename'] for r in responses))
        st.metric("Documents with Responses", resp_docs)
    
    # Show recommendations
    if recommendations:
        st.markdown("### 🎯 Recommendations Found")
        
        for i, rec in enumerate(recommendations[:10], 1):  # First 10
            doc_name = rec['document']['filename']
            page = rec['page_number']
            sentence = rec['sentence'][:100] + "..." if len(rec['sentence']) > 100 else rec['sentence']
            
            with st.expander(f"Recommendation {i} - {doc_name} (Page {page})", expanded=i <= 3):
                st.markdown(f"**Pattern:** {rec['pattern']}")
                st.markdown(f"**Content:** {sentence}")
    
    # Show responses
    if responses:
        st.markdown("### ↩️ Responses Found")
        
        for i, resp in enumerate(responses[:10], 1):  # First 10
            doc_name = resp['document']['filename']
            page = resp['page_number']
            sentence = resp['sentence'][:100] + "..." if len(resp['sentence']) > 100 else resp['sentence']
            
            with st.expander(f"Response {i} - {doc_name} (Page {page})", expanded=i <= 3):
                st.markdown(f"**Pattern:** {resp['pattern']}")
                st.markdown(f"**Content:** {sentence}")
    
    # Simple alignment attempt
    if recommendations and responses:
        st.markdown("### 🔗 Simple Alignment Analysis")
        
        # Group by document
        rec_by_doc = {}
        resp_by_doc = {}
        
        for rec in recommendations:
            doc = rec['document']['filename']
            if doc not in rec_by_doc:
                rec_by_doc[doc] = []
            rec_by_doc[doc].append(rec)
        
        for resp in responses:
            doc = resp['document']['filename']
            if doc not in resp_by_doc:
                resp_by_doc[doc] = []
            resp_by_doc[doc].append(resp)
        
        # Show documents with both
        common_docs = set(rec_by_doc.keys()) & set(resp_by_doc.keys())
        
        if common_docs:
            st.success(f"Found {len(common_docs)} documents with both recommendations and responses")
            
            for doc in common_docs:
                st.markdown(f"**📄 {doc}:**")
                st.write(f"  • {len(rec_by_doc[doc])} recommendations")
                st.write(f"  • {len(resp_by_doc[doc])} responses")
        else:
            st.warning("No documents contain both recommendations and responses")

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
                
                match = {
                    'sentence': sentence.strip(),
                    'similarity': similarity,
                    'document': doc,
                    'position': text.find(sentence),
                    'page_number': max(1, text.find(sentence) // 2000 + 1) if sentence in text else 1
                }
                
                matches.append(match)
    
    # Sort by similarity and limit
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches[:max_matches]

def display_manual_search_results(matches: List[Dict], target_sentence: str, show_scores: bool):
    """Display manual search results"""
    
    if not matches:
        st.warning("No similar content found")
        return
    
    st.success(f"Found {len(matches)} similar sentences")
    
    st.markdown("### 📝 Your Original Sentence:")
    st.info(target_sentence)
    
    st.markdown("### 🔍 Similar Content Found:")
    
    for i, match in enumerate(matches, 1):
        similarity = match['similarity']
        doc_name = match['document']['filename']
        page = match['page_number']
        
        confidence = "🟢 High" if similarity > 0.7 else "🟡 Medium" if similarity > 0.4 else "🔴 Low"
        score_text = f" (Score: {similarity:.3f})" if show_scores else ""
        
        with st.expander(f"{confidence} Match {i} - {doc_name} (Page {page}){score_text}", 
                        expanded=i <= 3):
            
            sentence = match['sentence']
            st.markdown(f"**Found:** {sentence}")
            
            if show_scores:
                st.caption(f"Similarity score: {similarity:.3f}")

def copy_results_simple(results: List[Dict], query: str):
    """Simple results copying"""
    
    output = f"Search Results for: {query}\n"
    output += f"Total Results: {len(results)}\n"
    output += "=" * 50 + "\n\n"
    
    for i, result in enumerate(results, 1):
        doc_name = result['document']['filename']
        score = result.get('score', 0)
        context = result.get('context', '')[:200]
        
        output += f"Result {i}:\n"
        output += f"Document: {doc_name}\n"
        output += f"Score: {score:.1f}\n"
        output += f"Context: {context}...\n\n"
    
    st.code(output, language="text")
    st.success("Results displayed above! Copy with Ctrl+A, Ctrl+C")

def export_results_csv_simple(results: List[Dict], query: str):
    """Simple CSV export"""
    
    csv_data = []
    
    for i, result in enumerate(results, 1):
        csv_data.append({
            'Match': i,
            'Query': query,
            'Document': result['document']['filename'],
            'Score': result.get('score', 0),
            'Page': result.get('page_number', 1),
            'Context': result.get('context', '')[:500]
        })
    
    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_alignment_info():
    """Show information about alignment features"""
    
    st.markdown("""
    ### 🎯 What This Feature Does:
    
    **🔍 Automatically finds:**
    - All recommendations in your documents
    - Corresponding responses to those recommendations
    - Aligns them using similarity matching
    
    **📊 Provides:**
    - Side-by-side view of recommendation + response
    - Confidence scores for alignments
    - Export options for further analysis
    
    **💡 Perfect for:**
    - Government inquiry reports
    - Policy documents and responses
    - Committee recommendations and outcomes
    - Audit findings and management responses
    
    ### 🚀 How to Use:
    1. Upload your documents in the **Upload** tab
    2. Return here to analyze recommendations and responses
    3. Configure search patterns for your specific documents
    4. Let the system find and align recommendation-response pairs
    5. Review results and export findings
    """)

# Additional utility functions for compatibility

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
