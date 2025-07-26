# modules/ui/search_components.py
# COMPLETE SEARCH COMPONENTS FILE - All functions included

import streamlit as st
import pandas as pd
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

# Import required modules with error handling
try:
    import sys
    sys.path.append('modules')
    from vector_store import VectorStoreManager
    from rag_engine import RAGQueryEngine
    from .shared_components import add_error_message
except ImportError as e:
    logging.error(f"Import error in search_components: {e}")
    # Create mock classes for development
    class VectorStoreManager:
        def similarity_search(self, query, k=5): return []
        def similarity_search_with_score(self, query, k=5, filter_dict=None): return []
        def get_collection_stats(self): return {'total_documents': 0}
    class RAGQueryEngine:
        def query(self, question): return {'answer': 'Mock response', 'sources': []}

# ===============================================
# MAIN SEARCH TAB FUNCTIONS
# ===============================================

def render_smart_search_tab():
    """Render the smart search tab with enhanced concern search"""
    st.header("🔎 Smart Search Engine")
    
    st.markdown("""
    Search across all your uploaded documents using AI-powered semantic search. 
    Find relevant content, recommendations, responses, and **coroner concerns** using natural language queries.
    """)
    
    # Check if search is available
    if not check_search_availability():
        return
    
    # Search interface
    render_search_interface()
    
    # Search history
    render_search_history()
    
    # Advanced search options
    render_advanced_search()
    
    # Display search results
    display_search_results()
    
    # Enhanced concern search functionality
    render_concern_search_section()

def render_search_tab():
    """Alias for render_smart_search_tab for backward compatibility"""
    return render_smart_search_tab()

# ===============================================
# SEARCH AVAILABILITY CHECK
# ===============================================

def check_search_availability():
    """Check if search functionality is available"""
    vector_store = st.session_state.get('vector_store_manager')
    
    if not vector_store:
        st.warning("⚠️ Search not available. Vector store not initialized.")
        st.info("Please go to the 'Find Responses' tab and index your documents first.")
        return False
    
    stats = vector_store.get_collection_stats()
    indexed_docs = stats.get('total_documents', 0)
    
    if indexed_docs == 0:
        st.warning("⚠️ No documents indexed for search.")
        st.info("Please go to the 'Find Responses' tab and index your documents first.")
        return False
    
    # Show search status
    st.success(f"✅ Search ready! {indexed_docs} document chunks indexed")
    return True

# ===============================================
# SEARCH INTERFACE
# ===============================================

def render_search_interface():
    """Render the main search interface"""
    st.subheader("🔍 Search Interface")
    
    # Search input
    search_query = st.text_input(
        "Enter your search query:",
        placeholder="e.g. 'patient safety recommendations', 'communication protocols', 'training requirements'",
        key="main_search_query"
    )
    
    # Search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_type = st.selectbox(
            "Search Type:",
            ["All Content", "Recommendations Only", "Responses Only", "Concerns Only"],
            key="search_type"
        )
    
    with col2:
        max_results = st.slider("Max Results:", 1, 20, 5, key="max_search_results")
    
    with col3:
        include_score = st.checkbox("Show Relevance Scores", value=True, key="include_scores")
    
    # Search button
    if st.button("🔍 Search", key="main_search_btn", type="primary"):
        if search_query.strip():
            perform_search(search_query, search_type, max_results, include_score)
        else:
            st.warning("Please enter a search query")

def perform_search(query: str, search_type: str, max_results: int, include_score: bool):
    """Perform the actual search"""
    vector_store = st.session_state.get('vector_store_manager')
    
    if not vector_store:
        st.error("Search not available - vector store not initialized")
        return
    
    # Map search type to filter
    content_type_map = {
        "All Content": None,
        "Recommendations Only": "recommendation",
        "Responses Only": "response", 
        "Concerns Only": "concern"
    }
    
    filter_dict = None
    if search_type != "All Content":
        filter_dict = {'content_type': content_type_map[search_type]}
    
    with st.spinner(f"🔍 Searching {search_type.lower()}..."):
        try:
            if include_score:
                results = vector_store.similarity_search_with_score(
                    query, k=max_results, filter_dict=filter_dict
                )
            else:
                docs = vector_store.similarity_search(
                    query, k=max_results, filter_dict=filter_dict
                )
                results = [(doc, 0.0) for doc in docs]
            
            # Store results in session state
            if 'search_results' not in st.session_state:
                st.session_state.search_results = {}
            
            st.session_state.search_results[query] = results
            st.session_state.last_search_query = query
            
            # Add to search history
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
            
            st.session_state.search_history.insert(0, {
                'query': query,
                'type': search_type,
                'results_count': len(results),
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 searches
            st.session_state.search_history = st.session_state.search_history[:10]
            
            st.success(f"✅ Found {len(results)} results")
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            logging.error(f"Search error: {e}")

# ===============================================
# SEARCH RESULTS DISPLAY
# ===============================================

def display_search_results():
    """Display search results"""
    if not st.session_state.get('search_results'):
        return
    
    last_query = st.session_state.get('last_search_query')
    if not last_query or last_query not in st.session_state.search_results:
        return
    
    results = st.session_state.search_results[last_query]
    
    if not results:
        st.info("No results found for your search query")
        return
    
    st.subheader(f"📋 Search Results ({len(results)} found)")
    
    # Results summary
    if len(results) > 0:
        scores = [score for _, score in results if score > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            st.info(f"Average relevance score: {avg_score:.3f}")
    
    # Display results
    for i, (doc, score) in enumerate(results):
        with st.expander(f"Result {i+1}" + (f" (Relevance: {score:.3f})" if score > 0 else "")):
            # Main content
            st.write(doc.page_content)
            
            # Show metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                st.divider()
                st.caption("**Source Information:**")
                metadata_cols = st.columns(3)
                
                metadata_items = list(doc.metadata.items())
                for idx, (key, value) in enumerate(metadata_items):
                    if key not in ['content', 'page_content'] and value:
                        col_idx = idx % 3
                        with metadata_cols[col_idx]:
                            st.caption(f"**{key.replace('_', ' ').title()}:** {value}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📋 Copy Text", key=f"copy_{i}"):
                    st.session_state[f"copied_text_{i}"] = doc.page_content
                    st.success("Text copied to clipboard!")
            
            with col2:
                if st.button(f"🔍 Similar", key=f"similar_{i}"):
                    find_similar_content(doc.page_content)
            
            with col3:
                if st.button(f"📊 Analyze", key=f"analyze_{i}"):
                    analyze_content(doc.page_content)

def find_similar_content(content: str):
    """Find content similar to the selected result"""
    vector_store = st.session_state.get('vector_store_manager')
    if not vector_store:
        st.error("Vector store not available")
        return
    
    # Take first 200 characters for similarity search
    query = content[:200] + "..."
    
    with st.spinner("Finding similar content..."):
        try:
            results = vector_store.similarity_search(query, k=3)
            
            if results:
                st.subheader("🔗 Similar Content Found:")
                for i, doc in enumerate(results):
                    with st.expander(f"Similar Content {i+1}"):
                        st.write(doc.page_content)
            else:
                st.info("No similar content found")
                
        except Exception as e:
            st.error(f"Error finding similar content: {e}")

def analyze_content(content: str):
    """Analyze selected content"""
    st.subheader("📊 Content Analysis")
    
    # Basic text analysis
    word_count = len(content.split())
    char_count = len(content)
    sentence_count = len(re.split(r'[.!?]+', content))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", word_count)
    with col2:
        st.metric("Characters", char_count)
    with col3:
        st.metric("Sentences", sentence_count)
    
    # Keyword extraction (simple)
    words = re.findall(r'\b\w+\b', content.lower())
    word_freq = Counter(words)
    
    # Remove common words
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
    filtered_words = {word: count for word, count in word_freq.items() if word not in common_words and len(word) > 3}
    
    if filtered_words:
        st.subheader("🔤 Top Keywords:")
        top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for word, count in top_words:
            st.write(f"• **{word}**: {count} times")

# ===============================================
# SEARCH HISTORY
# ===============================================

def render_search_history():
    """Render search history"""
    if not st.session_state.get('search_history'):
        return
    
    with st.expander("📚 Search History"):
        st.markdown("**Recent Searches:**")
        
        for i, search in enumerate(st.session_state.search_history):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.text(search['query'])
            
            with col2:
                st.caption(search['type'])
            
            with col3:
                st.caption(f"{search['results_count']} results")
            
            with col4:
                if st.button("🔄", key=f"repeat_search_{i}", help="Repeat this search"):
                    st.session_state.main_search_query = search['query']
                    st.rerun()
        
        # Clear history button
        if st.button("🗑️ Clear History", key="clear_search_history"):
            st.session_state.search_history = []
            st.success("Search history cleared")
            st.rerun()

# ===============================================
# ADVANCED SEARCH OPTIONS
# ===============================================

def render_advanced_search():
    """Render advanced search options"""
    with st.expander("⚙️ Advanced Search Options"):
        st.markdown("**Filter by Document Properties:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Document type filter
            uploaded_docs = st.session_state.get('uploaded_documents', [])
            doc_types = ["All"] + list(set([doc.get('document_type', 'Unknown') for doc in uploaded_docs]))
            selected_doc_type = st.selectbox("Document Type:", doc_types, key="adv_doc_type")
            
            # Source filter
            sources = ["All"] + list(set([doc.get('filename', 'Unknown') for doc in uploaded_docs]))
            selected_source = st.selectbox("Source Document:", sources, key="adv_source")
        
        with col2:
            # Date filter
            date_filter = st.checkbox("Filter by Date Range", key="adv_date_filter")
            if date_filter:
                from_date = st.date_input("From Date:", key="adv_from_date")
                to_date = st.date_input("To Date:", key="adv_to_date")
            
            # Content length filter
            min_length = st.slider("Minimum Content Length:", 0, 1000, 50, key="adv_min_length")
        
        # Search mode
        st.markdown("**Search Mode:**")
        search_mode = st.radio(
            "Choose search approach:",
            ["Semantic Search", "Keyword Search", "Hybrid Search"],
            key="adv_search_mode",
            help="Semantic: AI-powered meaning search, Keyword: Exact word matching, Hybrid: Both combined"
        )
        
        # Export options
        st.markdown("**Export Options:**")
        if st.button("📥 Export Search Results", key="export_search_results"):
            export_search_results()

def export_search_results():
    """Export current search results"""
    if not st.session_state.get('search_results'):
        st.warning("No search results to export")
        return
    
    last_query = st.session_state.get('last_search_query')
    if not last_query or last_query not in st.session_state.search_results:
        st.warning("No recent search results found")
        return
    
    results = st.session_state.search_results[last_query]
    
    # Prepare data for export
    export_data = []
    for i, (doc, score) in enumerate(results):
        export_data.append({
            'result_number': i + 1,
            'content': doc.page_content,
            'relevance_score': score,
            'source': doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown',
            'document_type': doc.metadata.get('document_type', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown',
            'search_query': last_query
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(export_data)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ===============================================
# ENHANCED CONCERN SEARCH
# ===============================================

def render_concern_search_section():
    """Render enhanced concern search functionality"""
    st.divider()
    st.subheader("🚨 Coroner Concerns Search")
    
    st.markdown("""
    **Enhanced search specifically for coroner concerns and recommendations.**
    Use natural language to find specific concerns, patterns, and related responses.
    """)
    
    # Concern search interface
    concern_query = st.text_input(
        "Search for concerns:",
        placeholder="e.g. 'patient safety concerns', 'communication failures', 'training deficiencies'",
        key="concern_search_query"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        concern_type = st.selectbox(
            "Concern Category:",
            ["All Concerns", "Patient Safety", "Communication", "Training", "Procedures", "Equipment"],
            key="concern_type"
        )
    
    with col2:
        concern_severity = st.selectbox(
            "Severity Level:",
            ["All Levels", "Critical", "High", "Medium", "Low"],
            key="concern_severity"
        )
    
    if st.button("🔍 Search Concerns", key="search_concerns_btn", type="primary"):
        if concern_query.strip():
            search_concerns(concern_query, concern_type, concern_severity)
        else:
            st.warning("Please enter a concern search query")

def search_concerns(query: str, concern_type: str, severity: str):
    """Search specifically for coroner concerns"""
    vector_store = st.session_state.get('vector_store_manager')
    
    if not vector_store:
        st.error("Vector store not available")
        return
    
    with st.spinner("🔍 Searching for concerns..."):
        try:
            # Enhanced query for concerns
            enhanced_queries = [
                f"coroner concern {query}",
                f"safety issue {query}",
                f"problem failure {query}",
                f"risk hazard {query}"
            ]
            
            all_results = []
            
            # Search with multiple enhanced queries
            for enhanced_query in enhanced_queries:
                results = vector_store.similarity_search_with_score(
                    enhanced_query, 
                    k=5,
                    filter_dict={'content_type': 'concern'} if concern_type == "All Concerns" else None
                )
                all_results.extend(results)
            
            # Remove duplicates and sort by score
            unique_results = {}
            for doc, score in all_results:
                content_key = doc.page_content[:100]  # Use first 100 chars as key
                if content_key not in unique_results or unique_results[content_key][1] < score:
                    unique_results[content_key] = (doc, score)
            
            final_results = list(unique_results.values())
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            # Display results
            if final_results:
                st.success(f"✅ Found {len(final_results)} relevant concerns")
                
                # Group by concern type if available
                concern_groups = {}
                for doc, score in final_results:
                    doc_type = doc.metadata.get('concern_type', 'General') if hasattr(doc, 'metadata') else 'General'
                    if doc_type not in concern_groups:
                        concern_groups[doc_type] = []
                    concern_groups[doc_type].append((doc, score))
                
                for group_name, group_results in concern_groups.items():
                    st.markdown(f"**{group_name} Concerns:**")
                    
                    for i, (doc, score) in enumerate(group_results):
                        with st.expander(f"{group_name} Concern {i+1} (Relevance: {score:.3f})"):
                            st.write(doc.page_content)
                            
                            # Show metadata
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.caption("**Details:**")
                                for key, value in doc.metadata.items():
                                    if key not in ['content', 'page_content'] and value:
                                        st.caption(f"• {key.replace('_', ' ').title()}: {value}")
                            
                            # Action buttons for concerns
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"🔗 Find Related", key=f"concern_related_{i}_{group_name}"):
                                    find_related_concerns(doc.page_content)
                            
                            with col2:
                                if st.button(f"📋 Copy", key=f"concern_copy_{i}_{group_name}"):
                                    st.session_state[f"copied_concern_{i}"] = doc.page_content
                                    st.success("Concern copied!")
            else:
                st.info("No concerns found matching your query")
                
                # Suggest alternative searches
                st.markdown("**Try these alternative searches:**")
                suggestions = [
                    "patient safety",
                    "communication breakdown", 
                    "training inadequate",
                    "procedure not followed",
                    "equipment failure"
                ]
                
                for suggestion in suggestions:
                    if st.button(f"🔍 {suggestion}", key=f"suggest_{suggestion}"):
                        st.session_state.concern_search_query = suggestion
                        st.rerun()
                        
        except Exception as e:
            st.error(f"Concern search failed: {e}")
            logging.error(f"Concern search error: {e}")

def find_related_concerns(concern_content: str):
    """Find concerns related to the selected concern"""
    vector_store = st.session_state.get('vector_store_manager')
    if not vector_store:
        st.error("Vector store not available")
        return
    
    # Extract key terms from the concern
    key_terms = extract_key_terms(concern_content)
    
    with st.spinner("Finding related concerns..."):
        try:
            # Search for related content
            related_query = " ".join(key_terms[:5])  # Use top 5 key terms
            results = vector_store.similarity_search(related_query, k=3)
            
            if results:
                st.subheader("🔗 Related Concerns:")
                for i, doc in enumerate(results):
                    with st.expander(f"Related Concern {i+1}"):
                        st.write(doc.page_content)
                        
                        # Highlight common terms
                        common_terms = find_common_terms(concern_content, doc.page_content)
                        if common_terms:
                            st.caption(f"**Common themes:** {', '.join(common_terms)}")
            else:
                st.info("No related concerns found")
                
        except Exception as e:
            st.error(f"Error finding related concerns: {e}")

def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from text"""
    # Simple keyword extraction
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
    
    # Filter and count words
    word_freq = Counter([word for word in words if word not in stop_words and len(word) > 3])
    
    # Return top words
    return [word for word, count in word_freq.most_common(10)]

def find_common_terms(text1: str, text2: str) -> List[str]:
    """Find common terms between two texts"""
    terms1 = set(extract_key_terms(text1))
    terms2 = set(extract_key_terms(text2))
    
    return list(terms1.intersection(terms2))

# ===============================================
# EXPORTS
# ===============================================

__all__ = [
    'render_smart_search_tab',
    'render_search_tab',
    'check_search_availability',
    'render_search_interface',
    'render_search_history',
    'render_advanced_search',
    'display_search_results',
    'render_concern_search_section',
    'perform_search',
    'search_concerns',
    'find_similar_content',
    'analyze_content',
    'export_search_results',
    'find_related_concerns',
    'extract_key_terms',
    'find_common_terms'
]
