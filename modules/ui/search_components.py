# ===============================================
# FILE: modules/ui/search_components.py
# ===============================================

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

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
        def get_collection_stats(self): return {'total_documents': 0}
    class RAGQueryEngine:
        def query(self, question): return {'answer': 'Mock response', 'sources': []}

def render_smart_search_tab():
    """Render the smart search tab"""
    st.header("🔎 Smart Search Engine")
    
    st.markdown("""
    Search across all your uploaded documents using AI-powered semantic search. 
    Find relevant content, recommendations, and responses using natural language queries.
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Indexed Documents", indexed_docs)
    
    with col2:
        available_docs = len(st.session_state.get('uploaded_documents', []))
        st.metric("📁 Available Documents", available_docs)
    
    with col3:
        search_ready = "✅ Ready" if indexed_docs > 0 else "❌ Not Ready"
        st.metric("🔍 Search Status", search_ready)
    
    return True

def render_search_interface():
    """Render the main search interface"""
    st.subheader("🔍 Search Query")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'recommendations about safety training' or 'implementation of ground radar systems'",
            help="Use natural language to search across all documents",
            key="search_query_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Quick search suggestions
    if not search_query:
        render_search_suggestions()
    
    # Execute search
    if search_button and search_query:
        execute_search(search_query)
    elif search_button and not search_query:
        st.warning("Please enter a search query.")

def render_search_suggestions():
    """Render search suggestions based on extracted content"""
    st.markdown("**💡 Search Suggestions:**")
    
    # Generate suggestions based on available content
    suggestions = []
    
    # From recommendations
    recommendations = st.session_state.get('extracted_recommendations', [])
    if recommendations:
        suggestions.extend([
            "safety recommendations",
            "training requirements",
            "implementation guidelines",
            "quality improvements"
        ])
    
    # From concerns
    concerns = st.session_state.get('extracted_concerns', [])
    if concerns:
        suggestions.extend([
            "safety concerns",
            "identified issues",
            "risk factors",
            "operational challenges"
        ])
    
    # From annotations
    annotations = st.session_state.get('annotation_results', {})
    if annotations:
        # Extract common themes
        themes = set()
        for result in annotations.values():
            for framework, theme_list in result.get('annotations', {}).items():
                for theme in theme_list:
                    theme_name = theme.get('theme', '').lower()
                    if theme_name:
                        themes.add(theme_name)
        
        suggestions.extend(list(themes)[:4])
    
    # Display as clickable buttons
    if suggestions:
        suggestion_cols = st.columns(min(len(suggestions), 4))
        
        for i, suggestion in enumerate(suggestions[:8]):  # Limit to 8 suggestions
            col_idx = i % 4
            with suggestion_cols[col_idx]:
                if st.button(f"🔗 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.search_query_input = suggestion
                    st.rerun()

def render_search_history():
    """Render search history"""
    search_history = st.session_state.get('search_history', [])
    
    if search_history:
        with st.expander("📜 Recent Searches"):
            st.markdown("Click on a previous search to run it again:")
            
            # Show recent searches as buttons
            for i, search_item in enumerate(reversed(search_history[-10:])):  # Last 10 searches
                query = search_item.get('query', '')
                timestamp = search_item.get('timestamp', '')
                result_count = search_item.get('result_count', 0)
                
                if st.button(
                    f"🔍 {query} ({result_count} results) - {timestamp[:19]}",
                    key=f"history_{i}",
                    use_container_width=True
                ):
                    st.session_state.search_query_input = query
                    st.rerun()
            
            # Clear history option
            if st.button("🗑️ Clear Search History", type="secondary"):
                st.session_state.search_history = []
                st.rerun()

def render_advanced_search():
    """Render advanced search options"""
    with st.expander("🔧 Advanced Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_mode = st.selectbox(
                "Search Mode:",
                ["Semantic Search", "RAG Query", "Hybrid"],
                index=2,  # Default to Hybrid
                help="Choose how to process your search",
                key="search_mode"
            )
            
            max_results = st.slider(
                "Max Results:",
                1, 50, 10,
                help="Maximum number of results to return",
                key="max_search_results"
            )
        
        with col2:
            content_type_filter = st.multiselect(
                "Content Type Filter:",
                ["recommendation", "response", "concern", "general"],
                default=[],
                help="Filter by content type",
                key="content_type_filter"
            )
            
            confidence_threshold = st.slider(
                "Similarity Threshold:",
                0.0, 1.0, 0.3, 0.05,
                help="Minimum similarity score",
                key="search_similarity_threshold"
            )
        
        with col3:
            document_filter = st.multiselect(
                "Document Filter:",
                [doc['filename'] for doc in st.session_state.get('uploaded_documents', [])],
                default=[],
                help="Search only in selected documents",
                key="document_filter"
            )
            
            include_metadata = st.checkbox(
                "Include Metadata",
                value=True,
                help="Include document metadata in results",
                key="include_search_metadata"
            )

def execute_search(query: str):
    """Execute search query and store results"""
    if not query.strip():
        st.warning("Please enter a search query.")
        return
    
    vector_store = st.session_state.vector_store_manager
    rag_engine = st.session_state.get('rag_engine')
    
    if not vector_store:
        st.error("Search not available - vector store not initialized.")
        return
    
    # Get search configuration
    search_mode = st.session_state.get('search_mode', 'Hybrid')
    max_results = st.session_state.get('max_search_results', 10)
    content_type_filter = st.session_state.get('content_type_filter', [])
    confidence_threshold = st.session_state.get('search_similarity_threshold', 0.3)
    document_filter = st.session_state.get('document_filter', [])
    include_metadata = st.session_state.get('include_search_metadata', True)
    
    with st.spinner(f"🔍 Searching for: '{query}'..."):
        try:
            search_results = []
            rag_response = None
            
            # Execute search based on mode
            if search_mode in ["Semantic Search", "Hybrid"]:
                # Semantic search
                filter_dict = {}
                
                if content_type_filter:
                    # For now, we'll filter results after retrieval
                    pass
                
                # Get semantic search results
                semantic_results = vector_store.similarity_search_with_score(
                    query, 
                    k=max_results * 2,  # Get more to allow for filtering
                    filter_dict=filter_dict if filter_dict else None
                )
                
                # Process and filter results
                for doc, score in semantic_results:
                    if score >= confidence_threshold:
                        # Apply content type filter
                        if content_type_filter:
                            doc_content_type = doc.metadata.get('content_type', 'general')
                            if doc_content_type not in content_type_filter:
                                continue
                        
                        # Apply document filter
                        if document_filter:
                            doc_source = doc.metadata.get('source', '')
                            if doc_source not in document_filter:
                                continue
                        
                        result_item = {
                            'type': 'semantic',
                            'content': doc.page_content,
                            'score': float(score),
                            'source': doc.metadata.get('source', 'Unknown'),
                            'metadata': doc.metadata if include_metadata else {},
                            'content_type': doc.metadata.get('content_type', 'general')
                        }
                        search_results.append(result_item)
                        
                        if len(search_results) >= max_results:
                            break
            
            if search_mode in ["RAG Query", "Hybrid"] and rag_engine:
                # RAG query
                try:
                    rag_response = rag_engine.query(query, include_sources=True)
                except Exception as e:
                    st.warning(f"RAG query failed: {str(e)}")
                    rag_response = None
            
            # Store search results
            search_result = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'mode': search_mode,
                'semantic_results': search_results,
                'rag_response': rag_response,
                'result_count': len(search_results),
                'configuration': {
                    'max_results': max_results,
                    'confidence_threshold': confidence_threshold,
                    'content_type_filter': content_type_filter,
                    'document_filter': document_filter
                }
            }
            
            # Update session state
            if 'search_results' not in st.session_state:
                st.session_state.search_results = {}
            
            search_id = f"search_{len(st.session_state.search_results) + 1}"
            st.session_state.search_results[search_id] = search_result
            
            # Add to search history
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
            
            history_item = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'result_count': len(search_results),
                'search_id': search_id
            }
            st.session_state.search_history.append(history_item)
            
            # Keep only last 50 searches
            if len(st.session_state.search_history) > 50:
                st.session_state.search_history = st.session_state.search_history[-50:]
            
            # Show results summary
            if search_results or rag_response:
                st.success(f"✅ Search completed! Found {len(search_results)} semantic results.")
                if rag_response:
                    st.info("🤖 RAG response generated successfully.")
            else:
                st.warning("⚠️ No results found. Try adjusting your search terms or filters.")
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            st.error(f"❌ {error_msg}")
            add_error_message(error_msg)
            logging.error(f"Search error: {e}", exc_info=True)

def display_search_results():
    """Display search results"""
    search_results = st.session_state.get('search_results', {})
    
    if not search_results:
        st.info("💡 No search results yet. Enter a query above to start searching.")
        return
    
    # Get the most recent search
    latest_search_id = max(search_results.keys(), key=lambda x: search_results[x]['timestamp'])
    latest_result = search_results[latest_search_id]
    
    st.subheader(f"🔍 Results for: '{latest_result['query']}'")
    
    # Results tabs
    semantic_results = latest_result.get('semantic_results', [])
    rag_response = latest_result.get('rag_response')
    
    if semantic_results and rag_response:
        semantic_tab, rag_tab, combined_tab = st.tabs(["📄 Document Results", "🤖 AI Response", "📊 Combined View"])
    elif semantic_results:
        semantic_tab = st.container()
        rag_tab = None
        combined_tab = None
    elif rag_response:
        rag_tab = st.container()
        semantic_tab = None
        combined_tab = None
    else:
        st.warning("No results to display.")
        return
    
    # Display semantic results
    if semantic_tab and semantic_results:
        with semantic_tab:
            display_semantic_results(semantic_results, latest_result)
    
    # Display RAG response
    if rag_tab and rag_response:
        with rag_tab:
            display_rag_response(rag_response, latest_result)
    
    # Display combined view
    if combined_tab and semantic_results and rag_response:
        with combined_tab:
            display_combined_search_view(semantic_results, rag_response, latest_result)

def display_semantic_results(results: List[Dict], search_info: Dict):
    """Display semantic search results"""
    if not results:
        st.info("No semantic results found.")
        return
    
    # Results summary
    st.markdown(f"**Found {len(results)} relevant document chunks**")
    
    # Filter and sort options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_filter = st.slider(
            "Min Score:",
            0.0, 1.0, 0.0, 0.05,
            key="semantic_score_filter"
        )
    
    with col2:
        content_type_display_filter = st.selectbox(
            "Content Type:",
            ['All'] + sorted(list(set(r['content_type'] for r in results))),
            key="semantic_content_type_display_filter"
        )
    
    with col3:
        sort_option = st.selectbox(
            "Sort by:",
            ["Relevance (High to Low)", "Relevance (Low to High)", "Source", "Content Type"],
            key="semantic_sort_option"
        )
    
    # Apply filters
    filtered_results = []
    for result in results:
        if result['score'] >= score_filter:
            if content_type_display_filter == 'All' or result['content_type'] == content_type_display_filter:
                filtered_results.append(result)
    
    # Apply sorting
    if sort_option == "Relevance (High to Low)":
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
    elif sort_option == "Relevance (Low to High)":
        filtered_results.sort(key=lambda x: x['score'])
    elif sort_option == "Source":
        filtered_results.sort(key=lambda x: x['source'])
    elif sort_option == "Content Type":
        filtered_results.sort(key=lambda x: x['content_type'])
    
    if not filtered_results:
        st.warning("No results match the current filters.")
        return
    
    st.write(f"Showing {len(filtered_results)} of {len(results)} results")
    
    # Display results
    for i, result in enumerate(filtered_results, 1):
        display_semantic_result_item(result, i)
    
    # Export option
    if st.button("📥 Export Semantic Results", use_container_width=True):
        export_semantic_results(filtered_results, search_info)

def display_semantic_result_item(result: Dict, index: int):
    """Display a single semantic search result"""
    score = result['score']
    content_type = result['content_type']
    source = result['source']
    content = result['content']
    
    # Color coding based on score
    if score >= 0.8:
        score_color = "🟢"
    elif score >= 0.6:
        score_color = "🟡"
    else:
        score_color = "🔴"
    
    # Content type emoji
    type_emoji = {
        'recommendation': '💡',
        'response': '✅',
        'concern': '⚠️',
        'general': '📄'
    }.get(content_type, '📄')
    
    with st.expander(f"{index}. {score_color} {type_emoji} {source} - {content_type} (score: {score:.3f})"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Content:**")
            st.write(content)
        
        with col2:
            st.markdown("**Details:**")
            st.write(f"**Source:** {source}")
            st.write(f"**Type:** {content_type}")
            st.write(f"**Score:** {score:.3f}")
            
            # Show metadata if available
            metadata = result.get('metadata', {})
            if metadata:
                st.markdown("**Metadata:**")
                for key, value in metadata.items():
                    if key not in ['source', 'content_type']:  # Don't repeat already shown info
                        st.write(f"**{key}:** {value}")

def display_rag_response(rag_response: Dict, search_info: Dict):
    """Display RAG query response"""
    answer = rag_response.get('answer', 'No answer generated')
    confidence = rag_response.get('confidence', 0)
    sources = rag_response.get('sources', [])
    
    # Answer
    st.subheader("🤖 AI-Generated Response")
    st.write(answer)
    
    # Confidence indicator
    if confidence >= 0.8:
        conf_color = "🟢"
        conf_text = "High"
    elif confidence >= 0.6:
        conf_color = "🟡"
        conf_text = "Medium"
    else:
        conf_color = "🔴"
        conf_text = "Low"
    
    st.markdown(f"**Confidence:** {conf_color} {conf_text} ({confidence:.2f})")
    
    # Sources
    if sources:
        st.subheader("📚 Sources Used")
        
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source.get('metadata', {}).get('source', 'Unknown')}"):
                content = source.get('content', 'No content available')
                st.write(content)
                
                metadata = source.get('metadata', {})
                if metadata:
                    st.markdown("**Metadata:**")
                    for key, value in metadata.items():
                        st.write(f"**{key}:** {value}")
    
    # Export option
    if st.button("📥 Export RAG Response", use_container_width=True):
        export_rag_response(rag_response, search_info)

def display_combined_search_view(semantic_results: List[Dict], rag_response: Dict, search_info: Dict):
    """Display combined view of semantic and RAG results"""
    st.subheader("📊 Combined Search Analysis")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Semantic Results", len(semantic_results))
    
    with col2:
        avg_score = sum(r['score'] for r in semantic_results) / len(semantic_results) if semantic_results else 0
        st.metric("Avg Relevance", f"{avg_score:.2f}")
    
    with col3:
        rag_confidence = rag_response.get('confidence', 0)
        st.metric("RAG Confidence", f"{rag_confidence:.2f}")
    
    with col4:
        source_count = len(rag_response.get('sources', []))
        st.metric("RAG Sources", source_count)
    
    # Content type distribution
    if semantic_results:
        st.subheader("📈 Content Type Distribution")
        type_counts = {}
        for result in semantic_results:
            content_type = result['content_type']
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        type_df = pd.DataFrame(list(type_counts.items()), columns=['Content Type', 'Count'])
        st.bar_chart(type_df.set_index('Content Type'))
    
    # Score distribution
    if semantic_results:
        st.subheader("📊 Relevance Score Distribution")
        scores = [r['score'] for r in semantic_results]
        score_df = pd.DataFrame({'Relevance Score': scores})
        st.histogram_chart(score_df, x='Relevance Score')
    
    # Export combined results
    if st.button("📥 Export All Results", use_container_width=True):
        export_combined_search_results(semantic_results, rag_response, search_info)

def export_semantic_results(results: List[Dict], search_info: Dict):
    """Export semantic search results as CSV"""
    if not results:
        st.warning("No results to export.")
        return
    
    # Prepare data for export
    export_data = []
    for i, result in enumerate(results, 1):
        export_data.append({
            'Rank': i,
            'Query': search_info['query'],
            'Content': result['content'],
            'Source': result['source'],
            'Content_Type': result['content_type'],
            'Relevance_Score': result['score'],
            'Search_Timestamp': search_info['timestamp']
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Semantic Results CSV",
        data=csv,
        file_name=f"semantic_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_rag_response(rag_response: Dict, search_info: Dict):
    """Export RAG response as text file"""
    query = search_info['query']
    answer = rag_response.get('answer', '')
    confidence = rag_response.get('confidence', 0)
    sources = rag_response.get('sources', [])
    timestamp = search_info['timestamp']
    
    # Format export content
    export_content = f"""RAG Query Response
==================

Query: {query}
Timestamp: {timestamp}
Confidence: {confidence:.2f}

Answer:
{answer}

Sources Used:
"""
    
    for i, source in enumerate(sources, 1):
        export_content += f"\n{i}. {source.get('metadata', {}).get('source', 'Unknown')}\n"
        export_content += f"   {source.get('content', '')[:200]}...\n"
    
    st.download_button(
        label="📥 Download RAG Response",
        data=export_content.encode('utf-8'),
        file_name=f"rag_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

def export_combined_search_results(semantic_results: List[Dict], rag_response: Dict, search_info: Dict):
    """Export combined search results as JSON"""
    combined_data = {
        'search_info': {
            'query': search_info['query'],
            'timestamp': search_info['timestamp'],
            'mode': search_info['mode'],
            'configuration': search_info.get('configuration', {})
        },
        'semantic_results': semantic_results,
        'rag_response': rag_response,
        'summary': {
            'semantic_result_count': len(semantic_results),
            'rag_confidence': rag_response.get('confidence', 0),
            'rag_source_count': len(rag_response.get('sources', []))
        }
    }
    
    json_str = json.dumps(combined_data, indent=2, ensure_ascii=False, default=str)
    
    st.download_button(
        label="📥 Download Combined Results JSON",
        data=json_str.encode('utf-8'),
        file_name=f"combined_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
