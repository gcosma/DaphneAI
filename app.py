"""
Streamlit UI for AI Document Search Engine
"""

import streamlit as st
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.search_engine import SmartSearchEngine
from src.core.document_processor import DocumentProcessor

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = SmartSearchEngine()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

def render_sidebar():
    """Render sidebar with system information"""
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        stats = st.session_state.search_engine.get_stats()
        
        # AI Status
        if stats['ai_available'] and stats['ai_enabled']:
            st.success("🤖 AI Search Ready")
        elif stats['ai_available']:
            st.warning("🤖 AI Available (No Index)")
        else:
            st.error("🤖 AI Unavailable")
            st.caption("Install: `pip install sentence-transformers scikit-learn`")
        
        # Document Stats
        st.markdown("### 📊 Document Stats")
        st.metric("Total Documents", stats['total_documents'])
        
        if stats['total_documents'] > 0:
            if stats['index_built']:
                st.success("✅ Search Index Built")
            else:
                st.info("⏳ Building Index...")
        
        # Supported Formats
        st.markdown("### 📁 Supported Formats")
        formats = st.session_state.processor.get_supported_formats()
        
        for fmt, available in formats.items():
            icon = "✅" if available else "❌"
            st.write(f"{icon} {fmt.upper()}")

def render_upload_section():
    """Render file upload interface"""
    st.header("📁 Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md', 'py', 'js', 'html', 'css', 'json'],
        help="Upload documents to search through. Supported formats: PDF, DOCX, TXT, MD, and more."
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            # Process files
            new_documents = st.session_state.processor.process_files(uploaded_files)
            
            # Add to existing documents
            st.session_state.documents.extend(new_documents)
            
            # Update search engine
            st.session_state.search_engine.add_documents(st.session_state.documents)
            
            st.success(f"✅ Processed {len(new_documents)} documents")
            
            # Show processing results
            for doc in new_documents:
                if 'error' in doc:
                    st.error(f"❌ {doc['filename']}: {doc['error']}")
                else:
                    st.info(f"✅ {doc['filename']} - {doc['word_count']} words")

def render_search_section():
    """Render search interface"""
    st.header("🔍 Smart Search")
    
    if not st.session_state.documents:
        st.warning("📝 Please upload documents first to enable search.")
        return
    
    # Search input
    query = st.text_input(
        "Search your documents",
        placeholder="Enter your search query...",
        help="Use natural language or keywords to find relevant information"
    )
    
    # Search button and options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    
    with col2:
        max_results = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
    
    # Perform search
    if search_button and query:
        with st.spinner("Searching..."):
            start_time = time.time()
            
            results = st.session_state.search_engine.search(query, max_results)
            
            search_time = time.time() - start_time
            
            # Add to search history
            st.session_state.search_history.append({
                'query': query,
                'results_count': len(results),
                'search_time': search_time
            })
        
        # Display results
        if results:
            st.success(f"Found {len(results)} results in {search_time:.2f} seconds")
            
            # Results
            for result in results:
                with st.expander(f"#{result.rank} - {result.filename} (Score: {result.score:.3f})"):
                    # Snippet
                    st.markdown("**Preview:**")
                    st.markdown(result.snippet)
                    
                    # Full content toggle
                    if st.button(f"Show full content", key=f"show_{result.rank}"):
                        st.markdown("**Full Content:**")
                        st.text_area(
                            "Document content",
                            value=result.content,
                            height=300,
                            key=f"content_{result.rank}"
                        )
        else:
            st.warning("No results found. Try different keywords or check your documents.")

def render_analytics():
    """Render search analytics"""
    st.header("📈 Search Analytics")
    
    if not st.session_state.search_history:
        st.info("No search history yet. Perform some searches to see analytics.")
        return
    
    # Basic stats
    total_searches = len(st.session_state.search_history)
    avg_results = sum(s['results_count'] for s in st.session_state.search_history) / total_searches
    avg_time = sum(s['search_time'] for s in st.session_state.search_history) / total_searches
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Searches", total_searches)
    
    with col2:
        st.metric("Avg Results", f"{avg_results:.1f}")
    
    with col3:
        st.metric("Avg Time", f"{avg_time:.2f}s")
    
    # Recent searches
    st.subheader("Recent Searches")
    
    for i, search in enumerate(reversed(st.session_state.search_history[-10:])):
        st.write(f"{i+1}. **{search['query']}** - {search['results_count']} results ({search['search_time']:.2f}s)")

def main():
    """Main application function"""
    # Initialize
    initialize_session_state()
    
    # Header
    st.title("🔍 AI Document Search Engine")
    st.markdown("**Upload documents and search with intelligent AI-powered ranking**")
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "🔍 Search", "📈 Analytics"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_search_section()
    
    with tab3:
        render_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("*AI-powered document search with semantic understanding*")

if __name__ == "__main__":
    main()
