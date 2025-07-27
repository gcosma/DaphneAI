# app.py
# Advanced Document Search Engine with RAG + Smart Search

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import modules with error handling
try:
    from modules.core_utils import setup_logging, log_action
    from modules.document_processor import process_uploaded_files, check_dependencies
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Some features may not be available. Please check your installation.")
    
    # Fallback functions
    def setup_logging():
        import logging
        return logging.getLogger(__name__)
    
    def log_action(action, data=None):
        pass
    
    def process_uploaded_files(files):
        return [{'filename': f.name, 'error': 'Processing not available'} for f in files]
    
    def check_dependencies():
        return {}

try:
    from modules.ui.search_components import render_search_interface
    ADVANCED_SEARCH_AVAILABLE = True
except ImportError:
    ADVANCED_SEARCH_AVAILABLE = False

# Setup
st.set_page_config(
    page_title="Advanced Document Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = setup_logging()

def initialize_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.documents = []
        st.session_state.search_results = {}
        st.session_state.search_history = []
        st.session_state.rag_index_built = False
        logger.info("Session state initialized")

def render_header():
    """Application header"""
    st.title("🔍 Advanced Document Search Engine")
    st.markdown("""
    **Intelligent Search with RAG + Smart Pattern Matching**
    
    Upload documents and search with AI-powered semantic understanding or traditional keyword matching.
    """)
    
    # Status indicators in sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Check dependencies
        deps = check_dependencies()
        
        # PDF processing
        if deps.get('pdfplumber') or deps.get('PyPDF2'):
            st.success("✅ PDF Processing Available")
        else:
            st.error("❌ PDF Processing Unavailable")
            st.caption("Install: `pip install pdfplumber`")
        
        # DOCX processing
        if deps.get('python-docx'):
            st.success("✅ DOCX Processing Available")
        else:
            st.warning("⚠️ DOCX Processing Unavailable")
            st.caption("Install: `pip install python-docx`")
        
        # RAG capabilities
        try:
            import sentence_transformers
            st.success("✅ RAG Available")
        except ImportError:
            st.error("❌ RAG Unavailable")
            st.caption("Install: `pip install sentence-transformers torch`")
        
        # Advanced search
        if ADVANCED_SEARCH_AVAILABLE:
            st.success("✅ Advanced Search Available")
        else:
            st.warning("⚠️ Basic Search Only")
        
        st.metric("Documents Loaded", len(st.session_state.documents))
        
        if st.session_state.documents:
            total_words = sum(doc.get('word_count', 0) for doc in st.session_state.documents)
            st.metric("Total Words", f"{total_words:,}")

def render_upload_section():
    """Document upload with progress tracking"""
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, DOCX files"
    )
    
    if uploaded_files:
        # Process documents with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            processed_docs = []
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Process single file
                doc_result = process_uploaded_files([file])
                if doc_result:
                    processed_docs.extend(doc_result)
            
            st.session_state.documents = processed_docs
            st.session_state.rag_index_built = False  # Reset RAG index
            
            progress_bar.progress(1.0)
            status_text.text("✅ All documents processed!")
            
            # Log the upload action
            log_action("documents_uploaded", {
                "count": len(processed_docs),
                "total_words": sum(doc.get('word_count', 0) for doc in processed_docs)
            })
            
            st.success(f"Successfully processed {len(processed_docs)} documents")
            
            # Show document summary
            if processed_docs:
                with st.expander("📊 Document Summary", expanded=False):
                    for doc in processed_docs:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**{doc['filename']}**")
                        with col2:
                            if 'error' in doc:
                                st.error(f"Error: {doc['error']}")
                            else:
                                st.write(f"{doc.get('word_count', 0):,} words")
                        with col3:
                            st.write(f"{doc.get('file_type', 'unknown').upper()}")
                            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            logger.error(f"Upload processing error: {e}")

def render_basic_search(documents):
    """Basic search interface if advanced search is not available"""
    st.header("🔍 Basic Search")
    
    query = st.text_input("Search documents:", placeholder="Enter your search terms...")
    
    if query and documents:
        results = []
        
        for doc in documents:
            if 'text' in doc and query.lower() in doc['text'].lower():
                # Simple relevance scoring
                text_lower = doc['text'].lower()
                query_lower = query.lower()
                
                # Count occurrences
                count = text_lower.count(query_lower)
                
                # Extract snippet
                index = text_lower.find(query_lower)
                start = max(0, index - 100)
                end = min(len(doc['text']), index + len(query) + 100)
                snippet = doc['text'][start:end]
                
                results.append({
                    'document': doc,
                    'count': count,
                    'snippet': snippet
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['count'], reverse=True)
        
        st.write(f"Found {len(results)} results:")
        
        for result in results:
            doc = result['document']
            with st.expander(f"📄 {doc['filename']} ({result['count']} matches)"):
                st.write(f"**File type:** {doc.get('file_type', 'unknown').upper()}")
                st.write(f"**Word count:** {doc.get('word_count', 0):,}")
                st.write("**Preview:**")
                st.write(result['snippet'])

def main():
    """Main application"""
    try:
        initialize_session_state()
        render_header()
        
        # Main layout
        tab1, tab2 = st.tabs(["📁 Upload Documents", "🔍 Search Engine"])
        
        with tab1:
            render_upload_section()
        
        with tab2:
            if st.session_state.documents:
                if ADVANCED_SEARCH_AVAILABLE:
                    # Use advanced search interface
                    render_search_interface(st.session_state.documents)
                else:
                    # Use basic search interface
                    render_basic_search(st.session_state.documents)
            else:
                st.info("👆 Please upload documents first in the Upload tab")
                
                # Sample data option
                if st.button("🎯 Try with Sample Data"):
                    sample_docs = [
                        {
                            'filename': 'sample_policy.txt',
                            'text': 'The government recommends implementing new healthcare policies to improve patient access and reduce waiting times. This policy should be considered for immediate implementation.',
                            'word_count': 25,
                            'file_type': 'txt'
                        },
                        {
                            'filename': 'sample_response.txt', 
                            'text': 'The department accepts the recommendation for healthcare reform. We will establish a task force to oversee the implementation of these critical changes.',
                            'word_count': 24,
                            'file_type': 'txt'
                        }
                    ]
                    st.session_state.documents = sample_docs
                    st.rerun()
        
        # Log app usage
        log_action("app_loaded", {
            "documents": len(st.session_state.documents),
            "has_documents": bool(st.session_state.documents),
            "advanced_search": ADVANCED_SEARCH_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ Application error: {str(e)}")
        
        # Error recovery
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("🐛 Show Debug Info"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
