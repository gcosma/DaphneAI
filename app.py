# app.py
# Complete Document Search Application

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import modules with error handling
try:
    from modules.core_utils import setup_logging, log_action
    from modules.document_processor import process_uploaded_files
    from modules.ui.search_components import render_search_interface
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    MODULES_AVAILABLE = False

# Setup
st.set_page_config(
    page_title="Document Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.documents = []
        st.session_state.search_results = {}
        st.session_state.search_history = []
        
        if MODULES_AVAILABLE:
            logger = setup_logging()
            logger.info("Session state initialized")

def render_header():
    """Application header"""
    st.title("🔍 Document Search Engine")
    st.markdown("""
    **Upload documents and search with intelligent pattern matching**
    
    Supports PDF, TXT, and DOCX files with smart relevance scoring.
    """)
    
    # Status indicators in sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Check dependencies
        try:
            import pdfplumber
            st.success("✅ PDF Processing Available")
        except ImportError:
            st.error("❌ PDF Processing Unavailable")
            st.caption("Install: `pip install pdfplumber`")
        
        try:
            import docx
            st.success("✅ DOCX Processing Available") 
        except ImportError:
            st.warning("⚠️ DOCX Processing Unavailable")
            st.caption("Install: `pip install python-docx`")
        
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
        if st.button("📤 Process Files", type="primary"):
            # Process documents with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if MODULES_AVAILABLE:
                    # Use proper document processor
                    processed_docs = []
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        doc_result = process_uploaded_files([file])
                        if doc_result:
                            processed_docs.extend(doc_result)
                else:
                    # Fallback simple processing
                    processed_docs = []
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        if file.name.endswith('.txt'):
                            text = file.getvalue().decode('utf-8')
                            processed_docs.append({
                                'filename': file.name,
                                'text': text,
                                'word_count': len(text.split()),
                                'file_type': 'txt'
                            })
                        else:
                            processed_docs.append({
                                'filename': file.name,
                                'error': 'File type not supported in fallback mode',
                                'file_type': file.name.split('.')[-1] if '.' in file.name else 'unknown'
                            })
                
                st.session_state.documents = processed_docs
                
                progress_bar.progress(1.0)
                status_text.text("✅ All documents processed!")
                
                # Log the upload action
                if MODULES_AVAILABLE:
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
                st.write("Error details:", str(e))

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
        
        if results:
            st.write(f"Found {len(results)} results:")
            
            for result in results:
                doc = result['document']
                with st.expander(f"📄 {doc['filename']} ({result['count']} matches)"):
                    st.write(f"**File type:** {doc.get('file_type', 'unknown').upper()}")
                    st.write(f"**Word count:** {doc.get('word_count', 0):,}")
                    st.write("**Preview:**")
                    st.write(result['snippet'])
        else:
            st.info("No results found.")

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
                # Check if we have advanced search available
                if MODULES_AVAILABLE:
                    try:
                        render_search_interface(st.session_state.documents)
                    except Exception as e:
                        st.error(f"Advanced search error: {e}")
                        st.info("Falling back to basic search...")
                        render_basic_search(st.session_state.documents)
                else:
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
        if MODULES_AVAILABLE:
            log_action("app_loaded", {
                "documents": len(st.session_state.documents),
                "has_documents": bool(st.session_state.documents),
                "modules_available": MODULES_AVAILABLE
            })
        
    except Exception as e:
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
