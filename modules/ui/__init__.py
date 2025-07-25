# ===============================================
# FILE: modules/ui/__init__.py
# ===============================================

"""
UI Components Package for Recommendation-Response Tracker

This package contains all the user interface components for the Streamlit application.
Each component is designed to be modular and can function independently with fallbacks.
"""

import streamlit as st
import logging
from typing import List, Dict, Any

__version__ = "2.0.0"
__description__ = "Modular UI components for Streamlit application"

# Initialize logging for UI package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# ===============================================
# COMPONENT AVAILABILITY TRACKING
# ===============================================

# Upload Components (CRITICAL)
try:
    from .upload_components import render_upload_tab
    UPLOAD_COMPONENTS_AVAILABLE = True
    logging.info("✅ Upload components imported successfully")
except ImportError as upload_error:
    UPLOAD_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import upload components: {upload_error}")
    
    def render_upload_tab():
        """Fallback upload tab with basic functionality"""
        st.header("📁 Document Upload & Management")
        st.warning("⚠️ Upload component not available - using fallback mode")
        
        st.markdown(f"""
        **Import Error:** `{str(upload_error)}`
        
        **Troubleshooting:**
        1. Check that `modules/ui/upload_components.py` exists
        2. Verify all dependencies are installed
        3. Check the application logs for detailed errors
        """)
        
        # Basic file uploader fallback
        st.subheader("🚧 Basic File Upload (Fallback Mode)")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload your PDF document here"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info("ℹ️ File uploaded successfully, but full processing requires the upload component to be working.")
            
            # Show file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
                
            # Store in session state
            if 'uploaded_documents' not in st.session_state:
                st.session_state.uploaded_documents = []
            
            # Add to uploaded documents if not already there
            doc_exists = any(doc['name'] == uploaded_file.name for doc in st.session_state.uploaded_documents)
            if not doc_exists:
                st.session_state.uploaded_documents.append({
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': uploaded_file.type,
                    'status': 'uploaded_fallback'
                })
                st.success("📋 Document added to session!")
        
        if st.button("🔄 Retry Import"):
            st.rerun()

# Extraction Components
try:
    from .extraction_components import render_extraction_tab
    EXTRACTION_COMPONENTS_AVAILABLE = True
    logging.info("✅ Extraction components imported successfully")
except ImportError as e:
    EXTRACTION_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import extraction components: {e}")
    
    def render_extraction_tab():
        """Fallback extraction tab"""
        st.header("🔍 Recommendation & Response Extraction")
        st.info("🚧 Extraction component not yet available")
        st.markdown("""
        **What this tab will do:**
        - Extract recommendations from uploaded documents using AI or pattern matching
        - Identify government responses to recommendations
        - Parse structured content from PDFs
        
        **Current status:** Component under development
        """)
        if st.button("📁 Go to Upload Tab"):
            st.rerun()

# Annotation Components
try:
    from .annotation_components import render_annotation_tab
    ANNOTATION_COMPONENTS_AVAILABLE = True
    logging.info("✅ Annotation components imported successfully")
except ImportError as e:
    ANNOTATION_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import annotation components: {e}")
    
    def render_annotation_tab():
        """Fallback annotation tab"""
        st.header("🏷️ Concept Annotation")
        st.info("🚧 Annotation component not yet available")
        st.markdown("""
        **What this tab will do:**
        - Annotate recommendations with conceptual themes using BERT
        - Apply multiple annotation frameworks (I-SIRch, House of Commons, etc.)
        - Generate semantic tags for better analysis
        
        **Current status:** Component under development
        """)

# Matching Components
try:
    from .matching_components import render_matching_tab
    MATCHING_COMPONENTS_AVAILABLE = True
    logging.info("✅ Matching components imported successfully")
except ImportError as e:
    MATCHING_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import matching components: {e}")
    
    def render_matching_tab():
        """Fallback matching tab"""
        st.header("🔗 Find Responses to Recommendations")
        st.info("🚧 Matching component not yet available")
        st.markdown("""
        **What this tab will do:**
        - Match recommendations to their corresponding government responses
        - Use semantic similarity and concept overlap
        - Provide confidence scores for matches
        
        **Current status:** Component under development
        """)

# Search Components
try:
    from .search_components import render_smart_search_tab
    SEARCH_COMPONENTS_AVAILABLE = True
    logging.info("✅ Search components imported successfully")
except ImportError as e:
    SEARCH_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import search components: {e}")
    
    def render_smart_search_tab():
        """Fallback search tab"""
        st.header("🔎 Smart Search")
        st.info("🚧 Search component not yet available")
        st.markdown("""
        **What this tab will do:**
        - Semantic search across all documents
        - RAG-powered question answering
        - Advanced filtering and discovery
        
        **Current status:** Component under development
        """)

# Dashboard Components
try:
    from .dashboard_components import render_dashboard_tab
    DASHBOARD_COMPONENTS_AVAILABLE = True
    logging.info("✅ Dashboard components imported successfully")
except ImportError as e:
    DASHBOARD_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import dashboard components: {e}")
    
    def render_dashboard_tab():
        """Fallback dashboard tab"""
        st.header("📊 Analytics Dashboard")
        st.info("🚧 Dashboard component not yet available")
        
        # Show basic stats from session state
        uploaded_docs = st.session_state.get('uploaded_documents', [])
        if uploaded_docs:
            st.subheader("📈 Basic Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Documents", len(uploaded_docs))
            with col2:
                total_sections = sum(len(doc.get('sections', [])) for doc in uploaded_docs)
                st.metric("📋 Sections", total_sections)
            with col3:
                total_size = sum(doc.get('file_size', 0) for doc in uploaded_docs)
                size_mb = total_size / (1024 * 1024)
                st.metric("💾 Total Size", f"{size_mb:.1f} MB")
        else:
            st.info("📄 No documents uploaded yet. Go to Upload tab to get started.")

# ===============================================
# SHARED UTILITY FUNCTIONS
# ===============================================

def get_component_status():
    """Get the status of all UI components"""
    return {
        'upload': 'available' if UPLOAD_COMPONENTS_AVAILABLE else 'fallback',
        'extraction': 'available' if EXTRACTION_COMPONENTS_AVAILABLE else 'fallback',
        'annotation': 'available' if ANNOTATION_COMPONENTS_AVAILABLE else 'fallback',
        'matching': 'available' if MATCHING_COMPONENTS_AVAILABLE else 'fallback',
        'search': 'available' if SEARCH_COMPONENTS_AVAILABLE else 'fallback',
        'dashboard': 'available' if DASHBOARD_COMPONENTS_AVAILABLE else 'fallback'
    }

def initialize_session_state():
    """Initialize session state with default values"""
    if 'ui_initialized' not in st.session_state:
        st.session_state.ui_initialized = True
        st.session_state.component_status = get_component_status()
        logging.info("✅ UI session state initialized")

# ===============================================
# PACKAGE EXPORTS
# ===============================================

__all__ = [
    # Version info
    '__version__',
    '__description__',
    
    # Component functions
    'render_upload_tab',
    'render_extraction_tab',
    'render_annotation_tab',
    'render_matching_tab',
    'render_smart_search_tab',
    'render_dashboard_tab',
    
    # Utility functions
    'get_component_status',
    'initialize_session_state',
    
    # Availability flags
    'UPLOAD_COMPONENTS_AVAILABLE',
    'EXTRACTION_COMPONENTS_AVAILABLE',
    'ANNOTATION_COMPONENTS_AVAILABLE',
    'MATCHING_COMPONENTS_AVAILABLE',
    'SEARCH_COMPONENTS_AVAILABLE',
    'DASHBOARD_COMPONENTS_AVAILABLE'
]

# Log package initialization
available_count = sum([
    UPLOAD_COMPONENTS_AVAILABLE,
    EXTRACTION_COMPONENTS_AVAILABLE,
    ANNOTATION_COMPONENTS_AVAILABLE,
    MATCHING_COMPONENTS_AVAILABLE,
    SEARCH_COMPONENTS_AVAILABLE,
    DASHBOARD_COMPONENTS_AVAILABLE
])

logging.info(f"✅ UI package initialized - {available_count}/6 components available")
if available_count < 6:
    logging.info("ℹ️ Missing components will use fallback implementations")
