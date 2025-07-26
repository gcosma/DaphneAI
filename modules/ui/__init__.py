# ===============================================
# COMPLETE modules/ui/__init__.py
# ===============================================

import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import all UI components with comprehensive fallbacks
try:
    from .upload_components import render_upload_tab
    UPLOAD_COMPONENTS_AVAILABLE = True
    logging.info("✅ Upload components imported successfully")
except ImportError as e:
    UPLOAD_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import upload components: {e}")
    
    def render_upload_tab():
        st.header("📁 Document Upload")
        st.info("🚧 Upload component not yet available")
        
        # Basic fallback uploader
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload government inquiry reports or response documents"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            st.session_state.uploaded_documents = [
                {
                    'filename': file.name,
                    'size': len(file.read()),
                    'type': 'pdf',
                    'content': file.read().decode('utf-8', errors='ignore') if file.type == 'text/plain' else None,
                    'uploaded_at': datetime.now().isoformat()
                }
                for file in uploaded_files
            ]

# Import enhanced extraction components
try:
    from .extraction_components import (
        render_extraction_tab,
        SmartExtractor,
        get_document_content_for_extraction,
        validate_documents_for_extraction
    )
    EXTRACTION_COMPONENTS_AVAILABLE = True
    logging.info("✅ Enhanced extraction components imported successfully")
except ImportError as e:
    EXTRACTION_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import extraction components: {e}")
    
    def render_extraction_tab():
        st.header("🔍 Recommendation & Response Extraction")
        st.error("❌ Enhanced extraction component not available")
        st.info("Please create modules/ui/extraction_components.py with the enhanced code")

# Import other components with fallbacks
try:
    from .annotation_components import render_annotation_tab
    ANNOTATION_COMPONENTS_AVAILABLE = True
    logging.info("✅ Annotation components imported successfully")
except ImportError as e:
    ANNOTATION_COMPONENTS_AVAILABLE = False
    logging.warning(f"⚠️ Annotation components not available: {e}")
    
    def render_annotation_tab():
        st.header("🏷️ Concept Annotation")
        st.info("🚧 Annotation component not yet available")

try:
    from .matching_components import render_matching_tab
    MATCHING_COMPONENTS_AVAILABLE = True
    logging.info("✅ Matching components imported successfully")
except ImportError as e:
    MATCHING_COMPONENTS_AVAILABLE = False
    logging.warning(f"⚠️ Matching components not available: {e}")
    
    def render_matching_tab():
        st.header("🔗 Response Matching")
        st.info("🚧 Matching component not yet available")

try:
    from .search_components import render_search_tab
    SEARCH_COMPONENTS_AVAILABLE = True
    logging.info("✅ Search components imported successfully")
except ImportError as e:
    SEARCH_COMPONENTS_AVAILABLE = False
    logging.warning(f"⚠️ Search components not available: {e}")
    
    def render_search_tab():
        st.header("🔎 Smart Search")
        st.info("🚧 Search component not yet available")

try:
    from .dashboard_components import render_dashboard_tab
    DASHBOARD_COMPONENTS_AVAILABLE = True
    logging.info("✅ Dashboard components imported successfully")
except ImportError as e:
    DASHBOARD_COMPONENTS_AVAILABLE = False
    logging.warning(f"⚠️ Dashboard components not available: {e}")
    
    def render_dashboard_tab():
        st.header("📊 Analytics Dashboard")
        st.info("🚧 Dashboard component not yet available")

# Session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
        # Document management
        st.session_state.uploaded_documents = []
        
        # Extraction results
        st.session_state.extracted_recommendations = []
        st.session_state.extracted_concerns = []
        st.session_state.extraction_results = {}
        
        # Analysis results
        st.session_state.annotation_results = {}
        st.session_state.matching_results = {}
        st.session_state.search_history = []
        st.session_state.search_results = {}
        
        # Processing states
        st.session_state.processing_status = "idle"
        st.session_state.last_processing_time = None
        st.session_state.error_messages = []
        
        # UI states
        st.session_state.selected_frameworks = []
        st.session_state.current_tab = "upload"
        st.session_state.export_ready = False
        
        logging.info("✅ Session state initialized")

# Export all components
__all__ = [
    'render_upload_tab',
    'render_extraction_tab', 
    'render_annotation_tab',
    'render_matching_tab',
    'render_search_tab',
    'render_dashboard_tab',
    'initialize_session_state',
    'UPLOAD_COMPONENTS_AVAILABLE',
    'EXTRACTION_COMPONENTS_AVAILABLE',
    'ANNOTATION_COMPONENTS_AVAILABLE',
    'MATCHING_COMPONENTS_AVAILABLE',
    'SEARCH_COMPONENTS_AVAILABLE',
    'DASHBOARD_COMPONENTS_AVAILABLE'
]
