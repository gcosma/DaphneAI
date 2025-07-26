# modules/ui/__init__.py
# COMPLETE FIXED VERSION - All missing components added

import streamlit as st
import logging

# ===============================================
# CORE UI FUNCTIONS - THESE WERE MISSING
# ===============================================

def render_header():
    """Render the application header - THIS WAS MISSING"""
    st.set_page_config(
        page_title="DaphneAI - Government Document Analyzer",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main header
    st.title("🏛️ DaphneAI")
    st.markdown("### AI-Powered Government Document Analysis Platform")
    
    # Status bar
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        uploaded_count = len(st.session_state.get('uploaded_documents', []))
        st.metric("📁 Documents", uploaded_count)
    
    with col2:
        rec_count = len(st.session_state.get('extracted_recommendations', []))
        st.metric("📋 Recommendations", rec_count)
    
    with col3:
        resp_count = len(st.session_state.get('extracted_responses', []))
        st.metric("💬 Responses", resp_count)
    
    with col4:
        status = st.session_state.get('processing_status', 'idle')
        status_emoji = "✅" if status == "complete" else "⏳" if status == "processing" else "⭕"
        st.metric("⚡ Status", f"{status_emoji} {status.title()}")
    
    st.divider()

def render_navigation_tabs():
    """Render the main navigation tabs - THIS WAS MISSING"""
    # Create the main navigation
    tabs = st.tabs([
        "📁 Upload Documents",
        "🔍 Extract Content", 
        "🏷️ Concept Annotation",
        "🔗 Find Responses",
        "🔎 Smart Search",
        "📊 Analytics Dashboard"
    ])
    
    # Store current tab in session state
    tab_names = ["upload", "extraction", "annotation", "matching", "search", "dashboard"]
    
    with tabs[0]:
        st.session_state.current_tab = "upload"
        render_upload_tab()
    
    with tabs[1]:
        st.session_state.current_tab = "extraction"
        render_extraction_tab()
    
    with tabs[2]:
        st.session_state.current_tab = "annotation"
        render_annotation_tab()
    
    with tabs[3]:
        st.session_state.current_tab = "matching"
        render_matching_tab()
    
    with tabs[4]:
        st.session_state.current_tab = "search"
        render_search_tab()
    
    with tabs[5]:
        st.session_state.current_tab = "dashboard"
        render_dashboard_tab()

def check_component_health():
    """Check the health of all UI components - THIS WAS MISSING"""
    health_status = {
        'upload_components': UPLOAD_COMPONENTS_AVAILABLE,
        'extraction_components': EXTRACTION_COMPONENTS_AVAILABLE,
        'annotation_components': ANNOTATION_COMPONENTS_AVAILABLE,
        'matching_components': MATCHING_COMPONENTS_AVAILABLE,
        'search_components': SEARCH_COMPONENTS_AVAILABLE,
        'dashboard_components': DASHBOARD_COMPONENTS_AVAILABLE,
        'overall_health': 'healthy'
    }
    
    # Calculate overall health
    available_count = sum(1 for status in health_status.values() if isinstance(status, bool) and status)
    total_count = sum(1 for status in health_status.values() if isinstance(status, bool))
    
    if available_count == total_count:
        health_status['overall_health'] = 'healthy'
    elif available_count >= total_count * 0.7:
        health_status['overall_health'] = 'warning'
    else:
        health_status['overall_health'] = 'critical'
    
    return health_status

# ===============================================
# IMPORT COMPONENTS WITH FALLBACKS
# ===============================================

# Upload components
try:
    from .upload_components import render_upload_tab
    UPLOAD_COMPONENTS_AVAILABLE = True
    logging.info("✅ Upload components imported successfully")
except ImportError as e:
    UPLOAD_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import upload components: {e}")
    
    def render_upload_tab():
        st.header("📁 Document Upload")
        st.error("❌ Upload component not available")
        st.info("Please check modules/ui/upload_components.py")

# Extraction components
try:
    from .extraction_components import render_extraction_tab
    EXTRACTION_COMPONENTS_AVAILABLE = True
    logging.info("✅ Enhanced extraction components imported successfully")
except ImportError as e:
    EXTRACTION_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import extraction components: {e}")
    
    def render_extraction_tab():
        st.header("🔍 Recommendation & Response Extraction")
        st.error("❌ Enhanced extraction component not available")
        st.info("Please create modules/ui/extraction_components.py with the enhanced code")

# Annotation components
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

# Matching components
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

# Search components - FIX THE FUNCTION NAME
try:
    from .search_components import render_smart_search_tab
    # Create alias for consistency
    render_search_tab = render_smart_search_tab
    SEARCH_COMPONENTS_AVAILABLE = True
    logging.info("✅ Search components imported successfully")
except ImportError as e:
    SEARCH_COMPONENTS_AVAILABLE = False
    logging.warning(f"⚠️ Search components not available: {e}")
    
    def render_search_tab():
        st.header("🔎 Smart Search")
        st.info("🚧 Search component not yet available")

# Dashboard components
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

# ===============================================
# SESSION STATE INITIALIZATION
# ===============================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
        # Document management
        st.session_state.uploaded_documents = []
        
        # Extraction results
        st.session_state.extracted_recommendations = []
        st.session_state.extracted_responses = []
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
        
        # Vector store and AI components
        st.session_state.vector_store_manager = None
        st.session_state.rag_engine = None
        st.session_state.bert_annotator = None
        
        logging.info("✅ Session state initialized")

# ===============================================
# EXPORTS - ALL REQUIRED FUNCTIONS
# ===============================================

__all__ = [
    # Core UI functions
    'render_header',
    'render_navigation_tabs', 
    'initialize_session_state',
    'check_component_health',
    
    # Tab rendering functions
    'render_upload_tab',
    'render_extraction_tab', 
    'render_annotation_tab',
    'render_matching_tab',
    'render_search_tab',
    'render_dashboard_tab',
    
    # Component availability flags
    'UPLOAD_COMPONENTS_AVAILABLE',
    'EXTRACTION_COMPONENTS_AVAILABLE',
    'ANNOTATION_COMPONENTS_AVAILABLE',
    'MATCHING_COMPONENTS_AVAILABLE',
    'SEARCH_COMPONENTS_AVAILABLE',
    'DASHBOARD_COMPONENTS_AVAILABLE'
]

# Log component initialization status
logging.info("🚀 UI module initialized")
logging.info(f"📊 Available components: {sum([
    UPLOAD_COMPONENTS_AVAILABLE,
    EXTRACTION_COMPONENTS_AVAILABLE, 
    ANNOTATION_COMPONENTS_AVAILABLE,
    MATCHING_COMPONENTS_AVAILABLE,
    SEARCH_COMPONENTS_AVAILABLE,
    DASHBOARD_COMPONENTS_AVAILABLE
])}/6")
