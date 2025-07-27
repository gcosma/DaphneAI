# ===============================================
# FILE: modules/ui/__init__.py (COMPLETE FIXED VERSION)
# Main UI module initialization with all components
# ===============================================

import streamlit as st
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# IMPORT ALL UI COMPONENTS WITH FALLBACKS
# ===============================================

# Upload components
try:
    from .upload_components import render_upload_tab
    UPLOAD_COMPONENTS_AVAILABLE = True
    logger.info("✅ Upload components imported successfully")
except ImportError as e:
    UPLOAD_COMPONENTS_AVAILABLE = False
    logger.error(f"❌ Failed to import upload components: {e}")
    
    def render_upload_tab():
        st.header("📁 Document Upload")
        st.error("❌ Upload component not available")
        st.info("Please check modules/ui/upload_components.py")

# Extraction components
try:
    from .extraction_components import render_extraction_tab
    EXTRACTION_COMPONENTS_AVAILABLE = True
    logger.info("✅ Enhanced extraction components imported successfully")
except ImportError as e:
    EXTRACTION_COMPONENTS_AVAILABLE = False
    logger.error(f"❌ Failed to import extraction components: {e}")
    
    def render_extraction_tab():
        st.header("🔍 Content Extraction")
        st.error("❌ Extraction component not available")
        st.info("Please check modules/ui/extraction_components.py")

# Annotation components  
try:
    from .annotation_components import render_annotation_tab
    ANNOTATION_COMPONENTS_AVAILABLE = True
    logger.info("✅ Annotation components imported successfully")
except ImportError as e:
    ANNOTATION_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Annotation components not available: {e}")
    
    def render_annotation_tab():
        st.header("🏷️ Concept Annotation")
        st.info("🚧 Annotation component not yet available")

# Matching components
try:
    from .matching_components import render_matching_tab
    MATCHING_COMPONENTS_AVAILABLE = True
    logger.info("✅ Matching components imported successfully")
except ImportError as e:
    MATCHING_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Matching components not available: {e}")
    
    def render_matching_tab():
        st.header("🔗 Response Matching")
        st.info("🚧 Matching component not yet available")

# Search components
try:
    from .search_components import render_search_tab
    SEARCH_COMPONENTS_AVAILABLE = True
    logger.info("✅ Search components imported successfully")
except ImportError as e:
    SEARCH_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Search components not available: {e}")
    
    def render_search_tab():
        st.header("🔎 Smart Search")
        st.info("🚧 Search component not yet available")

# Dashboard components
try:
    from .dashboard_components import render_dashboard_tab
    DASHBOARD_COMPONENTS_AVAILABLE = True
    logger.info("✅ Dashboard components imported successfully")
except ImportError as e:
    DASHBOARD_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Dashboard components not available: {e}")
    
    def render_dashboard_tab():
        st.header("📊 Analytics Dashboard")
        st.info("🚧 Dashboard component not yet available")

# ===============================================
# CORE UI FUNCTIONS
# ===============================================

def render_header():
    """Render the application header with status indicators"""
    st.set_page_config(
        page_title="DaphneAI - Government Document Analyzer",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ DaphneAI")
    st.markdown("### AI-Powered Government Document Analysis Platform")
    
    # Status metrics
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
    """Render the main navigation tabs"""
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
    """Check the health of all UI components"""
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
        
        logger.info("✅ Session state initialized")

# ===============================================
# SYSTEM STATUS RENDERING
# ===============================================

def render_system_status():
    """Render system status information"""
    if st.checkbox("🔧 Show System Status", key="show_system_status"):
        st.markdown("### 🔧 System Status")
        
        # Component availability
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Component Status")
            components = [
                ("Upload Components", UPLOAD_COMPONENTS_AVAILABLE),
                ("Extraction Components", EXTRACTION_COMPONENTS_AVAILABLE),
                ("Annotation Components", ANNOTATION_COMPONENTS_AVAILABLE),
                ("Matching Components", MATCHING_COMPONENTS_AVAILABLE),
                ("Search Components", SEARCH_COMPONENTS_AVAILABLE),
                ("Dashboard Components", DASHBOARD_COMPONENTS_AVAILABLE)
            ]
            
            for name, available in components:
                status = "✅" if available else "❌"
                st.markdown(f"{status} {name}")
        
        with col2:
            st.markdown("#### Session Data")
            data_items = [
                ("Documents", len(st.session_state.get('uploaded_documents', []))),
                ("Recommendations", len(st.session_state.get('extracted_recommendations', []))),
                ("Responses", len(st.session_state.get('extracted_responses', []))),
                ("Annotations", sum(len(v) for v in st.session_state.get('annotation_results', {}).values())),
                ("Search History", len(st.session_state.get('search_history', [])))
            ]
            
            for name, count in data_items:
                st.markdown(f"📊 {name}: {count}")
        
        # Health check
        health = check_component_health()
        overall_health = health['overall_health']
        
        if overall_health == 'healthy':
            st.success("✅ All systems operational")
        elif overall_health == 'warning':
            st.warning("⚠️ Some components unavailable")
        else:
            st.error("❌ Critical components missing")

# ===============================================
# FALLBACK FUNCTIONS FOR MISSING COMPONENTS
# ===============================================

def render_fallback_header():
    """Fallback header when UI components are not available"""
    st.set_page_config(
        page_title="DaphneAI - Government Document Analyzer",
        page_icon="🏛️",
        layout="wide"
    )
    
    st.title("🏛️ DaphneAI")
    st.markdown("### Government Document Analysis Platform")
    st.error("❌ Some UI components are not available. Please check the logs for details.")

def render_fallback_navigation():
    """Fallback navigation when components are missing"""
    st.markdown("### 🚧 Limited Mode")
    st.info("""
    The application is running in limited mode due to missing components.
    Available components will be shown below.
    """)
    
    # Show only available components
    available_tabs = []
    if UPLOAD_COMPONENTS_AVAILABLE:
        available_tabs.append(("📁 Upload", render_upload_tab))
    if EXTRACTION_COMPONENTS_AVAILABLE:
        available_tabs.append(("🔍 Extract", render_extraction_tab))
    if ANNOTATION_COMPONENTS_AVAILABLE:
        available_tabs.append(("🏷️ Annotate", render_annotation_tab))
    if MATCHING_COMPONENTS_AVAILABLE:
        available_tabs.append(("🔗 Match", render_matching_tab))
    if SEARCH_COMPONENTS_AVAILABLE:
        available_tabs.append(("🔎 Search", render_search_tab))
    if DASHBOARD_COMPONENTS_AVAILABLE:
        available_tabs.append(("📊 Dashboard", render_dashboard_tab))
    
    if available_tabs:
        tab_names = [tab[0] for tab in available_tabs]
        tab_functions = [tab[1] for tab in available_tabs]
        
        tabs = st.tabs(tab_names)
        
        for i, (tab, func) in enumerate(zip(tabs, tab_functions)):
            with tab:
                func()
    else:
        st.error("❌ No UI components are available. Please check the installation.")

# ===============================================
# EXPORTS - ALL REQUIRED FUNCTIONS
# ===============================================

__all__ = [
    # Core UI functions
    'render_header',
    'render_navigation_tabs', 
    'initialize_session_state',
    'check_component_health',
    'render_system_status',
    'render_fallback_header',
    'render_fallback_navigation',
    
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
logger.info("🚀 UI module initialized")
logger.info(f"📊 Available components: {sum([
    UPLOAD_COMPONENTS_AVAILABLE,
    EXTRACTION_COMPONENTS_AVAILABLE, 
    ANNOTATION_COMPONENTS_AVAILABLE,
    MATCHING_COMPONENTS_AVAILABLE,
    SEARCH_COMPONENTS_AVAILABLE,
    DASHBOARD_COMPONENTS_AVAILABLE
])}/6")
