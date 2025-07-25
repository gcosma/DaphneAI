# ===============================================
# FILE: modules/ui/__init__.py - COMPLETE FIXED VERSION
# ===============================================

"""
UI Components package for Recommendation-Response Tracker

This module provides all UI components with robust error handling and fallbacks.
All imports are designed to fail gracefully and provide fallback functionality.
"""

import streamlit as st
import logging
from typing import List, Dict, Any

# ===============================================
# IMPORT SHARED COMPONENTS WITH FALLBACKS
# ===============================================

try:
    from .shared_components import (
        initialize_session_state,
        render_header,
        render_navigation_tabs,
        render_sidebar_info,
        add_error_message,
        show_progress_indicator,
        render_progress_indicator,
        validate_api_keys,
        check_ai_availability,
        show_ai_status_message,
        log_user_action,
        display_debug_info,
        safe_filename,
        format_file_size,
        generate_mock_recommendations,
        generate_mock_annotation
    )
    SHARED_COMPONENTS_AVAILABLE = True
    logging.info("✅ Shared components imported successfully")
    
except ImportError as e:
    SHARED_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import shared components: {e}")
    
    # ===============================================
    # FALLBACK SHARED FUNCTIONS
    # ===============================================
    
    def initialize_session_state():
        """Fallback session state initialization"""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.uploaded_documents = []
            st.session_state.extracted_recommendations = []
            st.session_state.extracted_concerns = []
            st.session_state.annotation_results = {}
            st.session_state.matching_results = {}
            st.session_state.processing_status = "idle"
            st.session_state.error_messages = []
    
    def render_header():
        """Fallback header rendering"""
        st.title("📋 Recommendation-Response Tracker")
        st.markdown("**AI-Powered Document Analysis System** for UK Government Inquiries")
    
    def render_navigation_tabs():
        """Fallback navigation tabs"""
        return st.tabs([
            "📁 Upload", 
            "🔍 Extract", 
            "🏷️ Annotate", 
            "🔗 Match", 
            "🔎 Search", 
            "📊 Dashboard"
        ])
    
    def render_sidebar_info():
        """Fallback sidebar"""
        with st.sidebar:
            st.header("⚙️ System Status")
            st.info("Running in fallback mode")
    
    def show_progress_indicator(current=None, total=None, message="Processing..."):
        """Fallback progress indicator"""
        if current is not None and total is not None:
            if total > 0:
                progress = current / total
                st.progress(progress, text=f"{message}: {current}/{total}")
            else:
                st.info(f"{message}...")
            return None
        else:
            return st.spinner(message)
    
    def render_progress_indicator(current: int, total: int, description: str = "Processing"):
        """Fallback progress indicator"""
        if total > 0:
            progress = current / total
            st.progress(progress, text=f"{description}: {current}/{total}")
        else:
            st.info(f"{description}...")
    
    def add_error_message(message: str, error_type: str = "error"):
        """Fallback error message"""
        if error_type == "error":
            st.error(message)
        elif error_type == "warning":
            st.warning(message)
        elif error_type == "info":
            st.info(message)
        else:
            st.success(message)
    
    def validate_api_keys():
        """Fallback API key validation"""
        return False
    
    def check_ai_availability():
        """Fallback AI availability check"""
        return False
    
    def show_ai_status_message():
        """Fallback AI status"""
        st.info("🔧 Running in fallback mode")
    
    def log_user_action(action: str, details: str = ""):
        """Fallback logging"""
        logging.info(f"User Action: {action} - {details}")
    
    def display_debug_info():
        """Fallback debug info"""
        with st.expander("🐛 Debug Information"):
            st.json({
                'mode': 'fallback',
                'shared_components_available': False
            })
    
    def safe_filename(filename: str) -> str:
        """Fallback safe filename"""
        import re
        return re.sub(r'[^\w\-_\.]', '_', filename)
    
    def format_file_size(size_bytes: int) -> str:
        """Fallback file size formatting"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024*1024:
            return f"{size_bytes/1024:.1f} KB"
        else:
            return f"{size_bytes/(1024*1024):.1f} MB"
    
    def generate_mock_recommendations(doc_name: str, num_recommendations: int = 3):
        """Fallback mock recommendations"""
        return [
            {
                'id': f"REC-{i+1}",
                'text': f"Mock recommendation {i+1} from {doc_name}",
                'source': doc_name,
                'confidence': 0.8
            }
            for i in range(num_recommendations)
        ]
    
    def generate_mock_annotation(text: str, frameworks: List[str]):
        """Fallback mock annotation"""
        return {}, {}

# ===============================================
# IMPORT TAB COMPONENTS WITH FALLBACKS
# ===============================================

# Upload Components
try:
    from .upload_components import render_upload_tab
    UPLOAD_COMPONENTS_AVAILABLE = True
    logging.info("✅ Upload components imported successfully")
except ImportError as e:
    UPLOAD_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import upload components: {e}")
    
    def render_upload_tab():
        """Fallback upload tab"""
        st.header("📁 Document Upload & Management")
        st.error("❌ Upload component not available")
        st.info("""
        **Troubleshooting:**
        1. Check that `modules/ui/upload_components.py` exists
        2. Verify all dependencies are installed
        3. Check the application logs for detailed errors
        """)

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
        st.error("❌ Extraction component not available")
        st.info("Please check that the extraction_components.py file exists and dependencies are installed.")

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
        st.error("❌ Annotation component not available")
        st.info("Please check that the annotation_components.py file exists and dependencies are installed.")

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
        st.error("❌ Matching component not available")
        st.info("Please check that the matching_components.py file exists and dependencies are installed.")

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
        st.error("❌ Search component not available")
        st.info("Please check that the search_components.py file exists and dependencies are installed.")

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
        st.error("❌ Dashboard component not available")
        st.info("Please check that the dashboard_components.py file exists and dependencies are installed.")

# ===============================================
# MODULE STATUS AND HEALTH CHECK
# ===============================================

def get_module_status():
    """Get the status of all UI modules"""
    return {
        'shared_components': SHARED_COMPONENTS_AVAILABLE,
        'upload_components': UPLOAD_COMPONENTS_AVAILABLE,
        'extraction_components': EXTRACTION_COMPONENTS_AVAILABLE,
        'annotation_components': ANNOTATION_COMPONENTS_AVAILABLE,
        'matching_components': MATCHING_COMPONENTS_AVAILABLE,
        'search_components': SEARCH_COMPONENTS_AVAILABLE,
        'dashboard_components': DASHBOARD_COMPONENTS_AVAILABLE
    }

def render_module_health_check():
    """Render a health check for all modules"""
    st.subheader("🏥 Module Health Check")
    
    status = get_module_status()
    
    for module_name, is_available in status.items():
        if is_available:
            st.success(f"✅ {module_name.replace('_', ' ').title()}")
        else:
            st.error(f"❌ {module_name.replace('_', ' ').title()}")
    
    total_available = sum(status.values())
    total_modules = len(status)
    
    if total_available == total_modules:
        st.success(f"🎉 All {total_modules} modules are working correctly!")
    elif total_available > 0:
        st.warning(f"⚠️ {total_available}/{total_modules} modules are working. Some features may be limited.")
    else:
        st.error("❌ No modules are working correctly. Please check your installation.")

# ===============================================
# EXPORT ALL FUNCTIONS
# ===============================================

__all__ = [
    # Core functions
    'initialize_session_state',
    'render_header',
    'render_navigation_tabs',
    'render_sidebar_info',
    
    # Utility functions
    'add_error_message',
    'show_progress_indicator',
    'render_progress_indicator',
    'validate_api_keys',
    'check_ai_availability',
    'show_ai_status_message',
    'log_user_action',
    'display_debug_info',
    'safe_filename',
    'format_file_size',
    'generate_mock_recommendations',
    'generate_mock_annotation',
    
    # Tab rendering functions
    'render_upload_tab',
    'render_extraction_tab',
    'render_annotation_tab',
    'render_matching_tab',
    'render_smart_search_tab',
    'render_dashboard_tab',
    
    # Health and status functions
    'get_module_status',
    'render_module_health_check'
]

# ===============================================
# MODULE INFORMATION
# ===============================================

__version__ = "2.1.0"
__description__ = "Robust UI components for Recommendation-Response Tracker with comprehensive error handling"
__author__ = "Recommendation-Response Tracker Team"

# Log successful initialization
logging.info(f"✅ UI module initialized successfully (version {__version__})")
logging.info(f"📊 Module status: {sum(get_module_status().values())}/{len(get_module_status())} components available")
