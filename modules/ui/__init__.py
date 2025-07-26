# modules/ui/__init__.py

import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import all UI components with fallbacks
try:
    from .upload_components import render_upload_tab
    UPLOAD_COMPONENTS_AVAILABLE = True
    logging.info("✅ Upload components imported successfully")
except ImportError as e:
    UPLOAD_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import upload components: {e}")
    
    def render_upload_tab():
        st.error("❌ Upload component not available")
        st.info("Create modules/ui/upload_components.py")

# Import enhanced extraction components
try:
    from .extraction_components import (
        render_extraction_tab,
        SmartExtractor,
        EnhancedAIExtractor,
        check_ai_capabilities,
        estimate_openai_cost
    )
    EXTRACTION_COMPONENTS_AVAILABLE = True
    logging.info("✅ Enhanced extraction components imported successfully")
except ImportError as e:
    EXTRACTION_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ Failed to import extraction components: {e}")
    
    def render_extraction_tab():
        st.error("❌ Enhanced extraction component not available")
        st.info("Create modules/ui/extraction_components.py with the enhanced code")

# Import other components with fallbacks
try:
    from .shared_components import render_shared_sidebar
    SHARED_COMPONENTS_AVAILABLE = True
    logging.info("✅ Shared components imported successfully")
except ImportError as e:
    SHARED_COMPONENTS_AVAILABLE = False
    logging.warning(f"⚠️ Shared components not available: {e}")
    
    def render_shared_sidebar():
        st.sidebar.info("Shared components not available")

# Export all components
__all__ = [
    'render_upload_tab',
    'render_extraction_tab', 
    'render_shared_sidebar',
    'SmartExtractor',
    'EnhancedAIExtractor',
    'check_ai_capabilities',
    'estimate_openai_cost',
    'UPLOAD_COMPONENTS_AVAILABLE',
    'EXTRACTION_COMPONENTS_AVAILABLE',
    'SHARED_COMPONENTS_AVAILABLE'
]
