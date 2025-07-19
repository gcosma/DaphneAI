# ===============================================
# FILE: modules/streamlit_components.py (UPDATED MAIN FILE)
# ===============================================

"""
Updated main streamlit_components.py file that imports from the new modular UI structure.
This replaces the old large file and provides backward compatibility.
"""

# Import all UI components from the new modular structure
from .ui import (
    initialize_session_state,
    render_header,
    render_navigation_tabs,
    render_upload_tab,
    render_extraction_tab,
    render_annotation_tab,
    render_matching_tab,
    render_smart_search_tab,
    render_dashboard_tab
)

# Re-export everything for backward compatibility
__all__ = [
    'initialize_session_state',
    'render_header',
    'render_navigation_tabs',
    'render_upload_tab',
    'render_extraction_tab',
    'render_annotation_tab',
    'render_matching_tab',
    'render_smart_search_tab',
    'render_dashboard_tab'
]

# Version information
__version__ = "2.0.0"
__description__ = "Modular UI components for Recommendation-Response Tracker"
