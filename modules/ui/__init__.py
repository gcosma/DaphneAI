# ===============================================
# FILE: modules/ui/__init__.py
# ===============================================

"""
UI Components Package for Recommendation-Response Tracker

This package contains all Streamlit UI components organized by functionality.
"""

# Import all UI components for easy access
from .shared_components import (
    initialize_session_state,
    render_header,
    render_navigation_tabs
)

from .upload_components import render_upload_tab
from .extraction_components import render_extraction_tab
from .annotation_components import render_annotation_tab
from .matching_components import render_matching_tab
from .search_components import render_smart_search_tab
from .dashboard_components import render_dashboard_tab

# Re-export all functions for backward compatibility
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

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Streamlit UI Components for Document Analysis"
