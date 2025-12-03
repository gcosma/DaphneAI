"""
Streamlit UI package for DaphneAI.

Canonical home of the UI components; kept API-compatible with the former
``modules.ui`` package.
"""

from .search_components import (
    render_search_interface,
    render_recommendation_alignment_interface,
    check_rag_availability,
    filter_stop_words,
    STOP_WORDS,
)

__version__ = "2.0.0"
__author__ = "DaphneAI Team"

__all__ = [
    "render_search_interface",
    "render_recommendation_alignment_interface",
    "check_rag_availability",
    "filter_stop_words",
    "STOP_WORDS",
]
