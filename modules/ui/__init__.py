# modules/ui/__init__.py
"""
DaphneAI Search UI Components Package (legacy shim).

This re-exports the canonical ``ui`` package to preserve backward compatibility
with callers that still import via ``modules.ui``.
"""

from ui.search_components import (
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
