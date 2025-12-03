# modules/ui/__init__.py
"""
DaphneAI Search UI Components Package

Exports the active search/alignment interfaces and shared utilities.
"""

from .search_components import (
    render_search_interface,
    check_rag_availability,
    filter_stop_words,
    STOP_WORDS
)

__version__ = "2.0.0"
__author__ = "DaphneAI Team"

__all__ = [
    'render_search_interface',
    'check_rag_availability',
    'filter_stop_words',
    'STOP_WORDS'
]