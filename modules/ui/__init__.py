# modules/ui/__init__.py
"""
DaphneAI Search UI Components Package

This package provides enhanced document search capabilities including:
- Advanced search methods (Smart, Exact, Fuzzy, AI Semantic, Hybrid)
- Government recommendation-response alignment
- Rich result display and export functionality
"""

# Import main interfaces for easy access
from .search_interface import render_search_interface
from .recommendation_alignment import render_recommendation_alignment_interface

# Import commonly used utilities
from .search_utils import (
    check_rag_availability,
    filter_stop_words,
    STOP_WORDS
)

# Version information
__version__ = "2.0.0"
__author__ = "DaphneAI Team"

# Export main functions
__all__ = [
    'render_search_interface',
    'render_recommendation_alignment_interface',
    'check_rag_availability',
    'filter_stop_words',
    'STOP_WORDS'
]
